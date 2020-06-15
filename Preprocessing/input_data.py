import re
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from scipy import interp
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


os.environ['KERAS_BACKEND'] = 'tensorflow'
stopword_set = set(stopwords.words("dutch"))


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def preprocess(raw_text):
    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    # convert to lower case and split
    words = letters_only_text.lower().split()
    # remove stopwords
    meaningful_words = [w for w in words if w not in stopword_set]
    # join the cleaned words in a list
    cleaned_word_list = " ".join(meaningful_words)

    return cleaned_word_list


def preprocess2(raw_text):
    stopword_set_ = set(stopwords.words("dutch"))
    return " ".join([i for i in re.sub(r'[^a-zA-Z\s]', "", raw_text).lower().split() if i not in stopword_set_])


def get_index(key):
    if dict.has_key(key):
        return dict[key]
    else:
        dict_count = dict.values
        new_item = {key: dict_count}
        dict.update(new_item)
        return new_item


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum()
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int),
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(),
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(),
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"],
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


def roc_curve_multiclass(y_true, y_pred, n_classes):
    from scipy import interp
    import matplotlib.pyplot as plt
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc

    # Plot linewidth
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


run_model = 1
data = pd.read_csv("Data/I48.csv")
df1 = pd.DataFrame(data)

# age and sex
cols = ['age_norm', 'gender']
vars_dataframe = df1[cols]

new_texts = df1['text'].values.tolist()
# labels_chapter = df1['Diagnosechapter'].values.tolist()
# labels_digit = df1['Diagnosecodeerst'].values.tolist()
labels_code = df1['I48'].values.tolist()

unique_labels = np.unique(labels_code)

texts = []
for txt in new_texts:
    # txt = preprocess2(txt)
    texts.append(txt)

MAX_SEQUENCE_LENGTH = 156  # avg_len
MAX_NB_WORDS = 10000       # max_features
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 64
EPOCHS = 10


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(texts)
texts_sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

texts_sequences = np.asarray(texts_sequences)

indices = np.arange(texts_sequences.shape[0])
np.random.shuffle(indices)
data = texts_sequences[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
patient_id_test = df1['ID'].values.tolist()[-nb_validation_samples:]
patient_id_train = df1['ID'].values.tolist()[:-nb_validation_samples]

# # labels: labels_chapter
# labels_chapter = pd.get_dummies(labels_chapter)
# labels_chapter = np.asarray(labels_chapter)
# print('Shape of data tensor:', texts_sequences.shape)
# print('Shape of label tensor:', labels_chapter.shape)
#
# labels_chapter = labels_chapter[indices]
# y_train_chapter = labels_chapter[:-nb_validation_samples]
# y_test_chapter = labels_chapter[-nb_validation_samples:]
#
# # labels: labels_digit
# labels_digit = pd.get_dummies(labels_digit)
# labels_digit = np.asarray(labels_digit)
# print('Shape of label tensor:', labels_digit.shape)
# labels_digit = labels_digit[indices]
# y_train_digit = labels_digit[:-nb_validation_samples]
# y_test_digit = labels_digit[-nb_validation_samples:]

# labels_code: labels_code
labels_code = pd.get_dummies(labels_code)
labels_code = np.asarray(labels_code)
print('Shape of label tensor:', labels_code.shape)
labels_code = labels_code[indices]
y_train_code = labels_code[:-nb_validation_samples]
y_test_code = labels_code[-nb_validation_samples:]

# .....
x_train = data
patient_id_train = df1['ID'].values.tolist()
y_train_code = labels_code
# .....

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')


# def vectorize_sequences(sequences, dimension=10000):
#     results = np.zeros((sequences.shape[0], dimension))
#     for i in range(len(sequences)):
#         results[i, sequences[[i]]] = 1
#     return results


x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
print('Number of samples in the training and validation sets are:')
print(y_train_code.sum(axis=0))
print(y_test_code.sum(axis=0))
