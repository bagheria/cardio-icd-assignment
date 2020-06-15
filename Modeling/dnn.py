import csv
import re
from nltk.corpus import stopwords
import numpy as np
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from sklearn.metrics import classification_report
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
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


def readMyFile(filename):
    text = []
    categories = []
    dict = {}

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if row[1] in dict:
                ind = dict[row[1]]
            else:
                dict_count = len(dict)
                new_item = {row[1]: dict_count}
                dict.update(new_item)
                ind = dict_count

            categories.append(ind)
            text.append(row[0] + " ")

    return categories, text


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
data = pd.read_csv("Data/icd.csv")
df1 = pd.DataFrame(data)

new_texts = df1['deidentified'].values.tolist()
# labels = df1['Diagnosecodeerst'].values.tolist()
labels = df1['Diagnoseclass'].values.tolist()

texts = []
for txt in new_texts:
    # txt = preprocess2(txt)
    texts.append(txt)

MAX_SEQUENCE_LENGTH = 200  # max_len
MAX_NB_WORDS = 20000       # max_features
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(texts)
texts_sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

texts_sequences = np.asarray(texts_sequences)
labels = pd.get_dummies(labels)
labels = np.asarray(labels)
print('Shape of data tensor:', texts_sequences.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(texts_sequences.shape[0])
np.random.shuffle(indices)
data = texts_sequences[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

# n-grams
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))

ngram_range = 1
if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than MAX_NB_WORDS in order
    # to avoid collision with existing features.
    start_index = MAX_NB_WORDS + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # MAX_NB_WORDS is the highest integer that could be found in the dataset.
    MAX_NB_WORDS = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
print('Number of samples in the training and validation sets are:')
print(y_train.sum(axis=0))
print(y_test.sum(axis=0))


batch_size = 32
epochs = 5
model = Sequential()

# we start off with an embedding layer which maps vocab indices into embedding_dims dimensions
model.add(Embedding(MAX_NB_WORDS,  # len(word_index) + 1,
                    EMBEDDING_DIM,
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=True))  # trainable=False

# GlobalAveragePooling1D will average the embeddings of all words in the document
# model.add(GlobalAveragePooling1D())

model.add(Dropout(0.15))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Dropout(0.15))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
# from keras.layers import LSTM
# lstm_output_size = 70
# model.add(LSTM(lstm_output_size))
model.add(Dense(128, activation='relu'))
# We project onto a single unit output layer, and squash it with an activation function
model.add(Dense(3, activation='softmax'))  # , kernel_regularizer=l2(0.01))) # l2-svm
# squared_hinge is svm loss function
model.compile(loss='squared_hinge',  # 'categorical_crossentropy',
              optimizer='adam',  # optimizer='rmsprop'
              metrics=['accuracy'])
print("model fitting - convolutional neural network")
model.summary()

# patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))  # callbacks=[es, mc])

model.save('Data/my_model.h5')

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Validation Accuracy: {:.4f}".format(accuracy))
# print("Validation Loss:  {:.4f}".format(loss))
y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
# y_pred ... top three for multi-label classification
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(np.argmax(y_test, axis=1),
                            y_pred_bool))
print(confusion_matrix(np.argmax(y_test, axis=1, out=None),
                       y_pred_bool))
report_with_auc = class_report(y_true=np.argmax(y_test, axis=1),
                               y_pred=y_pred_bool,
                               y_score=y_pred)

print(report_with_auc)
roc_curve_multiclass(y_true=y_test,
                     y_pred=y_pred,
                     n_classes=8)
