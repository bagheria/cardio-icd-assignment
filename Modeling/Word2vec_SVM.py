import numpy as np
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# Sentence level embedding
class AverageEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = 100  # because we use 100 embedding points

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def eval_measures_multi_label(prob_predicted, y_train, threshold=0.08, level='chapter'):
    labels = list(set(y_train))
    dict_labels = {i: labels[i] for i in range(0, len(labels))}
    predicted_labels = []

    # Assign the threshold in such a way that the label cardinality for the test set is
    # in the same order as the label cardinality in the training set
    for i in range(X_test.shape[0]):
        lst = [np.argwhere(prob_predicted[i] > threshold)]
        lst_val = np.vectorize(dict_labels.get)(lst)[0]
        predicted_labels.append(lst_val)

    # A dataset containing patient id + corresponding predicted chapter labels
    predicted_set = list(zip(patient_id_test, predicted_labels))
    # new X_test for multi_label removing duplicates

    multi_label_patient_id = df_multi_label['PATIENT_ID_VOORKEUR']
    true_labels = []
    for j in range(df_multi_label.shape[0]):
        lst = df_multi_label[level][j].split(",")
        true_labels.append(lst)

    # A dataset containing patient id + corresponding true chapter labels
    true_set = list(zip(multi_label_patient_id, true_labels))
    sum_a = 0
    sum_p = 0
    sum_r = 0
    sum_f = 0
    n = 0
    for i in range(len(predicted_set)):
        # predicted_set[i][0] is the patient id
        pid = predicted_set[i][0]
        for j in range(len(true_set)):
            if true_set[j][0] == pid:
                YZ_intersection = len(np.intersect1d(predicted_set[i][1], true_set[j][1]))
                YZ_union = len(np.union1d(predicted_set[i][1], true_set[j][1]))
                Yi = len(predicted_set[i][1])
                Zi = len(true_set[j][1])
                break

        n = n + 1
        sum_a += YZ_intersection / YZ_union
        sum_p += YZ_intersection / Zi
        sum_r += YZ_intersection / Yi
        sum_f += (2 * YZ_intersection) / (Yi + Zi)

    acc = sum_a / n
    pre = sum_p / n
    rec = sum_r / n
    f1 = sum_f / n
    return acc, pre, rec, f1


data = pd.read_csv("Data/icd_data_singlelabel.csv")
df_single_label = pd.DataFrame(data)
print(df_single_label.head())

data = pd.read_csv("Data/icd_data_multilabel.csv", encoding="ISO-8859-1")
df_multi_label = pd.DataFrame(data)
print(df_multi_label.head())

text_df = df_single_label['deidentified'].values.tolist()
text_df = pd.Series((v for v in text_df))

labels_chapter = df_single_label['Diagnosechapter'].values.tolist()
labels_digit = df_single_label['DiagnosecodeerstNEW'].values.tolist()
labels_code = df_single_label['Diagnosecode'].values.tolist()
patient_id = df_single_label['PATIENT_ID_VOORKEUR'].values.tolist()

print('Split the data set: train and test')

X_train, X_test, y_train_chapter, y_test_chapter, y_train_digit, y_test_digit, \
y_train_code, y_test_code, patient_id_train, patient_id_test = \
    train_test_split(text_df, labels_chapter, labels_digit, labels_code, patient_id, test_size=0.25, random_state=0)


# Model2: Word2Vec with SVM
# combined_df = train_data['comment_text'].append(test_data['comment_text'])
Vocab_list = (text_df.apply(lambda x: str(x).strip().split()))
models = Word2Vec(Vocab_list, size=300, min_count=2)
WordVectorz = dict(zip(models.wv.index2word, models.wv.vectors))

AE = AverageEmbeddingVectorizer(WordVectorz)
pipe2 = Pipeline([("wordVectz", AE),
                  ("SVMProb", CalibratedClassifierCV(LinearSVC(random_state=0, max_iter=2000)))])


# training chapters
pipe2.fit(X_train, y_train_chapter)
predicted_chapter = pipe2.predict(X_test)

# single-label case
print("Classification report for the test set on chapter level:")
print("# single-label case")
print("accuracy = ", accuracy_score(y_test_chapter, predicted_chapter))
print("precision, recall, f-measure = ",
      precision_recall_fscore_support(y_test_chapter, predicted_chapter, average='micro'))

# multi-label case
print("# multi-label case")
prob_predicted_chapter = pipe2.predict_proba(X_test)
accuracy, precision, recall, f1_score = eval_measures_multi_label(prob_predicted_chapter,
                                                                  y_test_chapter,
                                                                  threshold=0.08,
                                                                  level='chapter')
print("accuracy, precision, recall, f-measure = ", accuracy, precision, recall, f1_score)

# training 3digits
pipe2.fit(X_train, y_train_digit)
predicted_digit = pipe2.predict(X_test)

# single-label case
print("Classification report for the test set on digit level")
print("# single-label case")
print("accuracy = ", accuracy_score(y_test_digit, predicted_digit))
print("precision, recall, f-measure = ",
      precision_recall_fscore_support(y_test_digit, predicted_digit, average='micro'))

# multi-label case
print("# multi-label case")
prob_predicted_3digit = pipe2.predict_proba(X_test)
accuracy, precision, recall, f1_score = eval_measures_multi_label(prob_predicted_3digit,
                                                                  y_test_digit,
                                                                  threshold=0.009,
                                                                  level='rolledup')
print("accuracy, precision, recall, f-measure = ", accuracy, precision, recall, f1_score)

# training full codes
pipe2.fit(X_train, y_train_code)
predicted_code = pipe2.predict(X_test)

# single-label case
print("Classification report for the test set on code level")
print("# single-label case")
print("accuracy = ", accuracy_score(y_test_code, predicted_code))
print("precision, recall, f-measure = ",
      precision_recall_fscore_support(y_test_code, predicted_code, average='micro'))

# multi-label case
print("# multi-label case")
prob_predicted_code = pipe2.predict_proba(X_test)
accuracy, precision, recall, f1_score = eval_measures_multi_label(prob_predicted_code,
                                                                  y_test_code,
                                                                  threshold=0.009,
                                                                  level='fullcode')
print("accuracy, precision, recall, f-measure = ", accuracy, precision, recall, f1_score)
