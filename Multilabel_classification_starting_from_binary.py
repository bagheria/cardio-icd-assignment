from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def plot_coefficients(classifier, feature_names, icd, top_features=20):
    coef = classifier.coef_.ravel()
    coef = np.delete(coef, -1)
    coef = np.delete(coef, -1)
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["darkgoldenrod" if c < 0 else "seagreen" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    # feature_names = np.array(feature_names)
    # key_list = list(vectorizer.vocabulary_.keys())
    # val_list = list(vectorizer.vocabulary_.values())
    key_list = list(feature_names.keys())
    val_list = list(feature_names.values())
    # N = [key_list[val_list.index(top_positive_coefficients[i])] for i in range(top_features)]
    plt.xticks(np.arange(0, 0 + 2 * top_features),
               [key_list[val_list.index(top_coefficients[i])] for i in range(2*top_features)],
               rotation=60, ha='right')
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(linestyle='--', linewidth='0.15', color='gray')
    plt.ylabel("Coefficients", fontsize=10)
    # plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    plt.title(icd)
    plt.savefig("D:/Github/ICD10 Classification/" + icd + ".png", bbox_inches="tight")
    plt.show()


items = "E11", "E78", "I10", "I21", "I25", "I42", "I48", "I50", "N18", "Z95"
icd = items[9]
data = pd.read_csv("D:/Github/ICD10 Classification/Data/" + icd + ".csv")
df = pd.DataFrame(data)
text_df = df['text'].values.tolist()
text_df = pd.Series((v for v in text_df))
labels_code = df[icd].values.tolist()
cols = ['age_norm', 'gender']
vars_df = df[cols]
x_train, x_test, y_train, y_test, vars_df_train, vars_df_test = \
    train_test_split(text_df, labels_code, vars_df, test_size=0.20, random_state=0)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(x_train)
df4 = pd.DataFrame(X.toarray())
train_data = pd.concat([df4.reset_index(drop=True), vars_df_train.reset_index(drop=True)], axis=1)
svc = LinearSVC(random_state=0)
svc.fit(train_data, y_train)
plot_coefficients(svc, vectorizer.vocabulary_, icd)

svm = OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(random_state=0)))
svm.fit(train_data, y_train)
vectorizer2 = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
X_t = vectorizer2.fit_transform(x_test)
df4_t = pd.DataFrame(X_t.toarray())
test_data = pd.concat([df4_t.reset_index(drop=True), vars_df_test.reset_index(drop=True)], axis=1)
p_label = svm.predict(test_data)
pd.DataFrame(p_label).to_csv("D:/Github/ICD10 Classification/Output with medic/pr" + icd + ".csv")
pd.DataFrame(y_test).to_csv("D:/Github/ICD10 Classification/Output with medic/tr" + icd + ".csv")


pipe1 = Pipeline([('TFidf', TfidfVectorizer()),
                  ("SVMProb", OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(random_state=0))))])

pipe1.fit(x_train, y_train)
predicted_label = pipe1.predict(x_test)
conf_matrix = confusion_matrix(y_test, predicted_label)
pd.DataFrame(predicted_label).to_csv("D:/Github/ICD10 Classification/Outputsamenvatting/pr" + icd + ".csv")
pd.DataFrame(y_test).to_csv("D:/Github/ICD10 Classification/Outputsamenvatting/tr" + icd + ".csv")

# ----
# predicted_label = pipe1.predict(text_df)
# pd.DataFrame(predicted_label).to_csv("D:/Github/ICD10 Classification/Output/pr_whole_" + icd + ".csv")
# pd.DataFrame(labels_code).to_csv("D:/Github/ICD10 Classification/Output/tr_whole_" + icd + ".csv")

print("done!")
