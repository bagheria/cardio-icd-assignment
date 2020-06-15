import input_data
import mlabels_eval_measures
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd


data = pd.read_csv("D:/Github/ICD10 Classification/icd_selected_data_singlelabel_edited.csv")
df_single_label = pd.DataFrame(data)

text_df = df_single_label['samenvatting'].values.tolist()
text_df = pd.Series((v for v in text_df))

labels_code = df_single_label['icdcode'].values.tolist()

# ------ Model1: TFiDf with SVM
pipe1 = Pipeline([('TFidf', TfidfVectorizer()),
                  ("SVMProb", OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(random_state=0))))])

pipe1.fit(text_df, labels_code)
predicted_code = pipe1.predict_proba(text_df)

items = "E11", "E78", "I10", "I21", "I25", "I42", "I48", "I50", "N18", "Z95"

print('Reports on test code')
print("multi-label case:")
threshold = 0.08
for t in range(10):
    print("threshold", threshold)
    print("\n sensitivity, specificity, ppv, npv, f1_score = ",
          mlabels_eval_measures.eval_measures_multi_label(predicted_code,
                                                          items,
                                                          input_data.patient_id_train,
                                                          threshold=threshold,
                                                          level='icdcode'))
    threshold += 0.01

print("Done!")


