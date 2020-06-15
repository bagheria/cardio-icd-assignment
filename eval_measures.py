import numpy as np
import pandas as pd


data = pd.read_csv("D:/Github/ICD10 Classification/icd_selected_data_multilabel.csv", encoding="ISO-8859-1")
df_multi_label = pd.DataFrame(data)


def eval_measures_multi_label(prob_predicted, y_train, patient_id_test, threshold=0.08, level='icdcode'):
    labels = list(set(y_train))
    dict_labels = {i: labels[i] for i in range(0, len(labels))}
    predicted_labels = []

    # Assign the threshold in such a way that the label cardinality for the test set is
    # in the same order as the label cardinality in the training set
    for i in range(prob_predicted.shape[0]):
        lst = [np.argwhere(prob_predicted[i] > threshold)]
        lst_val = np.vectorize(dict_labels.get)(lst)[0]
        predicted_labels.append(lst_val)

    # A dataset containing patient id + corresponding predicted chapter labels
    predicted_set = list(zip(patient_id_test, predicted_labels))
    # new X_test for multi_label removing duplicates

    multi_label_patient_id = df_multi_label['ID']
    true_labels = []
    for j in range(df_multi_label.shape[0]):
        lst = df_multi_label[level][j].split(",")
        true_labels.append(lst)

    # A dataset containing patient id + corresponding true chapter labels
    true_set = list(zip(multi_label_patient_id, true_labels))
    lengths = [len(true_labels[i]) for i in range(len(true_labels))]
    label_cardinality = np.mean(lengths)
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
