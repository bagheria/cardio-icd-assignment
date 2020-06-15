import numpy as np
import pandas as pd


data = pd.read_csv("Data/true_set.csv", encoding="ISO-8859-1")
obs = data['ID']
true_labels = []
for j in range(data.shape[0]):
    lst = data["labels"][j].split(",")
    true_labels.append(lst)
true_set = list(zip(obs, true_labels))

data = pd.read_csv("Data/pred_set.csv", encoding="ISO-8859-1")
obs = data['ID']
pred_labels = []
for j in range(data.shape[0]):
    lst = data["labels"][j].split(",")
    pred_labels.append(lst)
pred_set = list(zip(obs, pred_labels))


lengths = [len(true_labels[i]) for i in range(len(true_labels))]
label_cardinality = np.mean(lengths)
print("label cardinality for true set = ", label_cardinality)

lengths = [len(pred_labels[i]) for i in range(len(pred_labels))]
label_cardinality = np.mean(lengths)
print("label cardinality for pred set = ", label_cardinality)

items = "E11", "E78", "I10", "I21", "I25", "I42", "I48", "I50", "N18", "Z95"
sum_sen = 0
sum_spe = 0
sum_ppv = 0
sum_npv = 0
sum_f1 = 0
n = 0
for i in range(len(pred_set)):
    # predicted_set[i][0] is the combined patient letter id
    pid = pred_set[i][0]
    for j in range(len(true_set)):
        if true_set[j][0] == pid:
            YZ_intersection = len(np.intersect1d(pred_set[i][1], true_set[j][1]))
            tn = len(np.intersect1d(np.setdiff1d(items, pred_set[i][1]), np.setdiff1d(items, true_set[j][1])))
            sp_dn = len(np.setdiff1d(items, true_set[j][1]))
            npv_dn = len(np.setdiff1d(items, pred_set[i][1]))
            Yi = len(pred_set[i][1])
            Zi = len(true_set[j][1])
            break
    n = n + 1
    sum_sen += YZ_intersection / Zi
    sum_spe += tn / sp_dn
    sum_ppv += YZ_intersection / Yi
    sum_npv += tn / npv_dn
    sum_f1 += (2 * YZ_intersection) / (Yi + Zi)

sensitivity = sum_sen / n
specificity = sum_spe / n
ppv = sum_ppv / n
npv = sum_npv / n
f1_score = sum_f1 / n

print(sensitivity, specificity, ppv, npv, f1_score)

def eval_measures_multi_label(prob_predicted, labels, patient_id_test, threshold=0.08, level='icdcode'):

    # Assign the threshold in such a way that the label cardinality for the test set is
    # in the same order as the label cardinality in the training set
    predicted_labels = []
    for i in range(prob_predicted.shape[0]):
        lst = [np.argwhere(prob_predicted[i] > threshold)]
        lst_val = np.vectorize(dict_labels.get)(lst)[0]
        # temp_array = prob_predicted[i]
        # temp_array.sort()
        # max_1 = temp_array[-1]
        # max_2 = temp_array[-2]
        # max_3 = temp_array[-3]
        # ind_max_1 = np.argwhere(prob_predicted[i] == max_1)
        # ind_max_2 = np.argwhere(prob_predicted[i] == max_2)
        # ind_max_3 = np.argwhere(prob_predicted[i] == max_3)
        # max_1 = np.vectorize(dict_labels.get)(ind_max_1)[0]
        # max_2 = np.vectorize(dict_labels.get)(ind_max_2)[0]
        # max_3 = np.vectorize(dict_labels.get)(ind_max_3)[0]
        # lst_val = []
        # lst_val.extend((max_1, max_2, max_3))
        predicted_labels.append(lst_val)

    # A dataset containing patient id + corresponding predicted chapter labels
    # predicted_set = list(zip(patient_id_test, predicted_labels))
    # new X_test for multi_label removing duplicates

    multi_label_patient_id = df_multi_label['ID']
    true_labels = []
    for j in range(df_multi_label.shape[0]):
        lst = df_multi_label[level][j].split(",")
        true_labels.append(lst)

    # A dataset containing patient id + corresponding true chapter labels
    predicted_set = list(zip(multi_label_patient_id, predicted_labels))
    true_set = list(zip(multi_label_patient_id, true_labels))
    lengths = [len(true_labels[i]) for i in range(len(true_labels))]
    label_cardinality = np.mean(lengths)
    print("label cardinality = ", label_cardinality)

    sum_sen = 0
    sum_spe = 0
    sum_ppv = 0
    sum_npv = 0
    sum_f1 = 0
    n = 0
    for i in range(len(predicted_set)):
        # predicted_set[i][0] is the patient id
        pid = predicted_set[i][0]
        for j in range(len(true_set)):
            if true_set[j][0] == pid:
                YZ_intersection = len(np.intersect1d(predicted_set[i][1], true_set[j][1]))
                tn = len(np.intersect1d(np.setdiff1d(labels, predicted_set[i][1]), np.setdiff1d(labels, true_set[j][1])))
                sp_dn = len(np.setdiff1d(labels, true_set[j][1]))
                npv_dn = len(np.setdiff1d(labels, predicted_set[i][1]))
                Yi = len(predicted_set[i][1])
                Zi = len(true_set[j][1])
                break
        n = n + 1
        sum_sen += YZ_intersection / Zi
        sum_spe += tn / sp_dn
        sum_ppv += YZ_intersection / Yi
        sum_npv += tn / npv_dn
        sum_f1 += (2 * YZ_intersection) / (Yi + Zi)

    sensitivity = sum_sen / n
    specificity = sum_spe / n
    ppv = sum_ppv / n
    npv = sum_npv / n
    f1_score = sum_f1 / n
    return sensitivity, specificity, ppv, npv, f1_score
