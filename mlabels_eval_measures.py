import numpy as np
import pandas as pd


data = pd.read_csv("D:/Github/ICD10 Classification/icd_selected_data_multilabel.csv", encoding="ISO-8859-1")
df_multi_label = pd.DataFrame(data)


def eval_measures_multi_label(prob_predicted, labels, patient_id_test, threshold=0.08, level='icdcode'):
    # labels = list(set(labels))
    dict_labels = {i: labels[i] for i in range(0, len(labels))}
    # index_dict = [i for i in range(len(dict_labels)) if dict_labels[i]=='I10']
    index_E11 = 0
    index_E78 = 1
    index_I10 = 2
    index_I21 = 3
    index_I25 = 4
    index_I42 = 5
    index_I48 = 6
    index_I50 = 7
    index_N18 = 8
    index_Z95 = 9

    # Apply apriori rules
    for i in range(prob_predicted.shape[0]):
        # find max and second max icd probs
        temp_array = prob_predicted[i]
        temp_array.sort()
        max_1 = temp_array[-1]
        max_2 = temp_array[-2]
        ind_max_1 = np.argwhere(prob_predicted[i] == max_1)
        ind_max_2 = np.argwhere(prob_predicted[i] == max_2)
        max_1 = np.vectorize(dict_labels.get)(ind_max_1)[0]
        max_2 = np.vectorize(dict_labels.get)(ind_max_2)[0]

        # two itemset rules
        # if (max_1 == 'E78' and max_2 == 'I48') or (max_2 == 'E78' and max_1 == 'I48'):
        #     prob_predicted[i][index_I10] = prob_predicted[i][index_I10]*1.67
        #
        # if (max_1 == 'E78' and max_2 == 'Z95') or (max_2 == 'E78' and max_1 == 'Z95'):
        #     prob_predicted[i][index_I10] = prob_predicted[i][index_I10]*1.56
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25]*1.69
        #
        # if (max_1 == 'E78' and max_2 == 'I25') or (max_2 == 'E78' and max_1 == 'I25'):
        #     prob_predicted[i][index_I10] = prob_predicted[i][index_I10]*1.54
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95]*1.44
        #
        # if (max_1 == 'E78' and max_2 == 'I10') or (max_2 == 'E78' and max_1 == 'I10'):
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25]*1.31
        #
        # if (max_1 == 'I25' and max_2 == 'I48') or (max_2 == 'I25' and max_1 == 'I48'):
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95]*1.7
        #
        # if (max_1 == 'I25' and max_2 == 'I10') or (max_2 == 'I25' and max_1 == 'I10'):
        #     prob_predicted[i][index_E78] = prob_predicted[i][index_E78]*1.97
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95]*1.36
        #
        # if (max_1 == 'I50' and max_2 == 'I25') or (max_2 == 'I50' and max_1 == 'I25'):
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95]*1.68
        #
        # if (max_1 == 'I50' and max_2 == 'Z95') or (max_2 == 'I50' and max_1 == 'Z95'):
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25]*1.43
        #
        # if (max_1 == 'I50' and max_2 == 'I48') or (max_2 == 'I50' and max_1 == 'I48'):
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95]*1.49
        #
        # if (max_1 == 'I50' and max_2 == 'I10') or (max_2 == 'I50' and max_1 == 'I10'):
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25]*1.32
        #
        # if (max_1 == 'Z95' and max_2 == 'I10') or (max_2 == 'Z95' and max_1 == 'I10'):
        #     prob_predicted[i][index_E78] = prob_predicted[i][index_E78]*1.9
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25]*1.52
        #
        # if (max_1 == 'Z95' and max_2 == 'I48') or (max_2 == 'Z95' and max_1 == 'I48'):
        #     prob_predicted[i][index_I50] = prob_predicted[i][index_I50]*1.52
        #
        # if (max_1 == 'Z95' and max_2 == 'I25') or (max_2 == 'Z95' and max_1 == 'I25'):
        #     prob_predicted[i][index_I50] = prob_predicted[i][index_I50]*1.5
        #
        # # one itemset rules
        # if max_1 == 'E78':
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25] * 1.33
        #     prob_predicted[i][index_I10] = prob_predicted[i][index_I10] * 1.56
        #
        # if max_1 == 'E11':
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25] * 1.36
        #     prob_predicted[i][index_I10] = prob_predicted[i][index_I10] * 1.4
        #
        # if max_1 == 'I42':
        #     prob_predicted[i][index_I50] = prob_predicted[i][index_I50] * 1.91
        #
        # if max_1 == 'I21':
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25] * 1.39
        #
        # if max_1 == 'I50':
        #     prob_predicted[i][index_Z95] = prob_predicted[i][index_Z95] * 1.36
        #
        # if max_1 == 'Z95':
        #     prob_predicted[i][index_I25] = prob_predicted[i][index_I25] * 1.3
        #
        # if max_1 == 'I10':
        #     prob_predicted[i][index_E78] = prob_predicted[i][index_E78] * 1.56

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
