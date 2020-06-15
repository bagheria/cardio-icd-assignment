import pandas as pd
import apyori


data_arm = pd.read_csv("Data/data_arm.csv")
records = []
for i in range(0, data_arm.shape[0]):
    records.append([str(data_arm.values[i, j]) for j in range(1, data_arm.shape[1] - 1)])

for i, j in enumerate(records):
    while 'nan' in records[i]:
        records[i].remove('nan')

# results = list(apyori.apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2))
results = list(apyori.apriori(records, min_support=0.0045, min_confidence=0.1, min_lift=3, min_length=2))

for item in results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    # print("Rule: " + items[0] + " -> " + items[1])
    print("Rule: " + str(list(item.ordered_statistics[0].items_base)) + " -> " +
          str(list(item.ordered_statistics[0].items_add)))
    # second index of the inner list
    print("Support: " + str(item[1]))
    # third index of the list located at 0th
    # of the third index of the inner list
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
