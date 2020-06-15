import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx


data_arm = pd.read_csv("D:/Github/ICD10 Classification/pred_set_arm.csv")
records = []
for i in range(0, data_arm.shape[0]):
    records.append([str(data_arm.values[i, j]) for j in range(1, data_arm.shape[1] - 1)])

for i, j in enumerate(records):
    while 'nan' in records[i]:
        records[i].remove('nan')


items = "I25", "I10", "I48", "I50", "E78", "E11", "I42", "I21", "N18", "Z95"

encoded_vals = []
for index, row in data_arm.iterrows():
    labels = {}
    uncommons = list(set(items) - set(row))
    commons = list(set(items).intersection(row))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)


freq_items = apriori(ohe_df, min_support=0.05, use_colnames=True, verbose=1)
freq_items.head(7)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.1)
print(rules.head(50))

subset = rules[rules['conviction'] > 1.2]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(subset)

print((subset.sort_values(['conviction', 'lift'], ascending=(False, False))).to_string())
subset.to_csv("D:/Github/ICD10 Classification/pred_set_arm_out.csv")

z = subset['support']
y = subset['confidence']
plt.scatter(z, y, alpha=0.5, color='black')
n = np.arange(0, 12, 1)
for i, txt in enumerate(n):
    plt.annotate('R' + str(txt), (z.iloc[i] + 0.001, y.iloc[i] + 0.001))

plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()


def draw_graph(rules_g, rules_to_show, unique_items):
    g1 = nx.DiGraph()
    color_map = []
    n = 50
    colors = np.random.rand(n)

    for j in range(rules_to_show):
        g1.add_nodes_from(["R" + str(j)])
        for a in rules_g.iloc[j]['antecedents']:
            g1.add_nodes_from([a])
            g1.add_edge(a, "R" + str(j), color=colors[j], weight=2)
        for c in rules_g.iloc[j]['consequents']:
            g1.add_nodes_from([c])
            g1.add_edge("R" + str(j), c, color=colors[j], weight=2)

    for node in g1:
        found_a_string = False
        for item in unique_items:
            if node == item:
                found_a_string = True
        if found_a_string:
            color_map.append('red')
        else:
            color_map.append('black')

    edges = g1.edges()
    colors = [g1[u][v]['color'] for u, v in edges]
    weights = [g1[u][v]['weight'] for u, v in edges]

    pos = nx.spring_layout(g1, k=16, scale=1)
    nx.draw(g1, pos, edges=edges, node_color=color_map, edge_color=colors, width=weights, font_size=16,
            with_labels=False)

    for p in pos:  # raise text positions
        pos[p][1] += 0.07
    nx.draw_networkx_labels(g1, pos)
    plt.show()

draw_graph(subset, 12, items)
