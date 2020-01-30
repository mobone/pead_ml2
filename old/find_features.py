import pandas as pd
import sqlite3
from time import sleep
from itertools import combinations
conn = sqlite3.connect('earnings.db', timeout=120)


df = pd.read_sql('select * from optimized_results_with_selection_extratrees_adjusted', conn)
for cutoff in [2,2.25, 2.5,2.75, 3]:
    this_df = df[df['Avg Return']>cutoff]
    feature_list = []
    for index, row in this_df.iterrows():
        for feature in eval(row['Features']):
            feature_list.append(feature)
    feature_list = list(set(feature_list))

    feature_combinations = []
    for i in range(2,len(feature_list)):
        for combo in list(combinations(feature_list, i)):
            feature_combinations.append(combo)
    print(cutoff, len(feature_list), len(feature_combinations))
