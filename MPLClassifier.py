from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import sqlite3
from sklearn import metrics
import warnings
from datetime import datetime
import random
from itertools import combinations
from random import randint
from joblib import dump, load
from multiprocessing import cpu_count
from random import shuffle
from multiprocessing import Pool
from time import time
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)

def perform_ml():


    def find_cutoff():
        buy_cutoff = .02
        while True:
            train, test = prepare_data(buy_cutoff)
            print('starting', buy_cutoff)
            clf = train_model(train)

            predict(clf, test)
            mean, num_trades = get_results(clf, test)
            #scaler = int(num_trades/250)+1
            scaler = 1
            print('finding cutoff', mean, num_trades, buy_cutoff)
            if num_trades<150:
                print('Found cutoff.',mean, num_trades, buy_cutoff)
                return buy_cutoff, mean*100

            buy_cutoff = round(buy_cutoff + (.01*scaler), 4)


    def pre_prepare_data():
        this_df = df[features + ['10 Day Change Abnormal', '10 Day Change', 'Date Reported']]
        #this_df = df

        this_df = this_df.replace('-', np.nan)
        this_df = this_df.replace([np.inf, -np.inf], np.nan)
        this_df = this_df.dropna()
        return this_df


    def prepare_data(buy_cutoff):
        this_df['is_train'] = True
        #this_df['is_train'].values[this_df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = False
        this_df.loc[this_df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d'), ['is_train']] = False
        this_df['Action'] = 'None'
        #this_df['Action'].values[this_df['10 Day Change Abnormal'].values > buy_cutoff] = "Buy"
        this_df.loc[this_df['10 Day Change Abnormal'] > buy_cutoff, ['Action']] = "Buy"

        this_df['Action'] = this_df['Action'].astype('category')
        this_df["Action Code"] = this_df["Action"].cat.codes




        train, test = this_df[this_df['is_train']==True], this_df[this_df['is_train']==False]
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        scaler.fit(train[features])
        train[features] = scaler.transform(train[features])
        test[features] = scaler.transform(test[features])
        #print(train[['10 Day Change Abnormal','Action']])
        #print(test[['10 Day Change Abnormal','Action']])


        return train, test


    def train_model(train):
        #clf = ExtraTreesClassifier(n_jobs=int(cpu_count()/4), n_estimators=500)
        #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        clf = MLPClassifier()
        y = train['Action Code']
        clf.fit(train[features], y)
        return clf

    def predict(clf, test):
        preds = clf.predict(test[features])
        preds = pd.DataFrame(preds).astype(str)
        preds.columns = ['Predicted']
        preds = preds.replace('0','Buy').replace('1','None')
        test['Predicted'] = list(preds['Predicted'])


    def get_results(clf, test):
        #feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)

        chosen = test[test['Predicted']=='Buy']
        mean_return = round(chosen['10 Day Change'].mean()*100,4)
        print(mean_return, len(chosen))
        return mean_return, len(chosen)

    def get_total_results(means_list, num_trades_list):
        avg_return = round(sum(means_list)/len(means_list),4)
        avg_num_trades = round(sum(num_trades_list)/len(num_trades_list),4)
        return_stdev = round(np.std(means_list),4)
        trades_stdev = round(np.std(num_trades_list),4)
        out_df = pd.DataFrame([[buy_cutoff, avg_return, return_stdev, num_trades, trades_stdev, str(means_list), str(num_trades_list), str(features)]])
        out_df.columns = ['Buy Cutoff', 'Avg Return', 'Stdev Return', 'Num Trades', 'Stdev Num Trades', 'Returns', 'Trades', 'Features']
        return out_df


    start_time = time()

    #features = list(features)
    #features.append('Before and After')

    conn = sqlite3.connect('earnings.db', timeout=120)
    df = pd.read_sql('select * from aggregated_data_adjusted', conn, parse_dates = ['Date Reported'])
    features = list(df.columns)
    remove_list = ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                      '10 Day Change Abnormal',
                      'Date Reported', 'Time Reported',
                      'Symbol', 'Market Cap Text']
    for remove_me in remove_list:
        features.remove(remove_me)

    means_list = []
    num_trades_list = []
    iterations = 5
    this_df = pre_prepare_data()

    buy_cutoff, mean = find_cutoff()

    train, test = prepare_data(buy_cutoff)
    for i in range(iterations):
        clf = train_model(train)
        predict(clf, test)
        mean, num_trades = get_results(clf, test)
        print('run ', i, mean, num_trades)
        means_list.append(mean)
        num_trades_list.append(num_trades)

        out_df = get_total_results(means_list, num_trades_list)
        print('result',out_df)
        out_df.to_sql('MLPClassifier', conn, if_exists='append')
    end_time = time()
    total_time = ((end_time - start_time) * 32751)/4
    total_time = total_time/60/60

    print(total_time)

def get_combinations():
    conn = sqlite3.connect('earnings.db', timeout=120)
    df = pd.read_sql('select * from optimized_results_with_selection_extratrees_adjusted', conn)

    df = df[df['Avg Return']>2.25]
    feature_list = []
    for index, row in df.iterrows():
        for feature in eval(row['Features']):
            feature_list.append(feature)
    feature_list = list(set(feature_list))
    feature_list.remove('Before and After')
    feature_combinations = []
    for i in range(2,len(feature_list)):
        for combo in list(combinations(feature_list, i)):
            feature_combinations.append(combo)
    print('Num of models:', len(feature_combinations))
    shuffle(feature_combinations)

    return feature_combinations


if __name__ == '__main__':


    feature_combinations = get_combinations()

    #features = ['Before and After', 'Average Abnormal Change 10 Days', 'Revenue Actual', 'Revenue Num of Estimates']
    #p = Pool(4)
    #p.map(perform_ml, feature_combinations)
    perform_ml()
