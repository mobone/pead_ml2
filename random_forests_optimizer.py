from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)

class perform_ml():
    def __init__(self, df):

        self.df = df

        self.conn = sqlite3.connect('earnings.db', timeout=120)
        self.features = list(self.df.columns)
        #print(self.features)

        for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                          '10 Day Change Abnormal',
                          'Date Reported', 'Time Reported',
                          'Symbol', 'Market Cap Text']:

            self.features.remove(remove_me)

        self.first_run = True
        self.max_means = -90
        self.iterations = 5
        self.start_feature_imp = [0]
        while True:
            self.buy_cutoff = .03
            self.cutoff_found = False
            self.test_df = df
            self.prepare_data()

            self.means = []
            self.num_trades = []
            self.accuracys = []

            self.current_means = []
            self.current_num_trades = []
            self.current_accuracys = []
            # TODO: if we keep a feature, start over again
            print('======================')
            print('using features', self.features)

            for i in range(self.iterations):
                num_trades = 500

                mean = self.find_cutoff()
                if mean<0 and self.first_run == False:
                    break

                self.prepare_data()
                self.train_model()
                self.predict()

                mean_return,  num_trades, accuracy = self.get_results(self.test)

                self.means.append(mean_return)
                self.num_trades.append(num_trades)
                self.accuracys.append(accuracy)

                mean_return,  num_trades, accuracy = self.get_results(self.test_2019)

                self.current_means.append(mean_return)
                self.current_num_trades.append(num_trades)
                self.current_accuracys.append(accuracy)

            if self.first_run:
                #print('starting result', this_runs_avg, this_runs_num_trades,self.buy_cutoff, self.means)
                self.store_results()
                self.start_feature_imp = list(self.feature_imp.keys())
                self.start_feature_imp.insert(0, 'Before and After')
                self.initial_feature_imp = self.start_feature_imp.copy()
                self.features = []
                self.first_run = False

                self.add_feature()
                continue

            self.store_results()

            #self.add_feature()
            try:
                if self.this_runs_avg>self.max_means:
                    self.max_means = self.this_runs_avg
                    self.start_feature_imp = self.initial_feature_imp.copy()

                    self.add_feature()
                else:
                    self.remove_added_feature()
                    self.add_feature()
            except:
                break


    def find_cutoff(self):
        while True:

            if self.cutoff_found == True:
                mean_return = 1
                break
            self.prepare_data()
            self.train_model()
            self.predict()
            mean_return, num_trades, accuracy = self.get_results(self.test)
            scaler = int(num_trades/250)+1

            #print('finding cutoff', mean, num_trades, self.buy_cutoff, scaler)

            if num_trades<300:
                print('found cutoff')
                self.cutoff_found = True
                break

            self.buy_cutoff = round(self.buy_cutoff + (.005*scaler), 4)
        return mean_return


    def store_results(self):
        try:
            self.this_runs_avg = sum(self.means)/self.iterations
            this_runs_num_trades = sum(self.num_trades)/self.iterations
            accuracy = sum(self.accuracys)/self.iterations
            stddev = np.std(self.means)

            current_avg = sum(self.current_means)/self.iterations
            current_num_trades = sum(self.current_num_trades)/self.iterations
            current_accuracy = sum(self.current_accuracys)/self.iterations
            current_stddev = np.std(self.current_means)
            #print(self.this_runs_avg, this_runs_num_trades, self.buy_cutoff, self.means, stddev, self.this_years_avg, accuracy)

            out_df = pd.DataFrame([[self.this_runs_avg, stddev, this_runs_num_trades, accuracy, current_avg, current_stddev, current_num_trades, current_accuracy, self.buy_cutoff, str(self.means), str(self.num_trades), str(self.features)]])

            out_df.columns = ['Avg Return', 'Std Dev', 'Avg Num Trades', 'Accuracy', 'Current Avg Return', 'Current Std Dev', 'Current Avg Num Trades', 'Current Accuracy', 'Buy Cutoff', 'Returns','Num Trades', 'Features']
            print(out_df)
            print(self.max_means)
            #self.test.to_csv('test.csv')
            #input()

            out_df.to_sql('current_predictions', self.conn, if_exists='append')
        except Exception as e:
            print(e)
            pass

    def remove_added_feature(self):

        print('removing added feature ', self.feature_added)
        self.features.remove(self.feature_added)

    # TODO: add two features at a time
    def add_feature(self):

        self.feature_added = self.start_feature_imp.pop(0)
        while self.feature_added in self.features:
            print('not adding feature', self.feature_added, 'as it already exists')
            self.feature_added = self.start_feature_imp.pop(0)
        print('adding feature', self.feature_added)
        self.features.append(self.feature_added)


    def prepare_data(self):
        self.test_df['is_train'] = 'Train'
        self.test_df['is_train'].values[(self.test_df['Date Reported'] >= datetime.strptime('2018-01-01', '%Y-%m-%d')) & (self.test_df['Date Reported'] <= datetime.strptime('2018-12-31', '%Y-%m-%d'))] = 'Test 2018'
        self.test_df['is_train'].values[self.test_df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = 'Test 2019'

        self.test_df['Action'] = 'None'
        self.test_df['Action'].values[self.test_df['10 Day Change Abnormal'].values > self.buy_cutoff] = "Buy"
        self.test_df['Action'] = self.test_df['Action'].astype('category')
        self.test_df["Action Code"] = self.test_df["Action"].cat.codes


        self.test_df = self.test_df[self.features + ['Action', 'Action Code', 'is_train', '10 Day Change Abnormal', '10 Day Change', 'Date Reported', 'Symbol']]


        self.test_df = self.test_df.replace('-', np.nan)
        self.test_df = self.test_df.replace([np.inf, -np.inf], np.nan)
        self.test_df = self.test_df.dropna()

        self.train, self.test = self.test_df[self.test_df['is_train']=='Train'], self.test_df[self.test_df['is_train']=='Test 2018']

        self.train_2019 = pd.concat([self.test_df[self.test_df['is_train']=='Train'],self.test_df[self.test_df['is_train']=='Test 2018']])
        self.test_2019 = self.test_df[self.test_df['is_train']=='Test 2019']




    def train_model(self, fast=False):

        self.clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)

        #self.clf = RandomForestClassifier(n_jobs=-1)
        y = self.train['Action Code']

        train = self.train[self.features]
        self.clf.fit(train, y)

        self.clf_2019 = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)

        #self.clf = RandomForestClassifier(n_jobs=-1)
        y = self.train_2019['Action Code']

        train = self.train_2019[self.features]
        self.clf_2019.fit(train, y)

    def predict(self):
        preds = self.clf.predict(self.test[self.features])
        preds = pd.DataFrame(preds).astype(str)
        preds.columns = ['Predicted']
        preds = preds.replace('0','Buy').replace('1','None')
        self.test['Predicted'] = list(preds['Predicted'])

        preds = self.clf_2019.predict(self.test_2019[self.features])
        preds = pd.DataFrame(preds).astype(str)
        preds.columns = ['Predicted']
        preds = preds.replace('0','Buy').replace('1','None')
        self.test_2019['Predicted'] = list(preds['Predicted'])


    def get_results(self, test_data):
        self.feature_imp = pd.Series(self.clf.feature_importances_,index=self.features).sort_values(ascending=False)

        if self.first_run:
            print(self.feature_imp)

        chosen = test_data[test_data['Predicted']=='Buy']
        mean_return = round(chosen['10 Day Change'].mean()*100,4)

        accuracy = len(chosen[chosen['10 Day Change']>0])/float(len(chosen))

        return mean_return, len(chosen), accuracy






conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data_adjusted_ta', conn, parse_dates = ['Date Reported'])

df = df.sort_values(by='Date Reported')



#print(df)

perform_ml(df)
