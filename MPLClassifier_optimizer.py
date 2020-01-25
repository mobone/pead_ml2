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
        max_means = -90
        self.iterations = 5

        self.start_feature_imp = self.find_features()

        while True:
            self.buy_cutoff = .005
            self.cutoff_found = False
            self.test_df = df
            self.prepare_data()

            self.means = []
            self.num_trades = []
            # TODO: if we keep a feature, start over again
            print('======================')
            if self.first_run:
                self.features=['Before and After']
            print('using features', self.features)

            for i in range(self.iterations):
                num_trades = 300

                mean = self.find_cutoff()
                if mean<0 and self.first_run == False:
                    break

                self.prepare_data()
                self.train_model()
                self.predict()

                mean_return, num_trades = self.get_results()
                self.means.append(mean_return)
                self.num_trades.append(num_trades)

            if self.first_run:
                #print('starting result', this_runs_avg, this_runs_num_trades,self.buy_cutoff, self.means)
                self.store_results()
                #self.start_feature_imp = list(self.feature_imp.keys())
                self.initial_feature_imp = self.start_feature_imp.copy()
                self.features = []
                self.first_run = False

                self.add_feature()
                continue

            self.store_results()

            #self.add_feature()
            try:
                if self.this_runs_avg>max_means:
                    max_means = self.this_runs_avg
                    self.start_feature_imp = self.initial_feature_imp.copy()

                    self.add_feature()
                else:
                    self.remove_added_feature()
                    self.add_feature()
            except:
                break

    def find_features(self):
        conn = sqlite3.connect('earnings.db', timeout=120)
        df = pd.read_sql('select * from aggregated_data_adjusted_ta', conn, parse_dates = ['Date Reported'])

        df = df.sort_values(by='Date Reported')
        df = df.replace('-', np.nan)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        features = list(df.columns)
        for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                          'Symbol', '10 Day Change Abnormal',
                          'Date Reported', 'Time Reported', 'Market Cap Text']:

            features.remove(remove_me)
        df['Action'] = 'Dont Buy'
        df['Action'].values[df['10 Day Change Abnormal'].values > .06] = "Buy"
        df['Action'] = df['Action'].astype('category')
        df["Action Code"] = df["Action"].cat.codes

        X = df[features]
        y = df['Action Code']



        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier()
        model.fit(X,y)
        print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
        #plot graph of feature importances for better visualization
        feat_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(feat_importances)

        return list(feat_importances.keys())


    def find_cutoff(self):
        while True:

            if self.cutoff_found == True:
                mean = 1
                break
            self.prepare_data(init_run = True)
            self.train_model()
            self.predict()
            mean, num_trades = self.get_results()
            #scaler = int(num_trades/250)+1
            scaler = 1
            print('finding cutoff', mean, num_trades, self.buy_cutoff, scaler)

            if num_trades<200:
                print('found cutoff', mean, num_trades)
                self.cutoff_found = True
                break

            self.buy_cutoff = round(self.buy_cutoff + (.001*scaler), 4)
        return mean


    def store_results(self):
        try:
            self.this_runs_avg = sum(self.means)/self.iterations
            this_runs_num_trades = sum(self.num_trades)/self.iterations

            stddev = np.std(self.means)
            print(self.this_runs_avg, this_runs_num_trades, self.buy_cutoff, self.means, stddev)

            out_df = pd.DataFrame([[self.this_runs_avg, stddev, this_runs_num_trades, self.buy_cutoff, str(self.means), str(self.num_trades), str(self.features)]])

            out_df.columns = ['Avg Return', 'Std Dev', 'Avg Num Trades', 'Buy Cutoff', 'Returns','Num Trades', 'Features']
            print(out_df)

            out_df.to_sql('MPLClassifier_optimized', self.conn, if_exists='append')
        except:
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


    def prepare_data(self, init_run=False):
        self.test_df['is_train'] = True
        self.test_df['is_train'].values[self.test_df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = False
        self.test_df['Action'] = 'None'
        self.test_df['Action'].values[self.test_df['10 Day Change Abnormal'].values > self.buy_cutoff] = "Buy"
        self.test_df['Action'] = self.test_df['Action'].astype('category')
        self.test_df["Action Code"] = self.test_df["Action"].cat.codes


        self.test_df = self.test_df[self.features + ['Action', 'Action Code', 'is_train', '10 Day Change Abnormal', '10 Day Change', 'Date Reported']]


        self.test_df = self.test_df.replace('-', np.nan)
        self.test_df = self.test_df.replace([np.inf, -np.inf], np.nan)
        self.test_df = self.test_df.dropna()





        self.train, self.test = self.test_df[self.test_df['is_train']==True], self.test_df[self.test_df['is_train']==False]

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        scaler.fit(self.train[self.features])
        self.train[self.features] = scaler.transform(self.train[self.features])
        self.test[self.features] = scaler.transform(self.test[self.features])


    def train_model(self, fast=False):

        #print(self.train[self.features+['Action Code']])
        y = self.train['Action Code']

        train = self.train[self.features]

        self.clf = MLPClassifier()

        self.clf.fit(train, y)

    def predict(self):
        preds = self.clf.predict(self.test[self.features])
        preds = pd.DataFrame(preds).astype(str)
        preds.columns = ['Predicted']
        preds = preds.replace('0','Buy').replace('1','None')
        self.test['Predicted'] = list(preds['Predicted'])


    def get_results(self):
        #self.feature_imp = pd.Series(self.clf.feature_importances_,index=self.features).sort_values(ascending=False)

        #if self.first_run:
        #    print(self.feature_imp)

        chosen = self.test[self.test['Predicted']=='Buy']
        mean_return = round(chosen['10 Day Change'].mean()*100,4)
        print('result', mean_return, len(chosen))
        return mean_return, len(chosen)






conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data_adjusted_ta', conn, parse_dates = ['Date Reported'])

df = df.sort_values(by='Date Reported')



#print(df)

perform_ml(df)
