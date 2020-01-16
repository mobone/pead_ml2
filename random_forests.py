from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
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
        self.out_df = None



        while True:
            #self.buy_cutoff =  random.uniform(.02,.09)
            self.buy_cutoff = .06
            self.prepare_data()
            self.features = list(self.df.columns)
            for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                              'Date Reported', 'Time Reported',
                              'Symbol', 'Market Cap Text', 'Action',
                              'is_train']:
                self.features.remove(remove_me)
            for self.machine in ['Classifier']:
                self.train_model()
                self.predict()
                self.get_results()

    def prepare_data(self):
        self.df['is_train'] = True
        #self.df['is_train'] = np.random.uniform(0, 1, len(self.df)) <= .75
        self.df['is_train'].values[self.df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = False

        self.df['Action'] = 'Dont Buy'
        self.df['Action'].values[self.df['10 Day Change Abnormal'].values > self.buy_cutoff] = "Buy"
        #self.df['Action'].values[(self.df['10 Day Change Abnormal'].values >= self.regressor_min) & (self.df['10 Day Change Abnormal'].values <= self.regressor_max) ] = "Buy"
        self.df['Action'] = self.df['Action'].astype('category')
        self.df["Action Code"] = self.df["Action"].cat.codes
        self.df['Market Cap Text'] = self.df['Market Cap Text'].astype('category')
        self.df["Market Cap Code"] = self.df["Market Cap Text"].cat.codes
        print(self.df[['Action', 'Action Code', '10 Day Change Abnormal']])


        self.train, self.test = df[df['is_train']==True], df[df['is_train']==False]

    def train_model(self):
        if self.machine == 'Classifier':
            self.clf = RandomForestClassifier(n_jobs=-1)

            y = self.train['Action Code']
            self.features.remove('Action Code')
            self.features.remove('10 Day Change Abnormal')
            train = self.train[self.features]
            self.clf.fit(train[self.features], y)

            """
            # save model
            self.output_model = RandomForestClassifier(n_jobs=-1)
            y = self.df['Action Code']
            save_train = self.df[self.features]
            self.output_model.fit(save_train[self.features], y)
            dump(self.output_model, 'start_at_01_15_2020.joblib')
            input()
            """

        elif self.machine == 'Regressor':
            self.clf = RandomForestRegressor(n_jobs=-1)
            y = self.train['10 Day Change Abnormal']
            self.clf.fit(self.train[self.features], y)

    def predict(self):
        preds = self.clf.predict(self.test[self.features])
        self.test[self.machine + ' Predicted'] = preds


    def get_results(self):
        # View a list of the  and their importance scores
        feature_imp = pd.Series(self.clf.feature_importances_,index=self.features).sort_values(ascending=False)

        chosen = self.test[self.test['Classifier Predicted']==0]
        actual_winners = chosen[chosen['10 Day Change Abnormal'] > .0]
        accuracy = metrics.accuracy_score(self.test['Action Code'], self.test['Classifier Predicted'])
        profitable_percent = len(actual_winners)/float(len(chosen))
        mean_return = round(chosen['10 Day Change'].mean()*100,2)

        print(feature_imp)
        #print(pd.crosstab(self.test['Action'], self.test['Classifier Predicted'], rownames=['Actual Action'], colnames=['Predicted Action']))

        #print(chosen, actual_winners, accuracy, profitable_percent, mean_return)
        df_row = [[self.buy_cutoff, accuracy, profitable_percent, mean_return, len(chosen)]]
        print(df_row)
        if self.out_df is None:
            self.out_df = pd.DataFrame(df_row)
            print(self.out_df)
            self.out_df.columns = ['buy cutoff', 'accuracy', 'profitable percent', 'avg profit', 'num of trades']
        else:
            df_row = pd.DataFrame(df_row)
            df_row.columns = ['buy cutoff', 'accuracy', 'profitable percent', 'avg profit', 'num of trades']
            self.out_df = self.out_df.append(df_row)

        print(self.out_df)
        self.out_df.to_csv('results.csv')


conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data', conn, parse_dates = ['Date Reported'])



df = df.sort_values(by='Date Reported')

df = df.replace('-', np.nan)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna() #TODO: Deal with nan's

print(df)
max_accuracy = 0
max_correct = 0
max_mean_return = 0

perform_ml(df)

"""
if (accuracy > max_accuracy or correct > max_correct) and mean_return>max_mean_return:
    max_mean_return = mean_return
    if accuracy > max_accuracy:
        max_accuracy = accuracy
    if correct > max_correct:
        max_correct = correct
    try:
        test[['Date Reported', 'Action', 'Predicted', '10 Day Change', '10 Day Change Abnormal']].to_csv('predictions.csv')
    except:
        print('close excel')
        input()

    print("Accuracy:", round(accuracy,2), "Correct Accuracy:", round(correct,2),"Mean Return:", mean_return, "Trades:", len(chosen), "Actual Winners:", len(actual_winners), 'Buy Cutoff:', buy_cutoff)
    print(pd.crosstab(test['Action'], preds, rownames=['actual'], colnames=['preds']))
"""
