from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sqlite3
from sklearn import metrics
import warnings
from datetime import datetime
import random
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)

class perform_ml():
    def __init__(self, df, buy_cutoff):
        self.df = df
        self.buy_cutoff = buy_cutoff
        self.features = list(self.df.columns)
        self.features.remove('5 Day Change')
        self.features.remove('10 Day Change')
        self.features.remove('5 Day Change Abnormal')
        self.features.remove('10 Day Change Abnormal')
        print(self.features)
        self.prepare_data()
        for self.machine in ['Classifier', 'Regressor']:
            self.train()
            self.predict()
            self.get_results()

    def prepare_data(self):
        self.df['is_train'] = True
        #self.df['is_train'] = np.random.uniform(0, 1, len(self.df)) <= .75
        self.df['is_train'].values[self.df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = False

        self.df['Action'] = 'Buy'
        self.df['Action'].values[self.df['10 Day Change Abnormal'].values > self.buy_cutoff] = "Don't Buy"
        self.df['Action'] = self.df['Action'].astype('category')

        self.train, self.test = df[df['is_train']==True], df[df['is_train']==False]

    def train(self):
        if self.machine == 'Classifier':
            self.clf = RandomForestClassifier(n_jobs=-1)
            y, _ = pd.factorize(self.train['Action'])
            self.clf.fit(self.train[self.features], y)
        elif self.machine == 'Regressor':
            self.clf = RandomForestRegressor(n_jobs=-1)
            y = self.train['10 Day Change Abnormal']
            self.clf.fit(self.train[self.features], y)

    def predict(self):
        preds = self.clf.predict(self.test[features])
        self.test[self.machine + ' Predicted'] = preds


    def get_results(self):
        # View a list of the features and their importance scores
        feature_imp = pd.Series(self.clf.feature_importances_,index=self.features).sort_values(ascending=False)

        chosen = self.test[self.test['Predicted']==1]
        actual_winners = chosen[chosen['10 Day Change Abnormal'] > .0]
        accuracy = metrics.accuracy_score(self.test['Action'], self.test['Predicted'])
        profitable_percent = len(actual_winners)/float(len(chosen))
        mean_return = round(chosen['10 Day Change'].mean()*100,2)

        print(feature_imp)
        print(chosen, actual_winners, accuracy, profitable_percent, mean_return)

conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data', conn, parse_dates = ['Date Reported'])


df = df.dropna() #TODO: Deal with nan's
df = df.sort_values(by='Date Reported')
print(df)
max_accuracy = 0
max_correct = 0
max_mean_return = 0

perform_ml(df, .05)

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
