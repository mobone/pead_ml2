
import sqlite3
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class machine():
    def __init__(self, df, year, features):
        self.df = df
        self.year = year
        self.features = features

        self.buy_cutoff = .03



        self.find_cutoff()

        self.prepare_data()
        self.train_model()
        self.test_model()
        self.get_results()


    def prepare_data(self):
        if self.year == '2018':
            self.df = self.df[self.df['Date Reported']<'2019-01-01']
        start_date = datetime.strptime('%s-01-01' % self.year, '%Y-%m-%d')
        end_date = datetime.strptime('%s-12-31' % self.year, '%Y-%m-%d')
        self.df.loc[:,'is_train'] = 'Train'
        self.df.loc[(self.df['Date Reported'] >= start_date) & (self.df['Date Reported'] <= end_date), 'is_train'] = 'Test'

        self.df.loc[:,'Action'] = 'None'
        self.df.loc[self.df['10 Day Change Abnormal'] > self.buy_cutoff, 'Action'] = 'Buy'
        self.df.loc[:, 'Action Code'] = self.df['Action'].astype('category').cat.codes

        self.df = self.df[self.features + ['Action', 'Action Code', 'is_train', '10 Day Change Abnormal', '10 Day Change', 'Date Reported', 'Symbol']]
        self.df = self.df.replace('-', np.nan).replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna()

        self.train, self.test = self.df[self.df['is_train']=='Train'], self.df[self.df['is_train']=='Test']

    def find_cutoff(self):
        print('finding cutoff')
        while True:
            self.prepare_data()
            self.train_model()
            self.test_model()
            mean_return, accuracy, num_trades = self.get_results()
            scaler = int(num_trades/250)+1

            print('finding cutoff', mean_return, num_trades, self.buy_cutoff, scaler)

            if num_trades<400:
                print('found cutoff')
                break

            self.buy_cutoff = round(self.buy_cutoff + (.005*scaler), 4)

    def train_model(self):
        self.clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=500)
        y = self.train['Action Code']
        train = self.train[self.features]
        self.clf.fit(train, y)

    def test_model(self):
        preds = self.clf.predict(self.test[self.features])
        preds = pd.DataFrame(preds).astype(str).replace('0','Buy').replace('1','None')
        preds.columns = ['Predicted']
        self.test.loc[:, 'Predicted'] = list(preds['Predicted'])


    def get_results(self):
        chosen = self.test[self.test['Predicted']=='Buy']
        mean_return = round(chosen['10 Day Change'].mean()*100,4)

        accuracy = len(chosen[chosen['10 Day Change']>0])/float(len(chosen))

        print(mean_return, accuracy, len(chosen))
        return mean_return, accuracy, len(chosen)


def get_feature_importances(df, start_features):
    df = df.replace('-', np.nan).replace([np.inf, -np.inf], np.nan).dropna()

    df['Action'] = 'Dont Buy'
    df['Action'].values[df['10 Day Change Abnormal'].values > .06] = "Buy"
    df['Action Code'] = df['Action'].astype('category').cat.codes

    model = ExtraTreesClassifier()
    model.fit(df[start_features],df['Action Code'])

    feat_importances = pd.Series(model.feature_importances_, index=start_features).sort_values(ascending=False)

    return feat_importances




conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data_adjusted_ta', conn, parse_dates = ['Date Reported'])

df = df.sort_values(by='Date Reported')
start_features = list(df.columns)

for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                  '10 Day Change Abnormal',
                  'Date Reported', 'Time Reported',
                  'Symbol', 'Market Cap Text']:

    start_features.remove(remove_me)

print(get_feature_importances(df, start_features))

machine(df, '2018', ['Before and After'])
