
import sqlite3
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

class machine():
    def __init__(self, df, year, features, iterations,buy_cutoff=None):
        self.df = df
        self.year = year
        self.features = features

        self.buy_cutoff = buy_cutoff
        if self.buy_cutoff is None:
            self.buy_cutoff = .01

            self.find_cutoff()
        self.prepare_data()

        output_df = []
        for i in range(iterations):
            self.train_model()
            self.test_model()
            mean_return, accuracy, num_trades, score, mean_return_random = self.get_results()

            output_df.append([mean_return, accuracy, num_trades, score, mean_return_random, self.buy_cutoff])

        output_df = pd.DataFrame(output_df)
        output_df.columns = ['Returns', 'Accuracy', 'Num of Trades', 'Score', 'Random Return', 'Buy Cutoff']

        self.returns_mean = output_df['Returns'].mean()
        self.returns_stddev = output_df['Returns'].std()
        self.num_trades_mean = output_df['Num of Trades'].mean()
        self.score_mean = output_df['Score'].mean()

        self.output_df = output_df



    def prepare_data(self):

        if self.year == '2018':
            self.df = self.df[self.df['Date Reported']<datetime.strptime('%s-12-31' % self.year, '%Y-%m-%d')]
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
#        print('finding cutoff')
        while True:
            self.prepare_data()
            self.train_model()
            self.test_model()
            mean_return, accuracy, num_trades, score, mean_return_random = self.get_results()
            #print('finding cutoff', mean_return, accuracy, num_trades)
            #scaler = int(num_trades/500)+1

            if num_trades<500:
#                print('found cutoff')
                break

            self.buy_cutoff = self.buy_cutoff + (.001)

    def train_model(self):
        self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10)
        #self.clf = RandomForestClassifier(n_jobs=-1)
        y = self.train['Action Code']
        train = self.train[self.features]
        self.clf.fit(train, y)

    def plot_tree(self, name):
        from sklearn.tree import export_graphviz
        estimator = self.clf.estimators_[5]
        export_graphviz(estimator, out_file='tree.dot',
                feature_names = self.features,
                class_names = ['Buy', 'None'],
                rounded = True, proportion = False,
                precision = 2, filled = True)
        from subprocess import call
        name = str(round(name, 2)).replace('.','_')
        filename = '%s_tree.png' % name
        print(filename)
        call(['dot', '-Tpng', 'tree.dot', '-o', filename, '-Gdpi=600'])


    def test_model(self):
        preds = self.clf.predict(self.test[self.features])
        preds = pd.DataFrame(preds).astype(str).replace('0','Buy').replace('1','None')
        preds.columns = ['Predicted']
        self.test.loc[:, 'Predicted'] = list(preds['Predicted'])


    def get_results(self):
        score = accuracy_score(self.test['Action'], self.test['Predicted'])
        chosen = self.test[self.test['Predicted']=='Buy']
        mean_return = chosen['10 Day Change'].mean()*100

        accuracy = len(chosen[chosen['10 Day Change']>0])/float(len(chosen))

        mean_return_random = self.test.sample(n=len(chosen))['10 Day Change'].mean()*100

        #score = self.clf(self.test[self.features], self.test['Action Code'])

        #print(mean_return, accuracy, len(chosen))
        return mean_return, accuracy, len(chosen), score, mean_return_random


class run_models():
    def __init__(self, df, features):
        self.df = df
        print(features)

        self.initial_features = features
        self.start_feature_imp = features.copy()
        self.current_features = []
        self.max_mean = -10
        self.max_score = -10

        self.add_feature()
        while True:

            try:
                model_2015 = machine(self.df, '2018', self.current_features, 5)
            except:
                pass

            #print('model mean', model.returns_mean, 'max mean', self.max_mean)
            #total_score = sum([model_2015.score_mean, model_2016.score_mean])/2.0
            total_score = round(model_2015.score_mean,5)
            #print(self.current_features, total_score, self.max_score)
            if  total_score > self.max_score:
                #model.plot_tree(model.returns_mean)
                self.max_score = total_score
                self.current_max_feature = self.feature_added

                print('\nmax feature', self.max_score, self.feature_added)
                print(self.current_features)
                print(model_2015.output_df.mean())
                model_2016 = machine(self.df, '2019', self.current_features, 5, model_2015.buy_cutoff)
                print(model_2016.output_df.mean())
                print('')
                """
                print(self.current_features)
                print(model_2015.output_df)
                print(model_2015.output_df.mean())
                print(model_2016.output_df)
                print(model_2016.output_df.mean())
                print(model_2015.buy_cutoff, model_2016.buy_cutoff)
                """

                #self.start_feature_imp = self.initial_features.copy()

            self.remove_feature()

            if len(self.start_feature_imp) == 0:
                self.start_feature_imp = features.copy()
                if self.current_max_feature not in self.current_features:
                    self.current_features.append(self.current_max_feature)
                
                print('\nreset\n')
                #self.max_score = 0

            self.add_feature()




            """
            else:

                self.remove_feature()
                self.add_feature()


            """


    def add_feature(self):
        self.feature_added = self.start_feature_imp.pop(0)
        while self.feature_added in self.current_features:
            #print('not adding feature', self.feature_added, 'as it already exists')
            self.feature_added = self.start_feature_imp.pop(0)
        #print('adding feature', self.feature_added)
        self.current_features.append(self.feature_added)

    def remove_feature(self):
        #print('removing added feature ', self.feature_added)
        self.current_features.remove(self.feature_added)


def get_feature_importances(df, start_features):
    df = df.replace('-', np.nan).replace([np.inf, -np.inf], np.nan).dropna()

    df['Action'] = 'Dont Buy'
    df['Action'].values[df['10 Day Change Abnormal'].values > .06] = "Buy"
    df['Action Code'] = df['Action'].astype('category').cat.codes

    model = RandomForestClassifier()
    model.fit(df[start_features],df['Action Code'])

    feat_importances = pd.Series(model.feature_importances_, index=start_features).sort_values(ascending=False)
    features = list(feat_importances.keys())
    features.insert(0, 'Before and After')
    return features



conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data_adjusted_ta', conn, parse_dates = ['Date Reported'])

df = df.sort_values(by='Date Reported')
#df = df[df['Market Cap Text']=="Large"]
df = df[df['Market Cap Text'].isin(['Medium', 'Large'])]

initial_features = list(df.columns)

for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                  '10 Day Change Abnormal',
                  'Date Reported', 'Time Reported',
                  'Symbol', 'Market Cap Text']:

    initial_features.remove(remove_me)

initial_features = get_feature_importances(df, initial_features)
run_models(df, initial_features)
