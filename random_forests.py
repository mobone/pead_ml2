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
        self.buy_cutoff = .02
        self.conn = sqlite3.connect('earnings.db', timeout=120)
        self.prepare_data()
        self.features = list(self.df.columns)
        for remove_me in ['5 Day Change', '10 Day Change', '5 Day Change Abnormal',
                          '10 Day Change Abnormal',
                          'Date Reported', 'Time Reported',
                          'Symbol', 'Market Cap Text', 'Action',
                          'is_train']:
            self.features.remove(remove_me)
        for feature in self.features:
            if "Actual" in feature:
                self.features.remove(feature)
        #'Before and After', 'YoY Growth','Revenue Surprise'
        self.good_features = []
        for good_feature in self.good_features:
            self.features.remove(good_feature)

        self.n_features = randint(0,len(self.features)-2)

        self.feature_combinations = []
        for i in range(3,len(self.features)-1):
            for combo in list(combinations(self.features, i)):
                self.feature_combinations.append(combo)


        print(len(self.feature_combinations))
        self.chosen_means = [0]

        while True:
            self.buy_cutoff = round(random.uniform(.00,.04),3)
            #self.buy_cutoff = .045
            self.prepare_data()


            #self.n_features = randint(2,len(self.features)-1)


            for self.machine in ['Classifier']:
                self.train_model()
                self.predict()
                try:
                    self.get_results()
                except Exception as e:
                    print(e)
                    pass

                #self.visualize()
                #input()

    def prepare_data(self):
        self.df['is_train'] = True
        #self.df['is_train'] = np.random.uniform(0, 1, len(self.df)) <= .75
        self.df['is_train'].values[self.df['Date Reported'] >= datetime.strptime('2019-01-01', '%Y-%m-%d')] = False

        self.df['Action'] = 'None'
        #self.df['Action'].values[self.df['10 Day Change Abnormal'].values < self.short_cutoff] = "Short"
        self.df['Action'].values[self.df['10 Day Change Abnormal'].values > self.buy_cutoff] = "Buy"
        #self.df['Action'].values[(self.df['10 Day Change Abnormal'].values >= self.regressor_min) & (self.df['10 Day Change Abnormal'].values <= self.regressor_max) ] = "Buy"
        self.df['Action'] = self.df['Action'].astype('category')
        self.df["Action Code"] = self.df["Action"].cat.codes
        #self.df['Market Cap Text'] = self.df['Market Cap Text'].astype('category')
        #self.df["Market Cap Code"] = self.df["Market Cap Text"].cat.codes

        self.train, self.test = df[df['is_train']==True], df[df['is_train']==False]

    def train_model(self):
        if self.machine == 'Classifier':
            self.clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)

            y = self.train['Action Code']
            #print(self.train[['10 Day Change', 'Action', 'Action Code']])
            self.chosen_num = randint(0, len(self.feature_combinations)-1)
            self.chosen_features = self.feature_combinations[self.chosen_num]
            self.chosen_features = list(self.chosen_features)

            for good_feature in self.good_features:
                if good_feature not in self.chosen_features:
                    self.chosen_features.append(good_feature)
            self.n_features = len(self.chosen_features)
            if 'Action Code' in self.chosen_features:
                self.chosen_features.remove('Action Code')

            train = self.train[self.chosen_features]
            self.clf.fit(train, y)

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
        preds = self.clf.predict(self.test[self.chosen_features])
        preds = pd.DataFrame(preds).astype(str)
        preds.columns = ['Predicted']

        preds = preds.replace('0','Buy')
        preds = preds.replace('1','None')
        #preds = preds.replace('2','Sell')

        self.test[self.machine + ' Predicted'] = list(preds['Predicted'])
        #self.test.to_csv('ml_results.csv')
        # TODO: Possibly have machine provide probabilites as output
        """
        self.test['Prob 0'] = None
        self.test['Prob 1'] = None
        self.test[['Prob 0', 'Prob 1']] = self.clf.predict_proba(self.test[self.chosen_features])
        """
        # TODO: use target names, numpy nd array
        #preds = iris.target_names[clf.predict(test[features])]


    def get_results(self):
        # View a list of the  and their importance scores
        #self.test.to_csv('ml_results.csv')
        feature_imp = pd.Series(self.clf.feature_importances_,index=self.chosen_features).sort_values(ascending=False)

        chosen = self.test[self.test['Classifier Predicted']=='Buy']
        not_chosen = self.test[self.test['Classifier Predicted']=='None']
        actual_winners = chosen[chosen['10 Day Change'] > .0]
        #accuracy = round(metrics.accuracy_score(self.test['Action Code'], self.test['Classifier Predicted']),2)
        accuracy = 0
        profitable_percent = round(len(actual_winners)/float(len(chosen)), 2)
        mean_return = round(chosen['10 Day Change'].mean()*100,4)
        not_mean_return = round(not_chosen['10 Day Change'].mean()*100,4)

        avg_mean_total = sum(self.chosen_means)/len(self.chosen_means)
        #if len(chosen)< 100 or mean_return<avg_mean_total:
        #    return
        self.chosen_means.append(mean_return)
        #print(feature_imp)

        self.chosen_features = list(feature_imp.keys())
        #print(pd.crosstab(self.test['Action'], self.test['Classifier Predicted'], rownames=['Actual Action'], colnames=['Predicted Action']))

        #print(chosen, actual_winners, accuracy, profitable_percent, mean_return)
        df_row = [[self.buy_cutoff, self.n_features, profitable_percent, mean_return, not_mean_return, len(chosen), str(self.chosen_features)]]
        df_row = pd.DataFrame(df_row)
        df_row.columns = ['buy cutoff', 'n_features', 'profitable percent', 'avg profit', 'avg not profit', 'num of trades', 'chosen features']
        df_row.to_sql('ml_results', self.conn, if_exists='append')
        """
        if self.out_df is None:
            self.out_df = pd.DataFrame(df_row)

            self.out_df.columns = ['buy cutoff', 'n_features', 'profitable percent', 'avg profit', 'avg not profit', 'num of trades', 'chosen features']
        else:
            df_row = pd.DataFrame(df_row)
            print(df_row)
            df_row.columns = ['buy cutoff', 'n_features', 'profitable percent', 'avg profit', 'avg not profit','num of trades', 'chosen features']
            self.out_df = self.out_df.append(df_row)
        self.out_df = self.out_df.sort_values(by=['avg profit'])
        print(self.out_df)
        self.out_df.to_csv('results.csv')
        """

    def visualize(self):
        estimator = self.clf.estimators_[5]
        from sklearn.tree import export_graphviz
        # Export as dot file
        export_graphviz(estimator, out_file='tree.dot',
                        feature_names = iris.feature_names,
                        class_names = iris.target_names,
                        rounded = True, proportion = False,
                        precision = 2, filled = True)

        # Convert to png using system command (requires Graphviz)
        from subprocess import call
        call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


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
