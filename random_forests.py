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

conn = sqlite3.connect('earnings.db', timeout=120)
df = pd.read_sql('select * from aggregated_data', conn, parse_dates = ['Date Reported'])

df = df.replace('–', np.nan)
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna() #TODO: Deal with nan's
df = df.sort_values(by='Date Reported')
print(df)
max_accuracy = 0
max_correct = 0
max_mean_return = 0
while True:
    #df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

    df['is_train'] = True

    #df[df['Date Reported']>]['is_train'] = False
    df['is_train'].values[df['Date Reported'] >=datetime.strptime('2019-01-01', '%Y-%m-%d')] = False


    #print(df)
    buy_cutoff = round(random.uniform(.02, .08),2)
    df['Action'] = 0
    df['Action'].values[df['10 Day Change Abnormal'].values > random.uniform(.02, .08)] = 1
    df['Action'] = df['Action'].astype('category')

    #print(df.head())

    train, test = df[df['is_train']==True], df[df['is_train']==False]

    """features = ['EPS Num of Estimates', 'EPS Delta', 'EPS Surprise', 'EPS Wall St',
           'EPS Estimize', 'EPS Actual', 'Revenue Num of Estimates',
           'Revenue Delta', 'Revenue Surprise', 'Revenue Wall St',
           'Revenue Estimize', 'Revenue Actual',
           'Historical EPS Beat Ratio', 'Historical EPS Beat Percent',
           'Average Change 5 Days', 'Average Abnormal Change 5 Days', 'Average Change 10 Days', 'Average Abnormal Change 10 Days',
           'YoY Growth']"""

    features = ['EPS Surprise', 'EPS Wall St',
           'EPS Estimize', 'EPS Actual',
           'Revenue Delta', 'Revenue Surprise', 'Revenue Wall St',
           'Revenue Estimize', 'Revenue Actual',
           'Historical EPS Beat Percent',
           'Average Change 5 Days', 'Average Abnormal Change 5 Days', 'Average Change 10 Days', 'Average Abnormal Change 10 Days',
           'YoY Growth']


    clf = RandomForestClassifier(n_jobs=-1)
    y, _ = pd.factorize(train['Action'])
    clf.fit(train[features], y)

    # View the predicted probabilities of the first 10 observations
    #print(clf.predict_proba(test[features])[0:10])

    preds = clf.predict(test[features])
    test['Predicted'] = preds



    # View a list of the features and their importance scores
    #print(list(zip(train[features], clf.feature_importances_)))
    #print(clf.feature_importances_)
    feature_imp = pd.Series(clf.feature_importances_,index=features).sort_values(ascending=False)
    #print(feature_imp)
    chosen = test[test['Predicted']==1]
    actual_winners = chosen[chosen['10 Day Change Abnormal'] > .0]
    accuracy = metrics.accuracy_score(test['Action'], test['Predicted'])
    correct = len(actual_winners)/float(len(chosen))
    mean_return = round(chosen['10 Day Change'].mean()*100,2)
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
