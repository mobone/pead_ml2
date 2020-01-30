import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import sqlite3


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
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(15).plot(kind='barh')
plt.show()
