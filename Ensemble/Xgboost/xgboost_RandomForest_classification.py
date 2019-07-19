import datetime as dt
import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# ============ get data ============
df_in = pd.read_csv('data/Caravan_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
df_in['y_fac'] = pd.factorize(df_in.iloc[:, -1])[0]
print(df_in.head(5))
X_columns = list(df_in.columns[:-2])
X_orig = df_in.iloc[:, 0:-2].values
y_orig = df_in.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2)
print("X_train_size = {}, y_train_size = {}".format(X_train.shape, y_train.shape))
print("X_test_size  = {}, y_test_size  = {}".format(X_test.shape , y_test.shape))


N = 250
# ============ Random Forest =======
ts = dt.datetime.now()
forest = RandomForestClassifier(n_estimators=N)
fr_cv_acc = cross_val_score(forest, X_train, y_train, cv=5)
print('training CV accuracy of random forest: {}'.format(np.mean(fr_cv_acc)))
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
acc_test = np.mean(y_pred == y_test) 
print('test accuracy of random forest: {}'.format(acc_test))
te = dt.datetime.now()
print('time spending of random forest: {} seconds'.format((te-ts).total_seconds()))
  

# ============ Xgboost =======
ts = dt.datetime.now()
xgb = xgboost.XGBClassifier(n_estimators=N)
xgb_cv_acc = cross_val_score(xgb, X_train, y_train, cv=5)
print('training CV accuracy of xgboost: {}'.format(np.mean(xgb_cv_acc)))
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(y_pred)
acc_test = np.mean(y_pred == y_test) 
print('test accuracy of xgboost: {}'.format(acc_test))
te = dt.datetime.now()
print('time spending of random xgboost: {} seconds'.format((te-ts).total_seconds()))

