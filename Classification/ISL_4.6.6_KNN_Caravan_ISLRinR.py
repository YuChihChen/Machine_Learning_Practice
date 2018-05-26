import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import Visualization.classificationplot as vplt

# get data
df_in = pd.read_csv('data/Caravan_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
df_in['y_fac'] = pd.factorize(df_in.iloc[:, -1])[0]
print(df_in.head(5))

# standardization
stz = lambda se_: (se_-se_.mean())/se_.std()
df_stz = df_in.iloc[:, :-2].apply(stz)
df_stz['y_fac'] = df_in['y_fac'].copy()
print(df_stz.head())
print(df_stz.shape)

# split into training and test data
df_train = df_stz.sample(frac=0.8, replace=False)
df_test  = df_stz[~df_stz.index.isin(df_train.index)].copy()
print(df_train.shape)
print(df_test.shape)

# split into X and y
X_train = df_train.iloc[:, 0:-1].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, 0:-1].values
y_test = df_test.iloc[:, -1].values


# ============= KNN =============
knn = KNeighborsClassifier(n_neighbors=3)
classifier = knn.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.unique(y_pred, return_counts=True))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=3))


def knn_confusion_matrix(K_, X_train_=X_train, y_train_=y_train, X_test_=X_test, y_test_=y_test):
    knn = KNeighborsClassifier(n_neighbors=K_)
    model = knn.fit(X_train_, y_train_)
    y_pred = model.predict(X_test_)
    return confusion_matrix(y_pred, y_test_)


K = range(1, 10)
CMK = list(map(knn_confusion_matrix, K))  # confused matrix
for k in range(len(K)):
    print('k = {}, prediction = ({}, {}), predict_rate = {}'
          .format(k, CMK[k][1, 0], CMK[k][1, 1], CMK[k][1, 1]/(CMK[k][1, 0]+CMK[k][1, 1])))


"""
The results are exact the same with the one from logistic regression
"""

