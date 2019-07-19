import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_score
import classificationplot as vplt

# get data
df_in = pd.read_csv('data/Smarket_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
df_in['y_fac'] = pd.factorize(df_in.iloc[:, -1])[0]
print(df_in.head(5))
print(df_in.columns)
print(df_in.groupby('Year').size())

# split into training and test data
df_train = df_in[df_in.Year.isin([2001, 2002, 2003, 2004])].copy()
df_test  = df_in[df_in.Year == 2005]


# split into X and y
X_train = df_train.iloc[:, 1:3].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, 1:3].values
y_test = df_test.iloc[:, -1].values


# ======================== logistic regression ========================
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
print(classifier.predict_proba(X_train))
print("==================== training data ===========================")
y_pred = classifier.predict(X_train)
print(confusion_matrix(y_pred, y_train))
print(classification_report(y_train, y_pred, digits=3))
print("==================== test data ===========================")
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=3))

# ======================== Visualization ========================
# 1. for training data
fig1 = vplt.ClassificationPlot('S&P500 Up(0) and Down(1): Training Data', x1lab_='lag1', x2lab_='lag2')
fig1.plot_data_p2(X_train, y_train, classifier)
fig2 = vplt.ClassificationPlot('S&P500 Up(0) and Down(1): Test Data', x1lab_='lag1', x2lab_='lag2')
fig2.plot_data_p2(X_test, y_test, classifier)
fig2.plot_ellipse(np.array([0,0]), np.array([[1,0],[0,1]]), 'green')
plt.legend()
plt.show(block=False)
"""
From the figures, we find that there is no obvious pattern between today's movement and lags
"""
time.sleep(200)
print("The end of the code")