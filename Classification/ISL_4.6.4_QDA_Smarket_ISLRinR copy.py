import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
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

# ============= LDA =============
lda = QuadraticDiscriminantAnalysis(store_covariance=True)
classifier = lda.fit(X_train, y_train)
print('pi_k: {}'.format(classifier.priors_))
print('mu_k: \n{}'.format(classifier.means_))
print('Cov_k: \n{}'.format(classifier.covariance_))
y_pred = classifier.predict(X_test)
print(np.unique(y_pred, return_counts=True))
print(confusion_matrix(y_pred, y_test))
print(classification_report(y_test, y_pred, digits=3))
"""
The results are exact the same with the one from logistic regression
"""

# ======================= Visualization =========================
fig1 = vplt.ClassificationPlot('S&P500 Up(0) and Down(1): Training Data', x1lab_='lag1', x2lab_='lag2')
fig1.plot_data_p2(X_train, y_train, classifier)
fig1.plot_ellipse(classifier.means_[0], classifier.covariance_[0], 'red')
fig1.plot_ellipse(classifier.means_[1], classifier.covariance_[1], 'green')

fig2 = vplt.ClassificationPlot('S&P500 Up(0) and Down(1): Test Data', x1lab_='lag1', x2lab_='lag2')
fig2.plot_data_p2(X_test, y_test, classifier)
fig2.plot_ellipse(classifier.means_[0], classifier.covariance_[0], 'red')
fig2.plot_ellipse(classifier.means_[1], classifier.covariance_[1], 'green')

print(np.linalg.eig(classifier.covariance_[0]))
print(np.linalg.eig(classifier.covariance_[1]))

plt.show(block=False)
time.sleep(200)