import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import linear_model

n = 500
mean = 0
variance = 1
x1 = np.random.normal(loc=1 , scale=np.sqrt( 1*variance), size=n)
x2 = np.random.normal(loc=5 , scale=np.sqrt( 5*variance), size=n)
x3 = np.random.normal(loc=10, scale=np.sqrt(10*variance), size=n)
e  = np.random.normal(loc=0 , scale=np.sqrt( 2*variance), size=n)
y  = 2 + x1 + x2 + e
formula = 'y ~ x1 + x2 + x3'

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
print(df.head())
df_scaled = df[1:].apply(lambda x: (x-x.mean())/x.std())
df_scaled['y'] = df['y']
df_scaled = df_scaled[['y', 'x1', 'x2', 'x3']]
X_train = df_scaled.iloc[: , 1: ].values
y_train = df_scaled.iloc[: , 0:1].values

"""
(a) Ordinary Least Square
    After regularization, two method give the same results
"""
ols_sm = smf.ols(formula, df_scaled).fit()
print('======= OLS from sm ======')
print(ols_sm.params)
ols_sk = linear_model.LinearRegression().fit(X_train, y_train)
print('======= OLS from sk ======')
print(ols_sk.coef_)

"""
(b) Ridge Regression
    n * alpha_sm = alpha_sk
"""
ridge_sm = smf.ols(formula, df_scaled).fit_regularized(alpha=1, L1_wt=0)
print('======= Ridge from sm ======')
print(ridge_sm.params)
ridge_sk = linear_model.Ridge(alpha=n).fit(X_train, y_train)
print('======= Ridge from sk ======')
print(ridge_sk.coef_)

"""
(c) Lasso Regression
    alpha_sm = alpha_sk
"""
lasso_sm = smf.ols(formula, df_scaled).fit_regularized(alpha=1, L1_wt=1)
print('======= Lasso from sm ======')
print(lasso_sm.params)
lasso_sk = linear_model.Lasso(alpha=1).fit(X_train, y_train)
print('======= Lasso from sk ======')
print(lasso_sk.coef_)

