import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# get data
df_in = pd.read_csv('../Data/Boston_MASSinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace =True)
print(df_in.head(5))
print(df_in.columns)

X = df_in.iloc[:, 0:-1]
y = df_in.iloc[:, -1:].values

X2 = sm.add_constant(X)
lm = sm.OLS(y, X2)
est = lm.fit()
print(est.summary())