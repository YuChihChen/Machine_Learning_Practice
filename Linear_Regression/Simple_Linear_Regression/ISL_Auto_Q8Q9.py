import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# get data
df_in = pd.read_csv('../Data/Auto_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace =True)
print(df_in.head(5))
print(df_in.columns)

# ====================== Q8 ======================
x = df_in.iloc[:, 3:4].values
y = df_in.iloc[:, 0:1].values

X0 = sm.add_constant(x)
lm = sm.OLS(y, X0)
est = lm.fit()
print(est.summary())

yp = est.predict(X0)
#plt.scatter(x, y)
#plt.scatter(x, yp)

"""
(a)
i.  There is a relationship between mpg and horsepower. 
    This can be tested by using t-test or F-test
ii. R^2 is 0.606. There exist linear relation but to no strong in linear way 
iii.The relationship is negative.
iv. We can get it use textbook's formula
(b) The plot had been shown above
(c) let's plot the residuals vs. fitted value. 
    It shows there is a non-linear relationship between x and y
"""
res = y-yp
#plt.scatter(yp, res)


# ====================== Q9 ======================
"""
(a) produce a scatter plot matrix
"""
#scatter_matrix(df_in, alpha=0.2, figsize=(6, 6), diagonal='kde')

"""
(b) compute correlations between variables (also plot)
"""
dfx = df_in.drop(columns=['mpg', 'name'])
corr_matrix = dfx.corr()
print(corr_matrix)
#sns.heatmap(corr_matrix, linewidth=0.5)

"""
(c) 
i.  From the F-test, yes, there is a relation between X and y
ii. displacement, weight, year, and origin have significant relationship, 
    cylinders, horsepower, and acceleration do not.
iii.It suggests that car will increase efficiency as year increase 
"""
X = df_in.drop(columns=['mpg', 'name'])
y = df_in.iloc[:, 0:1].values
X0 = sm.add_constant(X)
lm = sm.OLS(y, X0)
est = lm.fit()
print(est.summary())

"""
(d) let's plot the residuals vs. fitted value. 
    It shows there is a non-linear relationship between x and y
"""
yp = est.predict(X0)
res = y.T-yp.values
print(res)
plt.scatter(yp, res)

"""
(e) here we do lm(mpg~cylinders*displacement+displacement*weight)
: => only use interaction terms
* => include interaction terms
"""
m1 = smf.ols(formula='mpg~cylinders:displacement+displacement:weight', data=df_in).fit()
print(m1.summary())
m2 = smf.ols(formula='mpg~cylinders*displacement+displacement*weight', data=df_in).fit()
print(m2.summary())

"""
(f) here we do lm(mpg~log(weight)+sqrt(horsepower)+acceleration+I(acceleration^2))
"""
mf = smf.ols(formula='mpg~np.log(weight)+np.sqrt(horsepower)+acceleration+np.square(acceleration)',
             data=df_in).fit()
print(mf.summary())

print("the end of the code")