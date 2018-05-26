import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


mean = 0
variance = 1
N = 100000
x1 = np.random.uniform(size=N)
x2 = 0.5*x1+np.random.normal(loc=mean, scale=np.sqrt(variance), size=N)/10
y = 2+2*x1+0.3*x2+np.random.normal(loc=mean, scale=np.sqrt(variance), size=N)
df_in = pd.DataFrame({'x1': x1, 'x2':x2, 'y': y})
print(df_in.head())
"""
(a) y = 2+2*x1+0.3*x2, b0=2, b1=2, b2=0.3
(b) Corr(x1, x2)=0.7917
"""
print(df_in['x1'].corr(df_in['x2']))
plt.scatter(x1, x2)

"""
(c) we can not reject null hypothesis: b2 = 0  
"""
m1 = smf.ols(formula='y ~ x1+x2', data=df_in).fit()
print(m1.summary())

"""
(d) The result is consistent with b1=2.15 
"""
m2 = smf.ols(formula='y ~ x1', data=df_in).fit()
print(m2.summary())

"""
(e) We can use the ranges of y an x2 to estimate the coefficient. 
    It is about 3.1. The result is consistent.
"""
m3 = smf.ols(formula='y ~ x2', data=df_in).fit()
print(m3.summary())

"""
(f) As we have discussed. The results are all consistent.
"""


print("The end of the code")