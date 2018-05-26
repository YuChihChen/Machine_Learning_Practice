import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


mean = 0
variance = 1
x = np.random.normal(loc=mean, scale=np.sqrt(variance), size=10000)
y = x+np.random.normal(loc=mean, scale=np.sqrt(variance), size=10000)
plt.scatter(x, y)
df_in = pd.DataFrame({'x': x, 'y': y})
print(df_in.head())
"""
(a) The bate^{hat} is 1.9226. It is statistical significant 
"""
m1 = smf.ols(formula='y ~ x+0', data=df_in).fit()
print(m1.summary())
"""
(b) 
"""
m2 = smf.ols(formula='x ~ y+0', data=df_in).fit()
print(m2.summary())



print("The end of the code")