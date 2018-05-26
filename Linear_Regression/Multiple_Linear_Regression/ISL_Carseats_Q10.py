import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# get data
df_in = pd.read_csv('../Data/Carseats_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace =True)
print(df_in.head(5))
print(df_in.columns)

"""
(a): sales ~ Price+Urban+US
"""
m1 = smf.ols(formula='Sales ~ Price+Urban+US', data=df_in).fit()
print(m1.summary())
m1_urban = smf.ols(formula='Sales ~ Urban', data=df_in).fit()
print(m1_urban.summary())
"""
(b)
Price: as price increase, the sales will decrease. Reasonable
US: If the store is in US, the sales will be larger
Urban: There is no significant relationships between urban and sales 

(c) from model m1, we have
sales = 13.0435 - 0.0545*Price + 1.2006*IsUS 

(d) We can reject H0 for Price and US

(e) from model m2
sales = 13.0308 - 0.0545*Price + 1.1996*IsUS 
"""
m2 = smf.ols(formula='Sales ~ Price+US', data=df_in).fit()
print(m2.summary())

"""
(f) model 1 and 2 have similar performance

(g) the confidence intervals are
"""
print(m2.conf_int(alpha=0.05, cols=None))
"""
(h) There are no outliers and high leverage points
"""
infl = m2.get_influence()
print('number of outliers is {}'.format(sum(infl.resid_studentized_external > 3)))
print(infl.cooks_distance)