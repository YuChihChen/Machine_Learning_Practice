import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf



"""
(a)(b) generate simulated data and plot
"""
n = 100
mean = 0
variance = 1
x = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
y = x-2*(x**2)+np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
plt.scatter(x, y)
plt.show()
df_in = pd.DataFrame({'x': x, 'y': y})
print(df_in.head())

"""
(c) 
Question:  
    Set a random seed, and then compute the LOOCV errors 
    that result from fitting the following four models using least squares:
    i.   Y = β0 + β1X + ε
    ii.  Y = β0 + β1X + β2X2 + ε
    iii. Y = β0 + β1X + β2X2 + β3X3 + ε
    iv.  Y = β0 + β1X + β2X2 + β3X3 + β4X4 + ε.
My though before simulation:
    1. LOOCV will underestimated the variance error of model
    2. LOOCV will give O(n) accurate of biase error of model
    3. LOOCV will give O(n) accurate of noise error of model
    Therefore, in general LOOCV will unserestimate testing error
    I will do more this research in other python files in this folder
    
"""
def MSE_LOOCV_ols(df_, formula_):
    pass
    n = len(df_.index)
    errors_LOOCV = list()
    yp = list()
    for i in range(n):
        df_test = df_[df_.index.isin([i])]
        df_train = df_[~df_.index.isin([i])]
        model = smf.ols(formula=formula_, data=df_train).fit()
        y_pred = model.predict(df_test.iloc[:, 0])
        yp.append(y_pred.values[0])
        errors_LOOCV.append(((df_test['y']-y_pred).values[0])**2)
    plt.scatter(df_.iloc[:,0].values, yp)
    plt.title("MSE: {}".format(sum(errors_LOOCV)/len(errors_LOOCV)))
    plt.show()
    return sum(errors_LOOCV)/len(errors_LOOCV)
formulas = ['y ~ x', 
            'y ~ x + np.power(x, 2)', 
            'y ~ x + np.power(x, 2) + np.power(x, 3)', 
            'y ~ x + np.power(x, 2) + np.power(x, 3) + np.power(x, 4)']
MSEs = [MSE_LOOCV_ols(df_in, formula) for formula in formulas]      
print(MSEs)

TrueMSEs = list()
# get true test errors
for _ in range(100):
    x = np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    y = x-2*(x**2)+np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    df_in = pd.DataFrame({'x': x, 'y': y})
    TrueMSEs.append(MSE_LOOCV_ols(df_in, 'y ~ x + np.power(x, 2)'))
print('true testing error is about: {}'.format(sum(TrueMSEs)/len(TrueMSEs))) 
plt.hist(TrueMSEs)     
plt.show()
