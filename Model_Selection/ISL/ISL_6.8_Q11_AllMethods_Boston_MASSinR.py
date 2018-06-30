import numpy as np
import pandas as pd
from sklearn import linear_model
import crossvalidation as acv
import dimensionreduction as add
from sklearn.cross_decomposition import PLSRegression


"""
(a) Split the data set into a training set and a test set.
"""
# ====== get data ======
df_in = pd.read_csv('data/Boston_MASSinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
print(df_in.head())

## ====== scale df and split train_test_dfs for CV ======
cv = acv.CrossValidation(df_in)
cv.scale_predictors('crim')
cv.train_test_split(k_folds_=5)


"""
(b) Fit a linear model using least squares on the training set, and
    report the test error obtained.
"""
print("============ Q(b) ============")
model_ols = linear_model.LinearRegression()
error_ols, error_std_ols = cv.cv_get_test_errors([model_ols], 'crim')
print('OLS: error_mean={}, error_std={}'.format(error_ols, error_std_ols))


"""
(c) Fit a ridge regression model on the training set, with λ chosen
    by cross-validation. Report the test error obtained.
"""
print("============ Q(c) ============")
lambdas = np.arange(0.1, 1, 0.1)
models_ridge = [linear_model.Ridge(lda) for lda in lambdas]
errors_ridge, errors_std_ridges = cv.cv_get_test_errors(models_ridge, 'crim', plot_=True,lambdas_=lambdas)
print('Ridge: error_mean={}, error_std={}'.format(errors_ridge[0], errors_std_ridges[0]))

"""
(d) Fit a lasso regression model on the training set, with λ chosen
    by cross-validation. Report the test error obtained.
"""
print("============ Q(d) ============")
lambdas = np.arange(0.1, 1, 0.1)
models_lasso = [linear_model.Lasso(lda) for lda in lambdas]
errors_lasso, errors_std_lasso = cv.cv_get_test_errors(models_lasso, 'crim', plot_=True,lambdas_=lambdas)
print('Lasso: error_mean={}, error_std={}'.format(errors_lasso[0], errors_std_lasso[0]))

"""
(e) Fit a PCR model on the training set, with M chosen by cross-validation. 
    Report the test error obtained, along with the value of M selected 
    by cross-validation.
"""
print("============ Q(e) ============")
m_dims = np.arange(1, len(df_in.columns), 1)
lm_ols = linear_model.LinearRegression()
models_pcr = [add.PCR(m_dim, lm_ols) for m_dim in m_dims]
errors_pcr, errors_std_pcr = cv.cv_get_test_errors(models_pcr, 'crim', plot_=True,lambdas_=m_dims)
print('PCR: error_mean={}, error_std={}'.format(errors_pcr[3], errors_std_pcr[3]))

"""
(f) Fit a PLS model on the training set, with M chosen by cross-validation. 
    Report the test error obtained, along with the value of M selected 
    by cross-validation.
"""
print("============ Q(f) ============")
m_dims = np.arange(1, len(df_in.columns), 1)
models_pls = [PLSRegression(n_components=m_dim) for m_dim in m_dims]
errors_pls, errors_std_pls = cv.cv_get_test_errors(models_pls, 'crim', plot_=True,lambdas_=m_dims)
print('PLS: error_mean={}, error_std={}'.format(errors_pls[0], errors_std_pls[0]))
