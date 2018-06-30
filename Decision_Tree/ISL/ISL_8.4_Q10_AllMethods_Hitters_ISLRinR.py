import numpy as np
import pandas as pd
import crossvalidation as acv
import ensemble as aes
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

# ====== get data ======
response = 'Salary'
df_in = pd.read_csv('data/Hitters_ISLRinR.csv', header=0)
df_in.drop(columns=['Unnamed: 0', 'League', 'Division', 'NewLeague'], inplace=True)
df_in.dropna(inplace=True)
print(df_in.head())

# ====== scale df and split train_test_dfs for CV ======
cv = acv.CrossValidation(df_in)
cv.train_test_split(k_folds_=5, scale_=False)


print("============ Regression Tree ============")
max_depths = np.arange(1, 20, 1)
models_RegrTree = [DecisionTreeRegressor(max_depth=m, ) for m in max_depths]
errors_RegrTree, errors_std_RegrTree = cv.cv_get_test_errors(models_RegrTree, response, plot_=True,lambdas_=max_depths)
print('Regression Tree: error_mean={}, error_std={}'.format(errors_RegrTree[2], errors_std_RegrTree[2]))


print("============ Bagging ============")
n_bags = np.arange(1, 20, 1)
models_bagging = list()
for b in n_bags:
    models = [DecisionTreeRegressor(max_depth=3, ) for _ in range(b) ]
    bagging_model = aes.Bagging(models)
    models_bagging.append(bagging_model)
errors_bag, errors_std_bag = cv.cv_get_test_errors(models_bagging, response, plot_=True,lambdas_=n_bags)
print('Bagging: error_mean={}, error_std={}'.format(errors_bag[4], errors_std_bag[4]))


print("============ Random Forest ============")
max_depths = np.arange(1, 20, 1)
models_RF = [RandomForestRegressor(max_depth=m) for m in max_depths]
errors_RF, errors_std_RF = cv.cv_get_test_errors(models_RF, response, plot_=True,lambdas_=max_depths)
print('Random Forest: error_mean={}, error_std={}'.format(errors_RF[2], errors_std_RF[2]))


print("============ Residual Boosting ============")
n_RB = np.arange(1, 20, 1)
models_RB = list()
for b in n_bags:
    models = [DecisionTreeRegressor(max_depth=3, ) for _ in range(b) ]
    RB_model = aes.ResidualBoosting(models)
    models_RB.append(RB_model)
errors_RB, errors_std_RB = cv.cv_get_test_errors(models_RB, response, plot_=True,lambdas_=n_RB)
print('Residual Boosting: error_mean={}, error_std={}'.format(errors_RB[3], errors_std_RB[3]))


print("============ Ada Boosting ============")
max_depths = np.arange(1, 50, 1)
models_Ada = [AdaBoostRegressor(DecisionTreeRegressor(max_depth=3, ), n_estimators=m) for m in max_depths]
errors_Ada, errors_std_Ada = cv.cv_get_test_errors(models_Ada, response, plot_=True,lambdas_=max_depths)
print('Regression Tree: error_mean={}, error_std={}'.format(errors_Ada[2], errors_std_Ada[2]))


