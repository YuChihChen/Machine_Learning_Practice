import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.ensemble import RandomForestRegressor

N = 500
data_size = 1000
s = 0
# ======================== Three Parameters ========================
#mean = [0, 0, 0]
#cov  = [[1, 0.9, 0], [0.9, 1, 0], [0, 0, 1]]
mean = [0, 0]
cov  = [[1, s], [s, 1]]
X = np.random.multivariate_normal(mean, cov, data_size)
y = X[:, 0] + np.random.rand(data_size)*2
Z = np.array([np.random.random(data_size), np.random.random(data_size)]).T
X = np.c_[X, X ** 2]
X = np.c_[X, Z]

forest = RandomForestRegressor(n_estimators=N)
forest.fit(X, y)
fi_forest = pd.Series(forest.feature_importances_).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_forest.index)), fi_forest)
plt.xticks(range(len(fi_forest.index)),fi_forest.index)
plt.xticks(rotation=30)
plt.show()

xgb = xgboost.XGBRegressor(n_estimators=N)
xgb.fit(X, y)
fi_xgb = pd.Series(xgb.feature_importances_).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_xgb.index)), fi_xgb)
plt.xticks(range(len(fi_xgb.index)),fi_xgb.index)
plt.xticks(rotation=30)
plt.show()


print(np.corrcoef(X.T))

plt.scatter(X[:, 0], y)
plt.show()

plt.scatter(X[:, 1], y)
plt.show()
