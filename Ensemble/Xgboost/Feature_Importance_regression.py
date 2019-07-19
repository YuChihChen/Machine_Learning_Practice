import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# ============ get data ============
response = 'Salary'
df_in = pd.read_csv('data/Hitters_ISLRinR.csv', header=0)
df_in.drop(columns=['Unnamed: 0', 'League', 'Division', 'NewLeague'], inplace=True)
df_in.dropna(inplace=True)
X_columns = list(df_in.columns)
X_columns.remove(response)
X_orig, y_orig = df_in.loc[:, X_columns].values, df_in.loc[:, [response]].values

print(X_orig.shape)
print(y_orig.shape)


N = 1000
# ============ sklearn: Random Forest ============
forest = RandomForestRegressor(n_estimators=N, max_depth=6)
forest.fit(X_orig, y_orig)
fi_forest = pd.Series(forest.feature_importances_, index=X_columns).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_forest.index)), fi_forest)
plt.xticks(range(len(fi_forest.index)),fi_forest.index)
plt.xticks(rotation=30)
plt.show()


# ============ xgboost ============
xgb = xgboost.XGBRegressor(n_estimators=N, max_depth=6)
xgb.fit(X_orig, y_orig)
fi_xgb = pd.Series(xgb.feature_importances_, index=X_columns).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_xgb.index)), fi_xgb)
plt.xticks(range(len(fi_xgb.index)), fi_xgb.index)
plt.xticks(rotation=30)
plt.show()
