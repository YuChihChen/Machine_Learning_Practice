import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from sklearn.ensemble import RandomForestRegressor

N = 500
data_size = 1000
s = 0.9
# ======================== Three Parameters ========================
response = 'Salary'
df_in = pd.read_csv('data/Hitters_ISLRinR.csv', header=0)
df_in.drop(columns=['Unnamed: 0', 'League', 'Division', 'NewLeague'], inplace=True)
df_in.dropna(inplace=True)
X_columns = list(df_in.columns)
X_columns.remove(response)
X_orig, y_orig = df_in.loc[:, X_columns].values, df_in.loc[:, [response]].values

se_fake = df_in['CHits'].copy()
r = 0.1
for i in range(len(df_in.index)):
    if np.random.rand() < r:
        se_fake.iloc[i] = np.random.rand(1)[0]

X_orig = np.c_[X_orig, se_fake]
X_columns = X_columns + ['CHits2']

print('correlation between Chits and fake: ', np.corrcoef(df_in['CHits'], se_fake)[0,1])
print(X_orig.shape)
print(y_orig.shape)


N = 1000
forest = RandomForestRegressor(n_estimators=N, max_depth=10)
forest.fit(X_orig, y_orig)
fi_forest = pd.Series(forest.feature_importances_, index=X_columns).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_forest.index)), fi_forest)
plt.xticks(range(len(fi_forest.index)),fi_forest.index)
plt.xticks(rotation=30)
plt.show()