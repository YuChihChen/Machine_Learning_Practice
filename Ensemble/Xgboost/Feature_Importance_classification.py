import numpy as np
import pandas as pd
import xgboost
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# ============ get data ============
df_in = pd.read_csv('data/Caravan_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace=True)
df_in['y_fac'] = pd.factorize(df_in.iloc[:, -1])[0]
print(df_in.head(5))
X_columns = list(df_in.columns[:-2])
X_orig = df_in.iloc[:, 0:-2].values
y_orig = df_in.iloc[:, -1].values




N = 250
# ============ sklearn: Random Forest ============
forest = RandomForestClassifier(n_estimators=N, max_depth=6)
forest.fit(X_orig, y_orig)
fi_forest = pd.Series(forest.feature_importances_, index=X_columns).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_forest.index)), fi_forest)
plt.xticks(range(len(fi_forest.index)),fi_forest.index)
plt.xticks(rotation=30)
plt.show()


## ============ xgboost ============
xgb = xgboost.XGBClassifier(n_estimators=N, max_depth=6)
xgb.fit(X_orig, y_orig)
fi_xgb = pd.Series(xgb.feature_importances_, index=X_columns).sort_values()
plt.figure(figsize=(10,8))
plt.bar(range(len(fi_xgb.index)), fi_xgb)
plt.xticks(range(len(fi_xgb.index)), fi_xgb.index)
plt.xticks(rotation=30)
plt.show()

print('========= random forest =======')
print(fi_forest)
print('========= xgboost =======')
print(fi_xgb)

forlist = list()
itr = 20
for _ in range(itr):
    xgb = xgboost.XGBClassifier(n_estimators=N, max_depth=6)
    xgb.fit(X_orig, y_orig)
    fi_xgb = pd.Series(xgb.feature_importances_, index=X_columns).sort_values(ascending=False)
    forlist.append(list(fi_xgb.index[0:3].values))
    print(fi_xgb.iloc[0:3])
print(pd.DataFrame(forlist))



