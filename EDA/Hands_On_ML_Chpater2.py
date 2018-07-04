import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ========== 1. read data and show basic information ===========
df_original = pd.read_csv('data/housing.csv', header=0)
print('================== data information ==================')
print('shape of data frame is {}'.format(df_original.shape))
print('---------------------------------------------------------------------')
print(df_original.head())
print('---------------------------------------------------------------------')
print(df_original.info())
print('---------------------------------------------------------------------')
print(df_original['ocean_proximity'].value_counts())
print('---------------------------------------------------------------------')
print(df_original.describe())
print('---------------------------------------------------------------------')



# ========== 2. train-test split ===========
df_in = df_original.sample(frac=0.8, replace=False).copy()
df_out = df_original[~df_original.index.isin(df_in.index)].copy()
print('shape of in sample      = {}'.format(df_in.shape))
print('shape of out of sample  = {}'.format(df_out.shape))
print('---------------------------------------------------------------------')
housing = df_in.copy()



# ========== 3. Visulization ===========
# --- a. histogram ---
housing.hist(bins=50, figsize=(20,15)) 
plt.show()
# --- b. scatter plot ---
housing.plot(kind='scatter', x="longitude", y="latitude", alpha=0.1
             , figsize=(10,8))
plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             figsize=(10,8))
plt.legend()
plt.show()
# --- c. correlation matrix ---
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
housing.plot(kind='scatter', x='median_income', y='median_house_value', figsize=(10,8))
plt.show()



# ========== 4. Handle data ===========
# --- a. Create Useful Predictors ---
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))
housing.plot(kind='scatter', x='median_income', y='total_rooms', figsize=(10,8))
plt.show()
housing.plot(kind='scatter', x='median_income', y='rooms_per_household', figsize=(10,8))
plt.show()

# --- b. Handling Missing Values ---
from sklearn.preprocessing import Imputer
housing_num = housing.drop(columns=["ocean_proximity", 'median_house_value'])
housing_response = housing['median_house_value'].copy()
imputer = Imputer(strategy='median')
imputer.fit(housing_num)
housing_tr = pd.DataFrame(imputer.transform(housing_num), 
                          columns=housing_num.columns, index=housing.index)

# --- c. Handling Text and Categorical Attributes ---
housing_cat = housing[["ocean_proximity"]]
housing_dummies = pd.get_dummies(housing_cat)
housing_dummies.rename(columns={'ocean_proximity_<1H OCEAN': '1HOcean',
                                'ocean_proximity_INLAND': 'Inland',
                                'ocean_proximity_ISLAND': 'Island',
                                'ocean_proximity_NEAR BAY': 'NearBay',
                                'ocean_proximity_NEAR OCEAN':'NearOcean'}, 
                        inplace=True)

housing_tr = pd.concat([housing_tr, housing_dummies], axis=1)

# --- D. Scale Features ---
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(housing_tr)
housing_tr_scale = pd.DataFrame(scaler.transform(housing_tr), 
                                columns=housing_tr.columns, index=housing_tr.index)



# ========== 4. Select and Train a Model ===========
# --- A. Fitting with lienar regression ---
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(housing_tr_scale, housing_response)
housing_preiction = lin_reg.predict(housing_tr_scale)
lin_mse = ((housing_response-housing_preiction)**2).mean()
print('RMSE of linear regression = {}'.format(np.sqrt(lin_mse)))

# --- B. Fitting Data with Regression Tree ---
from sklearn.tree import DecisionTreeRegressor 
tree_reg = DecisionTreeRegressor(max_depth=7)
tree_reg.fit(housing_tr_scale, housing_response)
housing_preiction = tree_reg.predict(housing_tr_scale)
tree_mse = ((housing_response-housing_preiction)**2).mean()
print('RMSE of tree regression   = {}'.format(np.sqrt(tree_mse)))

# --- C. Fitting Data with Random Forest ---
from sklearn.ensemble import RandomForestRegressor 
forest_reg = RandomForestRegressor(n_estimators=5)
forest_reg.fit(housing_tr_scale, housing_response)
housing_preiction = forest_reg.predict(housing_tr_scale)
forest_mse = ((housing_response-housing_preiction)**2).mean()
print('RMSE of random forest     = {}'.format(np.sqrt(forest_mse)))

# --- D. Cross-Validation ---
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_tr_scale, housing_response,
                         scoring="neg_mean_squared_error", cv=10)
lin_rmse = np.mean(-scores)
scores = cross_val_score(tree_reg, housing_tr_scale, housing_response,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse = np.mean(-scores)
scores = cross_val_score(forest_reg, housing_tr_scale, housing_response,
                         scoring="neg_mean_squared_error", cv=10)
forest_rmse = np.mean(-scores)
print('RMSE of linear regression from CV = {}'.format(np.sqrt(lin_rmse)))
print('RMSE of tree regression   from CV = {}'.format(np.sqrt(tree_rmse)))
print('RMSE of random forest     from CV = {}'.format(np.sqrt(forest_rmse)))



# ========== 5. Fine-Tune Your Model ===========
# --- A. Grid Search ---
from sklearn.model_selection import GridSearchCV
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_tr_scale, housing_response)
print('best parameters from grid search is', grid_search.best_params_)
#cvres = grid_search.cv_results_
#for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
#    print(np.sqrt(-mean_score), params)

# --- B. Randomized Search ---
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
param_dist = {
        'n_estimators': sp_randint(3, 30),
        "max_depth": [3, None],
        "max_features": sp_randint(1, 11),
        "min_samples_split": sp_randint(2, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "bootstrap": [True, False]}
n_iter_search = 20
forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg, param_distributions=param_dist,
                                   scoring='neg_mean_squared_error', n_iter=n_iter_search)
random_search.fit(housing_tr_scale, housing_response)
print('best parameters from random search is', random_search.best_params_)


