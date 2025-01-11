from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.utils import resample

data = pd.read_csv('important_features_data.csv', index_col=0)

# split data into X and y, where y is the pLC50 value
X = data.drop(['Compound ID', 'SMILES', 'pLC50'], axis=1)
y = data['pLC50']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_test.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_test.shape))
# Plot the distribution of pLC50 values
plt.hist(y_train, bins=50, edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()

model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}, R2: {r2}")
# parameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)


balanced_data = pd.read_csv('balanced_data_second_version.csv', index_col=0)
print(balanced_data)

# split data into X and y, where y is the pLC50 value
X = balanced_data.drop(['Compound ID', 'SMILES', 'pLC50', 'pLC50_bin'], axis=1)
y = balanced_data['pLC50_bin']
print(balanced_data['pLC50_bin'].value_counts())
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_test.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_test.shape))
print(y_train.value_counts())
print(y_test.value_counts())

model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}, R2: {r2}")
# parameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)


