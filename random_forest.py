from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('categorical_dataset.csv', index_col=0)

# split data into X and y, where y is the pLC50 value
X = data.drop(['compound_id', 'smiles', 'pLC50', 'toxicity_category', 'toxicity_numeric'], axis=1)
y = data['toxicity_numeric']

# Train Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_val.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_val.shape))

# Initialize Random Forest Model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV for Random Forest
rf_cv = GridSearchCV(rf_model, param_grid=rf_params, cv=kf, scoring='r2')
rf_cv.fit(X_train, y_train)

# Best Random Forest parameters
best_rf_params = rf_cv.best_params_
print("Best Random Forest Parameters:", best_rf_params)

# Train Random Forest with best parameters
rf_best = RandomForestRegressor(**best_rf_params, random_state=42)
rf_best.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = rf_best.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print("Random Forest Test MSE:", mse)
print("Random Forest Test R2 Score:", r2)

## To prevent overfitting, checl k-fold cross validation and the mean RMSE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(-scores)
print("Mean RMSE:", -scores.mean())

'''model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluate performance
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)
print(f"RMSE: {rmse}, R2: {r2}")
# parameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)
'''

'''balanced_data = pd.read_csv('balanced_data_version_two.csv', index_col=0)
print(balanced_data)

# split data into X and y, where y is the pLC50 value
X = balanced_data.drop(['Compound ID', 'SMILES', 'pLC50', 'pLC50_bin'], axis=1)
y = balanced_data['pLC50_bin']
print(balanced_data['pLC50_bin'].value_counts())
# Train Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_val.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_val.shape))
print(y_train.value_counts())
print(y_val.value_counts())

model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluate performance
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)
print(f"RMSE: {rmse}, R2: {r2}")
# parameter tuning
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(best_model)

# save the trained model into a file
joblib.dump(model, 'trained_random_forest_model.pkl')

## To prevent overfitting, checl k-fold cross validation and the mean RMSE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=20, n_estimators=50, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(-scores)
print("Mean RMSE:", -scores.mean())'''