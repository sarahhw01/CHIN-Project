import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data with important features
data = pd.read_csv('Projects/Project/important_features_data.csv')
print(data.head())

# Split data into X (features) and y (target), dropping unnecessary columns
X = data.drop(['Compound ID', 'SMILES', 'pLC50'], axis=1).values
y = data['pLC50'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Train Features Shape:", X_train.shape)
print("Test Features Shape:", X_test.shape)

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
y_pred = rf_best.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest Test MSE:", mse)
print("Random Forest Test R2 Score:", r2)

# Plot true vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("True vs Predicted pLC50")
plt.xlabel("True pLC50")
plt.ylabel("Predicted pLC50")
plt.grid(True)
plt.show()

###############
'''
# Subset the data with important features
X_train_important = pd.DataFrame(X_train, columns=features)[important_features].values
X_test_important = pd.DataFrame(X_test, columns=features)[important_features].values

# Save the subsetted data for future reference
data_important_features = data[important_features + ['Compound ID', 'SMILES', 'pLC50']]
#data_important_features.to_csv('important_features_data.csv', index=False)
#print("Subsetted data saved to important_features_data.csv")

# Initialize Random Forest Model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# GridSearchCV for Random Forest
rf_cv = GridSearchCV(rf_model, param_grid=rf_params, cv=kf, scoring='r2')
rf_cv.fit(X_train_important, y_train)

# Best Random Forest parameters
best_rf_params = rf_cv.best_params_
print("Best Random Forest Parameters:", best_rf_params)

# Train Random Forest with best parameters
rf_best = RandomForestRegressor(**best_rf_params, random_state=42)
rf_best.fit(X_train_important, y_train)

# Evaluate the model on test data
y_pred = rf_best.predict(X_test_important)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Random Forest Test MSE:", mse)
print("Random Forest Test R2 Score:", r2)

# Plot true vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("True vs Predicted pLC50")
plt.xlabel("True pLC50")
plt.ylabel("Predicted pLC50")
plt.grid(True)
plt.show()
'''