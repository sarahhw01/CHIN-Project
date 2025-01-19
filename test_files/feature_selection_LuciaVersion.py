import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Projects/Project/compound_data.csv', index_col=0)
print(data)

# split data into X and y, where y is the pLC50 value
X = data.drop(['Compound ID', 'SMILES', 'pLC50'], axis=1).values
y = data['pLC50'].values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_test.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_test.shape))

# parameters to be tested on GridSearchCV
params = {"alpha":np.logspace(-5,2,50)}

# Number of Folds and adding the random state for replication
kf=KFold(n_splits=5,shuffle=True, random_state=42)

# Initializing the Model
lasso = Lasso(max_iter=10000)

# GridSearchCV with model, params and folds.
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y)

#Best alpha from lasso
best_alpha = lasso_cv.best_params_['alpha']
print("Best Params: ", best_alpha) # --> best alpha value is 0.00001/0.05179474

# calling the model with the best parameter
lasso1 = Lasso(alpha=best_alpha, max_iter=1000)
lasso1.fit(X_train, y_train)

# Using np.abs() to make coefficients positive.  
lasso1_coef = np.abs(lasso1.coef_)

# plotting the Column Names and Importance of Columns. 
features = data.drop(['Compound ID', 'SMILES', 'pLC50'], axis=1).columns
print("Column Names: {}".format(features.values))

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(features, lasso1_coef)
fig.subplots_adjust(bottom=0.2)
plt.xticks(rotation=90, fontsize= 10)
plt.grid()
plt.title("Feature Selection Based on Lasso")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
'''
# --> LogP is by far the most important feature, after that comes aromatic bonds, the remaining features are similarly unimportant
# ---> Also MinPartialCharge and in lower amount MaxPartialCharge, more than aromatic bounds
# Subset data by including only important features, To-Do: try different combinations of features and check how this affects performance
data_important_features = data.drop(['MolecularWeight'], axis=1)
data_important_features = data.drop(['N_RadicalElectrons'], axis=1)
data_important_features = data.drop(['RingCount'], axis=1)
data_important_features = data.drop(['ExactMolWt'], axis=1)
print(data_important_features)
data_important_features.to_csv('important_features_data.csv')
'''

#Modification
# Filter features with non-zero importance
important_features = [features[i] for i in range(len(features)) if lasso1_coef[i] > 0]
print("Selected Features:", important_features)

# Subset the data with important features
data_important_features = data[important_features + ['Compound ID', 'SMILES', 'pLC50']]
data_important_features.to_csv('Projects/Project/important_features_data.csv', index=False)
print("Subsetted data saved to important_features_data.csv")



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