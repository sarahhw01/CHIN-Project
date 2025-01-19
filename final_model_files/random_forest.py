from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('balanced_data_same_sample_sizes.csv', index_col=0)

# split data into X and y, where y is the pLC50 value
X = data.drop(['compound_id', 'smiles', 'pLC50', 'toxicity_category', 'toxicity_numeric'], axis=1)
y = data['toxicity_numeric']

# Train Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
print("Shape of Train Features: {}".format(X_train.shape))
print("Shape of Test Features: {}".format(X_val.shape))
print("Shape of Train Target: {}".format(y_train.shape))
print("Shape of Test Target: {}".format(y_val.shape))

# Initialize Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
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
rf_best = RandomForestClassifier(**best_rf_params, random_state=42)
rf_best.fit(X_train, y_train)

# Evaluate the model on validation data
y_pred = rf_best.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
print("Random Forest Test MSE:", mse)
print("Random Forest Test R2 Score:", r2)

# to check if our model overfits, we compute the performance on the training data
# Evaluate the model on train data
y_pred = rf_best.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)
print("Random Forest Test MSE:", mse)
print("Random Forest Test R2 Score:", r2)

feature_importances = pd.Series(rf_best.feature_importances_, index=(X.columns).sort_values(ascending=False))
print(feature_importances)
