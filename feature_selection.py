import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold

data = pd.read_csv('compound_data.csv', index_col=0)
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
params = {"alpha":np.arange(0.00001, 10, 500)}

# Number of Folds and adding the random state for replication
kf=KFold(n_splits=5,shuffle=True, random_state=42)

# Initializing the Model
lasso = Lasso()

# GridSearchCV with model, params and folds.
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y)
print("Best Params {}".format(lasso_cv.best_params_)) # --> best alpha value is 0.00001

# calling the model with the best parameter
lasso1 = Lasso(alpha=0.00001)
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

# --> LogP is by far the most important feature, after that comes aromatic bonds, the remaining features are similarly unimportant
# Subset data by including only important features, To-Do: try different combinations of features and check how this affects performance
compound_id = data['Compound ID']
SMILES = data['SMILES']
logP = data['LogP']
aromatic_bonds = data['AromaticBonds']
plc50 = data['pLC50']
#data_important_features = pd.concat([compound_id, SMILES, logP, aromatic_bonds, plc50], axis=1)
data_important_features = data.drop(['MolecularWeight'], axis=1)
print(data_important_features)
data_important_features.to_csv('important_features_data.csv')