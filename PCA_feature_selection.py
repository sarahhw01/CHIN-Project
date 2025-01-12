from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 

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

# Standardize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA

# Initialize PCA, specify the number of components or explained variance
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Print the number of components selected
print(f"Number of principal components selected: {pca.n_components_}")