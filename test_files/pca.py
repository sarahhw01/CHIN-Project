import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('all_descriptors.csv', index_col=0)

# split data into X and y, where y is the pLC50 value
X = data.drop(['compound_id', 'smiles', 'pLC50'], axis=1)
y = data['pLC50']

# Step 1: Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # Extract 2 principal components
principal_components = pca.fit_transform(scaled_data)

import matplotlib.pyplot as plt

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Explained Variance vs. Number of Components")
plt.show()


# Get the absolute values of PCA components
pca_components = abs(pca.components_)

# Sum of component contributions for each original feature
feature_importance = pca_components.sum(axis=0)

# Create a DataFrame for easy viewing
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df = feature_importance_df[0:20]['Feature']
top_features_columns = list(feature_importance_df)
print(top_features_columns)

subset_descriptors_df = data[top_features_columns]
print(subset_descriptors_df.head())