import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('all_descriptors.csv', index_col=0)
data = data.drop(['compound_id', 'smiles', 'pLC50'], axis=1)
# Step 1: Standardize the features (mean=0, variance=1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Extract 2 principal components
principal_components = pca.fit_transform(scaled_data)

# Step 3: Create a DataFrame with the transformed data
pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])

# Step 4: Explained variance ratio (to decide number of components)
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")

# Step 5: Plot variance explained by each principal component
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()

# Step 6: Show PCA-transformed data
print(pca_df.head())