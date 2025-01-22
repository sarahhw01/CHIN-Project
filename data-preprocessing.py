from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
from sklearn.utils import resample 
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

## STEP 1: GET DATA AND FEATURES
# get the compounds
compounds = Chem.SDMolSupplier('chin-qspr-dataset.sdf')

# Function to compute descriptors for a molecule
def compute_descriptors(mol):
    # Extract descriptors
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptor_values = [desc[1](mol) for desc in Descriptors.descList]
    descriptors = dict(zip(descriptor_names, descriptor_values))
    
    # Add additional properties
    descriptors['smiles'] = Chem.MolToSmiles(mol)  # SMILES representation
    descriptors['pLC50'] = mol.GetProp('pLC50') if mol.HasProp('pLC50') else None  # pLC50
    descriptors['compound_id'] = mol.GetProp('_Name') if mol.HasProp('_Name') else "Unknown"  # Compound ID
    
    return descriptors

# Apply descriptor extraction to each molecule
descriptors = []
for mol in compounds:  # suppl is the SDF supplier loaded previously
    if mol is not None:
        #descriptors
        desc = compute_descriptors(mol)
        desc['compound_id'] = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
        descriptors.append(desc)

# Convert to DataFrame
descriptors_df = pd.DataFrame(descriptors)
print(descriptors_df.head())
descriptors_df.to_csv('all_descriptors.csv')

descriptors_df = pd.read_csv('all_descriptors.csv', index_col=0)
print(descriptors_df.head())

max_pLC50 = descriptors_df['pLC50'].max()
min_pLC50 = descriptors_df['pLC50'].min()
print('Max pLC50 value: ', max_pLC50)
print('Min pLC50 value: ', min_pLC50)

# STEP 2: TURN pLC50 INTO CATEGORICAL FEATURE
# group into different levels of toxicity
# Define the category edges and corresponding labels
plt.hist(descriptors_df['pLC50'], bins=50, edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()


bin_edges = [-float('inf'), 3.0, 4.0, 5.0, 6.0, float('inf')]
bin_labels = [
    "Practically non-toxic",
    "Slightly toxic",
    "Moderately toxic",
    "Highly toxic",
    "Extremely toxic"
]

# exclude the one outlier we have (pLC50 values usually range from 0 to 10)
print(sum(descriptors_df['pLC50'] > 10))  # we find one outlier
descriptors_df = descriptors_df[descriptors_df['pLC50'] < 10]
print(sum(descriptors_df['pLC50'] > 10)) # should be zero now
# Assign toxicity categories based on pLC50 ranges
descriptors_df["toxicity_category"] = pd.cut(descriptors_df["pLC50"], bins=bin_edges, labels=bin_labels, right=False)
print(descriptors_df.head())
print(descriptors_df['toxicity_category'].value_counts())

plt.hist(descriptors_df['pLC50'], edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()

# Define mapping of categories to numerical values
toxicity_mapping = {
    "Practically non-toxic": 1,
    "Slightly toxic": 2,
    "Moderately toxic": 3,
    "Highly toxic": 4,
    "Extremely toxic": 5
}

# Apply mapping
descriptors_df["toxicity_numeric"] = descriptors_df["toxicity_category"].map(toxicity_mapping)
print(descriptors_df)
descriptors_df.to_csv('categorical_dataset.csv')


# STEP 3: SELECT FEATURES 
#pLC50 to numerical
descriptors_df['pLC50'] = pd.to_numeric(descriptors_df['pLC50'], errors='coerce')


# Convert numerical columns to appropriate types
numerical_cols = descriptors_df.select_dtypes(include=['float64', 'int64']).columns

# Compute correlations
correlations = descriptors_df[numerical_cols].corr()['pLC50'].sort_values(ascending=False)

# Print the top correlated descriptors
print("Top positively correlated descriptors:")
print(correlations.head(20))
# get the top 20 positive correlations and subset the dataset by including them
top_correlations = correlations.head(20)  # Top 20 positive correlations
top_correlation_columns = list(top_correlations.index)
subset_descriptors_df = descriptors_df[top_correlation_columns]
subset_descriptors_df.insert(0, 'smiles', descriptors_df['smiles'])
subset_descriptors_df.insert(0, 'compound_id', descriptors_df['compound_id'])
subset_descriptors_df.insert(len(subset_descriptors_df.columns), 'toxicity_category', descriptors_df['toxicity_category'])
subset_descriptors_df.insert(len(subset_descriptors_df.columns), 'toxicity_numeric', descriptors_df['toxicity_numeric'])
print(subset_descriptors_df)
subset_descriptors_df.to_csv('subset_descriptors.csv')

print("\nTop negatively correlated descriptors:")
print(correlations.tail(10))  # Top 10 negative correlations

subset_descriptors_df = pd.read_csv('subset_descriptors.csv', index_col=0)
print(subset_descriptors_df.head())

max_pLC50 = subset_descriptors_df['pLC50'].max()
min_pLC50 = subset_descriptors_df['pLC50'].min()
print('Max pLC50 value: ', max_pLC50)
print('Min pLC50 value: ', min_pLC50)


# STEP 4: OVERSAMPLE SMALLER CLASSES SO THAT THE DATASET IS BALANCED
# Oversample categories with low sample count
print(subset_descriptors_df['toxicity_numeric'].value_counts())

category_1 = subset_descriptors_df[subset_descriptors_df['toxicity_numeric'] == 1]
category_2 = subset_descriptors_df[subset_descriptors_df['toxicity_numeric'] == 2]
category_3 = subset_descriptors_df[subset_descriptors_df['toxicity_numeric'] == 3]
category_4 = subset_descriptors_df[subset_descriptors_df['toxicity_numeric'] == 4]
category_5 = subset_descriptors_df[subset_descriptors_df['toxicity_numeric'] == 5]
# use the largest class as size we want all classes to have
target_category_size = category_2.shape[0]
#print(target_category_size)
# oversample smaller categories
category_1_sampled = resample(category_1, replace=True, n_samples=target_category_size, random_state=42)
category_2_sampled = resample(category_2, replace=True, n_samples=target_category_size, random_state=42)
category_3_sampled = resample(category_3, replace=True, n_samples=target_category_size, random_state=42)
category_4_sampled = resample(category_4, replace=True, n_samples=target_category_size, random_state=42)
category_5_sampled = resample(category_5, replace=True, n_samples=target_category_size, random_state=42)
# balance data
balanced_data = pd.concat([category_1_sampled, category_2_sampled, category_3_sampled, category_4_sampled, category_5_sampled])
# check if all categories have the same sample size
print(balanced_data['toxicity_numeric'].value_counts())
print(balanced_data['toxicity_category'].value_counts())

balanced_data.to_csv('balanced_data_same_sample_sizes.csv')