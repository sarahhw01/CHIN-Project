from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample 

## STEP 1: GET DATA AND FEATURES
# get the compounds
'''compounds = Chem.SDMolSupplier('chin-qspr-dataset.sdf')

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
descriptors_df.to_csv('all_descriptors.csv')'''

descriptors_df = pd.read_csv('all_descriptors.csv', index_col=0)
print(descriptors_df.head())

# STEP 2: SELECT FEATURES 
#pLC50 to numerical
descriptors_df['pLC50'] = pd.to_numeric(descriptors_df['pLC50'], errors='coerce')


# Convert numerical columns to appropriate types
numerical_cols = descriptors_df.select_dtypes(include=['float64', 'int64']).columns

# STEP 3: TURN pLC50 INTO CATEGORICAL FEATURE
# group into different levels of toxicity
# Define the category edges and corresponding labels
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

plt.hist(descriptors_df['pLC50'], bins=5, edgecolor='k')
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
descriptors_df.to_csv('categorical_dataset_all_Descriptors.csv')