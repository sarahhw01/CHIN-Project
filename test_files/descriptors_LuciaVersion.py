from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

# get the compounds
compounds = Chem.SDMolSupplier('Projects/Project/chin-qspr-dataset.sdf')

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

#pLC50 to numerical
descriptors_df['pLC50'] = pd.to_numeric(descriptors_df['pLC50'], errors='coerce')


# Convert numerical columns to appropriate types
numerical_cols = descriptors_df.select_dtypes(include=['float64', 'int64']).columns

# Compute correlations
correlations = descriptors_df[numerical_cols].corr()['pLC50'].sort_values(ascending=False)

# Print the top correlated descriptors
print("Top positively correlated descriptors:")
print(correlations.head(10))  # Top 10 positive correlations

print("\nTop negatively correlated descriptors:")
print(correlations.tail(10))  # Top 10 negative correlations



'''
# Function to compute Morgan fingerprint
def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)

# Add Morgan fingerprint to the DataFrame
fingerprints = []
for mol in compounds:
    if mol is not None:
        fp = compute_morgan_fingerprint(mol)
        compound_id = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
        fingerprints.append({'compound_id': compound_id, 'fingerprint': fp})

# Convert to DataFrame
fingerprints_df = pd.DataFrame(fingerprints)
print(fingerprints_df.head())

# Combine descriptors and fingerprints
combined_df = descriptors_df.merge(fingerprints_df, on='compound_id')
print(combined_df.head())
'''
