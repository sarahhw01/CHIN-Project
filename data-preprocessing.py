from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt 

'''# get the compounds
compounds = Chem.SDMolSupplier('chin-qspr-dataset.sdf')

data = []

# for each molecule in the sdf file, extract the smiles notation, the pLC50 value and its compound id
for mol in compounds:
        smiles = Chem.MolToSmiles(mol)
        plc50 = mol.GetProp('pLC50')
        compound_id = mol.GetProp('compound_id')
        # gather features from the sdf file 
        mol_weight = Descriptors.MolWt(mol)
        logP = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        N_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        N_H_donors = Descriptors.NumHDonors(mol)
        N_H_acceptors = Descriptors.NumHAcceptors(mol)
        heavy_atom_count = Descriptors.HeavyAtomCount(mol)
        ring_count = Descriptors.RingCount(mol)
        max_partial_charge = Descriptors.MaxAbsPartialCharge(mol)
        min_partial_charge = Descriptors.MinAbsPartialCharge(mol)
        N_radical_electrons = Descriptors.NumRadicalElectrons(mol)
        N_valence_electrons = Descriptors.NumValenceElectrons(mol)

        # append data 
        data.append({'Compound ID': compound_id, 'SMILES': smiles, 'MolecularWeight': mol_weight, 'LogP': logP, 'TPSA' : tpsa, 'N_RotatableBonds' : N_rotatable_bonds, 'AromaticBonds' : aromatic_rings, 'HeavyAtomCount' : heavy_atom_count, 'RingCount' : ring_count, 'N_H_Donors' : N_H_donors, 'N_H_Acceptors' : N_H_acceptors, 'MaxPartialCharge': max_partial_charge, 'MinPartialCharge': min_partial_charge, 'N_RadicalElectrons': N_radical_electrons, 'pLC50': plc50})   

# write result into a dataframe and csv file
compound_df = pd.DataFrame(data)
compound_df.to_csv('compound_data.csv')
print(compound_df)
 
# Look at distribution of the data
data = pd.read_csv('compound_data.csv') 
plt.hist(data['pLC50'], bins=50, edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()'''

## Lucias verion of getting the features
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

'''descriptors_df = pd.read_csv('all_descriptors.csv', index_col=0)
print(descriptors_df.head())
#pLC50 to numerical
descriptors_df['pLC50'] = pd.to_numeric(descriptors_df['pLC50'], errors='coerce')


# Convert numerical columns to appropriate types
numerical_cols = descriptors_df.select_dtypes(include=['float64', 'int64']).columns

# Compute correlations
correlations = descriptors_df[numerical_cols].corr()['pLC50'].sort_values(ascending=False)

# Print the top correlated descriptors
print("Top positively correlated descriptors:")
print(correlations.head(10))
# get the top 20 positive correlations and subset the dataset by including them
top_correlations = correlations.head(10)  # Top 20 positive correlations
top_correlation_columns = list(top_correlations.index)
subset_descriptors_df = descriptors_df[top_correlation_columns]
subset_descriptors_df.insert(0, 'smiles', descriptors_df['smiles'])
subset_descriptors_df.insert(0, 'compound_id', descriptors_df['compound_id'])
print(subset_descriptors_df)
subset_descriptors_df.to_csv('subset_descriptors.csv')

print("\nTop negatively correlated descriptors:")
print(correlations.tail(10))  # Top 10 negative correlations'''

# Calculate the LC50 values from the pLC5o values
'''subset_descriptors_df = pd.read_csv('subset_descriptors.csv', index_col=0)
print(subset_descriptors_df)

LC50_values = []
for value in subset_descriptors_df['pLC50']:
    LC50_value = 10 ** -value
    LC50_values.append(LC50_value)
print(LC50_values)

# convert to LC50 values with unit ppm (not sure if this is right)
LC50_mols = []
counter = 0
for mol_weight in subset_descriptors_df['MolWt']:
    LC50_mol = LC50_values[counter] * mol_weight 
    LC50_mols.append(LC50_mol)
    counter += 1
print(LC50_mols)

plt.hist(LC50_values,edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()'''


subset_descriptors_df = pd.read_csv('subset_descriptors.csv', index_col=0)
print(subset_descriptors_df.head())

max_pLC50 = subset_descriptors_df['pLC50'].max()
min_pLC50 = subset_descriptors_df['pLC50'].min()
print('Max pLC50 value: ', max_pLC50)
print('Min pLC50 value: ', min_pLC50)

# bin into different levels of toxicity
# Define the bin edges and corresponding labels
bin_edges = [-float('inf'), 3.0, 4.0, 5.0, 6.0, float('inf')]
bin_labels = [
    "Practically non-toxic",
    "Slightly toxic",
    "Moderately toxic",
    "Highly toxic",
    "Extremely toxic"
]

# exclude the one outlier we have (pLC50 values usually range from 0 to 10)
print(sum(subset_descriptors_df['pLC50'] > 10))  # we find one outlier
subset_descriptors_df = subset_descriptors_df[subset_descriptors_df['pLC50'] < 10]
print(sum(subset_descriptors_df['pLC50'] > 10)) # should be zero now
# Assign toxicity categories based on pLC50 ranges
subset_descriptors_df["toxicity_category"] = pd.cut(subset_descriptors_df["pLC50"], bins=bin_edges, labels=bin_labels, right=False)
print(subset_descriptors_df.head())
print(subset_descriptors_df['toxicity_category'].value_counts())

plt.hist(subset_descriptors_df['pLC50'], bins=5, edgecolor='k')
plt.xlabel('pLC50')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()

subset_descriptors_df.to_csv('categorical_dataset.csv')