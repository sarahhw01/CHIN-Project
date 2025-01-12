from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt 

# get the compounds
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
plt.show()

