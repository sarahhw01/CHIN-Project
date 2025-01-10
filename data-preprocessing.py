from rdkit import Chem
import pandas as pd

compounds = Chem.SDMolSupplier('chin-qspr-dataset.sdf')
print(compounds)

data = []

for mol in compounds:
        smiles = Chem.MolToSmiles(mol)
        plc50 = mol.GetProp('pLC50')
        compound_id = mol.GetProp('compound_id')
        data.append({'Compound ID': compound_id, 'SMILES': smiles, 'pLC50': plc50})

        

compound_df = pd.DataFrame(data)
print(compound_df)

