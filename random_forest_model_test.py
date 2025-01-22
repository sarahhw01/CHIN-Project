import argparse
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
from rdkit import RDLogger
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Test model function
def test_model(path_to_sdf_file):

    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore")

    # Load the data and prepare the features that we used for training our model
    compounds = Chem.SDMolSupplier(path_to_sdf_file)

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

    # use features that we decided were important during feature importance analysis
    subset_descriptors_df = pd.read_csv('subset_descriptors.csv', index_col=0)
    col_names = list(subset_descriptors_df.columns)
    col_names.remove('toxicity_category')
    col_names.remove('toxicity_numeric')
    data = descriptors_df[col_names]

    # convert pLC50 to numerical
    data['pLC50'] = pd.to_numeric(data['pLC50'], errors='coerce')

    # assign categorical classes to ranges of pLC values
    bin_edges = [-float('inf'), 3.0, 4.0, 5.0, 6.0, float('inf')]
    bin_labels = [
        "Practically non-toxic",
        "Slightly toxic",
        "Moderately toxic",
        "Highly toxic",
        "Extremely toxic"
    ]

    # Assign toxicity categories based on pLC50 ranges
    data["toxicity_category"] = pd.cut(data["pLC50"], bins=bin_edges, labels=bin_labels, right=False)
    print('')
    print('Value counts for the Toxicity classes:')
    print('')
    print(data['toxicity_category'].value_counts())

    # Define mapping of categories to numerical values
    toxicity_mapping = {
        "Practically non-toxic": 1,
        "Slightly toxic": 2,
        "Moderately toxic": 3,
        "Highly toxic": 4,
        "Extremely toxic": 5
    }

    # Apply mapping
    data["toxicity_numeric"] = data["toxicity_category"].map(toxicity_mapping)
    print('')
    print('Dataset:')
    print('')
    print(data)

    # Now we can test our trained model
    # Load the pre-trained model
    model = joblib.load('trained_random_forest_model.pkl')
    # create X and y
    X_test = data.drop(['compound_id', 'smiles', 'pLC50', 'toxicity_category', 'toxicity_numeric'], axis=1)
    y_test = data['toxicity_numeric']
    # predict the y values and compute the 
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print("")
    print("Performance Results on Test data:")
    print("")
    print("Random Forest Test RMSE:", rmse)
    print("Random Forest Test R2 Score:", r2)
    print("")

    toxicity_mapping_reversed = {
        1: "Practically non-toxic",
        2: "Slightly toxic",
        3: "Moderately toxic",
        4: "Highly toxic",
        5: "Extremely toxic"
    }

    # Store results in csv file
    compound_id = data['compound_id']
    prediction_results = pd.DataFrame({'compound_id': compound_id, 'Prediction': y_pred})
    prediction_results["pred_pLC50_category"] = prediction_results["Prediction"].map(toxicity_mapping_reversed)
    prediction_results = prediction_results.drop(['Prediction'], axis=1)
    print(prediction_results)
    prediction_results.to_csv("prediction_results.csv")

    # Plot heatmap
    # Compute Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix Heatmap")
    plt.show()

    print(y_test.value_counts())


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="SDF MOL input file")
    args = parser.parse_args()
    sdf_file = args.i
    test_model(sdf_file)
    ## In order to run this use the following call:
    # python3 random_forest_model_test.py -i test_file_name.sdf