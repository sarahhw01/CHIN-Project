import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('trained_random_forest_model.pkl')

# to-do: this script should take a parameter with the test data path
# read in the test data file
# test the model with this test data file