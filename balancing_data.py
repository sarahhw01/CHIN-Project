import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

data = pd.read_csv('important_features_data.csv', index_col=0)
print(data)
# balance the data
# Define bins (you can adjust bin edges based on your data)
bins = np.linspace(data['pLC50'].min(), data['pLC50'].max(), num=10)  # 10 bins
data['pLC50_bin'] = pd.cut(data['pLC50'], bins=bins, labels=False)
print(bins)
print(data['pLC50_bin'].value_counts())
plt.hist(data['pLC50_bin'], bins=50, edgecolor='k')
plt.xlabel('pLC50 Bins')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()

# drop na and outlier, which is bin 8 with only 1 occurence
data = data.dropna()
data = data[data['pLC50_bin'] != 8]

# check how many records we have per bin and then oversample minority bins
bin_counts = data['pLC50_bin'].value_counts()
# Separate the minority and majority bins
bin_1 = data[data['pLC50_bin'] == 1]
bin_0 = data[data['pLC50_bin'] == 0]
bin_2 = data[data['pLC50_bin'] == 2]

target_count = bin_1.shape[0]
#target_count = (bin_0.shape[0] + bin_1.shape[0] + bin_2.shape[0])/3
#target_count = round(target_count)
print(target_count)
bin_0_oversampled = resample(bin_0, replace=True, n_samples=target_count, random_state=42)
bin_2_oversampled = resample(bin_2, replace=True, n_samples=target_count, random_state=42)
balanced_data = pd.concat([bin_1, bin_0_oversampled, bin_2_oversampled])
print(balanced_data['pLC50_bin'].value_counts())

balanced_data.to_csv('balanced_data.csv')
print(balanced_data)