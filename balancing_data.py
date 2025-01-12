import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample

data = pd.read_csv('important_features_data.csv', index_col=0)
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



'''data = pd.read_csv('important_features_data.csv', index_col=0)
# balance the data
# Define bins (you can adjust bin edges based on your data)
bins = np.linspace(data['pLC50'].min(), data['pLC50'].max(), num=20)  # 20 bins
data['pLC50_bin'] = pd.cut(data['pLC50'], bins=bins, labels=False)
print(data['pLC50_bin'].value_counts())
print(bins)
plt.hist(data['pLC50_bin'], bins=50, edgecolor='k')
plt.xlabel('pLC50 Bins')
plt.ylabel('Frequency')
plt.title('Distribution of pLC50 Values')
plt.show()

# drop na and outlier, which is bin 6 with only 1 occurence
data = data.dropna()
data = data[data['pLC50_bin'] != 18]
data = data[data['pLC50_bin'] != 5]

# check how many records we have per bin and then oversample minority bins
bin_counts = data['pLC50_bin'].value_counts()
print(bin_counts)
# balance the two majority classes and the three minority classes
bin_0_max = data[data['pLC50_bin'] == 2]
bin_1_max = data[data['pLC50_bin'] == 3]
bin_2_min = data[data['pLC50_bin'] == 1]
bin_3_min = data[data['pLC50_bin'] == 4]
bin_4_min = data[data['pLC50_bin'] == 0]

target_count = bin_0_max.shape[0]
target_count_max = bin_0_max.shape[0]
target_count_min = bin_2_min.shape[0]
#target_count = (bin_0.shape[0] + bin_1.shape[0] + bin_2.shape[0])/3
#target_count = round(target_count)
print(target_count_max)
print(target_count_min)
bin_1_max_oversampled = resample(bin_1_max, replace=True, n_samples=target_count, random_state=42)
bin_2_min_oversampled = resample(bin_2_min, replace=True, n_samples=target_count, random_state=42)
bin_3_min_oversampled = resample(bin_3_min, replace=True, n_samples=target_count, random_state=42)
bin_4_min_oversampled = resample(bin_4_min, replace=True, n_samples=target_count, random_state=42)
balanced_data = pd.concat([bin_0_max, bin_1_max_oversampled, bin_2_min_oversampled, bin_3_min_oversampled, bin_4_min_oversampled])
print(balanced_data['pLC50_bin'].value_counts())

balanced_data.to_csv('balanced_data_second_version.csv')'''