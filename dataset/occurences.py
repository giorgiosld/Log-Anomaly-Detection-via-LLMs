import numpy as np

# Load the .npz file with allow_pickle=True
data = np.load("HDFSv1/preprocessed/HDFS.npz", allow_pickle=True)

# Get the y_data array
y_data = data['y_data']

# Count the frequency of each value (0 and 1) using np.unique
unique, counts = np.unique(y_data, return_counts=True)
distribution = dict(zip(unique, counts))

# Print the distribution in a readable format
for value, count in distribution.items():
    print(f"Value {value}: {count} occurrences")
