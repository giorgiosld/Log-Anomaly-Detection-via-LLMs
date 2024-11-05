import numpy as np

# Load the data from the .npz file
data = np.load('HDFSv1/preprocessed/HDFS.npz', allow_pickle=True)

# Extract sequences and labels
sequences = data['x_data']
labels = data['y_data']

# Verify the shape of the sequences and labels
print(f"Total number of sequences: {len(sequences)}")
print(f"Total number of labels: {len(labels)}")

# Iterate through labels to find the first anomalous sequence (label == 1)
anomalous_sequence_str = None
for i in range(len(labels)):
    if labels[i] == 1:
        anomalous_sequence = sequences[i]
        anomalous_sequence_str = " ".join(anomalous_sequence)
        print(f"Anomalous sequence found at index {i}: {anomalous_sequence}")
        print(f"Formatted anomalous sequence for testing: {anomalous_sequence_str}")
        break

