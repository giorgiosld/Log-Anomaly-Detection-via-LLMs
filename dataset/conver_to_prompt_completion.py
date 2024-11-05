import numpy as np
import json

# Load the data from the .npz file
data = np.load("HDFSv1/preprocessed/HDFS.npz", allow_pickle=True)

# Extract sequences and labels
sequences = data['x_data']
labels = data['y_data']

# Prepare data in prompt-completion format
data_list = []
for i in range(len(labels)):
    sequence = " ".join(sequences[i])
    label = "Normal" if labels[i] == 0 else "Anomaly"
    prompt = f"Log trace: {sequence}"
    completion = f"Label: {label}"
    
    # Create a dictionary for each example
    data_entry = {"prompt": prompt, "completion": completion}
    data_list.append(data_entry)

# Save to JSONL format
output_path = "HDFSv1/preprocessed/prompt_completion_data.jsonl"
with open(output_path, "w") as f:
    for entry in data_list:
        json.dump(entry, f)
        f.write("\n")

print(f"Data successfully converted to prompt-completion format and saved to {output_path}.")

