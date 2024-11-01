import numpy as np

# Load the .npz file with allow_pickle=True
data = np.load("HDFS.npz", allow_pickle=True)

# List all arrays in the file
print("Arrays in HDFS.npz:", data.files)

# Display the first 5 rows of each array in full detail
for array_name in data.files:
    array_data = data[array_name]
    print(f"\nArray '{array_name}':")
    print("  Shape:", array_data.shape)
    print("  Data type:", array_data.dtype)
    print("  First 5 rows:")

    # Display the first 5 rows without truncation
    for i in range(5):
        print(f"    Row {i + 1}: {array_data[i]}")
