import os
import pandas as pd
import numpy as np

# Function to add noise to a dataset
def add_noise(data, noise_level=0.05):
    noisy_data = data.copy()
    for col in data.select_dtypes(include=[np.number]).columns:  # Only apply to numeric columns
        noise = np.random.normal(0, noise_level * data[col].std(), size=data[col].shape)
        noisy_data[col] += noise
    return noisy_data

# Create output folder if it does not exist
output_folder = 'art_test'
os.makedirs(output_folder, exist_ok=True)

# List all CSV files in the current directory
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

# Process each CSV file
for file in csv_files:
    # Read the original data
    original_data = pd.read_csv(file)

    # Create three artificial datasets with varying noise levels
    for i, noise_level in enumerate([0.05, 0.1, 0.2], start=1):
        artificial_data = add_noise(original_data, noise_level=noise_level)
        new_file_name = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_artificial_{i}.csv")

        # Save the noisy dataset
        artificial_data.to_csv(new_file_name, index=False)
        print(f"Generated file: {new_file_name}")

print("Creation Done!")
