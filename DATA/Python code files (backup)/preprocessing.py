import pandas as pd

# Load the gathered data files and create preprocessed files

file_path = 'test.csv'
data = pd.read_csv(file_path, header=None, skiprows=1)  # Load all rows as raw data

data = data.iloc[1:].reset_index(drop=True)  # removing not needed row

# Set the first row as the new header (make sure it's the right row)
new_header = data.iloc[0]  # Use the first row as the header
data = data[1:].reset_index(drop=True)  # Drop the first row now that it's used as headers
data.columns = new_header  # Assign the first row as the new column headers

# Strip any extra whitespace from column names
data.columns = data.columns.str.strip()

# Handle any NaN or missing values in the column names by replacing them with empty strings or a placeholder
data.columns = data.columns.fillna('')

first_column = data.iloc[:, 0]  # First column by index (index 0)
# Suffixes I need
valid_suffixes = ['Name', ':Head', ':LUArm', ':HeadTop', ':RHand']

# Keep only the columns that end with the valid suffix
data = data.loc[:, data.columns.str.endswith(tuple(valid_suffixes))]
data = pd.concat([first_column, data], axis=1)
# Check the number of columns after filtering
print(f"Number of columns after filtering: {data.shape[1]}")

# Save the preprocessed data
data.to_csv('processed_file.csv', index=False)
print("Filtered Data:\n", data.head())
