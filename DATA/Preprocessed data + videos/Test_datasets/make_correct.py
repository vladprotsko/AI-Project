import os
import pandas as pd

# Get all files in the current directory
files = os.listdir('./prepare')

# Loop through each file
for file in files:
    if file.endswith('.csv'):
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)

        # Update the column names by removing the trailing number
        new_columns = []
        for col in df.columns:
            # Remove the trailing numbers (e.g., ".1", ".2", etc.)
            new_col = col.rsplit('.', 1)[0]  # Split from the right and keep everything before the last dot
            new_columns.append(new_col)

        # Assign the updated column names to the DataFrame
        df.columns = new_columns

        # Save the updated DataFrame back to a CSV file
        df.to_csv(file, index=False)

        print(f"Updated headers in {file}")
