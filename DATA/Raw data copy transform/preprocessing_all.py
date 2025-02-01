import pandas as pd
import os

directory = '.'  # current working directory

# Define valid suffixes
valid_suffixes = ['Name',
                  ':Head', ':LShoulder', ':LUArm', ':LFArm', ':LHand',
                  ':HeadTop', ':RShoulder', ':RUArm', ':RFArm', ':RHand',
                  ':BackTop', ':LElbowOut', ':LWristOut', ':WaistLFront', ':WaistLBack',
                  ':RElbowOut', ':RWristOut', ':WaistRFront', ':WaistRBack',
                  ':LThigh', ':LShin', ':LFoot', ':LTHI', ':LKNE', ':LANK',
                  ':RThigh', ':RShin', ':RFoot', ':RTHI', ':RKNE', ':RANK'
                  ]


# Function to process a single CSV file
def process_csv(file_path):
    data = pd.read_csv(file_path, header=None, skiprows=1)
    data = data.iloc[1:].reset_index(drop=True)

    # Set the first row as the new header (make sure it's the right row)
    new_header = data.iloc[0]  # Use the first row as the header
    data = data[1:].reset_index(drop=True)  # Drop the first row now that it's used as headers
    data.columns = new_header  # Assign the first row as the new column headers

    # Strip any extra whitespace from column names
    data.columns = data.columns.str.strip()
    data.columns = data.columns.fillna('')  # Handle any NaN or missing values
    first_column = data.iloc[:, 0]  # save first column in variable
    # Keep only the columns that end with one of the valid suffixes
    data = data.loc[:, data.columns.str.endswith(tuple(valid_suffixes))]
    data = pd.concat([first_column, data], axis=1)
    data = data.iloc[1:].reset_index(drop=True)
    
    base_name = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(base_name)
    processed_file_path = os.path.join(directory, f"{file_name}_processed{file_extension}")

    # Save the processed data to a new file with _processed ending
    data.to_csv(processed_file_path, index=False)
    print(f"Processed file saved as: {processed_file_path}")


# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Process only .csv files
        file_path = os.path.join(directory, filename)
        process_csv(file_path)
