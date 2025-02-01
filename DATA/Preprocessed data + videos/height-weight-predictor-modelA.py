# Author: Vladyslav Protsko k12147981
# Practical Work in AI WS24
# This program will work provided there is a .pkl file in the same directory as this python file
# and that the full Optitrack body suit was used to acquire the data
# as a result each preprocessed file will get the labels for height and weight in the name. ex.: light_floor.csv

import pandas as pd
import numpy as np
import os
import joblib

directory = './Predict'  # current working directory
directory_to_write = './Predict/Preprocessed'
os.makedirs(directory_to_write, exist_ok=True)

# Define valid suffixes (same as for the train/test data preprocessing)
valid_suffixes = ['Name',
                  ':Head', ':LShoulder', ':LUArm', ':LFArm', ':LHand',
                  ':HeadTop', ':RShoulder', ':RUArm', ':RFArm', ':RHand',
                  ':BackTop', ':LElbowOut', ':LWristOut', ':WaistLFront', ':WaistLBack',
                  ':RElbowOut', ':RWristOut', ':WaistRFront', ':WaistRBack',
                  ':LThigh', ':LShin', ':LFoot', ':LTHI', ':LKNE', ':LANK',
                  ':RThigh', ':RShin', ':RFoot', ':RTHI', ':RKNE', ':RANK'
                  ]     # it is important to preprocess the raw data the same way I have done it for
                        # test and train data, because that is the structure the model knows how to solve

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
    processed_file_path = os.path.join(directory_to_write, f"{file_name}_processed{file_extension}")

    # Save the processed data to a new file with _processed ending
    data.to_csv(processed_file_path, index=False)
    print(f"Processed file saved as: {processed_file_path}")


# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Process only .csv files
        file_path = os.path.join(directory, filename)
        process_csv(file_path)


# Now the code starts for loading the model and making predictions
# Load the trained model
model_path = 'height-weight-classification-modelA.pkl'
if not os.path.exists(model_path):
    print(f"Model file '{model_path}' not found! Make sure the .pkl file is in the current directory.")
    exit()

multi_output_model = joblib.load(model_path)
print("Model loaded successfully.")

# Get the label categories from the training process
weight_map = ['heavy', 'light', 'medium']  # Replace with the actual categories from training
height_map = ['floor', 'not_floor']       # Replace with the actual categories from training

# Directory containing preprocessed files
preprocessed_dir = './Predict/Preprocessed'
if not os.path.exists(preprocessed_dir):
    print(f"Preprocessed directory '{preprocessed_dir}' not found! Make sure the preprocessing step is completed.")
    exit()


# Function to calculate the angle between three points
def calculate_angle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    # Calculate vectors from points 2 -> 1 and 2 -> 3 (Knee -> Hip, Knee -> Ankle)
    v1 = np.array([x1 - x2, y1 - y2, z1 - z2])  # Vector from knee to hip
    v2 = np.array([x3 - x2, y3 - y2, z3 - z2])  # Vector from knee to ankle

    # Compute the cosine of the angle between the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure the cosine value is in the valid range for arccos
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    return np.degrees(angle)  # Convert from radians to degrees


def calculate_acceleration(positions, time_intervals):
    """
    Calculate acceleration based on positions and time intervals.

    Parameters:
        positions (np.ndarray): Array of position values (e.g., x, y, z).
        time_intervals (np.ndarray): Array of time intervals (time differences between frames).

    Returns:
        np.ndarray: Array of acceleration values.
    """
    # Calculate velocity as the difference in position over time
    velocities = np.diff(positions) / np.diff(time_intervals)

    # Calculate acceleration as the difference in velocity over time
    accelerations = np.diff(velocities) / np.diff(time_intervals[:-1])

    return accelerations


def estimate_force(accelerations, mass):
    """
    Estimate force using Newton's Second Law.

    Parameters:
        accelerations (np.ndarray): Array of acceleration values.
        mass (float): Mass of person + mass of object or just mass of person (in kg).

    Returns:
        np.ndarray: Array of force values.
    """
    forces = mass * accelerations  # F = m * a
    return forces


# Function to compute summary features (mean, std, min, max)
def compute_features(data, data_for_angle, data_for_force):
    features = {}

    # List of body parts to extract (I only use one side assuming the other is almost the same)
    body_parts = [
        'Skeleton_25_marker:WaistLBack',  # Hip
        'SkeletonConLow:LKNE',  # Knee
        'SkeletonConLow:LANK'  # Ankle
    ]
    # Extract the X, Y, Z coordinates for hip, knee, and ankle from the data
    # Assuming 'data' contains these columns and these names are consistent.
    # Filter columns based on body parts and 'Position'
    selected_columns = data_for_angle.loc[:,
                       (data_for_angle.columns.get_level_values(0).isin(body_parts)) &
                       (data_for_angle.columns.get_level_values(1) == 'Position')
                       ]

    # Extract the time column
    time = data_for_angle.iloc[3:, 1].reset_index(drop=True).astype(float)
    # Select columns containing the relevant strings for hip, knee, and ankle positions
    x1 = selected_columns.loc[:, ('Skeleton_25_marker:WaistLBack', 'Position', 'X')].iloc[3:].reset_index(drop=True).astype(float)
    y1 = selected_columns.loc[:, ('Skeleton_25_marker:WaistLBack', 'Position', 'Y')].iloc[3:].reset_index(drop=True).astype(float)
    z1 = selected_columns.loc[:, ('Skeleton_25_marker:WaistLBack', 'Position', 'Z')].iloc[3:].reset_index(drop=True).astype(float)

    x2 = selected_columns.loc[:, ('SkeletonConLow:LKNE', 'Position', 'X')].iloc[3:].reset_index(drop=True).astype(float)
    y2 = selected_columns.loc[:, ('SkeletonConLow:LKNE', 'Position', 'Y')].iloc[3:].reset_index(drop=True).astype(float)
    z2 = selected_columns.loc[:, ('SkeletonConLow:LKNE', 'Position', 'Z')].iloc[3:].reset_index(drop=True).astype(float)

    x3 = selected_columns.loc[:, ('SkeletonConLow:LANK', 'Position', 'X')].iloc[3:].reset_index(drop=True).astype(float)
    y3 = selected_columns.loc[:, ('SkeletonConLow:LANK', 'Position', 'Y')].iloc[3:].reset_index(drop=True).astype(float)
    z3 = selected_columns.loc[:, ('SkeletonConLow:LANK', 'Position', 'Z')].iloc[3:].reset_index(drop=True).astype(float)




    # Calculate the angle for each frame (time step)
    angles = [calculate_angle(x1[i], y1[i], z1[i], x2[i], y2[i], z2[i], x3[i], y3[i], z3[i]) for i in range(len(time))]

    # Compute summary statistics for angles
    features['angle_mean'] = np.mean(angles)
    features['angle_std'] = np.std(angles)
    features['angle_min'] = np.min(angles)
    features['angle_max'] = np.max(angles)

    ########
    # Code for force calculation
    # Filter columns based on body parts and 'Position'
    selected_columns_force = data_for_force.loc[:,
                       (data_for_force.columns.get_level_values(0).isin(body_parts)) &
                       (data_for_force.columns.get_level_values(1) == 'Position')
                       ]

    # Extract the time column
    time_force = data_for_force.iloc[3:, 1].reset_index(drop=True).astype(float)
    # Select columns containing the relevant strings for hip, knee, and ankle positions
    # Extract positions for a body part (e.g., knee)
    x_positions = selected_columns_force.loc[:, ('SkeletonConLow:LKNE', 'Position', 'X')].iloc[3:].reset_index(drop=True).astype(float)
    y_positions = selected_columns_force.loc[:, ('SkeletonConLow:LKNE', 'Position', 'Y')].iloc[3:].reset_index(drop=True).astype(float)
    z_positions = selected_columns_force.loc[:, ('SkeletonConLow:LKNE', 'Position', 'Z')].iloc[3:].reset_index(drop=True).astype(float)

    # Calculate accelerations for each axis
    x_acceleration = calculate_acceleration(x_positions, time_force)
    y_acceleration = calculate_acceleration(y_positions, time_force)
    z_acceleration = calculate_acceleration(z_positions, time_force)

    # Combine into total acceleration (magnitude)
    total_acceleration = np.sqrt(x_acceleration**2 + y_acceleration**2 + z_acceleration**2)

    # Estimate forces (assuming body_mass is the effective segment mass for now)
    body_mass = 75 # lets say the average weight is 75kg
    forces = estimate_force(total_acceleration, body_mass)

    # Compute summary statistics for forces
    features['force_mean'] = np.mean(forces)
    features['force_std'] = np.std(forces)
    features['force_min'] = np.min(forces)
    features['force_max'] = np.max(forces)

    for col in data.columns[2:]:
        if data[col].dtype in ['float64', 'int64']:  # Only numerical columns
            features[f'{col}_mean'] = data[col].mean()
            features[f'{col}_std'] = data[col].std()
            features[f'{col}_min'] = data[col].min()
            features[f'{col}_max'] = data[col].max()
    return features

# Iterate over processed files and make predictions
print("\nPredictions for preprocessed files:")
for filename in os.listdir(preprocessed_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(preprocessed_dir, filename)

        # Load the processed file
        data = pd.read_csv(file_path, skiprows=2)
        data_for_angle = pd.read_csv(file_path, header=[0, 1, 2])
        data_for_force = pd.read_csv(file_path, header=[0, 1, 2])
        # Compute features
        features = compute_features(data, data_for_angle, data_for_force)

        # Convert features to a DataFrame (single row)
        features_df = pd.DataFrame([features])
        features_df = features_df.fillna(0)  # Fill missing values with 0

        # Predict weight and height
        prediction = multi_output_model.predict(features_df)
        predicted_weight = weight_map[prediction[0][0]]
        predicted_height = height_map[prediction[0][1]]

        # Output the predictions
        print(f"File: {filename}")
        print(f"  Predicted Weight: {predicted_weight}")
        print(f"  Predicted Height: {predicted_height}\n")

         # Rename the file with predicted weight and height
        new_filename = f"{predicted_weight}_{predicted_height}_{filename}"
        new_file_path = os.path.join(preprocessed_dir, new_filename)
        os.rename(file_path, new_file_path)
        print(f"  Renamed file to: {new_filename}\n")

