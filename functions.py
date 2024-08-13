import re
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def process_files(file_dict, var1, var2):
    combined_df = pd.DataFrame()

    for key, files in file_dict.items():
        for file in files:
            # Read each file into a DataFrame
            df = pd.read_csv(file, comment='#', delimiter='\t')
            if 'monoE' not in df.columns:
                # Check if the DataFrame has a column called 'Energy'
                if 'Energy' in df.columns:
                    # Rename the column 'Energy' to 'monoE'
                    df.rename(columns={'Energy': 'monoE'}, inplace=True)
            # Check if var1 and var2 exist in the DataFrame
            if var1 not in df.columns or var2 not in df.columns:
                raise ValueError(f"Columns '{var1}' or '{var2}' are not in the DataFrame")

            # Create new column 'var1/var2'
            df[f'{var1}/{var2}'] = df[var1] / df[var2]
            # Combine DataFrames
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Apply Z-score based outlier removal
    #combined_df = remove_outliers(combined_df, [var1, var2, f'{var1}/{var2}'], z_threshold)
    combined_df = remove_outliers_iqr(combined_df, [var1, var2, f'{var1}/{var2}'])

    # Group by 'monoE' and calculate mean and std
    grouped = combined_df.groupby('monoE').agg({
        var1: ['mean', 'std'],
        var2: ['mean', 'std'],
        f'{var1}/{var2}': ['mean', 'std']
    }).reset_index()
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={'monoE_': 'monoE'}, inplace=True)
    
    return grouped
    
    #grouped = combined_df.groupby('monoE').agg({
    #    var1: ['mean', 'std'],
    #    var2: ['mean', 'std'],
    #    f'{var1}/{var2}': ['mean', 'std']
    #}).reset_index()
    
    # Flatten the column hierarchy
    #grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    #grouped.rename(columns={'monoE_': 'monoE'}, inplace=True)
    
    #return grouped

def remove_outliers(df, columns, z_threshold=3):
    """Remove outliers from a DataFrame based on Z-score."""
    for column in columns:
        df['z_score'] = np.abs((df[column] - df[column].mean()) / df[column].std(ddof=0))
        df = df[df['z_score'] < z_threshold]
    df = df.drop(columns=['z_score'])  # Drop the z_score column after filtering
    return df

def remove_outliers_iqr(df, columns):
    """Remove outliers from a DataFrame using the IQR method."""
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def process_files_OLD(file_dict):
    combined_df = pd.DataFrame()

    for key, files in file_dict.items():
        for file in files:
            # Read each file into a DataFrame
            df = pd.read_csv(file, comment='#', delimiter='\t')
            
            # Create new column 'roi1/mcaLt'
            df['roi1/mcaLt'] = df['roi1'] / df['mcaLt']
            
            # Combine DataFrames based on 'monoE'
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Group by 'monoE' and calculate mean and std
    grouped = combined_df.groupby('monoE').agg({
        'roi1': ['mean', 'std'],
        'mcaLt': ['mean', 'std'],
        'roi1/mcaLt': ['mean', 'std']
    }).reset_index()
    
    # Flatten the column hierarchy
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={'monoE_': 'monoE'}, inplace=True)
    
    return grouped

def load_energy(file_path):
    """Load the monoE column from a file and return the starting value."""
    try:
        df = pd.read_csv(file_path, comment='#', delimiter='\t')
        if 'monoE' in df.columns and not df['monoE'].empty:
            return df['monoE'].iloc[0]  # Return the starting value
        elif 'Energy' in df.columns and not df['Energy'].empty:
            print ("EXAFS file - ", file_path)
            return df['Energy'].iloc[0]  # Return the starting value
        else:
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def group_files_by_start(files):
    """Group files by their starting monoE values."""
    starting_values = defaultdict(list)
    
    # Extract starting values for each file
    for file in files:
        start_value = load_energy(file)
        if start_value is not None:
            starting_values[start_value].append(file)
    
    return dict(starting_values)

def read_esrf_spec_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    spec_data = {}
    current_scan = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith('#S'):
            # New scan section
            parts = line.split()
            scan_num = int(parts[1])
            command = ' '.join(parts[2:])
            current_scan = {
                'number': scan_num,
                'command': command,
                'metadata': {},
                'columns': [],
                'data': []
            }
            spec_data[scan_num] = current_scan
        elif line.startswith('#D') and current_scan is not None:
            # Date line
            date_str = line[2:].strip()
            current_scan['metadata']['date'] = date_str
        elif line.startswith('#L') and current_scan is not None:
            # Column labels
            column_labels = line[2:].strip().split('  ')  # Assume double space separation
            column_labels = [label for label in column_labels if label.strip()]
            current_scan['columns'] = column_labels
        elif line.startswith('#') and current_scan is not None:
            # Other metadata
            parts = line[1:].split(':', 1)
            if len(parts) == 2:
                key, value = parts
                current_scan['metadata'][key.strip()] = value.strip()
        elif re.match(r'^[0-9]', line) and current_scan is not None:
            # Data line
            data_values = line.split()
            current_scan['data'].append(data_values)

    return spec_data

def save_scan_to_ascii(scan_data, file_path):
    with open(file_path, 'w') as file:
        # Write metadata as comments
        file.write(f"# Scan number: {scan_data['number']}\n")
        file.write(f"# Command: {scan_data['command']}\n")
        file.write(f"# Date: {scan_data['metadata'].get('date')}\n")
        for key, value in scan_data['metadata'].items():
            file.write(f"# {key}: {value}\n")
        
        # Write column labels with quotes for those containing spaces
        quoted_columns = [f'"{col}"' if ' ' in col else col for col in scan_data['columns']]
        file.write('\t'.join(quoted_columns) + '\n')
        
        # Write data rows
        for row in scan_data['data']:
            file.write('\t'.join(row) + '\n')


def find_files_with_consecutive_numbers(folder_path, pattern):
    # Define the regex pattern to match the file names and extract numbers
    pattern_regex = re.compile(pattern)
    
    # List all files in the directory
    all_files = os.listdir(folder_path)
    # Filter files matching the pattern
    matching_files = [f for f in all_files if pattern_regex.match(f)]
    
    # Extract numbers from the filenames
    numbers = []
    file_dict = {}
    
    for file in matching_files:
        match = re.search(r'(\d+)', file)
        if match:
            num = int(match.group(1))
            numbers.append(num)
            file_dict[num] = file
    
    # Sort numbers to find consecutive sequences
    numbers.sort()
    
    # Find and filter out only strictly consecutive sequences
    consecutive_files = []
    i = 0
    while i < len(numbers):
        consecutive = [numbers[i]]
        while i < len(numbers) - 1 and numbers[i + 1] == numbers[i] + 1:
            consecutive.append(numbers[i + 1])
            i += 1
        # Add files corresponding to the consecutive numbers
        if len(consecutive) > 1:  # Ensure at least a pair of consecutive numbers
            for num in consecutive:
                consecutive_files.append(file_dict[num])
        i += 1
    
    return sorted(consecutive_files, key=lambda x: int(re.search(r'\d+', x).group()))


def find_edge_position(energy, intensity, smoothing_window=11, polyorder=3):
    """
    Finds the edge position in the given energy vs. intensity data.

    Parameters:
    - energy: numpy array of energy values
    - intensity: numpy array of intensity values
    - smoothing_window: window size for Savitzky-Golay filter (default is 11)
    - polyorder: polynomial order for Savitzky-Golay filter (default is 3)

    Returns:
    - edge_position_idx: index of the detected edge position
    """
    # Step 1: Normalize the intensity data
    normalized_data = intensity / np.max(intensity)
    
    # Optional: Smooth the data to reduce noise
    #smoothed_data = savgol_filter(normalized_data, smoothing_window, polyorder)
    
    # Step 2: Calculate the first derivative
    first_derivative = np.gradient(normalized_data)
    
    # Step 3: Find the index of the maximum in the first derivative
    max_index = np.argmax(first_derivative)
    
    # Step 4: Calculate the second derivative
    second_derivative = np.gradient(first_derivative)
    
    # Step 5: Find the zero-crossing in the second derivative near the maximum
    zero_crossing_indices = np.where(np.diff(np.sign(second_derivative)))[0]
    
    # Find the zero-crossing closest to the maximum in the first derivative
    edge_position_idx = zero_crossing_indices[np.abs(zero_crossing_indices - max_index).argmin()]
    
    return edge_position_idx

