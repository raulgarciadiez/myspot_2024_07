import re
import os
from collections import defaultdict
import pandas as pd
import numpy as np

def process_files(file_dict, var1, var2):
    combined_df = pd.DataFrame()

    for key, files in file_dict.items():
        for file in files:
            # Read each file into a DataFrame
            df = pd.read_csv(file, comment='#', delimiter='\t')
            
            # Check if var1 and var2 exist in the DataFrame
            if var1 not in df.columns or var2 not in df.columns:
                raise ValueError(f"Columns '{var1}' or '{var2}' are not in the DataFrame")

            # Create new column 'var1/var2'
            df[f'{var1}/{var2}'] = df[var1] / df[var2]
            
            # Combine DataFrames
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Group by 'monoE' and calculate mean and std
    grouped = combined_df.groupby('monoE').agg({
        var1: ['mean', 'std'],
        var2: ['mean', 'std'],
        f'{var1}/{var2}': ['mean', 'std']
    }).reset_index()
    
    # Flatten the column hierarchy
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped.rename(columns={'monoE_': 'monoE'}, inplace=True)
    
    return grouped

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

def load_monoE(file_path):
    """Load the monoE column from a file and return the starting value."""
    try:
        df = pd.read_csv(file_path, comment='#', delimiter='\t')
        if 'monoE' in df.columns and not df['monoE'].empty:
            return df['monoE'].iloc[0]  # Return the starting value
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
        start_value = load_monoE(file)
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

