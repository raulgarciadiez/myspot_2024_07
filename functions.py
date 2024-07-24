import re

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