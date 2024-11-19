import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns

def plot_rmsf(file_label_pairs, residue_indices=None):
    """
    Plots RMSF data from multiple files, aligning the first elements and truncating datasets
    to the length of the shortest dataset.

    Parameters:
    - file_label_pairs: List containing file paths and their corresponding labels.
    - residue_indices: List of residue indices to highlight on the plot.

    Returns:
    - None
    """
    # Set up the figure with appropriate size and resolution
    plt.figure(figsize=(12, 8), dpi=150)

    # Update the font settings for readability and compatibility
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'Arial'
    })

    # Create a color palette
    colors = sns.color_palette("tab10", n_colors=len(file_label_pairs) // 2)

    datasets = []
    min_length = None

    # Read all datasets and find the minimum length
    for i in range(0, len(file_label_pairs), 2):
        file_path = file_label_pairs[i]
        label = file_label_pairs[i+1]

        try:
            # Read the file
            data = pd.read_csv(file_path, sep='\s+')
        except Exception as e:
            print(f"Error reading '{file_path}': {e}")
            continue

        # Check if 'CA' column exists
        if 'CA' not in data.columns:
            print(f"'CA' column not found in '{file_path}'.")
            continue

        # Store the dataset along with its label
        datasets.append({'data': data, 'label': label})

        # Update the minimum length
        if min_length is None or len(data) < min_length:
            min_length = len(data)

    if len(datasets) == 0:
        print("No valid datasets to plot.")
        return

    # Truncate datasets to the minimum length and align first elements
    for idx, dataset in enumerate(datasets):
        data = dataset['data']

        # Truncate data to min_length
        truncated_data = data.iloc[:min_length].reset_index(drop=True)

        # Adjust x-values to align first elements
        # Subtract the first residue number to align datasets
        x_values = truncated_data[truncated_data.columns[0]] - truncated_data[truncated_data.columns[0]].iloc[0]

        # Update the dataset
        datasets[idx]['truncated_data'] = truncated_data
        datasets[idx]['x_values'] = x_values

    # Plot the datasets
    for idx, dataset in enumerate(datasets):
        truncated_data = dataset['truncated_data']
        x_values = dataset['x_values']
        label = dataset['label']

        plt.plot(x_values, truncated_data['CA'], label=label, color=colors[idx % len(colors)])

    # Handle residue_indices if provided
    if residue_indices:
        # Adjust residue_indices to align with the adjusted x_values
        adjusted_indices = [idx - datasets[0]['truncated_data'][datasets[0]['truncated_data'].columns[0]].iloc[0] for idx in residue_indices]
        for idx in adjusted_indices:
            plt.axvline(x=idx, color='r', linestyle='--', linewidth=1)

    plt.xlabel('Aligned Residue Index', fontsize=14)
    plt.ylabel('RMSF (Å)', fontsize=14)
    plt.title('Alpha Carbon (CA) RMSF Analysis', fontsize=16)

    # Place the legend inside the plot area in two columns
    plt.legend(loc='upper right', ncol=2, fontsize=12)

    plt.grid(True)

    # Adjust layout to ensure all elements fit within the figure area
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Command-line arguments: file1 label1 file2 label2 ... residue_indices
    # residue_indices are provided as a comma-separated list in the last argument
    if len(sys.argv) < 3:
        print("Usage: python script.py file1 label1 [file2 label2 ...] residue_indices")
        sys.exit(1)

    # The last argument may be residue_indices or another file-label pair
    residue_indices_input = sys.argv[-1]
    try:
        # Try to parse residue_indices
        residue_indices = list(map(int, residue_indices_input.split(',')))
        file_label_pairs = sys.argv[1:-1]
    except ValueError:
        # If parsing fails, treat it as a label and include it in file_label_pairs
        residue_indices = None
        file_label_pairs = sys.argv[1:]

    if len(file_label_pairs) % 2 != 0:
        print("Please provide a label for each file.")
        sys.exit(1)

    plot_rmsf(file_label_pairs, residue_indices)

