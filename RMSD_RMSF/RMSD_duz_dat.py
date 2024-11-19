import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np
import argparse
from itertools import cycle

def main():
    # Set up the logging module for error analysis and logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Plot RMSD data from multiple .dat files.')
    parser.add_argument('sim_time', type=float, help='Total simulation time in ns')
    parser.add_argument('files_and_labels', nargs='+', help='Pairs of .dat files and their labels')
    parser.add_argument('--output', type=str, default='rmsd_plot_all.png', help='Output filename for the plot')
    args = parser.parse_args()

    # Check that files and labels are provided in pairs
    if len(args.files_and_labels) % 2 != 0:
        logging.error("Files and labels must be provided in pairs.")
        sys.exit(1)

    sim_time = args.sim_time
    files_and_labels = args.files_and_labels
    output_filename = args.output

    # Parse files and labels
    file_labels = {}
    for i in range(0, len(files_and_labels), 2):
        dat_file = files_and_labels[i]
        label = files_and_labels[i + 1]
        file_labels[dat_file] = label

    # Log the files and labels
    logging.info(f"Files and labels: {file_labels}")

    # Generate a list of colors for plotting
    color_maps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors = []
    for cmap in color_maps:
        colors.extend(cmap(np.linspace(0, 1, 20)).tolist())
    color_cycle = cycle(colors)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    rmsd_values = []

    # Process each file
    for dat_file, label in file_labels.items():
        try:
            # Read the .dat file
            data = pd.read_csv(dat_file, sep='\s+')
            if data.empty:
                logging.warning(f"Data file {dat_file} is empty. Skipping.")
                continue

            # Check if 'Prot_Backbone' column exists
            if 'Prot_Backbone' not in data.columns:
                logging.error(f"Column 'Prot_CA' not found in {dat_file}. Skipping.")
                continue

            # Get RMSD values
            rmsd = data['Prot_CA']

            # Get time values
            if 'Time (ns)' in data.columns:
                time_ns = data['Time (ns)']
            else:
                # Generate time values based on sim_time
                time_ns = sim_time * data.index / len(data)

            mean_rmsd = rmsd.mean()
            rmsd_values.append({
                'mean_rmsd': mean_rmsd,
                'time_ns': time_ns,
                'rmsd': rmsd,
                'label': label,
            })
        except Exception as e:
            logging.error(f"Error reading or processing {dat_file}: {e}")
            continue

    if not rmsd_values:
        logging.error("No valid data to plot. Exiting.")
        sys.exit(1)

    # Sort datasets by mean RMSD (optional, can be removed if not needed)
    rmsd_values.sort(key=lambda x: x['mean_rmsd'])

    # Plot each dataset with the same line width
    line_width = 1  # Set desired line width here
    for val in rmsd_values:
        color = next(color_cycle)
        ax.plot(val['time_ns'], val['rmsd'], label=f"{val['label']} (Mean: {val['mean_rmsd']:.2f} Å)", color=color, linewidth=line_width)

    # Label the plot
    ax.set_xlabel('Time (ns)', fontsize=14)
    ax.set_ylabel('RMSD (Å)', fontsize=14)
    ax.set_title('RMSD Plot', fontsize=16)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Adjust layout and save the plot
    fig.tight_layout()
    fig.savefig(output_filename)
    plt.close(fig)

    logging.info(f"RMSD analysis and plotting completed successfully. Plot saved as '{output_filename}'.")

if __name__ == "__main__":
    main()

