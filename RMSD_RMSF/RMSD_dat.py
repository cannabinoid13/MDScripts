import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Configure logging for error analysis
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check and print command-line arguments
print(f"Command-line arguments: {sys.argv}")

if len(sys.argv) < 4 or (len(sys.argv) - 2) % 2 != 0:
    logging.error(f"Invalid number of arguments: {len(sys.argv)}. Usage: python <script.py> <sim_time> <file1.dat> <label1> <file2.dat> <label2> ...")
    sys.exit(1)

# Print the arguments
logging.info(f"Arguments: {sys.argv}")

# Get the simulation time
sim_time = float(sys.argv[1])

# Determine files and labels
file_labels = {}
try:
    for i in range(2, len(sys.argv), 2):
        dat_file = sys.argv[i]
        label = sys.argv[i + 1]
        file_labels[dat_file] = label
except IndexError as e:
    logging.error("IndexError: Make sure you provided all necessary files and labels.")
    sys.exit(1)

# Print files and labels
print(f"Files and labels: {file_labels}")

# Set seaborn style for scientific plots
sns.set(style="whitegrid", context="paper")

# Main colors for the plots
colors = sns.color_palette("tab10", n_colors=len(file_labels))

# Create the main figure with subplots for histograms
fig = plt.figure(figsize=(16, 12))  # Increase figure size

# Create gridspec for custom layout
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 2 rows, 1 column
ax = fig.add_subplot(gs[0])

# For statistical information
stat_table = []

# Prepare histogram axis
hist_ax = fig.add_subplot(gs[1])

# Process each file
for i, (dat_file, label) in enumerate(file_labels.items()):
    try:
        # Read the .dat file
        data = pd.read_csv(dat_file, sep='\s+')
        if 'Time' in data.columns:
            time_ns = data['Time']
        elif 'Time (ns)' in data.columns:
            time_ns = data['Time (ns)']
        else:
            time_ns = sim_time * (data.index + 1) / len(data)
            time_ns = time_ns.values  # Convert to NumPy array
        rmsd = data['Prot_CA']
    except Exception as e:
        logging.error(f"Error reading or processing {dat_file}: {e}")
        continue

    # Calculate rolling statistics for standard deviation band
    window_size = max(1, int(len(rmsd) * 0.10))  # 5% of data length
    rolling_mean = rmsd.rolling(window=window_size).mean()
    rolling_std = rmsd.rolling(window=window_size).std()

    # Calculate statistics
    mean_rmsd = np.mean(rmsd)
    std_rmsd = np.std(rmsd)
    # 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(rmsd) - 1
    sample_mean = mean_rmsd
    sample_standard_error = stats.sem(rmsd)
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

    # Append statistics to the table
    stat_table.append({
        'Label': label,
        'Mean RMSD (Å)': mean_rmsd,
        'Std RMSD (Å)': std_rmsd,
        '95% CI Lower (Å)': confidence_interval[0],
        '95% CI Upper (Å)': confidence_interval[1],
    })

    # Plot the RMSD over time
    ax.plot(time_ns, rmsd, label=f"{label}", color=colors[i], linewidth=1)

    # Plot the standard deviation band
    ax.fill_between(time_ns, rolling_mean - rolling_std, rolling_mean + rolling_std, color=colors[i], alpha=0.2)

    # Plot the mean RMSD as a horizontal dashed line
    ax.axhline(mean_rmsd, color=colors[i], linestyle='--', linewidth=2)

    # Annotate the mean value on the plot
    # Adjust position to avoid overlapping with data lines
    x_position = time_ns[-1] + (sim_time * 0.02)  # Slightly beyond the last time point
    # Ensure x_position does not exceed the plot limits
    if x_position > ax.get_xlim()[1]:
        x_position = ax.get_xlim()[1] * 0.99
    y_offset = std_rmsd * 0.1 * (-1)**i  # Alternate offsets for multiple datasets
    ax.text(x_position, mean_rmsd + y_offset, f"Mean: {mean_rmsd:.2f} Å", color=colors[i], fontsize=12,
            va='center', ha='left', backgroundcolor='white',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Plot KDE on the histogram axis
    sns.kdeplot(rmsd, color=colors[i], label=label, ax=hist_ax, linewidth=2)
    hist_ax.axvline(mean_rmsd, color=colors[i], linestyle='--', linewidth=1)

# Set labels and title for main plot
ax.set_xlabel('Time (ns)', fontsize=16)
ax.set_ylabel('RMSD (Å)', fontsize=16)
ax.set_title('RMSD over Time', fontsize=18)

# Create custom legend handles
dataset_handles = [Line2D([0], [0], color=colors[i], lw=1, label=file_labels[dat_file]) for i, dat_file in enumerate(file_labels.keys())]
mean_line_handles = [Line2D([0], [0], color=colors[i], linestyle='--', lw=2, label=f"{file_labels[dat_file]} Mean") for i, dat_file in enumerate(file_labels.keys())]
std_band_handle = Patch(facecolor='grey', alpha=0.2, label='Rolling Std Dev Band')

# Combine handles
handles = dataset_handles + mean_line_handles + [std_band_handle]

# Add legend to the plot
ax.legend(handles=handles, fontsize=12, loc='lower right')

ax.tick_params(axis='both', which='major', labelsize=14)

# Configure histogram axis
hist_ax.set_xlabel('RMSD (Å)', fontsize=14)
hist_ax.set_ylabel('Density', fontsize=14)
hist_ax.set_title('RMSD Distribution', fontsize=16)
hist_ax.legend(fontsize=12)
hist_ax.tick_params(axis='both', which='major', labelsize=12)

# Create a table for statistics
stat_df = pd.DataFrame(stat_table)
print("\nStatistical Summary:")
print(stat_df.to_string(index=False))

# Adjust layout
plt.tight_layout()

# Display the plot on screen
plt.show()

logging.info("RMSD analysis and plotting completed successfully.")

