import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
import warnings

def plot_mmgbsa(file_path, total_sim_time, total_frames, frame_skip=None, save_path=None):
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Set Seaborn style and palette
    sns.set(style="whitegrid")
    sns.set_palette("muted")

    # Load the data
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Check if the required column exists
    target_column = 'r_psp_MMGBSA_dG_Bind'
    if target_column not in data.columns:
        print(f"Error: '{target_column}' column not found in {file_path}")
        return

    # Extract the binding energy data and drop NaN values
    y_values = data[target_column].dropna().values

    # Compute statistical properties
    mean_energy = np.mean(y_values)
    std_energy = np.std(y_values, ddof=1)  # Sample standard deviation
    median_energy = np.median(y_values)
    min_energy = np.min(y_values)
    max_energy = np.max(y_values)
    energy_range = max_energy - min_energy
    sem_energy = std_energy / np.sqrt(len(y_values))  # Standard error of the mean

    # Generate time values for the x-axis
    if frame_skip is None or frame_skip == 0:
        time_step = total_sim_time / total_frames
    else:
        time_step = (total_sim_time / total_frames) * frame_skip

    x_values = np.arange(len(y_values)) * time_step

    # Create the plot
    plt.figure(figsize=(16, 10))
    plt.plot(x_values, y_values, marker='o', linestyle='-', linewidth=2, markersize=6, label='Binding Energy')

    # Plot the mean line
    plt.axhline(mean_energy, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_energy:.2f} kcal/mol')

    # Fill between (mean ± std)
    plt.fill_between(x_values, mean_energy - std_energy, mean_energy + std_energy, color='red', alpha=0.1,
                     label=f'Std Dev = {std_energy:.2f} kcal/mol')

    # Labels and title
    plt.xlabel('Simulation Time (ns)', fontsize=14)
    plt.ylabel('MMGBSA ΔG Bind (kcal/mol)', fontsize=14)
    plt.title('MMGBSA Binding Energy vs Simulation Time', fontsize=16, weight='bold')

    # Annotate statistical properties
    stats_text = (f'Mean = {mean_energy:.2f} kcal/mol\n'
                  f'Std Dev = {std_energy:.2f} kcal/mol\n'
                  f'Median = {median_energy:.2f} kcal/mol\n'
                  f'Min = {min_energy:.2f} kcal/mol\n'
                  f'Max = {max_energy:.2f} kcal/mol\n'
                  f'Range = {energy_range:.2f} kcal/mol\n'
                  f'SEM = {sem_energy:.2f} kcal/mol')

    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Grid settings
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)

    # Legend
    plt.legend(loc='best', fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Show or save the plot
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Generate MM-GBSA Binding Energy Plot with Statistical Analysis')
    parser.add_argument('-t', '--total_sim_time', type=float, required=True, help='Total simulation time (ns)')
    parser.add_argument('-f', '--total_frames', type=int, required=True, help='Total number of frames')
    parser.add_argument('-s', '--frame_skip', type=int, help='Number of frames skipped (default: no frame skip)')
    parser.add_argument('--save', type=str, help='Path to save the plot')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Generate the plot
    plot_mmgbsa(args.file_path, args.total_sim_time, args.total_frames, args.frame_skip, args.save)

