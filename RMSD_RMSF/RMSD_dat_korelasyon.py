import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
from matplotlib.lines import Line2D
from scipy.signal import welch, detrend
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf

def main():
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
    try:
        sim_time = float(sys.argv[1])
    except ValueError:
        logging.error("Simulation time must be a float.")
        sys.exit(1)

    # Determine files and labels
    file_labels = {}
    try:
        for i in range(2, len(sys.argv), 2):
            dat_file = sys.argv[i]
            label = sys.argv[i + 1]
            if not os.path.isfile(dat_file):
                logging.error(f"File not found: {dat_file}")
                sys.exit(1)
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

    # Create the main figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(32, 10), constrained_layout=True)  # Increased figure width and height

    # For statistical information
    stat_table = []

    # Initialize xf_max for setting x-axis limits later
    xf_max = 0

    # Process each file
    for i, (dat_file, label) in enumerate(file_labels.items()):
        try:
            # Read the .dat file
            data = pd.read_csv(dat_file, sep='\s+')
            if 'Time' in data.columns:
                time_ns = data['Time'].values
            elif 'Time(ns)' in data.columns:
                time_ns = data['Time(ns)'].values
            elif 'Time (ns)' in data.columns:
                time_ns = data['Time (ns)'].values
            else:
                # If time is not provided, assume uniform sampling
                time_ns = np.linspace(0, sim_time, len(data))
            rmsd = data['Prot_CA'].values
        except Exception as e:
            logging.error(f"Error reading or processing {dat_file}: {e}")
            continue

        # Check if time is uniformly sampled
        dt = np.median(np.diff(time_ns))
        if not np.allclose(np.diff(time_ns), dt):
            logging.warning(f"Time data in {dat_file} is not uniformly sampled. Resampling to uniform sampling.")
            uniform_time = np.arange(time_ns[0], time_ns[-1], dt)
            interp_func = interp1d(time_ns, rmsd, kind='linear', fill_value="extrapolate")
            rmsd_uniform = interp_func(uniform_time)
            time_ns = uniform_time
            rmsd = rmsd_uniform
        else:
            dt = time_ns[1] - time_ns[0]

        # Detrend the data to remove linear trends
        rmsd_detrended = detrend(rmsd)

        # Compute autocorrelation using statsmodels
        nlags = min(len(rmsd_detrended) // 2, 1000)
        autocorr_full, confint_full = acf(rmsd_detrended, nlags=nlags, alpha=0.05, fft=True)

        lag_times_full = np.arange(len(autocorr_full)) * dt

        # Compute correlation time (integral of autocorrelation function)
        correlation_time = np.trapz(autocorr_full, dx=dt)

        # Exclude lag zero for plotting to avoid issues with confidence intervals at lag zero
        autocorr = autocorr_full[1:]
        confint = confint_full[1:]
        lag_times = lag_times_full[1:]

        # Append statistics to the table
        stat_table.append({
            'Label': label,
            'Mean RMSD (Å)': np.mean(rmsd),
            'Std RMSD (Å)': np.std(rmsd),
            'SEM RMSD (Å)': np.std(rmsd) / np.sqrt(len(rmsd)),
            'Correlation Time (ns)': correlation_time,
        })

        # Plot the autocorrelation with confidence intervals
        axes[0].plot(lag_times, autocorr, label=label, color=colors[i], linewidth=2)
        axes[0].fill_between(lag_times, confint[:, 0], confint[:, 1], color=colors[i], alpha=0.2)

        # PSD Analysis using Welch's method
        try:
            fs = 1.0 / dt  # Sampling frequency
            nperseg = min(256, len(rmsd_detrended))
            frequencies, psd = welch(rmsd_detrended, fs=fs, nperseg=nperseg)

            # Update xf_max
            xf_max = max(xf_max, np.max(frequencies))

            # Plot the PSD
            axes[1].plot(frequencies, psd, label=label, color=colors[i], linewidth=2)
        except Exception as e:
            logging.error(f"Error performing PSD analysis on {dat_file}: {e}")
            continue

    # Set labels and title for autocorrelation plot
    axes[0].set_xlabel('Lag Time (ns)', fontsize=18, labelpad=15)
    axes[0].set_ylabel('Autocorrelation', fontsize=18, labelpad=15)
    axes[0].set_title('RMSD Autocorrelation Function', fontsize=22)

    # Set labels and title for PSD plot
    axes[1].set_xlabel('Frequency (1/ns)', fontsize=18, labelpad=15)
    axes[1].set_ylabel('Power Spectral Density', fontsize=18, labelpad=15)
    axes[1].set_title('RMSD PSD Analysis', fontsize=22)

    # Set x-axis limit for PSD plot if xf_max is greater than zero
    if xf_max > 0:
        axes[1].set_xlim(0, xf_max)  # Limit x-axis to positive frequencies

    # Create custom legend handles for data lines
    dataset_handles = [Line2D([0], [0], color=colors[i], lw=2, label=label) for i, label in enumerate(file_labels.values())]

    # Create a custom legend handle for confidence intervals
    ci_handle = Line2D([0], [0], color='gray', lw=2, alpha=0.2, label='95% Confidence Interval')

    # Combine all legend handles
    all_handles = dataset_handles + [ci_handle]

    # Add legend to the autocorrelation plot
    axes[0].legend(handles=all_handles, fontsize=14, loc='upper right', frameon=True)

    # Add legend to the PSD plot
    axes[1].legend(handles=dataset_handles, fontsize=14, loc='upper right', frameon=True)

    # Adjust tick parameters
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Create a table for statistics
    stat_df = pd.DataFrame(stat_table)
    print("\nStatistical Summary:")
    print(stat_df.to_string(index=False))

    # Display the plots on screen
    plt.show()

    logging.info("Autocorrelation and PSD analysis completed successfully.")

if __name__ == "__main__":
    main()

