import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
from deeptime.decomposition import TICA
from MDAnalysis.analysis import align

def load_data(pdb_file, xtc_file, selection):
    """
    Load and process molecular dynamics trajectory data.

    Parameters:
    - pdb_file (str): Path to the PDB file.
    - xtc_file (str): Path to the XTC file.
    - selection (str): Atom selection criteria.

    Returns:
    - data (np.ndarray): Reshaped position data suitable for TICA.
    """
    # Load the trajectory
    u = mda.Universe(pdb_file, xtc_file)
    # Align the trajectory
    align_trajectory(u, selection)
    # Extract selected atom positions
    protein = u.select_atoms(selection)
    positions = np.array([protein.positions.copy() for ts in u.trajectory])
    # Reshape for TICA
    data = np.reshape(positions, (positions.shape[0], -1))
    return data

def align_trajectory(universe, selection, reference_frame=0):
    """
    Align the trajectory to a reference frame based on the selection.

    Parameters:
    - universe (MDAnalysis.Universe): The Universe object containing the trajectory.
    - selection (str): Atom selection criteria for alignment.
    - reference_frame (int): Frame to use as the reference for alignment.
    """
    try:
        # Set universe to reference frame
        universe.trajectory[reference_frame]
        # Select the atoms for alignment
        reference = universe.select_atoms(selection)
        # Align trajectory
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        return aligner
    except Exception as e:
        print(f"Error aligning trajectory: {e}")
        raise

def run_tica(data, lag, n_components=3):
    """
    Perform TICA analysis on the data.

    Parameters:
    - data (np.ndarray): Input data for TICA.
    - lag (int): Lag time for TICA.
    - n_components (int): Number of TICA components to compute.

    Returns:
    - tica_traj (np.ndarray): Transformed data along TICA components.
    - tica_model: Fitted TICA model.
    """
    # Perform TICA analysis using deeptime
    tica = TICA(lagtime=lag, dim=n_components)
    tica_model = tica.fit(data)
    tica_traj = tica_model.transform(data)
    return tica_traj, tica_model

def calculate_free_energy(tica_traj, component_x=0, component_y=1, bins=25):
    """
    Calculate the free energy surface (FES) from TICA-transformed data.

    Parameters:
    - tica_traj (np.ndarray): TICA-transformed data.
    - component_x (int): Index of the TICA component for the x-axis.
    - component_y (int): Index of the TICA component for the y-axis.
    - bins (int): Number of bins for the histogram.

    Returns:
    - X, Y (np.ndarray): Meshgrid arrays for plotting.
    - Z (np.ndarray): Calculated free energy values.
    """
    # Extract the specified components
    x_data = tica_traj[:, component_x]
    y_data = tica_traj[:, component_y]

    # Compute free energy surface (FES)
    H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
    # Calculate bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xcenters, ycenters)
    Z = -np.log(H.T + 1)  # Adding 1 to avoid log(0)

    return X, Y, Z

def plot_free_energy_surface_2d(X, Y, Z, component_x, component_y, title, output_file=None):
    """
    Plot the 2D Free Energy Surface.

    Parameters:
    - X, Y (np.ndarray): Meshgrid arrays for plotting.
    - Z (np.ndarray): Free energy values.
    - component_x (int): Index of the TICA component for the x-axis.
    - component_y (int): Index of the TICA component for the y-axis.
    - title (str): Title of the plot.
    - output_file (str, optional): If provided, the plot will be saved to this file.
    """
    plt.figure(figsize=(10, 8))
    # Contour levels
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 100)
    # Plotting
    contour = plt.contourf(X, Y, Z, levels=levels, cmap='cividis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Free Energy (kT)', fontsize=14)
    plt.xlabel(f'TICA Component {component_x + 1}', fontsize=14)
    plt.ylabel(f'TICA Component {component_y + 1}', fontsize=14)
    plt.title(title, fontsize=16)
    # Increase tick label size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Adjust colorbar ticks
    cbar.ax.tick_params(labelsize=12)
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main(pdb_file, xtc_file, selection, lag, bins, output_prefix):
    """
    Main function to run the analysis and plotting.

    Parameters:
    - pdb_file (str): Path to the PDB file.
    - xtc_file (str): Path to the XTC file.
    - selection (str): Atom selection criteria.
    - lag (int): Lag time for TICA.
    - bins (int): Number of bins for the histogram.
    - output_prefix (str): Prefix for output plot files.
    """
    # Load the data
    data = load_data(pdb_file, xtc_file, selection)
    # Run TICA
    tica_traj, tica_model = run_tica(data, lag, n_components=3)

    # Define component pairs to plot
    component_pairs = [(0, 1), (0, 2), (1, 2)]

    for component_x, component_y in component_pairs:
        # Calculate free energy surface (FES)
        X, Y, Z = calculate_free_energy(tica_traj, component_x, component_y, bins)

        # Plot title
        title = f'2D Free Energy Surface from TICA Components {component_x + 1} and {component_y + 1}'

        # Output file name
        if output_prefix:
            output_file = f"{output_prefix}_TICA{component_x + 1}_vs_TICA{component_y + 1}.png"
        else:
            output_file = None

        # Plot the free energy surface
        plot_free_energy_surface_2d(X, Y, Z, component_x, component_y, title, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TICA analysis and plot 2D Free Energy Surfaces (FES) for component pairs.')
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file.')
    parser.add_argument('xtc_file', type=str, help='Path to the XTC file.')
    parser.add_argument('--select', type=str, default='name CA', help='Atom selection criteria (e.g., "name CA").')
    parser.add_argument('--lag', type=int, default=10, help='Lag time for TICA (default: 10).')
    parser.add_argument('--bins', type=int, default=25, help='Number of bins for histogram (default: 25).')
    parser.add_argument('--output_prefix', type=str, default=None, help='Prefix for output plot files (e.g., "fes").')
    args = parser.parse_args()

    main(args.pdb_file, args.xtc_file, args.select, args.lag, args.bins, args.output_prefix)

