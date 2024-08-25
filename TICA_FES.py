import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
from deeptime.decomposition import TICA
from MDAnalysis.analysis import align

def load_data(pdb_file, xtc_file, selection):
    # Load the trajectory
    u = mda.Universe(pdb_file, xtc_file)
    # Align the trajectory
    align_trajectory(u, selection)
    # Extract selected atom positions
    protein = u.select_atoms(selection)
    positions = np.array([protein.positions for ts in u.trajectory])
    # Reshape for TICA
    data = np.reshape(positions, (positions.shape[0], -1))
    return data

def align_trajectory(universe, selection, reference_frame=0):
    try:
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        return aligner
    except Exception as e:
        print(f"Error aligning trajectory: {e}")
        raise

def run_tica(data, lag):
    # Perform TICA analysis using deeptime
    tica = TICA(lagtime=lag, dim=2)
    tica_model = tica.fit(data)
    tica_traj = tica_model.transform(data)
    return tica_traj, tica_model

def calculate_free_energy(tica_traj, bins=25):
    # Convert tica_traj to numpy array if it's a list
    if isinstance(tica_traj, list):
        tica_traj = np.concatenate(tica_traj)

    # Compute free energy surface (FES)
    H, xedges, yedges = np.histogram2d(tica_traj[:, 0], tica_traj[:, 1], bins=bins)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = -np.log(H.T + 1)  # Adding 1 to avoid log(0)

    return X, Y, Z

def plot_free_energy_surface_2d(X, Y, Z, title):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Free Energy')
    plt.xlabel('TICA Component 1')
    plt.ylabel('TICA Component 2')
    plt.title(title)
    plt.show()

def main(pdb_file, xtc_file, selection, lag, bins):
    # Load the data
    data = load_data(pdb_file, xtc_file, selection)
    # Run TICA
    tica_traj, tica_model = run_tica(data, lag)
    # Calculate free energy surface (FES)
    X, Y, Z = calculate_free_energy(tica_traj, bins)

    # Plot the free energy surface
    plot_free_energy_surface_2d(X, Y, Z, '2D Free Energy Surface from TICA')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TICA analysis and plot 2D Free Energy Surface (FES).')
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file.')
    parser.add_argument('xtc_file', type=str, help='Path to the XTC file.')
    parser.add_argument('--select', type=str, default='name CA', help='Atom selection criteria (e.g., "name CA")')
    parser.add_argument('--lag', type=int, default=10, help='Lag time for TICA. Default is 10.')
    parser.add_argument('--bins', type=int, default=25, help='Number of bins for histogram (default: 25).')
    args = parser.parse_args()

    main(args.pdb_file, args.xtc_file, args.select, args.lag, args.bins)

