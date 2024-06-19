import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyemma


def load_data(pdb_file, xtc_file):
    # Load the trajectory
    u = mda.Universe(pdb_file, xtc_file)
    # Extract alpha carbon positions
    protein = u.select_atoms('name CA')
    positions = np.array([protein.positions for ts in u.trajectory])
    # Reshape for PyEMMA
    data = np.reshape(positions, (positions.shape[0], -1))
    return data


def run_tica(data, lag):
    # Perform TICA analysis
    tica_obj = pyemma.coordinates.tica(data, lag=lag)
    tica_traj = tica_obj.get_output()
    return tica_traj, tica_obj


def calculate_free_energy(tica_traj, bins=350):
    # Convert tica_traj to numpy array if it's a list
    if isinstance(tica_traj, list):
        tica_traj = np.concatenate(tica_traj)

    # Compute free energy surface (FES)
    H, xedges, yedges = np.histogram2d(tica_traj[:, 0], tica_traj[:, 1], bins=bins)
    X, Y = np.meshgrid(xedges, yedges)
    Z = -np.log(H.T + 1)  # Adding 1 to avoid log(0)

    fig, ax = plt.subplots()
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    fig.colorbar(c, ax=ax, label='Free Energy (kJ/mol)')
    return c


def main(pdb_file, xtc_file, lag):
    # Load the data
    data = load_data(pdb_file, xtc_file)
    # Run TICA
    tica_traj, tica_obj = run_tica(data, lag)
    # Calculate free energy surface (FES)
    calculate_free_energy(tica_traj)

    # Show the plot
    plt.title('2D Free Energy Surface')
    plt.xlabel('TICA Component 1')
    plt.ylabel('TICA Component 2')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TICA analysis and plot 2D Free Energy Surface (FES).')
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file.')
    parser.add_argument('xtc_file', type=str, help='Path to the XTC file.')
    parser.add_argument('--lag', type=int, default=10, help='Lag time for TICA. Default is 10.')
    args = parser.parse_args()

    main(args.pdb_file, args.xtc_file, args.lag)
