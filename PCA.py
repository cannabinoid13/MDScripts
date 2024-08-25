import MDAnalysis as mda
from MDAnalysis.analysis import pca, align
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

def load_simulation_data(topology_file, trajectory_file):
    try:
        u = mda.Universe(topology_file, trajectory_file)
    except Exception as e:
        print(f"Error loading simulation data: {e}")
        raise
    return u

def align_trajectory(universe, selection, reference_frame=0):
    try:
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        return aligner
    except Exception as e:
        print(f"Error aligning trajectory: {e}")
        raise

def perform_pca(universe, select):
    try:
        pca_analysis = pca.PCA(universe, select=select).run()
        return pca_analysis
    except Exception as e:
        print(f"Error performing PCA: {e}")
        raise

def calculate_free_energy_surface(transformed, bins=100):
    pc1 = transformed[:, 0]
    pc2 = transformed[:, 1]
    hist, xedges, yedges = np.histogram2d(pc1, pc2, bins=(bins, bins))
    energy = -np.log(hist.T + 1)
    return energy, xedges, yedges

def plot_free_energy_surface_3d(energy, xedges, yedges):
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = energy

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Free Energy')

    # Başlığı biraz yukarı kaydırarak grafiğin altına ekleme
    fig.text(0.5, 0.05, '3D Free Energy Surface from PCA', ha='center', va='center')

    # Renk skalası ekleme
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Free Energy')

    # Küçük 2D grafiği ekleme
    ax_inset = inset_axes(ax, width="30%", height="30%", loc=2)  # loc=2 sol üst köşe için
    ax_inset.contourf(X, Y, Z, cmap='viridis')
    ax_inset.set_xlabel('PC1')
    ax_inset.set_ylabel('PC2')
    ax_inset.set_title('2D Free Energy Surface')

    plt.show()

def main(topology_file, trajectory_file, selection, bins):
    u = load_simulation_data(topology_file, trajectory_file)
    if not u.trajectory:
        raise ValueError("Trajectory file does not contain any frames.")

    selected_atoms = u.select_atoms(selection)
    if len(selected_atoms) == 0:
        raise ValueError(f"No atoms selected with selection criteria: '{selection}'")

    # Trajektoriyi hizala
    align_trajectory(u, selection)

    # PCA analizi yap
    pca_analysis = perform_pca(u, selection)
    transformed = pca_analysis.transform(selected_atoms)
    energy, xedges, yedges = calculate_free_energy_surface(transformed, bins)
    plot_free_energy_surface_3d(energy, xedges, yedges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Analysis and 3D Free Energy Surface Plotting")
    parser.add_argument("topology", help="Topology file (PDB format)")
    parser.add_argument("trajectory", help="Trajectory file (XTC format)")
    parser.add_argument("--select", help="Atom selection criteria (e.g., 'name CA')", default="name CA")
    parser.add_argument("--bins", type=int, default=25, help="Number of bins for histogram (default: 25)")

    args = parser.parse_args()
    main(args.topology, args.trajectory, args.select, args.bins)

