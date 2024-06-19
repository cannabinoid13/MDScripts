import MDAnalysis as mda
import numpy as np
import plotly.express as px
import argparse
import os


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Calculate DCCM and plot heatmap from trajectory files.")
    parser.add_argument('topology', type=str, help='Topology file (PDB)')
    parser.add_argument('trajectory', type=str, help='Trajectory file (XTC)')
    parser.add_argument('output', type=str, help='Output file for DCCM matrix')
    return parser.parse_args()


def load_trajectory(trajectory_file, topology_file):
    """
    Load the trajectory and topology files using MDAnalysis.
    """
    try:
        u = mda.Universe(topology_file, trajectory_file)
        return u
    except Exception as e:
        print(f"Error loading files: {e}")
        return None


def calculate_dccm(universe):
    """
    Calculate the Dynamic Cross-Correlation Matrix (DCCM).
    """
    try:
        backbone = universe.select_atoms('backbone')
        positions = universe.trajectory.timeseries(atomgroup=backbone).transpose(1, 0, 2)
        positions = positions.reshape(positions.shape[0], -1)
        correlation_matrix = np.corrcoef(positions)

        num_residues = backbone.n_residues
        dccm = np.zeros((num_residues, num_residues))

        for i in range(num_residues):
            for j in range(num_residues):
                dccm[i, j] = correlation_matrix[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3].mean()

        return dccm
    except Exception as e:
        print(f"Error calculating DCCM: {e}")
        return None


def save_dccm_to_file(dccm, output_file):
    """
    Save the DCCM matrix to a file.
    """
    try:
        np.savetxt(output_file, dccm)
        print(f"DCCM matrix saved to {output_file}")
    except Exception as e:
        print(f"Error saving DCCM to file: {e}")


def plot_heatmap(matrix, title, trajectory_file):
    """
    Plot a heatmap of the given matrix using plotly.
    """
    try:
        file_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        fig = px.imshow(matrix, color_continuous_scale='RdBu_r', origin='lower', zmin=-1, zmax=1)
        fig.update_layout(
            title={
                'text': f"{title} for {file_name}",
                'y': 0.95,
                'x': 0.05,
                'xanchor': 'left',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            xaxis_title='Residue Index',
            yaxis_title='Residue Index',
            coloraxis_colorbar=dict(
                title="Correlation"
            )
        )
        fig.show()
    except Exception as e:
        print(f"Error plotting heatmap: {e}")


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the trajectory and topology
    universe = load_trajectory(args.trajectory, args.topology)
    if universe is None:
        return

    # Calculate DCCM
    dccm = calculate_dccm(universe)
    if dccm is not None:
        # Save DCCM to file
        save_dccm_to_file(dccm, args.output)
        # Plot heatmap
        plot_heatmap(dccm, 'Dynamic Cross-Correlation Map (DCCM)', args.trajectory)


if __name__ == '__main__':
    main()
