import numpy as np
import MDAnalysis as mda
from sklearn.neighbors import BallTree
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
from MDAnalysis.analysis import align
import random
import pandas as pd
from scipy.stats import gaussian_kde
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import Cividis256, YlOrRd

# Fix random seeds for reproducibility
np.random.seed(13)
random.seed(13)

def log_message(message):
    print(f"[INFO] {message}")

def load_md_data(topology_file, trajectory_file):
    log_message("Loading MD data...")
    try:
        u = mda.Universe(topology_file, trajectory_file)
        log_message("MD data loaded successfully.")
        return u
    except Exception as e:
        log_message(f"Error loading MD data: {e}")
        sys.exit(1)

def align_trajectory(universe, selection, reference_frame=0):
    try:
        universe.trajectory[reference_frame]
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        log_message("Trajectory aligned successfully.")
        return aligner
    except Exception as e:
        log_message(f"Error aligning trajectory: {e}")
        sys.exit(1)

def compute_features(universe, selection):
    log_message("Computing features...")
    try:
        selected_atoms = universe.select_atoms(selection)
        num_frames = len(universe.trajectory)
        features = np.zeros((num_frames, len(selected_atoms.positions.flatten())))

        for i, ts in enumerate(universe.trajectory):
            features[i] = selected_atoms.positions.flatten()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        log_message("Features computed and normalized successfully.")
        return features
    except Exception as e:
        log_message(f"Error computing features: {e}")
        sys.exit(1)

def compute_local_epsilons(features, k=5):
    log_message("Computing local epsilon values using Ball Tree...")
    try:
        tree = BallTree(features)
        distances, _ = tree.query(features, k=k+1)
        local_epsilons = distances[:, k]
        mean_epsilon = np.mean(local_epsilons)
        log_message(f"Using epsilon value (mean of local epsilons): {mean_epsilon}")
        return local_epsilons
    except Exception as e:
        log_message(f"Error computing local epsilon values: {e}")
        sys.exit(1)

def compute_diffusion_map(features, n_components=3, epsilon=None):
    log_message("Computing diffusion map...")
    try:
        if epsilon is None:
            local_epsilons = compute_local_epsilons(features)
        else:
            log_message(f"Using provided epsilon value: {epsilon}")
            local_epsilons = np.full(features.shape[0], epsilon)
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features))

        epsilon_product = np.outer(local_epsilons, local_epsilons)
        # Prevent division by zero
        epsilon_product[epsilon_product == 0] = 1e-10
        kernel_matrix = np.exp(-distances ** 2 / epsilon_product)

        log_message(f"Kernel matrix computed. Shape: {kernel_matrix.shape}")

        D = np.sum(kernel_matrix, axis=1)
        # Prevent division by zero
        D_sqrt_inv = np.where(D > 0, 1 / np.sqrt(D), 0)
        kernel_matrix = D_sqrt_inv[:, None] * kernel_matrix * D_sqrt_inv[None, :]

        log_message("Kernel matrix normalized successfully.")

        kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

        embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
        diffusion_map = embedding.fit_transform(kernel_matrix)

        log_message("Diffusion map computed successfully.")
        return diffusion_map
    except Exception as e:
        log_message(f"Error computing diffusion map: {e}")
        sys.exit(1)

def cluster_states(diffusion_map, n_clusters):
    log_message("Clustering states using KMeans...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=13)
        states = kmeans.fit_predict(diffusion_map)
        log_message("KMeans clustering completed.")
        return states
    except Exception as e:
        log_message(f"Error during KMeans clustering: {e}")
        sys.exit(1)

def read_energy_csv(energy_csv_path, num_frames):
    log_message(f"Reading energy data from {energy_csv_path}...")
    try:
        # Read the CSV file without headers
        df = pd.read_csv(energy_csv_path, header=None)
        # Check if at least two columns exist
        if df.shape[1] < 2:
            log_message("Energy CSV must have at least two columns.")
            sys.exit(1)
        # Check if there are enough rows to read from B2 onwards
        if df.shape[0] < 1 + num_frames:
            log_message(f"Energy CSV has {df.shape[0] - 1} energy entries starting from cell B2, which is less than the number of frames ({num_frames}).")
            sys.exit(1)
        # Read energies from B2 onwards
        energies = df.iloc[1:1+num_frames, 1].astype(float).values
        if len(energies) != num_frames:
            log_message(f"Mismatch between number of energy entries ({len(energies)}) and number of frames ({num_frames}).")
            sys.exit(1)
        log_message(f"Successfully read {len(energies)} energy values.")
        return energies
    except Exception as e:
        log_message(f"Error reading energy CSV: {e}")
        sys.exit(1)

def visualize_2d_diffusion_map(diffusion_map, states, energies):
    log_message("Creating 2D visualizations...")
    try:
        # Prepare DataFrame
        df = pd.DataFrame({
            'Diffusion Component 1': diffusion_map[:, 0],
            'Diffusion Component 2': diffusion_map[:, 1],
            'Diffusion Component 3': diffusion_map[:, 2],
            'Energy (kcal/mol)': energies,
            'Frame': np.arange(len(diffusion_map))
        })

        components = [('Diffusion Component 1', 'Diffusion Component 2'),
                     ('Diffusion Component 2', 'Diffusion Component 3'),
                     ('Diffusion Component 3', 'Diffusion Component 1')]

        for i, (x_comp, y_comp) in enumerate(components):
            x = df[x_comp].values
            y = df[y_comp].values
            energy = df['Energy (kcal/mol)'].values
            frames = df['Frame'].values

            # Compute min and max for color mapping
            energy_min = energy.min()
            energy_max = energy.max()

            # KDE calculation
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            x_padding = (xmax - xmin) * 0.10 if xmax > xmin else 1.0  # Increased padding to 10%
            y_padding = (ymax - ymin) * 0.10 if ymax > ymin else 1.0  # Increased padding to 10%
            plot_xmin = xmin - x_padding
            plot_xmax = xmax + x_padding
            plot_ymin = ymin - y_padding
            plot_ymax = ymax + y_padding

            N = 1000
            X, Y_grid = np.mgrid[xmin:xmax:N*1j, ymin:ymax:N*1j]
            positions = np.vstack([X.ravel(), Y_grid.ravel()])
            values = np.vstack([x, y])
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)

            # Contour levels and palette
            num_levels = 7
            levels = np.linspace(Z.min(), Z.max(), num_levels)
            contour_palette = YlOrRd[9][::-1]

            # Create figure with dynamic ranges
            p = figure(
                width=1000, height=1000,
                x_axis_label=x_comp, y_axis_label=y_comp,
                title=f"{x_comp} vs {y_comp}",
                tools="pan,wheel_zoom,box_zoom,reset,save",
                x_range=(plot_xmin, plot_xmax),
                y_range=(plot_ymin, plot_ymax)
            )

            # Scatter plot with energy color mapping
            color_mapper = LinearColorMapper(palette=Cividis256, low=energy_min, high=energy_max)
            source = ColumnDataSource(data=dict(
                x=x,
                y=y,
                energy=energy,
                frame=frames
            ))
            p.scatter('x', 'y', size=12, source=source, fill_color={'field': 'energy', 'transform': color_mapper},
                      line_color='black', fill_alpha=0.8)

            # Add contour lines
            p.contour(X, Y_grid, Z, levels=levels, fill_color=None, line_color=contour_palette, line_width=3)

            # Add hover tool
            hover = HoverTool(
                tooltips=[
                    ('Frame', '@frame'),
                    ('Energy (kcal/mol)', '@energy{0.00}')
                ],
                mode='mouse'
            )
            p.add_tools(hover)

            # Add color bar
            color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0),
                                 title='MM-GBSA DeltaG (kcal/mol)')
            p.add_layout(color_bar, 'right')

            # Save plot
            output_file(f"diffusion_map_{i+1}.html")
            save(p)

            log_message(f"Plot {i+1} created and saved as 'diffusion_map_{i+1}.html'.")

        log_message("2D visualizations created successfully.")
    except Exception as e:
        log_message(f"Error creating visualizations: {e}")
        sys.exit(1)

def main(topology_file, trajectory_file, selection, energy_csv, n_components=3, epsilon=None, n_clusters=5):
    universe = load_md_data(topology_file, trajectory_file)
    
    align_trajectory(universe, selection)
    
    features = compute_features(universe, selection)
    
    diffusion_map = compute_diffusion_map(features, n_components, epsilon)
    
    states = cluster_states(diffusion_map, n_clusters)
    
    num_frames = len(universe.trajectory)
    energies = read_energy_csv(energy_csv, num_frames)
    
    visualize_2d_diffusion_map(diffusion_map, states, energies)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        log_message("Usage: python diffusion_map.py <topology_file> <trajectory_file> <selection> <energy_csv> [n_components] [epsilon] [n_clusters]")
        sys.exit(1)

    topology_file = sys.argv[1]
    trajectory_file = sys.argv[2]
    selection = sys.argv[3]
    energy_csv = sys.argv[4]
    n_components = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    epsilon = float(sys.argv[6]) if len(sys.argv) > 6 else None
    n_clusters = int(sys.argv[7]) if len(sys.argv) > 7 else 5

    main(topology_file, trajectory_file, selection, energy_csv, n_components, epsilon, n_clusters)

