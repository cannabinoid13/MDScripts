import numpy as np
import MDAnalysis as mda
from sklearn.neighbors import BallTree
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
from MDAnalysis.analysis import align
import pyemma
import random

# Rastgelelikleri sabitle
np.random.seed(13)
random.seed(13)

def log_message(message):
    print(f"[INFO] {message}")

def load_md_data(topology_file, trajectory_file):
    log_message("MD verileri yükleniyor...")
    try:
        u = mda.Universe(topology_file, trajectory_file)
        log_message("MD verileri başarıyla yüklendi.")
        return u
    except Exception as e:
        log_message(f"MD verileri yüklenirken hata oluştu: {e}")
        sys.exit(1)

def align_trajectory(universe, selection, reference_frame=0):
    try:
        universe.trajectory[reference_frame]
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        log_message("Yörünge başarıyla hizalandı.")
        return aligner
    except Exception as e:
        log_message(f"Yörünge hizalanırken hata oluştu: {e}")
        sys.exit(1)

def compute_features(universe, selection):
    log_message("Özellikler hesaplanıyor...")
    try:
        selected_atoms = universe.select_atoms(selection)
        num_frames = len(universe.trajectory)
        features = np.zeros((num_frames, len(selected_atoms.positions.flatten())))

        for i, ts in enumerate(universe.trajectory):
            features[i] = selected_atoms.positions.flatten()

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        log_message("Özellikler başarıyla hesaplandı ve normalleştirildi.")
        return features
    except Exception as e:
        log_message(f"Özellikler hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def compute_local_epsilons(features, k=5):
    log_message("Ball Tree kullanarak lokal epsilon değerleri hesaplanıyor...")
    try:
        tree = BallTree(features)
        distances, _ = tree.query(features, k=k+1)
        local_epsilons = distances[:, k]
        mean_epsilon = np.mean(local_epsilons)
        log_message(f"Epsilon değeri (lokal epsilonların ortalaması) kullanılıyor: {mean_epsilon}")
        return local_epsilons
    except Exception as e:
        log_message(f"Lokal epsilon değerleri hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def compute_diffusion_map(features, n_components=3, epsilon=None):
    log_message("Diffusion map hesaplanıyor...")
    try:
        if epsilon is None:
            local_epsilons = compute_local_epsilons(features)
        else:
            log_message(f"Verilen epsilon değeri kullanılıyor: {epsilon}")
            local_epsilons = np.full(features.shape[0], epsilon)
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features))

        epsilon_product = np.outer(local_epsilons, local_epsilons)
        # Öncelikle sıfırları önlemek için epsilon_product'da sıfır değerleri kontrol ediliyor
        epsilon_product[epsilon_product == 0] = 1e-10  # Küçük bir sayı ile değiştirilir
        kernel_matrix = np.exp(-distances ** 2 / epsilon_product)

        log_message(f"Kernel matrisi hesaplandı. Boyut: {kernel_matrix.shape}")

        D = np.sum(kernel_matrix, axis=1)
        # Sıfır bölme hatalarını önlemek için D değerlerini kontrol ediyoruz
        D_sqrt_inv = np.where(D > 0, 1 / np.sqrt(D), 0)
        kernel_matrix = D_sqrt_inv[:, None] * kernel_matrix * D_sqrt_inv[None, :]

        log_message("Kernel matrisi başarıyla normalize edildi.")

        kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

        embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
        diffusion_map = embedding.fit_transform(kernel_matrix)

        log_message("Diffusion map başarıyla hesaplandı.")
        return diffusion_map
    except Exception as e:
        log_message(f"Diffusion map hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def cluster_states(diffusion_map, n_clusters):
    log_message("KMeans ile durumlar kümeleniyor...")
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=13)
        states = kmeans.fit_predict(diffusion_map)
        log_message("KMeans kümeleme tamamlandı.")
        return states
    except Exception as e:
        log_message(f"KMeans kümeleme sırasında hata oluştu: {e}")
        sys.exit(1)

def build_msm(states, lag_time):
    log_message("MSM inşa ediliyor...")
    try:
        msm = pyemma.msm.estimate_markov_model([states], lag=lag_time)
        log_message("MSM başarıyla inşa edildi.")
        return msm
    except Exception as e:
        log_message(f"MSM inşa edilirken hata oluştu: {e}")
        sys.exit(1)

def compute_free_energy_msm(msm, temperature=300):
    """MSM'den serbest enerji hesapla."""
    log_message("MSM kullanarak serbest enerji hesaplanıyor...")
    try:
        R = 1.98720425864083e-3  # kcal/(mol·K)
        stationary_distribution = msm.pi
        # Sıfır dağılımlarını önlemek için küçük bir sayı ekleniyor
        stationary_distribution = np.where(stationary_distribution > 0, stationary_distribution, 1e-10)
        free_energy = -R * temperature * np.log(stationary_distribution)
        free_energy -= np.min(free_energy)
        # Serbest enerjiyi 0 ile 5 arasında normalize et
        max_energy = np.max(free_energy)
        if max_energy > 0:
            free_energy = (free_energy / max_energy) * 5
        else:
            free_energy = free_energy  # Tüm değerler aynı ise değişiklik yapma
        log_message("MSM kullanarak serbest enerji hesaplaması başarıyla tamamlandı.")
        return free_energy
    except Exception as e:
        log_message(f"Serbest enerji hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def visualize_2d_diffusion_map(diffusion_map, states, msm, free_energy):
    log_message("2D grafikler oluşturuluyor...")
    try:
        import pandas as pd
        from scipy.stats import gaussian_kde
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, ColorBar, Label
        from bokeh.palettes import Cividis256, YlOrRd
        import bokeh

        # Durumları MSM indekslerine eşle
        state_to_msm_index = {state_label: idx for idx, state_label in enumerate(msm.active_set)}
        msm_indices = np.array([state_to_msm_index.get(s, -1) for s in states])
        valid_frames = msm_indices != -1
        msm_indices_valid = msm_indices[valid_frames]
        diffusion_map_valid = diffusion_map[valid_frames]
        free_energy_valid = free_energy[msm_indices_valid]

        # Veri hazırlığı
        df = pd.DataFrame({
            'Diffusion Component 1': diffusion_map_valid[:, 0],
            'Diffusion Component 2': diffusion_map_valid[:, 1],
            'Diffusion Component 3': diffusion_map_valid[:, 2],
            'Free Energy (kcal/mol)': free_energy_valid,
            'Frame': np.arange(len(diffusion_map))[valid_frames]
        })

        components = [('Diffusion Component 1', 'Diffusion Component 2'),
                      ('Diffusion Component 2', 'Diffusion Component 3'),
                      ('Diffusion Component 3', 'Diffusion Component 1')]

        for i, (x_comp, y_comp) in enumerate(components):
            x = df[x_comp].values
            y = df[y_comp].values
            energy = df['Free Energy (kcal/mol)'].values
            frames = df['Frame'].values

            # KDE hesapla
            xmin, xmax = x.min(), x.max()
            ymin, ymax = y.min(), y.max()
            N = 300
            X, Y = np.mgrid[xmin:xmax:N*1j, ymin:ymax:N*1j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([x, y])
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)

            # Kontur seviyeleri ve renk paleti
            num_levels = 7
            levels = np.linspace(Z.min(), Z.max(), num_levels)
            # Renk paletini ters çevir
            contour_palette = YlOrRd[9][::-1]

            # Grafik oluştur
            p = figure(
                width=1200, height=1200,
                x_axis_label=x_comp, y_axis_label=y_comp,
                title=f"{x_comp} vs {y_comp}",
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            # Veri noktalarını ekle
            point_color_mapper = LinearColorMapper(palette=Cividis256, low=energy.min(), high=energy.max())
            source = ColumnDataSource(data=dict(
                x=x,
                y=y,
                energy=energy,
                frame=frames
            ))
            p.scatter('x', 'y', size=12, source=source, fill_color={'field': 'energy', 'transform': point_color_mapper},
                      line_color='black', fill_alpha=0.8)

            # Kontur çizgilerini ekle ve kalınlığı 4 yap
            p.contour(X, Y, Z, levels=levels, fill_color=None, line_color=contour_palette, line_width=3)


            # Hover aracı ekle
            hover = HoverTool(
                tooltips=[
                    ('Frame', '@frame'),
                    ('Free Energy (kcal/mol)', '@energy{0.00}')
                ],
                mode='mouse'
            )
            p.add_tools(hover)

            # Renk skalası (color bar) ekle
            color_bar = ColorBar(color_mapper=point_color_mapper, label_standoff=12, location=(0,0),
                                 title='Free Energy (F = -kT ln(π)) (kcal/mol)')
            p.add_layout(color_bar, 'right')

            # Enerji hesaplama formülünü grafikten kaldırıldı

            # Grafiği kaydet
            output_file(f"diffusion_map_{i+1}.html")
            save(p)

            log_message(f"Grafik {i+1} oluşturuldu ve 'diffusion_map_{i+1}.html' olarak kaydedildi.")

        log_message("2D grafikler başarıyla oluşturuldu.")
    except Exception as e:
        log_message(f"2D grafikler oluşturulurken hata oluştu: {e}")
        sys.exit(1)

def main(topology_file, trajectory_file, selection, n_components=3, epsilon=None, n_clusters=5, lag_time=1):
    universe = load_md_data(topology_file, trajectory_file)
    
    align_trajectory(universe, selection)
    
    features = compute_features(universe, selection)
    
    diffusion_map = compute_diffusion_map(features, n_components, epsilon)
    
    states = cluster_states(diffusion_map, n_clusters)
    
    msm = build_msm(states, lag_time)
    
    frame_energies = compute_free_energy_msm(msm)
    
    visualize_2d_diffusion_map(diffusion_map, states, msm, frame_energies)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        log_message("Kullanım: python diffusion_map.py <topoloji_dosyası> <yörünge_dosyası> <seçim> [n_components] [epsilon] [n_clusters] [lag_time]")
        sys.exit(1)

    topology_file = sys.argv[1]
    trajectory_file = sys.argv[2]
    selection = sys.argv[3]
    n_components = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    epsilon = float(sys.argv[5]) if len(sys.argv) > 5 else None
    n_clusters = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    lag_time = int(sys.argv[7]) if len(sys.argv) > 7 else 1

    main(topology_file, trajectory_file, selection, n_components, epsilon, n_clusters, lag_time)

