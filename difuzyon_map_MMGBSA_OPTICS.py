import numpy as np
import MDAnalysis as mda
from sklearn.neighbors import BallTree
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import sys
from MDAnalysis.analysis import align
import random
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, ColorBar, BasicTicker
from bokeh.palettes import Category10, Category20, Turbo256, Set2
import argparse
import traceback
import logging

# Rastgelelik için sabit tohumlar
np.random.seed(13)
random.seed(13)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Protein Konformasyonlarını OPTICS ile Kümeleme")
    parser.add_argument("topology_file", help="Topoloji dosyası (örn. topology.pdb)")
    parser.add_argument("trajectory_file", help="Trajektori dosyası (örn. trajectory.dcd)")
    parser.add_argument("selection", help="MDAnalysis ile atom seçimi (örn. 'protein and name CA')")
    parser.add_argument("energy_csv", help="Enerji verilerini içeren CSV dosyası (örn. mmgbsa-prime-out.csv)")
    parser.add_argument("--n_components", type=int, default=3, help="Diffusion map için bileşen sayısı (varsayılan: 3)")
    parser.add_argument("--epsilon", type=float, default=None, help="OPTICS için epsilon değeri (varsayılan: Otomatik hesaplama)")
    parser.add_argument("--min_samples", type=int, default=5, help="OPTICS için minimum komşu sayısı (varsayılan: 5)")
    parser.add_argument("--xi", type=float, default=0.05, help="OPTICS için küme sınırları belirleme parametresi (varsayılan: 0.05)")
    parser.add_argument("--min_cluster_size", type=float, default=0.05, help="OPTICS için minimum küme büyüklüğü (varsayılan: 0.05)")
    return parser.parse_args()

def load_md_data(topology_file, trajectory_file):
    """
    MD verilerini yükler.
    """
    logger.info("MD verileri yükleniyor...")
    try:
        u = mda.Universe(topology_file, trajectory_file)
        logger.info("MD verileri başarıyla yüklendi.")
        return u
    except Exception as e:
        logger.error(f"MD verileri yüklenirken hata oluştu: {e}")
        sys.exit(1)

def align_trajectory(universe, selection, reference_frame=0):
    """
    Trajektoriyi referans bir çerçeveye göre hizalar.
    """
    try:
        universe.trajectory[reference_frame]
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        logger.info("Trajektori başarıyla hizalandı.")
        return aligner
    except Exception as e:
        logger.error(f"Trajektori hizalarken hata oluştu: {e}")
        sys.exit(1)

def read_energy_csv(energy_csv_path, num_frames):
    """
    Enerji verilerini CSV dosyasından okur.
    """
    logger.info(f"{energy_csv_path} dosyasından enerji verileri okunuyor...")
    try:
        df = pd.read_csv(energy_csv_path)
        if df.shape[1] < 2:
            logger.error("Enerji CSV en az iki sütuna sahip olmalıdır.")
            sys.exit(1)
        # Enerji verilerini ikinci sütunun ikinci hücresinden itibaren okuyun
        energies = df.iloc[1:, 1].astype(float).values  # İkinci sütun, ikinci hücreden itibaren
        if len(energies) < num_frames:
            logger.error(f"Enerji girdileri sayısı ({len(energies)}) çerçeve sayısından ({num_frames}) az.")
            sys.exit(1)
        elif len(energies) > num_frames:
            # Enerji verilerini çerçeve sayısına göre kırpın
            energies = energies[:num_frames]
        logger.info(f"{len(energies)} enerji değeri başarıyla okundu.")
        return energies
    except Exception as e:
        logger.error(f"Enerji CSV okunurken hata oluştu: {e}")
        sys.exit(1)

def compute_features(universe, selection, energies):
    """
    Özellikleri hesaplar ve normalleştirir.
    """
    logger.info("Özellikler hesaplanıyor...")
    try:
        selected_atoms = universe.select_atoms(selection)
        num_frames = len(universe.trajectory)
        num_features = selected_atoms.positions.size
        features = np.zeros((num_frames, num_features + 1))  # Enerji için +1

        for i, ts in enumerate(universe.trajectory):
            # Atom koordinatlarını ve enerji değerini birleştiriyoruz
            features[i, :-1] = selected_atoms.positions.flatten()
            features[i, -1] = energies[i]  # Enerji değerini ekliyoruz

        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        logger.info("Özellikler başarıyla hesaplandı ve normalleştirildi.")
        return features
    except Exception as e:
        logger.error(f"Özellikler hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def compute_local_epsilons(features, k=5):
    """
    Yerel epsilon değerlerini hesaplar.
    """
    logger.info("Yerel epsilon değerleri Ball Tree kullanarak hesaplanıyor...")
    try:
        tree = BallTree(features)
        distances, _ = tree.query(features, k=k+1)
        local_epsilons = distances[:, k]
        mean_epsilon = np.mean(local_epsilons)
        logger.info(f"Epsilon değeri (yerel epsilonların ortalaması): {mean_epsilon}")
        return local_epsilons
    except Exception as e:
        logger.error(f"Yerel epsilon değerleri hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def compute_diffusion_map(features, n_components=3, epsilon=None):
    """
    Diffusion map hesaplar.
    """
    logger.info("Diffusion map hesaplanıyor...")
    try:
        if epsilon is None:
            local_epsilons = compute_local_epsilons(features)
        else:
            logger.info(f"Verilen epsilon değeri kullanılıyor: {epsilon}")
            local_epsilons = np.full(features.shape[0], epsilon)
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features))

        epsilon_product = np.outer(local_epsilons, local_epsilons)
        # Sıfıra bölmeyi engelle
        epsilon_product[epsilon_product == 0] = 1e-10
        kernel_matrix = np.exp(-distances ** 2 / epsilon_product)

        logger.info(f"Kernel matrisi hesaplandı. Şekil: {kernel_matrix.shape}")

        D = np.sum(kernel_matrix, axis=1)
        # Sıfıra bölmeyi engelle
        D_sqrt_inv = np.where(D > 0, 1 / np.sqrt(D), 0)
        kernel_matrix = D_sqrt_inv[:, None] * kernel_matrix * D_sqrt_inv[None, :]

        logger.info("Kernel matrisi başarıyla normalleştirildi.")

        kernel_matrix = (kernel_matrix + kernel_matrix.T) / 2

        embedding = SpectralEmbedding(n_components=n_components, affinity='precomputed')
        diffusion_map = embedding.fit_transform(kernel_matrix)

        logger.info("Diffusion map başarıyla hesaplandı.")
        return diffusion_map
    except Exception as e:
        logger.error(f"Diffusion map hesaplanırken hata oluştu: {e}")
        sys.exit(1)

def cluster_states(diffusion_map, min_samples=5, xi=0.05, min_cluster_size=0.05):
    """
    OPTICS algoritması ile kümeleri belirler.
    """
    logger.info("OPTICS ile durumlar kümeleme yapılıyor...")
    try:
        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
        optics.fit(diffusion_map)
        states = optics.labels_
        reachability = optics.reachability_[optics.ordering_]
        logger.info("OPTICS kümeleme tamamlandı.")
        return states, reachability, optics.ordering_
    except Exception as e:
        logger.error(f"Kümeleme sırasında hata oluştu: {e}")
        sys.exit(1)

def visualize_2d_diffusion_map(diffusion_map, states, energies, reachability, ordering):
    """
    2D görselleştirmeleri oluşturur ve kaydeder.
    """
    logger.info("2D görselleştirmeler oluşturuluyor...")
    try:
        # Veri Çerçevesini Hazırlayın
        df = pd.DataFrame({
            'Diffusion Component 1': diffusion_map[:, 0],
            'Diffusion Component 2': diffusion_map[:, 1],
            'Diffusion Component 3': diffusion_map[:, 2] if diffusion_map.shape[1] > 2 else np.zeros(len(diffusion_map)),
            'Energy (kcal/mol)': energies,
            'Frame': np.arange(len(diffusion_map)),
            'Cluster': states
        })

        # Küme etiketlerini sayısal olarak tutun (gürültü için -1)
        clusters = df['Cluster'].values
        unique_clusters = np.unique(clusters)
        
        # Gürültü noktaları (-1) dahil, kümeleri indekslere eşleyin
        non_noise_clusters = unique_clusters[unique_clusters != -1]
        cluster_mapping = {label: idx for idx, label in enumerate(non_noise_clusters)}
        if -1 in unique_clusters:
            cluster_mapping[-1] = len(cluster_mapping)  # Gürültü için en son indeks
        
        # cluster_indices: kümeleri indekslerle eşleştirin
        cluster_indices = np.array([cluster_mapping[label] for label in clusters])

        # Veri çerçevesine yeni küme indekslerini ekleyin
        df['ClusterIndex'] = cluster_indices

        components = [('Diffusion Component 1', 'Diffusion Component 2'),
                     ('Diffusion Component 2', 'Diffusion Component 3'),
                     ('Diffusion Component 3', 'Diffusion Component 1')]

        for i, (x_comp, y_comp) in enumerate(components):
            if x_comp not in df or y_comp not in df:
                continue  # Eğer yeterli bileşen yoksa atla

            x = df[x_comp].values
            y = df[y_comp].values
            energy = df['Energy (kcal/mol)'].values
            frames = df['Frame'].values
            clusters = df['Cluster'].values
            cluster_indices = df['ClusterIndex'].values

            # Renk paletini kümelere göre ayarlayın
            num_clusters = len(unique_clusters)
            if num_clusters == 1:
                # Sadece bir küme varsa, belirli bir renk
                palette = ['#1f77b4']
            elif num_clusters == 2:
                # İki küme için manuel renk paleti
                palette = ['#1f77b4', '#ff7f0e']  # Mavi ve turuncu
            elif num_clusters <= 8:
                palette = Set2[num_clusters]
            elif num_clusters <= 10:
                palette = Category10[10][:num_clusters]
            elif num_clusters <= 20:
                palette = Category20[20][:num_clusters]
            else:
                palette = Turbo256[:num_clusters]

            # Gürültü için ayrı bir renk (son renge gri ekle)
            if -1 in unique_clusters:
                # Son renge gri ekleyerek gürültüyü temsil edin
                palette = list(palette[:-1]) + ['#7f7f7f']  # Liste olarak birleştiriyoruz

            color_mapper = LinearColorMapper(palette=palette, low=cluster_indices.min(), high=cluster_indices.max())

            source = ColumnDataSource(data=dict(
                x=x,
                y=y,
                energy=energy,
                frame=frames,
                cluster=clusters,
                cluster_index=cluster_indices
            ))

            p = figure(
                width=1000, height=1000,
                x_axis_label=x_comp, y_axis_label=y_comp,
                title=f"{x_comp} vs {y_comp}",
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            )

            p.scatter('x', 'y', size=8, source=source,
                      fill_color={'field': 'cluster_index', 'transform': color_mapper},
                      line_color='black', fill_alpha=0.8)

            # Hover aracını güncelleyin
            hover = p.select_one(HoverTool)
            hover.tooltips = [
                ('Frame', '@frame'),
                ('Energy (kcal/mol)', '@energy{0.00}'),
                ('Cluster', '@cluster')
            ]

            # Renk çubuğunu ekleyin
            # Create tick labels
            tick_labels = {}
            for label in unique_clusters:
                tick_value = cluster_mapping[label]
                tick_labels[str(tick_value)] = str(label)
            
            color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0),
                                 title='Cluster',
                                 ticker=BasicTicker(desired_num_ticks=num_clusters),
                                 major_label_overrides=tick_labels)
            p.add_layout(color_bar, 'right')

            output_file(f"diffusion_map_{i+1}.html")
            save(p)

            logger.info(f"Plot {i+1} oluşturuldu ve 'diffusion_map_{i+1}.html' olarak kaydedildi.")

        # Reachability grafiğini oluşturun
        logger.info("Reachability grafiği oluşturuluyor...")
        try:
            p_reachability = figure(
                width=1000, height=400,
                x_axis_label='Sample',
                y_axis_label='Reachability Distance',
                title='OPTICS Reachability Plot',
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            # x_values olarak ordering kullanın
            reachability_plot = reachability.copy()
            reachability_plot[np.isinf(reachability_plot)] = reachability_plot[np.isfinite(reachability_plot)].max()
            x_values = np.arange(len(reachability_plot))
            p_reachability.line(x_values, reachability_plot, line_width=2)

            output_file("reachability_plot.html")
            save(p_reachability)

            logger.info("Reachability grafiği oluşturuldu ve 'reachability_plot.html' olarak kaydedildi.")
        except Exception as e:
            logger.error(f"Reachability grafiği oluşturulurken hata oluştu: {e}")
            traceback.print_exc()
            sys.exit(1)

        logger.info("2D görselleştirmeler başarıyla oluşturuldu.")
    except Exception as e:
        logger.error(f"Görselleştirme sırasında hata oluştu: {e}")
        traceback.print_exc()
        sys.exit(1)

def main():
    args = parse_arguments()
    try:
        universe = load_md_data(args.topology_file, args.trajectory_file)
        align_trajectory(universe, args.selection)
        num_frames = len(universe.trajectory)
        energies = read_energy_csv(args.energy_csv, num_frames)
        features = compute_features(universe, args.selection, energies)
        diffusion_map = compute_diffusion_map(features, args.n_components, args.epsilon)
        states, reachability, ordering = cluster_states(
            diffusion_map, 
            min_samples=args.min_samples, 
            xi=args.xi, 
            min_cluster_size=args.min_cluster_size
        )
        visualize_2d_diffusion_map(diffusion_map, states, energies, reachability, ordering)
    except Exception as e:
        logger.error(f"Program sırasında hata oluştu: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

