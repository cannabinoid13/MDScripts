import sys
import argparse
import MDAnalysis as mda
import numpy as np
import logging
from MDAnalysis.analysis import align
from sklearn.cluster import AgglomerativeClustering
from pyemma import msm
import matplotlib.pyplot as plt

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="MSM oluşturma ve analizi için PyEMMA ve MDAnalysis kullanımı.")
    parser.add_argument('files', type=str, nargs='+', help='PDB ve XTC dosyalarının listesi (PDB1 XTC1 PDB2 XTC2 ...)')
    parser.add_argument('--n_clusters', type=int, default=100, help='Agglomerative Clustering için küme sayısı')
    parser.add_argument('--selection', type=str, default='name CA', help='Hizalama ve özellik çıkarımı için atom seçimi')
    return parser.parse_args()

def load_trajectory(pdb_file, xtc_file):
    try:
        logger.info(f"{pdb_file} ve {xtc_file} dosyalarından yörünge yükleniyor")
        u = mda.Universe(pdb_file, xtc_file)
        logger.info("Yörünge başarıyla yüklendi")
        return u
    except Exception as e:
        logger.error(f"Yörünge yüklenirken hata oluştu: {e}")
        sys.exit(1)

def align_trajectory(universe, selection):
    try:
        logger.info("Yörünge hizalanıyor")
        aligner = align.AlignTraj(universe, universe, select=selection, in_memory=True).run()
        logger.info("Yörünge başarıyla hizalandı")
    except Exception as e:
        logger.error(f"Yörünge hizalanırken hata oluştu: {e}")
        sys.exit(1)

def extract_features(universe, selection):
    try:
        logger.info("Özellikler çıkarılıyor")
        selected_atoms = universe.select_atoms(selection)
        distances = []
        for ts in universe.trajectory:
            positions = selected_atoms.positions.copy()
            distance = np.linalg.norm(positions - positions[0], axis=1)
            distances.append(distance)
        features = np.array(distances)
        logger.info("Özellikler başarıyla çıkarıldı")
        return features
    except Exception as e:
        logger.error(f"Özellikler çıkarılırken hata oluştu: {e}")
        sys.exit(1)

def cluster_features(features, n_clusters=100):
    try:
        logger.info(f"Özellikler Agglomerative Clustering kullanılarak {n_clusters} kümeye ayrılıyor")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        dtrajs = clustering.fit_predict(features)
        logger.info("Kümeleme başarıyla tamamlandı")
        return dtrajs
    except Exception as e:
        logger.error(f"Özellikler kümeleme sırasında hata oluştu: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()

    # Dosya sayısının çift olup olmadığını kontrol edin
    if len(args.files) % 2 != 0:
        logger.error("PDB ve XTC çiftleri olarak çift sayıda dosya sağlamalısınız.")
        sys.exit(1)

    file_pairs = list(zip(args.files[::2], args.files[1::2]))

    # Simülasyon zamanı ve frame sayısını isteyin
    try:
        simulation_time = float(input("Lütfen simülasyon zamanını girin (örneğin, nanosecond cinsinden): "))
        number_of_frames = int(input("Lütfen frame sayısını girin: "))
    except ValueError:
        logger.error("Simülasyon zamanı veya frame sayısı için geçersiz giriş.")
        sys.exit(1)

    time_per_frame = simulation_time / number_of_frames
    logger.info(f"Frame başına zaman: {time_per_frame} ns")

    all_dtrajs = []
    labels = []
    for idx, (pdb_file, xtc_file) in enumerate(file_pairs):
        label = input(f"Simülasyon {idx+1} için bir etiket girin: ")
        labels.append(label)
        logger.info(f"Simülasyon {idx+1} işleniyor: {pdb_file}, {xtc_file}")
        # Yörüngeyi yükle
        universe = load_trajectory(pdb_file, xtc_file)

        # Yörüngeyi hizala
        align_trajectory(universe, args.selection)

        # Özellikleri çıkar
        features = extract_features(universe, args.selection)

        # Özellikleri kümele
        dtrajs = cluster_features(features, args.n_clusters)

        all_dtrajs.append(dtrajs)

    # Lag zamanlarını tanımla
    max_lag = int(number_of_frames / 5)
    if max_lag < 1:
        max_lag = 1
    lag_times = np.arange(1, max_lag, max(1, int(max_lag / 50)))
    logger.info(f"Lag zamanları: {lag_times}")

    # Her simülasyon için implied timescales hesapla
    implied_timescales_list = []
    for idx, dtrajs in enumerate(all_dtrajs):
        logger.info(f"Simülasyon {idx+1} için implied timescales hesaplanıyor")
        its = msm.its([dtrajs], lags=lag_times, nits=1)
        implied_timescales_list.append(its)

    # Implied timescales grafiğini Matplotlib kullanarak çiz
    plt.figure(figsize=(10, 6))
    for idx, its in enumerate(implied_timescales_list):
        lag_times_in_ns = its.lagtimes * time_per_frame
        timescales_in_ns = its.timescales[:, 0] * time_per_frame  # Sadece ilk implied timescale kullanılır
        plt.plot(lag_times_in_ns, timescales_in_ns, label=labels[idx], linewidth=2)
    plt.xlabel('Lag Zamanı (ns)')
    plt.ylabel('Implied Timescale (ns)')
    plt.title('Implied Timescales')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

