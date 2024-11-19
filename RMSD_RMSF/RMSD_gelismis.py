import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import logging

# Hata analizi ve loglama için logging modülü ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if len(sys.argv) < 2:
    logging.error("Usage: python <script.py> <file1.pdb> <file1.xtc> <label1> <file2.pdb> <file2.xtc> <label2> ...")
    sys.exit(1)

# Dosya çiftlerini belirleyin
file_pairs = {}
labels = {}

try:
    for i in range(1, len(sys.argv), 3):
        pdb_file = sys.argv[i]
        xtc_file = sys.argv[i + 1]
        label = sys.argv[i + 2]

        pdb_filename = os.path.basename(pdb_file)
        xtc_filename = os.path.basename(xtc_file)
        base_name_pdb, ext_pdb = os.path.splitext(pdb_filename)
        base_name_xtc, ext_xtc = os.path.splitext(xtc_filename)

        if base_name_pdb != base_name_xtc:
            logging.error(f"Base names do not match for {pdb_file} and {xtc_file}")
            sys.exit(1)

        file_pairs[base_name_pdb] = {'.pdb': pdb_file, '.xtc': xtc_file}
        labels[base_name_pdb] = label
except IndexError as e:
    logging.error("IndexError: Make sure you provided all necessary files and labels.")
    sys.exit(1)

# Ana grafik renkleri
colors = sns.color_palette("tab10", n_colors=len(file_pairs))

# Grafik oluşturma
fig, ax = plt.subplots(figsize=(20, 16))
inset_axes = []

# Her dosya çifti için işlemleri yapın
for i, (base_name, files) in enumerate(file_pairs.items()):
    if '.pdb' in files and '.xtc' in files:
        pdb_file = files['.pdb']
        xtc_file = files['.xtc']

        try:
            # MDAnalysis universe oluşturma
            u = mda.Universe(pdb_file, xtc_file)
        except Exception as e:
            logging.error(f"Error creating MDAnalysis universe for {base_name}: {e}")
            continue

        try:
            # RMSD analizi
            R = rms.RMSD(u, u, select="backbone")  # Backbone atomlarını kullanarak RMSD hesapla
            R.run()
        except Exception as e:
            logging.error(f"Error during RMSD calculation for {base_name}: {e}")
            continue

        try:
            # RMSD verilerini pandas DataFrame'e çevir
            rmsd_df = pd.DataFrame(R.results.rmsd, columns=["Frame", "Time (ps)", "RMSD"])
            rmsd_df["Time (ns)"] = rmsd_df["Time (ps)"] / 1000  # Zamanı nanosaniye cinsine çevir
        except Exception as e:
            logging.error(f"Error converting RMSD results to DataFrame for {base_name}: {e}")
            continue

        # Ana grafiği oluşturun
        label = labels[base_name] if base_name in labels else base_name
        ax.plot(rmsd_df["Time (ns)"], rmsd_df["RMSD"], label=f"RMSD - {label}", color=colors[i], linewidth=2)

        # Histogram grafiklerini oluşturun
        num_insets = len(file_pairs)
        inset_width = 0.22 / num_insets * 3 if num_insets > 3 else 0.22
        inset_height = 0.3 / num_insets * 3 if num_insets > 3 else 0.3
        inset_spacing = 0.04  # Aralıkları artırmak için bu değeri ekliyoruz

        # Histogram grafikleri için pozisyon ayarlaması
        inset_ax = fig.add_axes(
            [0.15 + i * (inset_width + inset_spacing) + 0.02, 0.1 + 0.05, inset_width, inset_height])
        sns.histplot(rmsd_df["RMSD"], color=colors[i], kde=True, ax=inset_ax)
        inset_ax.set_xlabel('RMSD Value (Å)', fontsize=12)
        inset_ax.set_ylabel('Frequency', fontsize=12)
        inset_ax.set_title(f'{label}', fontsize=14)
        inset_axes.append(inset_ax)

# Ana grafiği etiketleyin
ax.set_xlabel('Time (ns)', fontsize=20)
ax.set_ylabel('RMSD (Å)', fontsize=20)
ax.set_title('RMSD Plot', fontsize=20)
ax.legend(fontsize=16, loc='upper right', title_fontsize='16')

ax.tick_params(axis='both', which='major', labelsize=16)

# Alt grafikleri hareket ettirin
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Grafiği kaydet
fig.savefig('rmsd_plot_all.png')
plt.close(fig)

logging.info("RMSD analysis and plotting completed successfully.")
