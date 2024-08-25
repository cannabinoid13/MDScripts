import sys
import argparse
import MDAnalysis as mda
import pyemma
import numpy as np
import logging
from MDAnalysis.analysis import align
from sklearn.cluster import AgglomerativeClustering

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Markov State Model (MSM) construction and analysis using PyEMMA and MDAnalysis.")
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file')
    parser.add_argument('xtc_file', type=str, help='Path to the XTC file')
    parser.add_argument('--n_clusters', type=int, default=100, help='Number of clusters for Agglomerative Clustering')
    parser.add_argument('--lag_time', type=int, default=100, help='Lag time for MSM')
    parser.add_argument('--selection', type=str, default='name CA', help='Atom selection for alignment and feature extraction')
    return parser.parse_args()

def load_trajectory(pdb_file, xtc_file):
    try:
        logger.info(f"Loading trajectory from {pdb_file} and {xtc_file}")
        u = mda.Universe(pdb_file, xtc_file)
        logger.info("Trajectory loaded successfully")
        return u
    except Exception as e:
        logger.error(f"Error loading trajectory: {e}")
        sys.exit(1)

def align_trajectory(universe, selection):
    try:
        logger.info("Aligning trajectory")
        reference = universe.select_atoms(selection)
        aligner = align.AlignTraj(universe, reference, select=selection, in_memory=True).run()
        logger.info("Trajectory aligned successfully")
    except Exception as e:
        logger.error(f"Error aligning trajectory: {e}")
        sys.exit(1)

def extract_features(universe, selection):
    try:
        logger.info("Extracting features")
        selected_atoms = universe.select_atoms(selection)
        distances = np.array([np.linalg.norm(selected_atoms.positions - selected_atoms.positions[0], axis=1) for ts in universe.trajectory])
        logger.info("Features extracted successfully")
        return distances
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        sys.exit(1)

def cluster_features(features, n_clusters=100):
    try:
        logger.info(f"Clustering features into {n_clusters} clusters using Agglomerative Clustering")
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        dtrajs = clustering.fit_predict(features)
        logger.info("Clustering completed successfully")
        return dtrajs
    except Exception as e:
        logger.error(f"Error clustering features: {e}")
        sys.exit(1)

def build_msm(dtrajs, lag_time=100):
    try:
        logger.info(f"Building MSM with lag time {lag_time}")
        msm = pyemma.msm.estimate_markov_model(dtrajs, lag=lag_time)
        logger.info("MSM built successfully")
        return msm
    except Exception as e:
        logger.error(f"Error building MSM: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()

    # Load the trajectory
    universe = load_trajectory(args.pdb_file, args.xtc_file)

    # Align the trajectory
    align_trajectory(universe, args.selection)

    # Extract features
    features = extract_features(universe, args.selection)

    # Cluster the features
    dtrajs = cluster_features(features, args.n_clusters)

    # Build the MSM
    msm = build_msm(dtrajs, args.lag_time)

    # MSM analysis
    logger.info(f"Number of states: {msm.nstates}")
    logger.info(f"Active set fraction: {msm.active_state_fraction}")
    logger.info(f"Implied timescales: {msm.timescales()}")

    # Save the MSM object
    try:
        msm.save('msm_model.pkl', overwrite=True)
        logger.info("MSM model saved to msm_model.pkl")
    except Exception as e:
        logger.error(f"Error saving MSM model: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

