MDScripts
----------

The MDScripts repository is a comprehensive collection of scripts developed for the analysis of molecular dynamics (MD) simulations. These scripts utilize the **MDAnalysis** library to extract data from trajectories and perform various analyses. The analyses include **RMSD** (Root-Mean-Square Deviation) and **RMSF** (Root-Mean-Square Fluctuation) calculations, as well as principal component analysis (PCA) and time-independent component analysis (TICA) combined with free energy surface extraction, UMAP, diffusion maps, and other dimension reduction techniques, along with dynamic cross-correlation matrix (DCCM) calculations and the construction of Markov state models (MSM). Additionally, the integration and visualization of MM/GBSA(Molecular Mechanics/Generalized Born Surface Area) binding energy results into the analysis is also possible. This repository enables comprehensive and flexible MD analysis by executing the scripts via the terminal. Each script is designed to automate a specific analysis and its visualization.

Citations
-----
Please [cite us](https://scholar.google.com/citations?user=OvpMySIAAAAJ&hl=tr) if you use this software in your research:

**Articles published in peer-reviewed journals that have used this software** 

Ortaakarsu, A. B., Boğa, Ö. B., & Kurbanoğlu, E. B. (2025). Cardaria draba subspecies Shalepensis exerts in vitro and in silico inhibition of α-glucosidase, TRP1, and DLD-1 proliferation. Scientific Reports, 15(1), 10402. [DOI link](https://doi.org/10.1038/s41598-025-95538-1)

Hari, A., Lahlali, R., Ortaakarsu, A. B., Taarji, N., Laasli, S. E., Karaaoui, K., ... & Echchgadda, G. (2024). Inhibitory effects of wild Origanum elongatum extracts on Fusarium oxysporum mycelium growth and spores germination: Evidence from in-vitro, in-planta, and in-silico experiments. Journal of Natural Pesticide Research, 10, 100096. [DOI link](https://doi.org/10.1016/j.napere.2024.100096)


Installation
------
To use the MDScripts, you must first install the required **Python packages**. Ideally, in a Python 3 environment, install the following **dependencies** (for example, using the `pip install` command):

```
pip install MDAnalysis pandas matplotlib seaborn scipy statsmodels scikit-learn umap-learn plotly bokeh deeptime pyemma cupy
```
The above command will install all the basic libraries required by the scripts in this repository (see the Dependencies section below). Then, clone the GitHub repository or download the script files into your working directory. Each script can run independently; to perform a specific analysis, simply execute the corresponding Python file from the terminal.

For example, after installing all dependencies, you can run an advanced RMSD analysis script in the repository directory as follows:

```
python RMSD_RMSF/RMSD_gelismis.py structure.pdb trajectory.xtc Label
```

(For more detailed information regarding this command, please refer to the **Usage** and **Script Descriptions** sections.)

Usage
-----
Each script is designed to be executed via command-line arguments from the terminal. The usage depends on the input files and parameters required by the specific analysis. In general, when a script is executed with the `-h` or `--help parameter`, it will display its usage information (using argparse). For example, running `python PCA.py -h` will list the accepted arguments and their default values for that script.

When running the analysis scripts, you typically follow these steps:

- **Prepare the input files:** Ensure that the trajectory file (e.g., .xtc, .dcd), structure file (e.g., .pdb or .gro), and, if necessary, pre-calculated data files (such as CSV energies, .dat RMSD/RMSF data) are prepared in the correct format.

- **Execute the script:** Run the command `python <script_name>.py <arguments>` from the terminal. The required arguments vary for each script (detailed explanations for each script are provided below).

- **Review the analysis output:** The scripts generally generate **graphical outputs**. Some scripts display the graphs interactively (via plt.show() or in a browser), while others **save them to a file** (e.g., PNG file or HTML report). Additionally, many scripts print important numerical results (statistical tables, free energy values, etc.) to the terminal.

The following sections provide detailed explanations for each script regarding its specific usage, the expected input formats, and the interpretation of the outputs. These descriptions will serve as a guide when applying the scripts to your own data.

When running the analysis scripts, you typically follow these steps:

Prepare the input files: Ensure that the trajectory file (e.g., .xtc, .dcd), structure file (e.g., .pdb or .gro), and, if necessary, pre-calculated data files (such as CSV energies, .dat RMSD/RMSF data) are prepared in the correct format.

Execute the script: Run the command python <script_name>.py <arguments> from the terminal. The required arguments vary for each script (detailed explanations for each script are provided below).

Review the analysis output: The scripts generally generate graphical outputs. Some scripts display the graphs interactively (via plt.show() or in a browser), while others save them to a file (e.g., PNG file or HTML report). Additionally, many scripts print important numerical results (statistical tables, free energy values, etc.) to the terminal.

The following sections provide detailed explanations for each script regarding its specific usage, the expected input formats, and the interpretation of the outputs. These descriptions will serve as a guide when applying the scripts to your own data.

Dependencies
-----

For the MDScripts analyses to run smoothly, the following Python libraries are required:

- **MDAnalysis** – For reading trajectories and performing atom selection (to load and process MD simulation data)

- **NumPy** – For numerical computations and array operations

- **Pandas** – For reading and processing .csv and .dat files

- **Matplotlib** – For plotting graphs (2D plots, subplots, etc.)

- **Seaborn** – For graphical styling and advanced plots (Seaborn-themed plots)

- **SciPy** – For statistical computations (especially scipy.stats and scipy.signal in RMSD analysis) and distance calculations (pdist, squareform in diffusion maps)

- **Statsmodels** – For computing time-series autocorrelation (using statsmodels.tsa.stattools.acf in RMSD analysis)

- **Scikit-Learn (sklearn)** – For machine learning algorithms (Clustering: KMeans, OPTICS; dimension reduction: SpectralEmbedding; scaling: StandardScaler; etc.)

- **UMAP (umap-learn)** – For the UMAP dimension reduction algorithm

- **Plotly** – For interactive visualizations (e.g., interactive heat maps for DCCM)

- **Bokeh** – For interactive visualizations and HTML reports (especially for UMAP and OPTICS diffusion map graphs)

- **Deeptime** – For TICA analysis (time-lagged ICA computations)

- **PyEMMA** – For constructing and analyzing Markov state models (used in the MSM scripts)

- **Cupy (optional)** – To accelerate some operations with GPU (if not available, the scripts will run on the CPU)

Ensure that all of the above dependencies are installed either individually or via a requirements file. Otherwise, you may encounter an `ImportError` when executing a script.

Script Descriptions
--------

Below are detailed descriptions for each script or group of scripts in the repository. For each script, the type of **analysis performed**, the expected **input files**, the **accepted parameters**, and how to interpret the **output** are specified.

RMSD and RMSF Analysis Scripts
----

These scripts in this directory calculate and visualize the deviations and fluctuations in protein structures throughout the simulation.

- **RMSD_gelismis.py:** *Advanced RMSD calculation and multi-plot generation*.
  
**Inputs:** A pair consisting of a structure file (.pdb) and a trajectory file (.xtc) for each system, along with a label (provided as three consecutive command-line arguments; for example: `python RMSD_gelismis.py prot1.pdb prot1.xtc A prot2.pdb prot2.xtc B`).
  
**Operation:** Each trajectory is aligned to its first frame, and the RMSD values for the backbone atoms are calculated. The RMSD curves for all systems are plotted in a single graph against time. Additionally, an inset histogram (with a KDE curve) is generated for the RMSD distribution of each system.
Output: An image file named rmsd_plot_all.png is automatically saved. This image displays the RMSD vs. time curves in the main panel with a horizontal dashed line representing the average RMSD and a ±1 standard deviation band. In the inset panels, the RMSD distribution (histogram + kernel density) is shown. Moreover, the script prints statistics such as the average RMSD, standard deviation, and the 95% confidence interval for each system to the console. This script is ideal for comparing RMSD results from multiple simulations simultaneously.

- **RMSD_dat.py:** *Plotting from pre-calculated RMSD data*.

**Inputs:** At least one `.dat` file containing RMSD data along with a label for each file, plus the total simulation time (in ns). For example: `python RMSD_dat.py 100 sim1_rmsd.dat Simulation1 sim2_rmsd.dat Simulation2`. The .dat files should contain space-separated columnar data; the time column should be labeled `Time` or `Time (ns)`, and the RMSD column should be named something like `Prot_CA` (typically representing the Cα atom RMSD value).
  
**Operation:** Each file is read, and if a time axis is present, it is used; otherwise, a time axis is automatically generated based on the total duration. For each RMSD curve, the mean and standard deviation are computed, the 95% confidence interval is determined, and these are added to the main plot.
  
**Output:** The script displays an RMSD vs. time graph in a window. Each dataset is represented by a line, with the mean RMSD shown as a dashed line and the ±1 standard deviation band indicated by a colored area. In the lower part, a density graph (KDE) showing the RMSD distribution is plotted, with vertical lines marking the means. The legend includes each simulation’s label and its average value. Additionally, a table (titled Statistical Summary) with the average, standard deviation, and confidence interval for each dataset is printed to the console. This script is useful for quickly visualizing RMSD data that has been pre-calculated (for example, from GROMACS or other tools).

- **RMSD_dat_korelasyon.py:** *Autocorrelation and spectral analysis for RMSD time series*.
  
**Inputs:** The inputs are identical to those of `RMSD_dat.py` (total simulation time, one or more RMSD `.dat` files, and their labels).
  
**Operation:** For each RMSD series, the data is first read and a time axis is constructed. Then, if there is a linear trend in the signal, the RMSD values are detrended. The **autocorrelation function** for each series is computed (using FFT acceleration) and obtained along with 95% confidence intervals. The correlation time is calculated from the autocorrelation integral of each series. Additionally, the **Power Spectral Density (PSD)** of each RMSD series is computed using Welch’s method (to analyze the frequency components of the RMSD signal).
  
**Output:** The script displays a plot with two panels side by side. The left panel plots the RMSD autocorrelation functions versus time lag, with confidence intervals represented as semi-transparent bands. The right panel displays the PSD for each RMSD series (power versus frequency). The plot includes titles, axis labels, and a legend comparing multiple series. Additionally, a table with the average RMSD, standard deviation, SEM, and correlation time for each dataset is printed to the console. This script is used to analyze the time scales (i.e., how long the signal remains correlated) and potential periodic behavior (via the PSD) of the RMSD signal.

- **RMSD_duz_dat.py**: *Overlaying multiple RMSD datasets in a single graph*.
  
**Inputs:** The total simulation time (in ns), one or more `.dat` RMSD files, and their labels. Optionally, the parameter `--output <filename.png>` can be provided to specify the output file name (default: rmsd_plot_all.png).
  
**Operation:** Each file is read, and a time axis is generated (using the `Time (ns)` column if available). For each curve, the mean RMSD is calculated, and the curves are sorted in ascending order based on their mean RMSD values. Then, all RMSD curves are plotted in a single graph with thin, differently colored lines; the legend displays the label and the mean value (in parentheses) for each curve. The graph’s axes are appropriately labeled and formatted.
Output: The plot is saved as a PNG file with the specified filename, and the script prints a success message to the terminal. This script is useful for comparing multiple simulations at a glance, especially when you wish to overlay RMSD curves rather than display separate histograms.

- **RMSF.py:** *Alpha carbon RMSF (fluctuation) plot*.
  
**Inputs:** One or more RMSF data files (typically containing columns for residue number and CA RMSF value) and a label for each file. Optionally, a final argument with specific residue indices can be provided; these residues will be highlighted in the graph. For example: `python RMSF.py system1_rmsf.dat Sim1 system2_rmsf.dat Sim2 50,100` (highlighting residues 50 and 100).
  
**Operation:** Each file is read, and it is expected that a column named `CA` (representing the Cα atom RMSF value for each residue) is present. When multiple datasets are provided, if there are differences in length, they are trimmed or aligned using the dataset with the fewest residues as the reference, ensuring the starting residue numbers are aligned. All RMSF curves are then plotted in a single graph in different colors. If the `residue_indices` argument is provided, vertical lines are added at those indices to highlight whether the corresponding residues have especially high RMSF values.
  
**Output:** An **RMSF vs. Residue Number** graph is displayed, titled “Alpha Carbon (CA) RMSF Analysis”, with the x-axis representing the aligned residue indices and the y-axis representing the RMSF values (Å). The legend, listing each dataset’s label, appears in the upper right corner (arranged in two columns if necessary). The graph includes a grid and appropriate labeling; if specific residues are selected, they are marked with red dashed lines. This script is useful for comparing structural fluctuations under different conditions or for comparing RMSF with experimental B-factors.

  Markov State Model (MSM) Scripts
  ----
These scripts are used to construct and analyze **Markov State Models (MSM)** from MD simulation data. MSMs discretize the system’s configuration space into states and compute the transition probabilities between them, enabling the analysis of long-timescale behaviors.

- **markov.py:** *Construction of an MSM from a single simulation*.
  
**Inputs:** A structure file (usually .pdb) and a trajectory file (xtc/dcd) along with necessary parameters. The script expects detailed arguments via argparse (e.g., backbone atom selection, desired number of clusters, MSM lag time, etc.—the full list can be viewed with `-h`).
  
**Operation:** The trajectory is read using MDAnalysis, and coordinates for the selected atoms (typically the entire protein) are extracted. Then, using the PyEMMA library, clustering (usually via k-means) is performed on the coordinate space (often after PCA or TICA transformation). After converting each frame into a state sequence (discrete trajectory, dtraj), the MSM is computed using PyEMMA’s `msm.estimate_markov_model` function at the desired lag time.
  
**Output:** The computed MSM model (an object containing the transition matrix, state populations, etc.) is saved to a file (typically in a pickled .pyemma model format). The script prints key outputs (such as the number of metastable states and the slowest timescales) to the console and saves the model for further analysis.

- **markov_multi.py:** *Construction of a combined MSM from multiple simulations*.
  
**Inputs:** A text file listing multiple PDB+trajectory paths or multiple inputs provided via the command line (the script may also be designed to read all trajectory files in a directory).
  
**Operation:** Essentially similar to markov.py, but it utilizes PyEMMA’s support for multiple trajectories by combining several discrete trajectories into the estimate_markov_model function. In this way, similar states sampled in different simulations are aggregated under a single MSM.
  
**Output:** A single MSM model (with combined data) is saved. This model increases the data count, thereby yielding more reliable transition statistics.

- **analiz.py:** Analysis of a saved MSM model.
  
**Inputs:** The path to the MSM model file, the total simulation time (in ns), and the total number of frames (information required to scale the time axis).
  
**Operation:** The MSM model file is loaded using PyEMMA. Then, the **implied timescales (ITS)** are computed from the MSM’s eigenvalues and eigenvectors (representing the slowest dynamic processes). Additionally, the free energy of the equilibrium state can be computed from the model (e.g., using the formula $ \Delta G = -RT \ln(\pi_i)$ for each metastable state). The script likely visualizes this information: one plot for the slowest timescales (for instance, plotting the second and third eigenvalue corresponding timescales) and another plot for the free energy profile of the metastable states (with adjustable color parameters via `--timescale_color` and `--energy_color`).
  
**Output:** The script likely opens a window displaying two overlaid plots: one comparing the primary timescales of the MSM (e.g., tens of nanoseconds vs. nanoseconds) and another showing the free energy levels of the metastable states. In addition, these timescales are printed to the terminal along with state populations or energy information. This script assists in understanding the slow processes and energy differences between states in the MSM.

(Note: Since the Markov scripts utilize PyEMMA, ensure that the `pyemma` library is installed. The models are saved as .msm or .h5 files and are directly loaded by the analysis script.)

Dynamic Cross-Correlation Matrix (DCCM)
---
- **DCCM_dosya.py:** *Calculation of correlations between atomic motions within a protein.*
  
**Inputs:** A topology file (e.g., a protein PDB) and a trajectory file (e.g., XTC), along with the desired output filename for the matrix. Usage: `python DCCM_dosya.py protein.pdb traj.xtc dccm_output.txt`.
  
**Operation:** The trajectory and structure are loaded using MDAnalysis. The script selects all backbone atoms, and the coordinate information is obtained for each frame. The 3D motion vectors of the Cα atom of each residue are extracted, and the Pearson correlation coefficients between these vector series over the entire simulation are computed. The result is an NxN correlation matrix (where N is the number of residues), with each element `i,j` representing the correlation (ranging from -1 to +1) between the motions of residues i and j. The script saves this matrix to the specified output file using `numpy.savetxt`. Additionally, for visualization, an **interactive heat map** is generated using the Plotly library. This heat map displays the correlation matrix with a red-blue color scale (ranging from -1 to 1).
  
**Output:** The script prints progress messages to the terminal (e.g., file loading, DCCM calculation, etc.). The numeric DCCM matrix is saved in the specified output file (as text, with each row representing a residue). Furthermore, when executed, an interactive graph titled Dynamic Cross-Correlation Map (DCCM) is displayed in a browser window. In this graph, the x and y axes represent residue indices; the cell colors indicate the correlation values, and hovering over a cell shows its numerical value (using Plotly interactivity). This allows, for example, easy identification of regions within the protein that move together (high positive correlation) or in opposite directions (negative correlation).

MM-GBSA Energy Analysis and Plotting
--
**MM-GBSA.py:** *Time series analysis of binding free energy results from MM/GBSA.*

**Inputs:** A CSV file containing the energy data and simulation time parameters. Required arguments: `-t <total_time_ns>` (total simulation time in ns), `-f <total_number_of_frames>` (the total number of structures for which energy was calculated), and `file_path` (the path to the CSV file containing energy values). Optional arguments include: `-s <frame_skip>` (the step size if the energy is calculated every few frames; defaults to 1 if not provided) and `--save <output.png>` (the filename for saving the plot; if not provided, the plot is displayed on screen).

**Operation:** The CSV file is read using pandas. The expected column name is `'r_psp_MMGBSA_dG_Bind'`, which contains the binding free energy for each frame (in the Schrödinger Prime MMGBSA output format). Any NaN values in this column are discarded, and the remaining values are converted into a numpy array. Basic statistics are then computed: mean, median, minimum, maximum, range (max-min), standard deviation (sample std), and standard error (SEM). The time axis is generated based on the `total_sim_time` and `total_frames` information: if `frame_skip` is provided, the time step is scaled accordingly. For example, if the total time is 100 ns, 1000 frames, and frame_skip=10, then the time interval between energy points is 0.1 ns * 10 = 1 ns. Then, using Matplotlib, the energy values are plotted against time; each point is marked (using marker='o') and connected by lines. A horizontal dashed line representing the average energy level (in red, with an annotated numerical value) is added. Additionally, a semi-transparent red band representing ±1 standard deviation is added to the plot. The axes are labeled “Simulation Time (ns)” and “MMGBSA ΔG Bind (kcal/mol)”, and a title is provided. An annotation box is added in the upper left corner of the plot, containing a statistical summary (Mean, Std Dev, Median, Min, Max, Range, SEM).

**Output:** If the `--save` parameter is provided, the plot is saved as a high-resolution (600 DPI) file with the specified name, and a “Plot saved to: ...” message is printed to the terminal. Otherwise, the plot is displayed interactively using `plt.show()`. The plot can be interpreted as follows: the fluctuations of the energy values over time are displayed, with the average clearly indicated by the dashed line. If the system has reached equilibrium, the energy curve will fluctuate around a specific average value. The red band indicates the typical range of fluctuations. This script allows, for example, monitoring the trend of a ligand’s binding free energy throughout the simulation or comparing energy differences across replicates.

Principal Component Analysis (PCA)
----
- *PCA.py:* *PCA and free energy surface visualization for protein conformations.*
  
**Inputs:** A structure file (PDB) and a trajectory file (XTC) are required arguments. Optionally, `--select "<atom selection>"` (default is `"name CA"`, meaning only Cα atoms are selected) can be used to specify the atoms for PCA, and `--bins <number>` (default 25) to set the number of bins for the 2D histogram when calculating the free energy surface.
  
**Operation:** The script loads the trajectory using MDAnalysis, selects the desired atoms (e.g., all Cα atoms), and treats each frame as a data point. The entire trajectory is aligned to the first frame (removing translation/rotation). Then, PCA is performed using MDAnalysis’s PCA implementation: `MDAnalysis.analysis.pca.PCA` is used to compute the first three principal components. The PCA-transformed coordinates (the `transformed` matrix) for each frame are obtained. A 2D histogram is generated on the first two PCA components (PC1 vs. PC2) using `np.histogram2d`, and the free energy for each bin is calculated as $-k_B T \ln P$ (with k_B T set to 1 for relative comparisons). A meshgrid is created for the resulting free energy matrix, and a *3D surface plot* is drawn using Matplotlib’s 3D plotting tools. The z-axis is labeled “Free Energy”, and the x and y axes are labeled “Principal Component 1” and “Principal Component 2”, respectively. A viridis colormap is used, and a color bar is added. Additionally, an inset 2D contour map is included (displaying the free energy isoclines on the PC1-PC2 plane).

**Output:** When executed, the script opens an interactive 3D plot window. The user can interact with the graph (rotate, zoom). The peaks and valleys of the 3D surface indicate the high and low free energy regions in the PCA space; for example, if two distinct wells are present, this may indicate the existence of two metastable states. The 2D contour inset provides a top-down view of the same surface. The script may also output PCA variance information (e.g., the percentage of variance explained by the first three components) to the terminal.

Time-Independent Component Analysis (TICA) and Free Energy
-----
- **TICA_FES.py:** *Identifying slow modes using TICA and generating 2D free energy surfaces*.
  
**Inputs:** A structure file (PDB), a trajectory file (XTC), and optionally `--select "<atom selection>"` (default `"name CA"`), `--lag <lag>` (default 10, in frames, for TICA lag time), and `--bins <number>` (default 25 for the number of bins in the free energy histogram). Additionally, if `--output_prefix <prefix>` is provided, the graphs are saved to files with the given prefix; otherwise, the graphs are displayed interactively.
  
**Operation:** First, the trajectory is loaded using MDAnalysis, and the desired atoms are selected and aligned to the first frame (similar to the PCA script). Then, the TICA module from the `deeptime` library is used: the coordinates of the selected atoms for each frame are vectorized, and `TICA(lagtime=lag, dim=3)` computes a three-dimensional TICA space. The `fit_transform` method of the TICA model is then used to obtain the coordinates of three independent components for each frame. Three pairs of components are considered: (IC1 vs IC2), (IC1 vs IC3), and (IC2 vs IC3). For each pair, a 2D histogram is computed to generate a free energy matrix (using the –ln P method). Using Matplotlib, a separate 2D contour map is drawn for each pair: the x and y axes correspond to the respective TICA components, and a continuous colormap (using cividis) displays the free energy in kT units (relative, not absolute). The title of each plot is “2D Free Energy Surface from TICA Components i and j” (with appropriate component numbers).
  
**Output:** If an `--output_prefix` is provided, separate PNG files are generated for each pair of components (for example, `prefix_TICA1_vs_TICA2.png, prefix_TICA1_vs_TICA3.png, prefix_TICA2_vs_TICA3.png`). If no output_prefix is given, the script displays each contour map sequentially on the screen. Terminal messages indicate the completion of the TICA process and provide details for each generated graph. This script enables the visualization of the combinations of slow modes in protein dynamics through free energy landscapes; for example, collective motions related to ligand binding/dissociation may be revealed as energy barriers in the TICA space.

UMAP (Unified Manifold Approximation and Projection)
----
- **UMAP.py:** Dimension reduction of high-dimensional conformational space using UMAP and visualization of the free energy map.
  
**Inputs:** A structure file (PDB), a trajectory file (XTC), and optionally `-s "<atom_selection>"` (default is `"protein"`, meaning all protein atoms are used) to specify the atoms for the analysis. Additionally, `-o <output.html>` (default `"umap_free_energy.html"`) can be provided to specify the output file name for the interactive result.
  
**Operation:** The structure and trajectory are loaded using MDAnalysis, and all frames are aligned to the first frame using `align.AlignTraj` (so that only internal motion differences are compared). For each frame, the positions of the selected atoms are flattened into a feature vector (with dimensions equal to the number of selected atoms × 3 coordinates). These vectors are scaled using StandardScaler (to have mean 0 and variance 1). Then, the **UMAP algorithm** is applied: `umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)` is used to compute a 2D embedding. UMAP reduces each frame (and thus each conformation) to a 2D point. Subsequently, a **density estimation** is performed on these 2D points using `scipy.stats.gaussian_kde`. The density values for each point are computed; high density indicates a free energy well, and low density indicates higher free energy. Therefore, the density values (or equivalently, $- \ln(\text{density})$) are used to assign colors. The script then creates an interactive plot using the Bokeh library: a 1000x1000 scatter plot is prepared. Each frame is plotted as a point; the color is determined by the point’s free energy level based on a color mapper (using `LinearColorMapper`), and a color bar is added. Interactive tools such as pan, zoom, and hover are enabled. The resulting graph is saved as an HTML file with the specified name (default *umap_free_energy.html*), which can be manually opened in a browser.

**Output:** When the HTML file is opened in a browser, it displays an interactive graph: the x and y axes correspond to UMAP component 1 and 2, and each point represents a simulation frame. A hover tool may display the frame index or time (if implemented, as the code includes a HoverTool). In the background, the script may also compute and overlay a contour fill to better illustrate the free energy surface (using KDE results on a 2D grid via a `p.contour(...)` call). This graph enables the identification of clusters within the sampled conformations; for example, distinct clusters in the UMAP space indicate different conformational substates. The color scale provides the free energy topography of the UMAP plane (with warm colors indicating high energy sparse regions and cool colors indicating low energy dense regions). This script is a powerful tool for capturing complex separations that linear methods like PCA/TICA might not reveal, and it allows for interactive exploration of the results.

Diffusion Map Analysis Scripts
------

Diffusion maps apply a graph-based dimension reduction method to capture important collective modes of the system in a non-linear manner. This directory contains scripts for the analysis and visualization of MD data using diffusion maps. These analyses may also integrate MM/GBSA energy data and alternative clustering methods.

- **difuzyon_map_2D.py:** *2D embedding using diffusion maps, K-means clustering, and MSM analysis*.
  
**Inputs:** A structure file, a trajectory file, and an atom selection expression (e.g., `"protein and name CA"`). Optional arguments include `n_components` (number of components in the diffusion map, default is 3), `epsilon` (the scaling parameter for the RBF kernel; if not provided, it is calculated automatically), `n_clusters` (number of clusters for K-means, default is 5), and lag_time (MSM lag time, default 1 frame). Usage example: `python difuzyon_map_2D.py prot.pdb traj.xtc "name CA" 3 None 5 5`.

**Operation:** First, the structure and trajectory are loaded using MDAnalysis, and a feature matrix is generated from the coordinates of the selected atoms (resulting in a matrix of dimensions: frames × 3N). These features are scaled using StandardScaler. If `epsilon` is not provided, the script calculates a local epsilon value for each frame: using sklearn’s BallTree, it finds the distance to the 5th nearest neighbor for each point and reports the average as `mean_epsilon`. (This adaptive scale selection is used to automatically determine the kernel width for the diffusion map.) Next, the pairwise Euclidean distance matrix is computed (using pdist and squareform). An RBF kernel matrix is constructed as $K_{ij} = \exp(-d_{ij}^2 / (\epsilon_i \epsilon_j))$, where $\epsilon_i$ and $\epsilon_j$ are the local epsilon values for each point, or a constant epsilon if specified. This kernel matrix is normalized row-wise and symmetrized to obtain the diffusion operator ($D^{-1/2} K D^{-1/2}$). Then, using sklearn’s SpectralEmbedding(affinity='precomputed'), the top eigenvectors corresponding to the desired number of components are extracted. The result, referred to as `diffusion_map`, contains the diffusion coordinates for each frame (e.g., the first two components, diffusion coordinate 1 and 2). Next, K-means clustering is performed in the diffusion space, assigning each frame to one of `n_clusters` clusters. Using the obtained state sequence (cluster assignments) and `lag_time`, an MSM is computed using PyEMMA (transition probability matrix). From this MSM, the equilibrium populations are obtained, and the **free energy for each frame** is calculated using the formula $\Delta G_i = -RT \ln(\pi_{cluster(i)})$, where $\pi_{cluster(i)}$ is the equilibrium probability of the cluster to which the frame belongs. Finally, the function `visualize_2d_diffusion_map` is called to generate 2D graphs in the diffusion space. Typically, two graphs are produced: one is a scatter plot of the diffusion space (with points colored according to cluster assignments), and the other is a summary diagram showing cluster centers and possibly the main MSM transitions. Additionally, the free energy levels of each cluster may be represented by different color intensities.
  
**Output:** The script prints progress messages to the terminal at each stage (e.g., scale calculation, kernel matrix dimensions, normalization, diffusion map computed). Graphical outputs are displayed interactively via matplotlib. The expected graphs include: *Diffusion Map 2D Scatter* – where the x and y axes represent diffusion components 1 and 2, with each point corresponding to a frame colored according to its cluster. This allows visualization of the separation of states in the diffusion space. Additionally, a Free Energy per State graph may be produced, possibly showing cluster centers overlaid with free energy values (alternatively, a bar chart showing the free energy for each cluster). Since the free energy is calculated via the MSM, these values represent the actual energy landscape of the system (in kT units, e.g., if R=0.001987 kcal/(mol·K) and T=300K, then 0.592 kcal/mol ≈ 1 kT). As a result, this script automatically applies the diffusion map + MSM chain to reveal the metastable states of the system and the energy barriers between them.

- **difuzyon_map_MMGBSA.py:** *Diffusion map + K-means with integration of MMGBSA energy data.*
  
**Inputs:** A structure file, a trajectory file, an atom selection, and a CSV file containing energy data are required arguments. Optionally, parameters such as `n_components`, `epsilon`, and `n_clusters` are also accepted (note: there is no lag_time here since no MSM is constructed). For example: `python difuzyon_map_MMGBSA.py prot.pdb traj.xtc "name CA" energies.csv 3 None 5`.
  
**Operation:** The trajectory is loaded and aligned, and the feature matrix is computed similarly to the previous script. The diffusion map is then computed using the same procedure (via a call to `compute_diffusion_map`). In this case, instead of constructing an MSM, the energy data is read from the CSV (using a function `read_energy_csv`). The `read_energy_csv` function likely reads the CSV with pandas, extracts the `r_psp_MMGBSA_dG_Bind` column, and verifies its consistency with the number of frames. Then, the function `cluster_states` is called to perform K-means clustering in the diffusion space (using `n_clusters`). Finally, the function `visualize_2d_diffusion_map(diffusion_map, states, energies)` is called. This function produces an interactive 2D scatter plot in the diffusion space: each frame is represented as a point, with its color determined directly by its MMGBSA energy value (with lower energy represented by one color and higher energy by another). Additionally, the cluster centers may be indicated by a different marker or outline. The output of this script is likely an HTML file (e.g., diffusion_map_energy.html) generated using Bokeh. In the code, Bokeh’s `output_file(f"diffusion_map_{i+1}.html")` indicates that if `n_components=3` (or higher), the script creates separate HTML graphs for each pair of components (using an incrementing counter). These HTML files contain interactive scatter plots where the x and y axes represent the corresponding diffusion components, and a color bar indicates the energy scale. This allows the user to visually inspect which regions in the diffusion space have higher or lower binding energies. This script facilitates the analysis of the relationship between conformational substates and binding energy.

- **difuzyon_map_MMGBSA_OPTICS.py:** *Diffusion map + energy with OPTICS clustering and interactive analysis.*
  
**Inputs:** A structure file, a trajectory file, an atom selection, and a CSV file containing energy data are required. Optional arguments include: `--n_components` (number of diffusion map components, default is 3), `--epsilon` (maximum neighborhood distance for OPTICS; optional, default is None for automatic calculation), `--min_samples` (minimum number of samples for OPTICS, default is 5), `--xi` (sensitivity parameter for OPTICS cluster extraction, default is 0.05), and `--min_cluster_size` (minimum cluster size as a proportion of the total data, default is 0.05 i.e., 5%).
  
**Operation:** The trajectory and energy data are loaded, and the diffusion map is computed as in previous scripts (calculating frame features, normalization, kernel matrix construction, spectral embedding). Then, sklearn’s OPTICS algorithm is applied to the diffusion map results, using the provided parameters (`min_samples`, `xi`, and `min_cluster_size`). The OPTICS algorithm outputs cluster labels for each frame (`optics.labels_`) along with the reachability distances and ordering (`optics.reachability_` and `ordering_`). These data are passed to the function `visualize_2d_diffusion_map(diffusion_map, states, energies, reachability, ordering)`. In the visualization step, Bokeh is used: an initial scatter plot is prepared (e.g., via `p = figure(...)`), where the x and y axes represent diffusion components 1 and 2, and all frames are plotted as points.
  
**Output:** The color coding here is noteworthy: it is likely that, instead of directly mapping binding energy to color, cluster labels are used, or vice versa, as the code shows that energy may be included as an extra dimension (via `features = compute_features(universe, args.selection, energies`)【121†L1533-L1539]). This extra energy dimension can affect the diffusion distances. Additionally, a reachability plot is generated: the OPTICS reachability plot is created as a separate Bokeh figure【121†L1471-L1479】. In this plot, the x-axis represents the OPTICS ordering of data points, and the y-axis represents the reachability distance; thus, valleys in this plot indicate cluster boundaries. The code replaces infinite reachability values with the largest finite value for plotting, and this graph is saved as *reachability_plot.html*. The scatter plot is likely saved as one or more HTML files (e.g., as `diffusion_map_{i+1}.html` in a loop if n_components > 2).
  
**Output:** Upon execution, the script prints progress messages to the terminal and produces **interactive graph files**. For example, files such as `diffusion_map_1.html`, `diffusion_map_2.html`, ..., and `reachability_plot.html` are generated. Opening `reachability_plot.html` in a browser allows you to view the OPTICS cluster hierarchy; distinct valleys indicate different clusters. Each `diffusion_map_X.html` file contains a scatter plot of the corresponding pair of diffusion components, allowing the user to hover over points to view frame information (and possibly energy). If clusters are distinct, they will appear grouped in similar colors or shapes. The color palette is likely based on a continuous mapping of binding energy (using a `LinearColorMapper` with a free energy mapper). Consequently, the user can simultaneously analyze the clustering in the diffusion space and the energy characteristics of these clusters. For example, by selecting an epsilon cutoff from the reachability plot, one can determine which points form a cluster in the scatter plot. Since energy is included as a dimension, even if points are distant in diffusion space, they may cluster together if they have similar energy, or vice versa, which is reflected in the plot. This script provides a powerful and interactive tool for exploring complex conformational-energy landscapes.

-------------------------------------

With the above detailed explanations of each script, the theoretical context of the analyses and the expected outputs have been provided. Users can perform a comprehensive MD simulation post-analysis using these scripts with their own datasets. In summary, the MDScripts repository includes tools that automate everything from RMSD/RMSF calculations to advanced dimension reduction, MSM construction, and energy analysis. By following the instructions in this README and the help messages within each script, you can efficiently execute your analyses. Each script is designed to clearly reveal its working principles with outputs that are intuitively interpretable and suitable for scientific reporting. Ensure that all the dependencies listed above are installed and that your input files are correctly formatted to effectively use these scripts. We wish you success in your analysis!

