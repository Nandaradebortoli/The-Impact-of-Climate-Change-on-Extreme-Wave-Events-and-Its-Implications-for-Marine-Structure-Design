# The-Impact-of-Climate-Change-on-Extreme-Wave-Events-and-Its-Implications-for-Marine-Structure-Design
This repository provides Python scripts for processing, analyzing, and visualizing extreme wave climate in the Western South Atlantic based on global climate model projections. It focuses on statistical analysis of significant wave height (Hs), including distribution fitting, return level estimation, and spatial and basin-scale visualization.

Due to the large size of the model output datasets (NetCDF files), all input data are managed using Git Large File Storage (Git LFS). The repository therefore contains lightweight pointer to the datasets, allowing full access to the underlying data.

The repository includes scripts to:

- Select the best-fit univariate probability distribution for Hs using the Anderson–Darling statistic, comparing Weibull, Lognormal, Normal, Exponentiated Weibull, Generalized Gamma, and Von Mises distributions.
- Compute return levels of significant wave height (Hs) at each grid point using the Exponentiated Weibull distribution, considering: 20-year periods and 10-year moving blocks.
- Generate basin-scale return period curves (up to 100 years) for key offshore regions.
- Produce publication-ready visualizations, including: Spatial maps of Hs return levels, difference maps (ΔHs) between future periods and comparative bar plots of best-fit distributions by model and scenario.
Parallel processing is employed to ensure computational efficiency over large spatial grids.

This repository is designed to support reproducible extreme wave climate analysis and accompanies the datasets used in the associated peer-reviewed publication.


### Requirements for data access
Make sure Git LFS is installed before cloning the repository:

```bash
git lfs install
git clone https://github.com/Nandaradebortoli/The-Impact-of-Climate-Change-on-Extreme-Wave-Events-and-Its-Implications-for-Marine-Structure-Design.git
git lfs pull
