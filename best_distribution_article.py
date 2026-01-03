"""
Best-Fit Univariate Distribution Mapping for Significant Wave Height (Hs)
=========================================================================

This script processes multiple NetCDF files containing significant wave height (Hs) data 
and identifies the best-fit distribution among a predefined set using the Anderson-Darling 
statistic as the sole selection criterion.

Workflow
--------
- Loads Hs data from NetCDF files for multiple climate models/scenarios.
- Uses pre-selected random spatial points stored in a text file.
- Extracts daily maxima (2006–2024) at each point.
- Fits multiple univariate distributions to each time series.
- Selects the best fit using the Anderson-Darling statistic.
- Aggregates and visualizes results across all grid points.
- Saves summary CSVs and barplot figures for each file processed.

Dependencies
------------
- numpy
- pandas
- xarray
- matplotlib
- scipy
- joblib
- tqdm
- virocon (with custom distributions)

Author
------
Created by Rafael A. N. Reis, 2025
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from virocon import (
    WeibullDistribution, LogNormalDistribution, NormalDistribution,
    ExponentiatedWeibullDistribution, GeneralizedGammaDistribution, VonMisesDistribution
)

# --- Paths
DATA_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/'
OUTPUT_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/'
POINTS_FILE = os.path.join(OUTPUT_FOLDER, 'pontos_aleatorios.txt')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- NetCDF files to process
nc_files = [
    "Hadgen_hs_all_45.nc",
    "Hadgen_hs_all_85.nc",
    "MPI_hs_all_45.nc",
    "MPI_hs_all_85.nc"
]

# --- Candidate distributions
all_distributions = {
    'Weibull': WeibullDistribution(),
    'LogNormal': LogNormalDistribution(),
    'Normal': NormalDistribution(),
    'ExponentiatedWeibull': ExponentiatedWeibullDistribution(),
    'GeneralizedGamma': GeneralizedGammaDistribution(),
    'VonMises': VonMisesDistribution()
}

# --- Best-fit distribution selection (Anderson-Darling only)
def select_best_distribution_name(data: np.ndarray) -> str:
    """
    Select the best distribution based only on the Anderson-Darling statistic.

    Parameters
    ----------
    data : np.ndarray
        1D array of observed values.

    Returns
    -------
    str
        Name of the distribution with the lowest AD statistic.
    """
    results = []
    data_sorted = np.sort(data)
    n = len(data_sorted)
    i = np.arange(1, n + 1)
    eps = 1e-10

    for name, dist in all_distributions.items():
        try:
            dist.fit(data)
            cdf_vals = dist.cdf(data_sorted, **dist.parameters)
            cdf_vals = np.clip(cdf_vals, eps, 1 - eps)
            ad_stat = -n - np.sum((2 * i - 1) / n * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
            if not np.isnan(ad_stat):
                results.append((name, ad_stat))
        except Exception:
            continue

    if not results:
        return 'None'

    best_name = min(results, key=lambda x: x[1])[0]
    return best_name

# --- Load sampled (lat, lon) points
def load_sampled_points(lat_array, lon_array):
    points = []
    with open(POINTS_FILE, 'r') as f:
        for line in f:
            lat, lon = map(float, line.strip().split(','))
            i = np.argmin(np.abs(lat - lat_array))
            j = np.argmin(np.abs(lon - lon_array))
            points.append((i, j))
    return points

# --- Main function to process each NetCDF file
def process_nc_file(nc_filename):
    print(f"\n📂 Processing: {nc_filename}")
    ds = xr.open_dataset(os.path.join(DATA_FOLDER, nc_filename), decode_times=True)
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    points = load_sampled_points(lats, lons)

    def process_point(i, j):
        try:
            hs_series = ds['hs'].isel(latitude=i, longitude=j).to_series().resample('1D').max().dropna()
            hs_early = hs_series['2006':'2024'].values
            if len(hs_early) < 100:
                return 'None'
            return select_best_distribution_name(hs_early)
        except Exception as e:
            print(f"Error at point ({i},{j}): {e}")
            return 'None'

    labels = Parallel(n_jobs=28)(
        delayed(process_point)(i, j) for i, j in tqdm(points, desc=f"Processing {nc_filename}", total=len(points))
    )

    # --- Count occurrences and generate bar plot
    counter = Counter(labels)
    df_result = pd.DataFrame(counter.items(), columns=['Distribution', 'Count'])
    df_result = df_result[df_result['Distribution'] != 'None']
    df_result.sort_values('Count', ascending=False, inplace=True)

    tag = os.path.splitext(nc_filename)[0]
    df_result.to_csv(os.path.join(OUTPUT_FOLDER, f'distribution_summary_{tag}.csv'), index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_result['Distribution'], df_result['Count'], color='skyblue')
    plt.xlabel('Distribution', fontsize=12)
    plt.ylabel('Number of Points', fontsize=12)
    plt.title(f'Best Fit Distribution - {tag}', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'best_distribution_barplot_{tag}.png'))
    plt.close()

    print(f"✅ Completed: {tag}")

# --- Run all files
for nc_file in nc_files:
    process_nc_file(nc_file)
