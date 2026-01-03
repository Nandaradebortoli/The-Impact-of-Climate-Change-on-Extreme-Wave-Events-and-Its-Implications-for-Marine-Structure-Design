"""
Computation of Return Levels Using Exponentiated Weibull Distribution (HadGEM RCP 8.5)
=======================================================================================

This script calculates return levels of significant wave height (Hs) for 
the HadGEM model under the RCP 8.5 scenario, based on daily maxima 
and using the Exponentiated Weibull distribution.

Features
--------
- Parallel processing of all grid points using `ProcessPoolExecutor`.
- Calculates 20-year return levels for two time spans:
    - 2007–2027
    - 2027–2047
- Calculates 10-year return levels for four decadal blocks:
    - 2007–2017
    - 2017–2027
    - 2027–2037
    - 2037–2047

Assumptions
-----------
- 375 sea states per year (used in return period calculation).
- Daily maximum time series is stored in a NetCDF file: `daily_max_hadgen_hs_85.nc`.

Output
------
- One NetCDF file with six 2D fields (lat × lon) of return levels:
  `hs_return_10_hadgen_85.nc`

Author
------
Created by Rafael A. N. Reis, 2025
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from virocon.distributions import ExponentiatedWeibullDistribution

# --- Configuration
DATA_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/'
OUTPUT_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/results_hs_tp/'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

RETURN_PERIOD_20Y = 20
RETURN_PERIOD_10Y = 10
N_WORKERS = 20  # Number of CPU cores to use

# --- Load full dataset
dataset_path = os.path.join(DATA_FOLDER, 'daily_max_hadgen_hs_85.nc')
ds = xr.open_dataset(dataset_path, decode_times=True)
hs_all = ds['hs'].load()  # Load into memory
lats = hs_all['latitude'].values
lons = hs_all['longitude'].values
times = hs_all['time'].values
ds.close()

# --- Function to compute return levels for one grid point
def process_point(i_j):
    i, j = i_j
    try:
        hs_series = hs_all[:, i, j].values

        if np.all(np.isnan(hs_series)) or np.count_nonzero(~np.isnan(hs_series)) < 2000:
            return [np.nan] * 6

        hs_df = pd.DataFrame({'time': times, 'hs': hs_series})
        hs_df['time'] = pd.to_datetime(hs_df['time'])
        hs_df = hs_df.set_index('time')

        results = []

        # --- 20-year return levels
        for period in [('2007', '2027'), ('2027', '2047')]:
            hs_period = hs_df[period[0]:period[1]].dropna().values
            dist = ExponentiatedWeibullDistribution()
            dist.fit(hs_period)
            rl = dist.icdf(1 - 1 / (RETURN_PERIOD_20Y * 375))
            results.append(rl)

        # --- 10-year return levels
        periods_10y = [('2007', '2017'), ('2017', '2027'), ('2027', '2037'), ('2037', '2047')]
        for start, end in periods_10y:
            hs_period = hs_df[start:end].dropna().values
            if len(hs_period) < 100:
                results.append(np.nan)
            else:
                dist = ExponentiatedWeibullDistribution()
                dist.fit(hs_period)
                rl = dist.icdf(1 - 1 / (RETURN_PERIOD_10Y * 375))
                results.append(rl)

        return results
    except Exception:
        return [np.nan] * 6

# --- List of grid points
total_points = [(i, j) for i in range(len(lats)) for j in range(len(lons))]

# --- Parallel execution
print(f"🌊 Starting parallel computation with {N_WORKERS} workers...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    results = list(tqdm(executor.map(process_point, total_points), total=len(total_points)))

# --- Reshape results into arrays
results = np.array(results)

hs_20y_2007_2027 = results[:, 0].reshape((len(lats), len(lons)))
hs_20y_2027_2047 = results[:, 1].reshape((len(lats), len(lons)))
hs_10y_2007_2017 = results[:, 2].reshape((len(lats), len(lons)))
hs_10y_2017_2027 = results[:, 3].reshape((len(lats), len(lons)))
hs_10y_2027_2037 = results[:, 4].reshape((len(lats), len(lons)))
hs_10y_2037_2047 = results[:, 5].reshape((len(lats), len(lons)))

# --- Export to NetCDF
final_ds = xr.Dataset(
    {
        'hs_20y_2007_2027': (['latitude', 'longitude'], hs_20y_2007_2027),
        'hs_20y_2027_2047': (['latitude', 'longitude'], hs_20y_2027_2047),
        'hs_10y_2007_2017': (['latitude', 'longitude'], hs_10y_2007_2017),
        'hs_10y_2017_2027': (['latitude', 'longitude'], hs_10y_2017_2027),
        'hs_10y_2027_2037': (['latitude', 'longitude'], hs_10y_2027_2037),
        'hs_10y_2037_2047': (['latitude', 'longitude'], hs_10y_2037_2047),
    },
    coords={
        'latitude': lats,
        'longitude': lons
    }
)

output_path = os.path.join(OUTPUT_FOLDER, 'hs_return_10_hadgen_85.nc')
final_ds.to_netcdf(output_path)

print(f"\n✅ Final file saved at: {output_path}")
