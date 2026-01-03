"""
Computation of Return Levels Using Exponentiated Weibull Distribution (MPI RCP 4.5)
====================================================================================

This script calculates return levels of significant wave height (Hs) for 
the MPI model under the RCP 4.5 scenario, using the Exponentiated Weibull distribution.

Details
-------
- Processes all grid points in parallel using ProcessPoolExecutor.
- Calculates 20-year return levels for:
    - 2007–2027
    - 2027–2047
- Calculates 10-year return levels for:
    - 2007–2017
    - 2017–2027
    - 2027–2037
    - 2037–2047

Assumptions
-----------
- 375 sea states per year used in return level estimation.
- Daily maxima of Hs are stored in the NetCDF file: `daily_max_mpi_hs_45.nc`.

Output
------
- A NetCDF file named `hs_return_10_mpi_45.nc`, containing:
    - 6 return level fields (lat × lon)
    - One for each period listed above

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

# --- Load daily maximum Hs dataset
dataset_path = os.path.join(DATA_FOLDER, 'daily_max_mpi_hs_45.nc')
ds = xr.open_dataset(dataset_path, decode_times=True)
hs_all = ds['hs'].load()
lats = hs_all['latitude'].values
lons = hs_all['longitude'].values
times = hs_all['time'].values
ds.close()

# --- Processing function for a single grid point
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

# --- Create list of all grid points
total_points = [(i, j) for i in range(len(lats)) for j in range(len(lons))]

# --- Parallel loop
print(f"🌊 Starting parallel computation using {N_WORKERS} workers...")
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    results = list(tqdm(executor.map(process_point, total_points), total=len(total_points)))

# --- Reshape results to 2D arrays
results = np.array(results)

hs_20y_2007_2027 = results[:, 0].reshape((len(lats), len(lons)))
hs_20y_2027_2047 = results[:, 1].reshape((len(lats), len(lons)))
hs_10y_2007_2017 = results[:, 2].reshape((len(lats), len(lons)))
hs_10y_2017_2027 = results[:, 3].reshape((len(lats), len(lons)))
hs_10y_2027_2037 = results[:, 4].reshape((len(lats), len(lons)))
hs_10y_2037_2047 = results[:, 5].reshape((len(lats), len(lons)))

# --- Save as NetCDF
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

output_path = os.path.join(OUTPUT_FOLDER, 'hs_return_10_mpi_45.nc')
final_ds.to_netcdf(output_path)

print(f"\n✅ Final file saved at: {output_path}")
