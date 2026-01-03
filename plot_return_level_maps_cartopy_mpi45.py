"""
Return Level Maps - MPI RCP 4.5 Scenario
========================================

This script generates a series of maps for the MPI climate model under the RCP 4.5 scenario.
It visualizes both the significant wave height (Hs) return levels and their changes over time.

Map Types
---------
- Absolute return levels (Hs) for 10-year periods
- ΔHs maps showing differences between time blocks (change projections)

Features
--------
- Publication-quality maps using Cartopy
- Gray land background, black coastlines and borders
- Blue-white-red diverging color map for differences
- Sequential Viridis for absolute Hs values
- Automatically saved as transparent PNGs

Input
-----
NetCDF file:
- `hs_return_10_mpi_45.nc`

Output
------
- PNG figures saved in `results_hs_tp/plots/` with the prefix `mapa_`

Author
------
Created by Rafael A. N. Reis, 2025
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Paths
DATA_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/results_hs_tp'
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'plots')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NC_PATH = os.path.join(DATA_FOLDER, 'hs_return_10_mpi_45.nc')

# --- Load NetCDF data
ds = xr.open_dataset(NC_PATH)

# --- Variables to plot
maps = {
    'ΔHs_20y_2027_2047_minus_2007_2026_mpi_45': ds['hs_20y_2027_2047'] - ds['hs_20y_2007_2027'],
    'Hs_10y_2007_2017_mpi_45': ds['hs_10y_2007_2017'],
    'ΔHs_10y_2017_2027_minus_2007_2016_mpi_45': ds['hs_10y_2017_2027'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2027_2037_minus_2007_2017_mpi_45': ds['hs_10y_2027_2037'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2037_2047_minus_2007_2017_mpi_45': ds['hs_10y_2037_2047'] - ds['hs_10y_2007_2017'],
}

# --- Define diverging colormap
bluewhitered = LinearSegmentedColormap.from_list(
    'bluewhitered',
    [(0.0, '#0000aa'), (0.5, 'white'), (1.0, '#aa0000')]
)

# --- Coordinates
lats = ds['latitude'].values
lons = ds['longitude'].values
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --- Plot loop
for name, data_array in maps.items():
    data = data_array.values

    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Colormap settings
    if 'Δ' in name:
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        cmap = bluewhitered
    else:
        norm = Normalize(vmin=0, vmax=20)
        cmap = 'viridis'

    # --- Map base (gray land, black coastlines)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='black', zorder=1)

    # --- Plot data
    img = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, norm=norm,
                        shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # --- Map extent
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    # --- Gridlines and labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- Title and colorbar
    title_str = name.replace('_', ' ').replace('mpi', 'MPI')
    ax.set_title(title_str, fontsize=14)
    label_cb = 'ΔHs (m)' if 'Δ' in name else 'Max Hs (m)'
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.85, pad=0.05)
    cbar.set_label(label_cb, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # --- Save figure
    output_img = os.path.join(OUTPUT_FOLDER, f'mapa_{name}_cartopy.png')
    fig.savefig(output_img, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ MPI 4.5 Hs maps successfully generated using standardized scientific layout.")
