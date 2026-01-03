"""
Return Level Maps - MPI RCP 8.5 Scenario
========================================

This script generates publication-quality maps of significant wave height (Hs) return levels
and their projected changes under the MPI model for the RCP 8.5 climate scenario.

Map Types
---------
- Hs return levels for 10-year blocks
- ΔHs difference maps comparing future periods to 2007–2017

Visualization Features
----------------------
- Gray landmass with black coastlines and borders
- Diverging colormap (blue-white-red) for ΔHs maps
- Viridis colormap for absolute Hs values
- Scientific layout with labeled gridlines and colorbars
- Output in transparent, high-resolution PNGs

Input
-----
NetCDF:
- `hs_return_10_mpi_85.nc` (stored in `results_hs_tp/`)

Output
------
- Figures in `results_hs_tp/plots/`, named `mapa_<name>_cartopy.png`

Author
------
Rafael A. N. Reis, 2025
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

NC_PATH = os.path.join(DATA_FOLDER, 'hs_return_10_mpi_85.nc')

# --- Load dataset
ds = xr.open_dataset(NC_PATH)

# --- Maps to generate
maps = {
    'ΔHs_20y_2027_2047_minus_2007_2026_mpi_85': ds['hs_20y_2027_2047'] - ds['hs_20y_2007_2027'],
    'Hs_10y_2007_2017_mpi_85': ds['hs_10y_2007_2017'],
    'ΔHs_10y_2017_2027_minus_2007_2016_mpi_85': ds['hs_10y_2017_2027'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2027_2037_minus_2007_2017_mpi_85': ds['hs_10y_2027_2037'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2037_2047_minus_2007_2017_mpi_85': ds['hs_10y_2037_2047'] - ds['hs_10y_2007_2017'],
}

# --- Custom diverging colormap
bluewhitered = LinearSegmentedColormap.from_list(
    'bluewhitered',
    [(0.0, '#0000aa'), (0.5, 'white'), (1.0, '#aa0000')]
)

# --- Coordinates
lats = ds['latitude'].values
lons = ds['longitude'].values
lon_grid, lat_grid = np.meshgrid(lons, lats)

# --- Loop over maps
for name, data_array in maps.items():
    data = data_array.values

    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Choose color scale
    if 'Δ' in name:
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        cmap = bluewhitered
    else:
        norm = Normalize(vmin=0, vmax=20)
        cmap = 'viridis'

    # --- Map base
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='black', zorder=1)

    # --- Plot
    img = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, norm=norm,
                        shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # --- Map extent
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    # --- Grid with labels
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

print("✅ MPI 8.5 Hs maps successfully generated in standard scientific layout.")
