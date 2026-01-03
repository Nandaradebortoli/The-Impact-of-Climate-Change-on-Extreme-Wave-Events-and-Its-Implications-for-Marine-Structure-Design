"""
Map Plotting of Return Levels and Changes - HadGEM RCP 8.5 Scenario
===================================================================

This script generates scientific-style maps (e.g., for publication or reports) showing:
- Return levels of significant wave height (Hs) for 10-year periods
- Differences (ΔHs) between periods to assess projected changes

Features
--------
- Uses Cartopy for clean, minimal base maps (gray land, black coastlines)
- Automatically handles diverging or sequential color mapping
- Blue-white-red diverging scale for differences (centered at 0)
- Viridis sequential scale for absolute return levels
- Saves plots with transparent background and appropriate size for paper layout

Input
-----
- NetCDF file: `hs_return_10_hadgen_85.nc` with 10-year and 20-year return levels
- Location: `/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/results_hs_tp/`

Output
------
- PNG maps saved to `results_hs_tp/plots/`
- File names follow: `mapa_<VARIABLE_NAME>_cartopy.png`

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

NC_PATH = os.path.join(DATA_FOLDER, 'hs_return_10_hadgen_85.nc')

# --- Load NetCDF
ds = xr.open_dataset(NC_PATH)

# --- Variables to plot
maps = {
    'ΔHs_20y_2027_2047-2007_2026_hadgen_85': ds['hs_20y_2027_2047'] - ds['hs_20y_2007_2027'],
    'Hs_10y_2007_2017_hadgen_85': ds['hs_10y_2007_2017'],
    'ΔHs_10y_2017_2027-2007_2016_hadgen_85': ds['hs_10y_2017_2027'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2027_2037-2007_2017_hadgen_85': ds['hs_10y_2027_2037'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2037_2047-2007_2017_hadgen_85': ds['hs_10y_2037_2047'] - ds['hs_10y_2007_2017'],
}

# --- Custom blue-white-red colormap for ΔHs
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

    # --- Colormap and normalization
    if 'Δ' in name:
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_min > 0:
            data_min = 0
        if data_max < 0:
            data_max = 0
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        cmap = bluewhitered
    else:
        norm = Normalize(vmin=0, vmax=20)
        cmap = 'viridis'

    # --- Base map (clean style)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='black', zorder=1)

    # --- Plot the data
    img = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, norm=norm,
                        shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # --- Extent and gridlines
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- Title and colorbar
    title_str = name.replace('_', ' ').replace('hadgen', 'HadGEM').replace('mpi', 'MPI')
    ax.set_title(title_str, fontsize=14)
    label_cb = 'ΔHs (m)' if 'Δ' in name else 'Max Hs (m)'
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.85, pad=0.05)
    cbar.set_label(label_cb, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # --- Save figure
    output_img = os.path.join(OUTPUT_FOLDER, f'mapa_{name}_cartopy.png')
    fig.savefig(output_img, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ HadGEM 8.5 Hs maps generated with clean scientific style.")
