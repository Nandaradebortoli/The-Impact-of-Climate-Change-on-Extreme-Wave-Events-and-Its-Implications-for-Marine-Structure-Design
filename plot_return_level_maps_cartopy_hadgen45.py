"""
Map Plotting of Return Levels and Their Differences Using Cartopy
==================================================================

This script generates publication-ready maps (half-page style) for visualizing:
- Return levels of significant wave height (Hs) for specific 10-year periods
- Differences in Hs return levels across different future periods

Features
--------
- Uses Cartopy for clean, minimal maps (no bathymetry or street detail)
- Plots are saved with a transparent background
- Includes blue-white-red diverging colormap for deltas (ΔHs)
- Extent, coastlines, borders, and lat/lon gridlines are included

Input
-----
- NetCDF file: `hs_return_10_hadgen_45.nc` (daily max Hs return levels)
- Stored in: `/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/results_hs_tp/`

Output
------
- PNG figures saved under: `results_hs_tp/plots/`
- Filenames follow: `mapa_<VARIABLE_NAME>_cartopy.png`

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

NC_PATH = os.path.join(DATA_FOLDER, 'hs_return_10_hadgen_45.nc')

# --- Load NetCDF
ds = xr.open_dataset(NC_PATH)

# --- Variables to plot
maps = {
    'ΔHs_20y_2027_2047-2007_2026_hadgen_45': ds['hs_20y_2027_2047'] - ds['hs_20y_2007_2027'],
    'Hs_10y_2007_2017_hadgen_45': ds['hs_10y_2007_2017'],
    'ΔHs_10y_2017_2027-2007_2016_hadgen_45': ds['hs_10y_2017_2027'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2027_2037-2007_2017_hadgen_45': ds['hs_10y_2027_2037'] - ds['hs_10y_2007_2017'],
    'ΔHs_10y_2037_2047-2007_2017_hadgen_45': ds['hs_10y_2037_2047'] - ds['hs_10y_2007_2017'],
}

# --- Custom colormap: blue-white-red
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

    # --- Figure (half-page style)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # --- Colormap settings
    if 'Δ' in name:
        # Diverging scale centered at 0
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        if data_min > 0:
            data_min = 0
        if data_max < 0:
            data_max = 0
        norm = TwoSlopeNorm(vcenter=0, vmin=-2, vmax=2)
        cmap = bluewhitered
    else:
        # Single-sided scale
        norm = Normalize(vmin=0, vmax=20)
        cmap = 'viridis'

    # --- Map background: clean gray land
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black', zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='black', zorder=1)

    # --- Plot data
    img = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, norm=norm,
                        shading='auto', transform=ccrs.PlateCarree(), zorder=2)

    # --- Extent
    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

    # --- Gridlines and labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # --- Title and colorbar
    title_str = name.replace('_', ' ')
    title_str = title_str.replace('hadgen', 'HadGEM').replace('mpi', 'MPI')
    ax.set_title(title_str, fontsize=14)

    colorbar_label = 'ΔHs (m)' if 'Δ' in name else 'Max Hs (m)'
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.85, pad=0.05)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # --- Save figure
    output_img = os.path.join(OUTPUT_FOLDER, f'mapa_{name}_cartopy.png')
    plt.savefig(output_img, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()

print("✅ All maps have been generated in MATLAB-style (gray continents and coastlines).")
