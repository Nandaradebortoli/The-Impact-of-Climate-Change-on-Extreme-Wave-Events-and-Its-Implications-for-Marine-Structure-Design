"""
Hs Return Period Curves for Key Offshore Basins (HadGEM & MPI, RCP 4.5/8.5)
===========================================================================

This script generates return period plots (up to 100 years) for significant wave heights (Hs)
at specified offshore basin coordinates under different climate scenarios and models.

Key Features
------------
- Models: HadGEM and MPI
- Scenarios: RCP 4.5 and 8.5
- Periods: Four 10-year blocks from 2007 to 2046
- Fitted distribution: Exponentiated Weibull (via Virocon)
- Includes geographic inset maps with basin location
- Output as 2x2 panels with one figure per basin

Dependencies
------------
- xarray
- numpy
- pandas
- matplotlib
- virocon
- cartopy

Author
------
Rafael A. N. Reis, 2025
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from virocon.distributions import ExponentiatedWeibullDistribution
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.mpl.geoaxes

# ========== CONFIGURATION ==========
MODELS = ["HadGEN", "MPI"]
SCENARIOS = ["45", "85"]
BLOCKS = [
    ('2007-01-01', '2016-12-31'),
    ('2017-01-01', '2026-12-31'),
    ('2027-01-01', '2036-12-31'),
    ('2037-01-01', '2046-12-31')
]
COLORS = ['blue', 'green', 'orange', 'red']

BASE_PATH = "/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO"
OUTPUT_PATH = os.path.join(BASE_PATH, "results_plots/figuras_bacias/")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ========== TARGET BASINS ==========
basins = {
    "Pelotas Basin": {"lat": -33.5, "lon": -51.0},
    "Foz do Amazonas Basin": {"lat": 3.0, "lon": -48.0}
}

# ========== UTILITY FUNCTION ==========
def find_nearest_grid_point(lat_array, lon_array, target_lat, target_lon):
    i = (np.abs(lat_array - target_lat)).argmin()
    j = (np.abs(lon_array - target_lon)).argmin()
    return i, j

# ========== PLOT FUNCTION FOR ONE SUBPLOT ==========
def plot_return_period(ax, ds_ret_path, ds_daily_path, i, j, title):
    ds_ret = xr.open_dataset(ds_ret_path)
    ds_daily = xr.open_dataset(ds_daily_path)

    lat = ds_daily["latitude"].values
    lon = ds_daily["longitude"].values

    hs_all = ds_daily["hs"][:, i, j].to_dataframe().dropna()
    hs_all.index = pd.to_datetime(hs_all.index)

    hs_grid = np.linspace(0, hs_all["hs"].max() * 10, 500)

    for idx, (start, end) in enumerate(BLOCKS):
        hs_block = hs_all.loc[start:end]["hs"].dropna().values
        if len(hs_block) < 10:
            continue  # skip sparse blocks

        dist = ExponentiatedWeibullDistribution()
        dist.fit(hs_block)

        prob_exceed = 1 - dist.cdf(hs_grid)
        prob_exceed[prob_exceed <= 1e-10] = 1e-10
        return_periods = 1 / prob_exceed / 365

        mask = (return_periods <= 100)
        ax.plot(return_periods[mask], hs_grid[mask], '-', color=COLORS[idx],
                label=f'{start[:4]}–{end[:4]}')

    # Reference vertical lines
    for T in [1, 5, 10, 50, 100]:
        ax.axvline(x=T, color='black', linestyle=':', linewidth=1)
        ymax = ax.get_ylim()[1]
        ax.text(T, ymax * 0.98, f'{T}y', rotation=90, va='top', ha='center',
                fontsize=9, color='black', weight='bold')

    # Axis setup
    ax.set_xscale('log')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel(r'$Hs_{\max}$ (m)')
    ax.set_title(f"{title}\n(lat={lat[i]:.2f}, lon={lon[j]:.2f})", fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    # Inset map
    axins = inset_axes(ax, width="40%", height="40%", loc='lower right',
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(map_projection=ccrs.PlateCarree()))
    axins.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
    axins.add_feature(cfeature.COASTLINE, linewidth=0.5)
    axins.add_feature(cfeature.BORDERS, linewidth=0.2)
    axins.plot(lon[j], lat[i], 'ro', markersize=5, transform=ccrs.PlateCarree())

# ========== MAIN FIGURE GENERATOR ==========
def generate_basin_figure(basin_name, target_lat, target_lon):
    sample_ds = xr.open_dataset(f"{BASE_PATH}/daily_max_hadgen_hs_45.nc")
    lat_array = sample_ds["latitude"].values
    lon_array = sample_ds["longitude"].values

    i, j = find_nearest_grid_point(lat_array, lon_array, target_lat, target_lon)
    print(f"📍 {basin_name}: nearest grid -> lat={lat_array[i]:.2f}, lon={lon_array[j]:.2f} (i={i}, j={j})")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.suptitle(f"{basin_name} – Return Periods (Hs) for HadGEM and MPI", fontsize=14, weight="bold")

    for m_idx, model in enumerate(MODELS):
        for c_idx, scenario in enumerate(SCENARIOS):
            ret_path = f"{BASE_PATH}/results_hs_tp/hs_return_10_{model.lower()}_{scenario}.nc"
            daily_path = f"{BASE_PATH}/daily_max_{model.lower()}_hs_{scenario}.nc"

            model_title = 'HadGEM' if model == 'HadGEN' else model
            scenario_label = f"{model_title} RCP {scenario}"
            plot_return_period(axs[m_idx, c_idx], ret_path, daily_path, i, j, scenario_label)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    fig_path = os.path.join(OUTPUT_PATH, f"return_periods_{basin_name.replace(' ', '_').lower()}.png")
    plt.savefig(fig_path, dpi=300, transparent=True)
    plt.close()
    print(f"✅ Figure saved: {fig_path}")

# ========== EXECUTION ==========
for name, coords in basins.items():
    generate_basin_figure(name, coords["lat"], coords["lon"])
