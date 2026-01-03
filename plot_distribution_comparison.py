"""
Distribution Comparison Plot by Model and Emission Scenario
===========================================================

This script loads summary CSVs with the best-fit univariate distribution (Hs) 
for each point in multiple NetCDF datasets, and plots a grouped bar chart 
showing the percentage occurrence of each distribution across:

- Two global climate models (HadGEM and MPI)
- Two emission scenarios (RCP 4.5 and RCP 8.5)

Usage
-----
- Requires prior execution of distribution fitting script which produces
  `distribution_summary_<model>.csv` files.
- Output: a PNG file comparing the distributions side-by-side.

Author
------
Created by Rafael A. N. Reis, 2025
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Paths and input filenames
OUTPUT_FOLDER = '/media/rafael/HDD/DADOS/nandara/RESULTADOS ARTIGO/'
files = {
    'HadGEM 4.5': 'distribution_summary_Hadgen_hs_all_45.csv',
    'HadGEM 8.5': 'distribution_summary_Hadgen_hs_all_85.csv',
    'MPI 4.5': 'distribution_summary_MPI_hs_all_45.csv',
    'MPI 8.5': 'distribution_summary_MPI_hs_all_85.csv'
}

# --- Load each CSV and convert raw counts to percentage
data = {}
for label, filename in files.items():
    df = pd.read_csv(os.path.join(OUTPUT_FOLDER, filename))
    total = df['Count'].sum()
    df['Percent'] = df['Count'] / total * 100
    data[label] = df.set_index('Distribution')['Percent']

# --- Combine into one DataFrame (distributions x scenarios)
df_all = pd.DataFrame(data).fillna(0).sort_index()

# --- Plot
plt.figure(figsize=(5.5, 4))  # Half-page width
ax = df_all.plot(kind='bar', ax=plt.gca())

plt.ylabel('Occurrence Percentage (%)', fontsize=12)
plt.xlabel('Distribution', fontsize=12)
plt.title('Comparison of Best-Fit Distributions by Model and Scenario', fontsize=12)
plt.xticks(rotation=60, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Scenario', fontsize=11, title_fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# --- Layout adjustment
plt.subplots_adjust(left=0.12, right=0.98, top=0.85, bottom=0.44)
plt.savefig(os.path.join(OUTPUT_FOLDER, 'distribution_comparison_percentage_v2.png'), transparent=True, dpi=300)
plt.show()
