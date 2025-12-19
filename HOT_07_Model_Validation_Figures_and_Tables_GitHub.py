"""
HOT - Model Validation, Figures and Tables for Methods Paper

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""
#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
from scipy.stats import spearmanr, linregress
from scipy.stats import median_abs_deviation as MAD
# Import specific modules from packages
from dateutil import relativedelta
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "Arial"  # Set font for all plots
from sklearn.metrics import mean_squared_error
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore');

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    rd = relativedelta.relativedelta( pd.to_datetime( end ), pd.to_datetime( start ) )
    #date_len = str(rd.years)+"yrs"+str(rd.months)+"m"+str(rd.days+"d")
    date_len  = '{}y{}m{}d'.format(rd.years,rd.months,rd.days)
    #return rd.years, rd.months, rd.days
    return date_len

def aggregate(rows,columns,df):
    column_keys = df[columns].unique()
    row_keys = df[rows].unique()

    agg = { key : [ len(df[(df[rows]==value) & (df[columns]==key)]) for value in row_keys]
               for key in column_keys }

    aggdf = pd.DataFrame(agg,index = row_keys)
    aggdf.index.rename(rows,inplace=True)

    return aggdf

#%%

### READ & EXTRACT BOTTLE PIGMENTS DATA & MODEL RESULTS ###

### Read/Import cleaned Bottle data from CSV including model fit results ###
# CSV filename
filename_1 = 'data/HOT_Bottle_Pigments_ModelResults.csv'
# Load data from CSV. "index_col = 0" makes the first column the index.
bottle_6 = pd.read_csv(filename_1, index_col=0)

# Sort dataframe by Cruise_ID and depth
bottle_6 = bottle_6.sort_values(by=['Cruise_ID', 'depth'])
# Reset dataframe index, replacing the old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Extract Cruise_ID as numpy array
b_ID           = bottle_6['Cruise_ID'].to_numpy()

### Generate a list of unique Cruise_IDs ###
ID_list_6 = pd.unique(b_ID)
print(f"Unique HPLC Chl-a Cruise_IDs: {len(ID_list_6)}")  # Print the count of unique Cruise_IDs

#%%

### READ & EXTRACT BOTTLE POC & PON DATA & MODEL RESULTS ###

# Load and preprocess POC data
filename_poc = 'data/HOT_Bottle_POC_ModelResults.csv'

# Load data from CSV, using the first column as the index
bottle_poc = pd.read_csv(filename_poc, index_col=0)
#bottle_poc.info()

# Sort dataframe by Cruise_ID and depth, then reset the index
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID', 'depth']).reset_index(drop=True)

# Extract Cruise_ID as numpy array
b2_ID         = bottle_poc['Cruise_ID'].to_numpy()      

### Cruise_ID List ###
ID_list_poc = pd.unique(b2_ID)  # Create unique Cruise_ID list
print(f"Unique Particulate Carbon & Nitrogen Cruise_IDs: {len(ID_list_poc)}")  # Print the count of unique Cruise_IDs

#%%

### READ & EXTRACT POP DATA & MODEL RESULTS ###

# Import required libraries

# Load and preprocess POP data
filename_pop = 'data/HOT_Bottle_POP_ModelResults.csv'

# Load data from CSV, using the first column as the index
bottle_pop = pd.read_csv(filename_pop, index_col=0)
#bottle_pop.info()

# Sort dataframe by Cruise_ID and depth, then reset the index
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID', 'depth']).reset_index(drop=True)

# Extract Cruise_ID as numpy array
b3_ID        = bottle_pop['Cruise_ID'].to_numpy()         

### Cruise_ID List ###
ID_list_pop = pd.unique(b3_ID)  # Create unique Cruise_ID list
print(f"Unique Particulate Phosphorus Cruise_IDs: {len(ID_list_pop)}")  # Print the count of unique Cruise_IDs

#%%

### IMPORT BOTTLE PROF DATA

# CSV filename
filename_1 = 'data/HOT_Bottle_profData_Int.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)
#bottle_prof.info()
print(len(bottle_prof))

# Sort dataframe by Cruise_ID
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])

# Reset dataframe index, replacing the old index column
bottle_prof = bottle_prof.reset_index(drop=True)

# Extract bottle MLD with corresponding time ###
ID_list_bottle  = bottle_prof['Cruise_ID'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

#%%

### PLOT PROFILES PER Year colour by Month - PAR int Licor

# Aggregate profile counts for Chl, PC and PN
df = bottle_prof
b_df_count_y = aggregate('yyyy','mm',df)
b_df_count_y = b_df_count_y.sort_values(by='yyyy')
b_df_count_y = b_df_count_y.sort_index(axis=1)

count_bottle = len(bottle_prof)

bottle_pop_prof = bottle_prof.dropna(subset = 'POP_data_int').copy()

count_pop = len(bottle_pop_prof)

df = bottle_pop_prof
pop_df_count_y = aggregate('yyyy','mm',df)
pop_df_count_y = pop_df_count_y.sort_values(by='yyyy')
pop_df_count_y = pop_df_count_y.sort_index(axis=1)

bottle_prof_par = bottle_prof.dropna(subset = 'PAR_int').copy()

count_par = len(bottle_prof_par)

df = bottle_prof_par
par_df_count_y = aggregate('yyyy','mm',df)
par_df_count_y = par_df_count_y.sort_values(by='yyyy')
par_df_count_y = par_df_count_y.sort_index(axis=1)

#Figure setup
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(14, 12),
    gridspec_kw={'hspace': 0.18})

#my_cmap = sns.color_palette("hls", 12)
my_cmap = sns.color_palette("tab20", 12)


# (a) Chl, PC & PN profiles
b_df_count_y.plot(kind='bar', stacked=True, ax=ax1, width=0.7, color=my_cmap)
ax1.legend(bbox_to_anchor=(1.01, 0.5), loc='center left',
           fontsize=15, title='Month', title_fontsize=15)
ax1.set_xlabel('Year', fontsize=16, color='k')
ax1.set_ylabel('Chl, PC & PN Profiles', fontsize=16, color='k')
ax1.set_title('(a)', fontsize=20, color='k')
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.get_legend().remove()
ax1.text(0.985, 0.94, f"n = {count_bottle}",
         transform=ax1.transAxes, ha='right', va='top',fontsize=14, fontweight='bold')

# (b) PP Profiles
pop_df_count_y.plot(kind='bar', stacked=True, ax=ax2, width=0.7, color=my_cmap)
ax2.legend(bbox_to_anchor=(1.01, 0.5), loc='center left',
           fontsize=15, title='Month', title_fontsize=15)
ax2.set_xlabel('Year', fontsize=16, color='k')
ax2.set_ylabel('PP Profiles', fontsize=16, color='k')
ax2.set_title('(b)', fontsize=20, color='k')
ax2.xaxis.set_tick_params(labelsize=15)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.text(0.985, 0.94, f"n = {count_pop}",
         transform=ax2.transAxes, ha='right', va='top',fontsize=14, fontweight='bold')

# (c) Matched PAR
par_df_count_y.plot(kind='bar', stacked=True, ax=ax3, width=0.7, color=my_cmap)
ax3.set_xlabel('Year', fontsize=16, color='k')
ax3.set_ylabel('Number of Matched PAR', fontsize=16, color='k')
ax3.set_title('(c)', fontsize=20, color='k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3.get_legend().remove()
ax3.text(0.985, 0.94, f"n = {count_par}",
         transform=ax3.transAxes, ha='right', va='top',fontsize=14, fontweight='bold')

#plt.tight_layout()
plt.show()

fig.savefig(
    'plots/HOT_ProfileCounts_YearMonth_PAR_Methods.jpeg',
    format='jpeg', dpi=300, bbox_inches="tight")

#%%

### Full Profile Model vs Data Comparison

#Chl Model Results DF
#bottle_6.info()
chl_data  = bottle_6['Chla'].values
chl_model = bottle_6['CHL_mod_total'].values
chl_depth = bottle_6['depth'].values

#PC & PN Model Results DF
#bottle_poc.info()
poc_depth = bottle_poc['depth'].values
poc_data  = bottle_poc['POC'].values
poc_model = bottle_poc['POC_mod_total'].values
pon_data  = bottle_poc['PON'].values
pon_model = bottle_poc['PON_mod_total'].values

#PP Model Results DF
#bottle_pop.info()
pop_depth = bottle_pop['depth'].values
pop_data  = bottle_pop['POP'].values
pop_model = bottle_pop['POP_mod_total'].values

                   # optional, shows accepted args

def build_stats_text(x, y):
    """
    Calculate Spearman stats, regression, RMSE, centered RMSE (CRMSE), and bias,
    returning a formatted multiline string for annotation.
    """
    # Mask out NaNs for paired data
    mask = np.isfinite(x) & np.isfinite(y)
    x_vals = x[mask]
    y_vals = y[mask]
    N = len(x_vals)

    # Spearman correlation
    rho, pval = spearmanr(x_vals, y_vals)

    # Linear regression
    regression = linregress(x_vals, y_vals)
    slope = regression.slope
    intercept = regression.intercept

    # RMSE
    rmse = np.sqrt(mean_squared_error(x_vals, y_vals))
    #rmse_b = mean_squared_error(x_vals, y_vals, squared=False)

    # Bias: mean difference (model - data)
    bias = np.mean(y_vals - x_vals)

    # Format and return with Spearman at top and N included
    return (
        f"R = {rho:.2f}, p = {pval:.3f}\n"
        f"Slope = {slope:.2f}\n"
        f"Intercept = {intercept:.2f}\n"
        f"RMSD = {rmse:.2f}\n"
        #f"Centered RMSE: {crmse:.2f}\n"
        f"Bias = {bias:.2f}\n"
        f"n = {N}"
    )

def plot_reference_lines(ax, x, y, one_to_one_style='k--', reg_style='r-'):
    mask = np.isfinite(x) & np.isfinite(y)
    x_vals = x[mask]
    y_vals = y[mask]
    if len(x_vals) < 2:
        return ax  # Avoid regression with <2 points

    min_val = np.nanmin([x_vals, y_vals])
    max_val = np.nanmax([x_vals, y_vals])

    ax.plot([min_val, max_val], [min_val, max_val], one_to_one_style,
            lw=1.5, label='1:1')

    reg = linregress(x_vals, y_vals)
    slope, intercept = reg.slope, reg.intercept
    x_fit = np.linspace(min_val, max_val, 100)
    ax.plot(x_fit, intercept + slope * x_fit, reg_style,
            lw=1.5, label='Regression')

    return ax


# Create the four-panel figure
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 12))
fig.subplots_adjust(wspace=0.26, hspace=0.3)
fig.patch.set_facecolor('white')

# ========================
# Chl-a Panel (ax1) with Extra Annotations
# ========================

# Scatter plot (without a label)
im1 = ax1.scatter(chl_data, chl_model, c=chl_depth, alpha=0.7, cmap='viridis_r')
ax1.set_title('(a) Chl-a', fontsize=20, color='darkgreen')
ax1.set_ylabel('Model Chl-a (mg m$^{-3}$)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Data Chl-a (mg m$^{-3}$)', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)

textbox_text = build_stats_text(chl_data, chl_model)
ax1.text(0.05, 0.95, textbox_text, transform=ax1.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
ax1 = plot_reference_lines(ax1, chl_data, chl_model)

ax1.locator_params(nbins=7)
# Add a colorbar for the scatter plot
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.ax.locator_params(nbins=6)
cbar1.set_label("Depth (m)", size=16)
cbar1.ax.tick_params(labelsize=15)

# Add a legend for the lines (1:1 and regression) in the lower right
ax1.legend(loc="lower right", fontsize=12, title=None)


# ========================
# POC Panel (ax2)
# ========================
im2 = ax2.scatter(poc_data, poc_model, c=poc_depth, alpha=0.7, cmap='viridis_r')
ax2.set_title('(b) PC', fontsize=20, color='darkorange')
ax2.set_ylabel('Model PC (mg m$^{-3}$)', fontsize=16)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Data PC (mg m$^{-3}$)', fontsize=16)
ax2.xaxis.set_tick_params(labelsize=15)

textbox_text = build_stats_text(poc_data, poc_model)
ax2.text(0.05, 0.95, textbox_text, transform=ax2.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
ax2 = plot_reference_lines(ax2, poc_data, poc_model)

ax2.locator_params(nbins=8)
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.ax.locator_params(nbins=6)
cbar2.set_label("Depth (m)", size=16)
cbar2.ax.tick_params(labelsize=15)

# Add a legend for the lines (1:1 and regression) in the lower right
ax2.legend(loc="lower right", fontsize=12, title=None)


# ========================
# PON Panel (ax3)
# ========================
im3 = ax3.scatter(pon_data, pon_model, c=poc_depth, alpha=0.7, cmap='viridis_r')
ax3.set_title('(c) PN', fontsize=20, color='m')
ax3.set_ylabel('Model PN (mg m$^{-3}$)', fontsize=16)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Data PN (mg m$^{-3}$)', fontsize=16)
ax3.xaxis.set_tick_params(labelsize=15)
#ax3.legend(loc="upper left", fontsize=10, title='Spearman Correlation')

textbox_text = build_stats_text(pon_data, pon_model)
ax3.text(0.05, 0.95, textbox_text, transform=ax3.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
ax3 = plot_reference_lines(ax3, pon_data, pon_model)

ax3.locator_params(nbins=8)
cbar3 = fig.colorbar(im3, ax=ax3)
cbar3.ax.locator_params(nbins=6)
cbar3.set_label("Depth (m)", size=16)
cbar3.ax.tick_params(labelsize=15)

# Add a legend for the lines (1:1 and regression) in the lower right
ax3.legend(loc="lower right", fontsize=12, title=None)


# ========================
# POP Panel (ax4)
# ========================
im4 = ax4.scatter(pop_data, pop_model, c=pop_depth, alpha=0.7, cmap='viridis_r')
ax4.set_title('(d) PP', fontsize=20, color='c')
ax4.set_ylabel('Model PP (mg m$^{-3}$)', fontsize=16)
ax4.yaxis.set_tick_params(labelsize=15)
ax4.set_xlabel('Data PP (mg m$^{-3}$)', fontsize=16)
ax4.xaxis.set_tick_params(labelsize=15)
#ax4.legend(loc="upper left", fontsize=10, title='Spearman Correlation')

textbox_text = build_stats_text(pop_data, pop_model)
ax4.text(0.05, 0.95, textbox_text, transform=ax4.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
ax4 = plot_reference_lines(ax4, pop_data, pop_model)

# Add a legend for the lines (1:1 and regression) in the lower right
ax4.legend(loc="lower right", fontsize=12, title=None)

ax4.locator_params(nbins=8)
cbar4 = fig.colorbar(im4, ax=ax4)
cbar4.ax.locator_params(nbins=6)
cbar4.set_label("Depth (m)", size=16)
cbar4.ax.tick_params(labelsize=15)

# ========================
# Save and show the final figure
# ========================
fig.savefig('plots/HOT_Scatter_ModelvsData_CHL_POC_PON_POP_Stats.jpeg', 
            format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### Integrated Model vs Data Comparison

#Get Integrated Results from DF
#print(list(bottle_prof.columns))
chl_data  = bottle_prof['Chla_data_Int'].values
chl_model = bottle_prof['Chla_mod_Int'].values

#PC & PN Model Results DF
#bottle_poc.info()
poc_data  = bottle_prof['POC_data_int'].values
poc_model = bottle_prof['POC_mod_total_int'].values
pon_data  = bottle_prof['PON_data_int'].values
pon_model = bottle_prof['PON_mod_total_int'].values

#PP Model Results DF
#bottle_pop.info()
pop_data  = bottle_prof['POP_data_int'].values
pop_model = bottle_prof['POP_mod_total_int'].values

# Create the four-panel figure
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(15, 12))
fig.subplots_adjust(wspace=0.26, hspace=0.3)
fig.patch.set_facecolor('white')

# ========================
# Chl-a Panel (ax1) with Extra Annotations
# ========================

# Scatter plot (without a label)
im1 = ax1.scatter(chl_data, chl_model, c='g', alpha=0.7)
ax1.set_title('(a) Chl-a', fontsize=20, color='darkgreen')
ax1.set_ylabel('Model Chl-a (mg m$^{-2}$)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Data Chl-a (mg m$^{-2}$)', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)

textbox_text = build_stats_text(chl_data, chl_model)
ax1.text(0.05, 0.95, textbox_text, transform=ax1.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax1 = plot_reference_lines(ax1, chl_data, chl_model)

ax1.locator_params(nbins=7)

# Add a legend for the lines (1:1 and regression) in the lower right
ax1.legend(loc="lower right", fontsize=12, title=None)


# ========================
# POC Panel (ax2)
# ========================
im2 = ax2.scatter(poc_data, poc_model, c='orange', alpha=0.7)
ax2.set_title('(b) PC', fontsize=20, color='darkorange')
ax2.set_ylabel('Model PC (g m$^{-2}$)', fontsize=16)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Data PC (g m$^{-2}$)', fontsize=16)
ax2.xaxis.set_tick_params(labelsize=15)

textbox_text = build_stats_text(poc_data, poc_model)
ax2.text(0.05, 0.95, textbox_text, transform=ax2.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax2 = plot_reference_lines(ax2, poc_data, poc_model)

ax2.locator_params(nbins=8)
# Add a legend for the lines (1:1 and regression) in the lower right
ax2.legend(loc="lower right", fontsize=12, title=None)


# ========================
# PON Panel (ax3)
# ========================
im3 = ax3.scatter(pon_data, pon_model, c='m', alpha=0.7)
ax3.set_title('(c) PN', fontsize=20, color='m')
ax3.set_ylabel('Model PN (g m$^{-2}$)', fontsize=16)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Data PN (g m$^{-2}$)', fontsize=16)
ax3.xaxis.set_tick_params(labelsize=15)
#ax3.legend(loc="upper left", fontsize=10, title='Spearman Correlation')

textbox_text = build_stats_text(pon_data, pon_model)
ax3.text(0.05, 0.95, textbox_text, transform=ax3.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax3 = plot_reference_lines(ax3, pon_data, pon_model)

ax3.locator_params(nbins=8)
# Add a legend for the lines (1:1 and regression) in the lower right
ax3.legend(loc="lower right", fontsize=12, title=None)


# ========================
# POP Panel (ax4)
# ========================
im4 = ax4.scatter(pop_data, pop_model, c='c', alpha=0.7)
ax4.set_title('(d) PP', fontsize=20, color='c')
ax4.set_ylabel('Model PP (mg m$^{-2}$)', fontsize=16)
ax4.yaxis.set_tick_params(labelsize=15)
ax4.set_xlabel('Data PP (mg m$^{-2}$)', fontsize=16)
ax4.xaxis.set_tick_params(labelsize=15)
#ax4.legend(loc="upper left", fontsize=10, title='Spearman Correlation')

textbox_text = build_stats_text(pop_data, pop_model)
ax4.text(0.05, 0.95, textbox_text, transform=ax4.transAxes, fontsize=13,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
ax4 = plot_reference_lines(ax4, pop_data, pop_model)

# Add a legend for the lines (1:1 and regression) in the lower right
ax4.legend(loc="lower right", fontsize=12, title=None)

ax4.locator_params(nbins=8)

# ========================
# Save and show the final figure
# ========================
fig.savefig('plots/HOT_Scatter_ModelvsData_Int_Stats.jpeg', 
            format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### MODEL-DATA DIFFERENCE - CHL

### FIGURE NOT USED IN METHODS PAPER

# ── Adjustable Parameters ────────────────────────────────────────────────
bin_size       = 20           # Depth bin size in meters
max_depth      = 220          # Maximum depth shown
label_fontsize = 13           # Axis label font size
tick_fontsize  = 12           # Tick label font size
title_fontsize = 15           # Title font size

# ── Data Columns ─────────────────────────────────────────────────────────
depth_col = 'depth'
model_col = 'CHL_mod_total'
data_col  = 'Chla'

# ── Plot Settings ────────────────────────────────────────────────────────
figsize    = (6, 8)
zero_line  = dict(color='gray', linestyle='--', linewidth=1.5)
box_width  = 15

# ── 1. Bin Definitions ───────────────────────────────────────────────────
bin_edges = np.arange(0, max_depth + bin_size, bin_size)   # 0, 20, ..., 220
bin_mids  = bin_edges[:-1] + bin_size / 2                  # 10, 30, ..., 210

# ── 2. Assign Bins and Calculate Difference ──────────────────────────────
bottle_6['depth_bin'] = pd.cut(
    bottle_6[depth_col],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True
)
bottle_6['diff'] = bottle_6[model_col] - bottle_6[data_col]

#print(np.nanmin(bottle_6['diff']))

# ── 3. Gather Data Per Bin ───────────────────────────────────────────────
boxplot_data = [
    bottle_6.loc[bottle_6['depth_bin'] == mid, 'diff'].dropna().values
    for mid in bin_mids
]

# ── 4. Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=figsize)
# Zero line
ax.axvline(0, **zero_line)

ax.boxplot(
    boxplot_data,
    vert=False,
    positions=bin_mids,
    widths=box_width
)

# Y-axis ticks and direction
yticks = np.arange(0, max_depth + 1, bin_size)
ax.set_ylim(max_depth, 0)
ax.set_yticks(yticks)
ax.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax.tick_params(axis='x', labelsize=tick_fontsize)


# Labels and title directly in plotting section
ax.set_xlabel('Chl-a Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax.set_title('(a)', fontsize=title_fontsize)

# Layout and save
plt.tight_layout()
plt.savefig('plots/HOT_ModelDifference_Chl.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#%%

### MODEL-DATA DIFFERENCE - ALL

# ── Adjustable Parameters ────────────────────────────────────────────────
bin_size       = 20           # Depth bin size in meters
max_depth      = 220          # Maximum depth shown
label_fontsize = 14           # Axis label font size
tick_fontsize  = 12           # Tick label font size
title_fontsize = 16           # Title font size
show_outliers = True
line_thickness = 1.5

# ── Plot Settings ────────────────────────────────────────────────────────
zero_line  = dict(color='k', linestyle='--', linewidth=1.8)
box_width  = 12

# 1. Bin Definitions ───────────────────────────────────────────────────
bin_edges = np.arange(0, max_depth + bin_size, bin_size)   # 0, 20, ..., 220
bin_mids  = bin_edges[:-1] + bin_size / 2                  # 10, 30, ..., 210

# 2. Assign Bins and Calculate Difference
#CHL
bottle_6['depth_bin'] = pd.cut(
    bottle_6['depth'],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True)
bottle_6['diff'] = bottle_6['CHL_mod_total'] - bottle_6['Chla']
#PC
bottle_poc['depth_bin'] = pd.cut(
    bottle_poc['depth'],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True)
bottle_poc['diff_PC'] = bottle_poc['POC_mod_total'] - bottle_poc['POC']
#PN
bottle_poc['depth_bin'] = pd.cut(
    bottle_poc['depth'],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True)
bottle_poc['diff_PN'] = bottle_poc['PON_mod_total'] - bottle_poc['PON']
#PP
bottle_pop['depth_bin'] = pd.cut(
    bottle_pop['depth'],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True)
bottle_pop['diff_PP'] = bottle_pop['POP_mod_total'] - bottle_pop['POP']

# 3. Gather Data Per Bin
boxplot_data_chl = [
    bottle_6.loc[bottle_6['depth_bin'] == mid, 'diff'].dropna().values
    for mid in bin_mids]
boxplot_data_poc = [
    bottle_poc.loc[bottle_poc['depth_bin'] == mid, 'diff_PC'].dropna().values
    for mid in bin_mids]
boxplot_data_pon = [
    bottle_poc.loc[bottle_poc['depth_bin'] == mid, 'diff_PN'].dropna().values
    for mid in bin_mids]
boxplot_data_pop = [
    bottle_pop.loc[bottle_pop['depth_bin'] == mid, 'diff_PP'].dropna().values
    for mid in bin_mids]


# 4. Plot 
fig, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2,figsize=(12,14))
fig.subplots_adjust(wspace=0.3,hspace=0.3)
# CHL
ax1.axvline(0, **zero_line)
bp = ax1.boxplot(
    boxplot_data_chl,
    vert=False,
    positions=bin_mids,
    widths=box_width,
    showfliers = show_outliers,
    patch_artist=True,                                 # <-- allow facecolor
    boxprops=dict(facecolor='green', edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
    flierprops=dict(markeredgecolor='black', markersize=5)           # only used if show_outliers=True
)

# Y-axis ticks and direction
yticks = np.arange(0, max_depth + 1, bin_size)
ax1.set_ylim(max_depth, 0)
ax1.set_yticks(yticks)
ax1.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

# Labels and title directly in plotting section
ax1.set_xlabel('Chl-a Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax1.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax1.set_title('(a)', fontsize=title_fontsize)

# PC
ax2.axvline(0, **zero_line)
bp = ax2.boxplot(
    boxplot_data_poc,
    vert=False,
    positions=bin_mids,
    widths=box_width,
    showfliers = show_outliers,
    patch_artist=True,                                 # <-- allow facecolor
    boxprops=dict(facecolor='darkorange', edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
    flierprops=dict(markeredgecolor='black', markersize=5)           # only used if show_outliers=True
)

# Y-axis ticks and direction
yticks = np.arange(0, max_depth + 1, bin_size)
ax2.set_ylim(max_depth, 0)
ax2.set_yticks(yticks)
ax2.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax2.tick_params(axis='x', labelsize=tick_fontsize)

# Labels and title directly in plotting section
ax2.set_xlabel('PC Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax2.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax2.set_title('(b)', fontsize=title_fontsize)

# PN
ax3.axvline(0, **zero_line)
bp = ax3.boxplot(
    boxplot_data_pon,
    vert=False,
    positions=bin_mids,
    widths=box_width,
    showfliers = show_outliers,
    patch_artist=True,                                 # <-- allow facecolor
    boxprops=dict(facecolor='m', edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
    flierprops=dict(markeredgecolor='black', markersize=5)           # only used if show_outliers=True
)

# Y-axis ticks and direction
yticks = np.arange(0, max_depth + 1, bin_size)
ax3.set_ylim(max_depth, 0)
ax3.set_yticks(yticks)
ax3.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax3.tick_params(axis='x', labelsize=tick_fontsize)

# Labels and title directly in plotting section
ax3.set_xlabel('PN Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax3.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax3.set_title('(c)', fontsize=title_fontsize)

# PP
ax4.axvline(0, **zero_line)
bp = ax4.boxplot(
    boxplot_data_pop,
    vert=False,
    positions=bin_mids,
    widths=box_width,
    showfliers = show_outliers,
    patch_artist=True,                                 # <-- allow facecolor
    boxprops=dict(facecolor='c', edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
    flierprops=dict(markeredgecolor='black', markersize=5)           # only used if show_outliers=True
)

# Y-axis ticks and direction
yticks = np.arange(0, max_depth + 1, bin_size)
ax4.set_ylim(max_depth, 0)
ax4.set_yticks(yticks)
ax4.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax4.tick_params(axis='x', labelsize=tick_fontsize)

# Labels and title directly in plotting section
ax4.set_xlabel('PP Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax4.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax4.set_title('(d)', fontsize=title_fontsize)

# Layout and save
fig.savefig('plots/HOT_ModelDifference_All.jpeg', dpi=300, bbox_inches='tight')
plt.show()


#%%

# ── Adjustable Parameters ────────────────────────────────────────────────
bin_size       = 20           # Depth bin size in meters
max_depth      = 220          # Maximum depth shown
label_fontsize = 13           # Axis label font size
tick_fontsize  = 12           # Tick label font size
title_fontsize = 15           # Title font size

# ── Data Columns ─────────────────────────────────────────────────────────
depth_col = 'depth'
model_col = 'CHL_mod_total'
data_col  = 'Chla'

# ── Plot Settings ────────────────────────────────────────────────────────
figsize    = (12, 8)  # wider to accommodate both subplots
zero_line  = dict(color='gray', linestyle='--', linewidth=1.5)
box_width  = 15

# ── 1. Bin Definitions ───────────────────────────────────────────────────
bin_edges = np.arange(0, max_depth + bin_size, bin_size)   # 0, 20, ..., 220
bin_mids  = bin_edges[:-1] + bin_size / 2                  # 10, 30, ..., 210

# ── 2. Assign Bins and Calculate Differences ─────────────────────────────
bottle_6['depth_bin'] = pd.cut(
    bottle_6[depth_col],
    bins=bin_edges,
    labels=bin_mids,
    include_lowest=True
)
bottle_6['diff'] = bottle_6[model_col] - bottle_6[data_col]
bottle_6['rel_diff'] = bottle_6['diff'] / bottle_6[data_col]  # relative difference

# ── 3. Gather Data Per Bin ───────────────────────────────────────────────
abs_diff_data = [
    bottle_6.loc[bottle_6['depth_bin'] == mid, 'diff'].dropna().values
    for mid in bin_mids
]
rel_diff_data = [
    bottle_6.loc[bottle_6['depth_bin'] == mid, 'rel_diff'].dropna().values
    for mid in bin_mids
]

# ── 4. Plot ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

# Absolute Difference Plot
ax1.axvline(0, **zero_line)
ax1.boxplot(abs_diff_data, vert=False, positions=bin_mids, widths=box_width)
ax1.set_ylim(max_depth, 0)
yticks = np.arange(0, max_depth + 1, bin_size)
ax1.set_yticks(yticks)
ax1.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)
ax1.set_xlabel('Chl-a Difference (Model − Data)\n(mg m$^{-3}$)', fontsize=label_fontsize)
ax1.set_ylabel('Depth (m)', fontsize=label_fontsize)
ax1.set_title('(a)', fontsize=title_fontsize)

# Relative Difference Plot
ax2.axvline(0, **zero_line)
ax2.boxplot(rel_diff_data, vert=False, positions=bin_mids, widths=box_width)
ax2.set_ylim(max_depth, 0)
ax2.set_yticks(yticks)  # shared y-axis, but set ticklabels only once
ax2.set_yticklabels([str(y) for y in yticks], fontsize=tick_fontsize)
ax2.tick_params(axis='x', labelsize=tick_fontsize)
ax2.set_xlabel('Relative Difference\n(Model − Data) / Data', fontsize=label_fontsize)
ax2.set_title('(b)', fontsize=title_fontsize)

# Layout and save
plt.tight_layout()
plt.savefig('plots/HOT_ModelDifference_Chl_Relative.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#%%

### SUMMARY TABLES FOR METHOD PAPER - Concentrations & Integrated

#bottle_prof.info()

#print(bottle_prof.columns.tolist())
bottle_prof['Chla_int_sub_ratio'] = bottle_prof['Chla_mod_sub_Int']/bottle_prof['Chla_mod_Int']
bottle_prof['PhytoC_int_sub_ratio'] = bottle_prof['POC_mod_sub_int']/bottle_prof['PhytoC_int_discreet']
bottle_prof['PhytoN_int_sub_ratio'] = bottle_prof['PON_mod_sub_int']/bottle_prof['PhytoN_int_discreet']
bottle_prof['PhytoP_int_sub_ratio'] = bottle_prof['POP_mod_sub_int']/bottle_prof['PhytoP_int_discreet']

print("=== Subsurface contribution to Phyto stocks ====")
sub_Chl_ratio    = round(bottle_prof['Chla_int_sub_ratio'].median(),2)
sub_Cphyto_ratio = round(bottle_prof['PhytoC_int_sub_ratio'].median(),2)
sub_Nphyto_ratio = round(bottle_prof['PhytoN_int_sub_ratio'].median(),2)
sub_Pphyto_ratio = round(bottle_prof['PhytoP_int_sub_ratio'].median(),2)

print("Subsurface Chl vs Total Chl     = ",sub_Chl_ratio)
print("Subsurface Carbon vs Cphyto     = ",sub_Cphyto_ratio)
print("Subsurface Nitrogen vs Nphyto   = ",sub_Nphyto_ratio)
print("Subsurface Phosphorus vs Pphyto = ",sub_Pphyto_ratio)

bottle_prof[['Chla_int_sub_ratio','PhytoC_int_sub_ratio','PhytoN_int_sub_ratio','PhytoP_int_sub_ratio']].describe()


ratio_list = ['DCM_depth','chla_mod_surf_conc','chla_mod_sub_conc',
              'POC_mod_surf_conc','POC_mod_sub_conc','POC_mod_bk',
              'PON_mod_surf_conc','PON_mod_sub_conc','PON_mod_bk',
              'POP_mod_surf_conc','POP_mod_sub_conc','POP_mod_bk',
              'Chla_data_Int','Chla_mod_Int', 'Chla_mod_surf_Int', 'Chla_mod_sub_Int',
              'POC_data_int','POC_mod_total_int','PhytoC_int_discreet', 'POC_mod_surf_int', 'POC_mod_sub_int',
              'PON_data_int','PON_mod_total_int','PhytoN_int_discreet', 'PON_mod_surf_int', 'PON_mod_sub_int',
              'POP_data_int','POP_mod_total_int','PhytoP_int_discreet', 'POP_mod_surf_int', 'POP_mod_sub_int',
              'PhytoC_ratio_median_mld', 'PhytoC_ratio_dcm',
              'PhytoN_ratio_median_mld', 'PhytoN_ratio_dcm',
              'PhytoP_ratio_median_mld', 'PhytoP_ratio_dcm']
#biomass_list

# Compute summary statistics
summary_stats = bottle_prof[ratio_list].describe()

# Compute MAD for each column and add as a new row
mad_values = bottle_prof[ratio_list].apply(lambda x: MAD(x, nan_policy='omit'))

# Convert MAD to a DataFrame row
mad_df = pd.DataFrame([mad_values], index=['MAD'])

# Insert MAD as the fourth row
summary_stats = pd.concat([summary_stats.iloc[:3], mad_df, summary_stats.iloc[3:]])

# Round all values to 2 decimal places
summary_stats = summary_stats.round(2)

# Transpose for portrait (so original columns become rows)
summary_stats = summary_stats.T

# Create rename dictionary for cleaner row names
rename_dict = {
    # Depth
    'DCM_depth': 'DCM depth (m)',
    
    # Chlorophyll concentrations
    'chla_mod_surf_conc': 'Chl-a Model Surface Conc',
    'chla_mod_sub_conc': 'Chl-a Model Subsurface Conc',
    
    # POC concentrations
    'POC_mod_surf_conc': 'PC Model PhytoC-Surface Conc',
    'POC_mod_sub_conc': 'PC Model PhytoC-Subsurface Conc',
    'POC_mod_bk': 'PC Model Non-Algal Bk Conc',
    
    # PON concentrations
    'PON_mod_surf_conc': 'PN Model PhytoN-Surface Conc',
    'PON_mod_sub_conc': 'PN Model PhytoN-Subsurface Conc',
    'PON_mod_bk': 'PN Model Non-Algal Bk Conc',
    
    # POP concentrations
    'POP_mod_surf_conc': 'PP Model PhytoP-Surface Conc',
    'POP_mod_sub_conc': 'PP Model PhytoP-Subsurface Conc',
    'POP_mod_bk': 'PP Model Non-Algal Bk Conc',
    
    # DEPTH-INTEGRATED
    
    # Chlorophyll integrals
    'Chla_data_Int': 'Chl-a Data integral',
    'Chla_mod_Int': 'Chl-a Model Total Int',
    'Chla_mod_surf_Int': 'Chl-a Model Surface Int',
    'Chla_mod_sub_Int': 'Chl-a Model Subsurface Int',
    
    # POC integrals
    'POC_data_int': 'PC Data Int',
    'POC_mod_total_int': 'PC Model Total Int',
    'PhytoC_int_discreet': 'PC Model PhytoC Int',
    'POC_mod_surf_int': 'PC Model Surface Int',
    'POC_mod_sub_int': 'PC Model Subsurface Int',
    
    # PON integrals
    'PON_data_int': 'PN Data Int',
    'PON_mod_total_int': 'PN Model Total Int',
    'PhytoN_int_discreet': 'PN Model PhytoN Int',
    'PON_mod_surf_int': 'PN Model Surface Int',
    'PON_mod_sub_int': 'PN Model Subsurface Int',
    
    # POP integrals
    'POP_data_int': 'PP Data Int',
    'POP_mod_total_int': 'PP Model Total Int',
    'PhytoP_int_discreet': 'PP Model PhytoP Int',
    'POP_mod_surf_int': 'PP Model Surface Int',
    'POP_mod_sub_int': 'PP Model Subsurface Int',
    
    # Ratios at MLD and DCM
    'PhytoC_ratio_median_mld': 'PhytoC:PC ML-median',
    'PhytoC_ratio_dcm': 'PhytoC:PC DCM',
    'PhytoN_ratio_median_mld': 'PhytoN:P ML-median',
    'PhytoN_ratio_dcm': 'PhytoN:PN DCM',
    'PhytoP_ratio_median_mld': 'PhytoP:PP ML-median',
    'PhytoP_ratio_dcm': 'PhytoP:PP DCM'
}

units_dict = {
    # Depth
    'DCM_depth': 'm',
    
    # Chlorophyll concentrations
    'chla_mod_surf_conc': 'mg m⁻³',
    'chla_mod_sub_conc': 'mg m⁻³',
    
    # POC concentrations
    'POC_mod_surf_conc': 'mg m⁻³',
    'POC_mod_sub_conc': 'mg m⁻³',
    'POC_mod_bk': 'mg m⁻³',
    
    # PON concentrations
    'PON_mod_surf_conc': 'mg m⁻³',
    'PON_mod_sub_conc': 'mg m⁻³',
    'PON_mod_bk': 'mg m⁻³',
    
    # POP concentrations
    'POP_mod_surf_conc': 'mg m⁻³',
    'POP_mod_sub_conc': 'mg m⁻³',
    'POP_mod_bk': 'mg m⁻³',
    
    # Chlorophyll integrals
    'Chla_data_Int': 'mg m⁻²',
    'Chla_mod_Int': 'mg m⁻²',
    'Chla_mod_surf_Int': 'mg m⁻²',
    'Chla_mod_sub_Int': 'mg m⁻²',
    
    # POC integrals
    'POC_data_int': 'g m⁻²',
    'POC_mod_total_int': 'g m⁻²',
    'PhytoC_int_discreet': 'g m⁻²',
    'POC_mod_surf_int': 'g m⁻²',
    'POC_mod_sub_int': 'g m⁻²',
    
    # PON integrals
    'PON_data_int': 'g m⁻²',
    'PON_mod_total_int': 'g m⁻²',
    'PhytoN_int_discreet': 'g m⁻²',
    'PON_mod_surf_int': 'g m⁻²',
    'PON_mod_sub_int': 'g m⁻²',
    
    # POP integrals
    'POP_data_int': 'mg m⁻²',
    'POP_mod_total_int': 'mg m⁻²',
    'PhytoP_int_discreet': 'mg m⁻²',
    'POP_mod_surf_int': 'mg m⁻²',
    'POP_mod_sub_int': 'mg m⁻²',
    
    # Ratios (dimensionless)
    'PhytoC_ratio_median_mld': '–',
    'PhytoC_ratio_dcm': '–',
    'PhytoN_ratio_median_mld': '–',
    'PhytoN_ratio_dcm': '–',
    'PhytoP_ratio_median_mld': '–',
    'PhytoP_ratio_dcm': '–'
}

# Save to CSV
summary_stats.to_csv("Tables/HOT_Summary_Stats_Methods.csv")

# Print summary stats
print(summary_stats)

#%%

# 2. Compute the basic describe() table
descr = bottle_prof[ratio_list].describe()

# 3. Extract the needed series
count = descr.loc['count']
mean  = descr.loc['mean']
std   = descr.loc['std']
median= descr.loc['50%']
q25   = descr.loc['25%']
q75   = descr.loc['75%']
mn    = descr.loc['min']
mx    = descr.loc['max']
mad   = bottle_prof[ratio_list].apply(lambda x: MAD(x, nan_policy='omit'))

# 4. Build the final DataFrame, one row per variable
stats_table = pd.DataFrame(index=ratio_list)

# count as integer
stats_table['count'] = count.round(0).astype(int)

# median ± MAD
stats_table['Median±MAD'] = median.map(lambda v: f"{v:.2f}") + " ± " + mad.map(lambda d: f"{d:.2f}")

# mean ± std
stats_table['Mean±Std']   = mean.map(lambda v: f"{v:.2f}")  + " ± " + std.map(lambda s: f"{s:.2f}")

# range min–max
stats_table['Range']      = mn.map(lambda v: f"{v:.2f}")    + " – "  + mx.map(lambda v: f"{v:.2f}")

# 25th and 75th percentiles, numeric
stats_table['25%']        = q25.round(2)
stats_table['75%']        = q75.round(2)

# Add Units column (with superscripts)
stats_table.insert(0, "Units", stats_table.index.map(units_dict))

# Apply the same renaming dictionary for cleaner row labels
stats_table.rename(index=rename_dict, inplace=True)

# 5. Save and print
stats_table.to_csv("Tables/HOT_Summary_Stats_Custom_Paper.csv")
stats_table.to_excel("Tables/HOT_Summary_Stats_Custom_Paper.xlsx", index=True)
print(stats_table)

#%%

### ML vs DCM BOXPLOTS - Temp, Nitrate and PAR

#bottle_prof.info()

# Font size settings
label_fontsize = 18
tick_fontsize = 16
title_fontsize = 22

# Extract data
DATA_temp_surface = bottle_prof['Temp_ML'].to_numpy()
DATA_temp_dcm = bottle_prof['Temp_DCM'].to_numpy()  # same as surface

DATA_nitrate_surface = bottle_prof['Nitrate_ML_median'].to_numpy()
DATA_nitrate_dcm = bottle_prof['Nitrate_DCM_conc'].to_numpy()

DATA_par_surface = bottle_prof['PAR_MLD'].to_numpy()
DATA_par_dcm = bottle_prof['PAR_DCM'].to_numpy()

# Set up the figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))
fig.patch.set_facecolor('white')
fig.subplots_adjust(wspace=0.5, hspace=0.5)

line_width = 1.2
box_colors = ['r', 'royalblue']

# Temperature
box1 = ax1.boxplot(
    [DATA_temp_surface[~np.isnan(DATA_temp_surface)], DATA_temp_dcm[~np.isnan(DATA_temp_dcm)]],
    positions=[-1.7, 0.0], widths=0.35, patch_artist=True,
    boxprops=dict(color='black', linewidth=line_width),
    whiskerprops=dict(color='black', linewidth=line_width),
    capprops=dict(color='black', linewidth=line_width),
    medianprops=dict(color='k', linewidth=line_width)
)
ax1.set_xticks([-1.7, 0.0])
ax1.set_xticklabels(['ML', 'DCM'], fontsize=tick_fontsize)
ax1.set_ylabel('Temperature ($^o$C)', fontsize=label_fontsize)
ax1.set_title('(a)', fontsize=title_fontsize)
ax1.tick_params(axis='y', labelsize=tick_fontsize)
ax1.grid(True, linestyle='--', linewidth=0.5)
for i, patch in enumerate(box1['boxes']):
    patch.set_facecolor(box_colors[i])

# Nitrate
box2 = ax2.boxplot(
    [DATA_nitrate_surface[~np.isnan(DATA_nitrate_surface)], DATA_nitrate_dcm[~np.isnan(DATA_nitrate_dcm)]],
    positions=[-1.7, 0.0], widths=0.35, patch_artist=True,
    boxprops=dict(color='black', linewidth=line_width),
    whiskerprops=dict(color='black', linewidth=line_width),
    capprops=dict(color='black', linewidth=line_width),
    medianprops=dict(color='k', linewidth=line_width)
)
ax2.set_xticks([-1.7, 0.0])
ax2.set_xticklabels(['ML', 'DCM'], fontsize=tick_fontsize)
ax2.set_ylabel('Nitrate (nmol L$^{-1}$)', fontsize=label_fontsize)
ax2.set_title('(b)', fontsize=title_fontsize)
ax2.tick_params(axis='y', labelsize=tick_fontsize)
ax2.grid(True, linestyle='--', linewidth=0.5)
for i, patch in enumerate(box2['boxes']):
    patch.set_facecolor(box_colors[i])

# PAR
box3 = ax3.boxplot(
    [DATA_par_surface[~np.isnan(DATA_par_surface)], DATA_par_dcm[~np.isnan(DATA_par_dcm)]],
    positions=[-1.7, 0.0], widths=0.35, patch_artist=True,
    boxprops=dict(color='black', linewidth=line_width),
    whiskerprops=dict(color='black', linewidth=line_width),
    capprops=dict(color='black', linewidth=line_width),
    medianprops=dict(color='k', linewidth=line_width)
)
ax3.set_xticks([-1.7, 0.0])
ax3.set_xticklabels(['ML', 'DCM'], fontsize=tick_fontsize)
ax3.set_ylabel('PAR (mol quanta m$^{-2}$ d$^{-1}$)', fontsize=label_fontsize)
ax3.set_title('(c)', fontsize=title_fontsize)
ax3.tick_params(axis='y', labelsize=tick_fontsize)
ax3.grid(True, linestyle='--', linewidth=0.5)
ax3.set_yscale('log')
for i, patch in enumerate(box3['boxes']):
    patch.set_facecolor(box_colors[i])

# Save
fig.savefig('plots/HOT_Boxplot_TempNitratePAR_SurfaceDCM_paper.jpeg', dpi=300, bbox_inches='tight')
#fig.savefig('plots/HOT_Boxplot_TempNitratePAR_SurfaceDCM.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%

### BOXPLOTS - CONCENTRATION RESULTS

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

# Plot styling variables
labsize = 20
titlesize = 25
ticksize = 18
legendsize = 12
line_thickness = 1.5
box_width = 0.3
show_outliers = True

# Redfield C:N ratio
redfield_ratio = 106 / 16

# Melt data for C:Chl and C:N
df_chl      = bottle_prof[['chla_mod_surf_conc', 'chla_mod_sub_conc']].melt(var_name='Measurement', value_name='Chla')
df_carbon   = bottle_prof[['POC_mod_surf_conc','POC_mod_sub_conc','POC_mod_bk',]].melt(var_name='Measurement', value_name='Carbon')
df_nitrogen = bottle_prof[['PON_mod_surf_conc','PON_mod_sub_conc','PON_mod_bk',]].melt(var_name='Measurement', value_name='Nitrogen')
df_phos = bottle_prof[['POP_mod_surf_conc','POP_mod_sub_conc','POP_mod_bk',]].melt(var_name='Measurement', value_name='Phosphorus')

# Create figure with 3 subplots
fig, ([ax1, ax2], [ax3,ax4]) = plt.subplots(ncols=2,nrows=2, figsize=(16, 12))#, gridspec_kw={'width_ratios': [1, 1, 1,1]}
fig.subplots_adjust(wspace=0.3,hspace=0.3)
# --- Chl subplot ---
sns.boxplot(
    x='Measurement',
    y='Chla',
    data=df_chl,
    palette=["r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=0.22,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax1.set_xticklabels(['Surface', 'Subsurface'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylabel("Chl-a (mg m$^{-3}$)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a) Chl-a", fontsize=titlesize, color = 'g')
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Carbon subplot ---
sns.boxplot(
    x='Measurement',
    y='Carbon',
    data=df_carbon,
    palette=["r", "royalblue", "orchid"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax2,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax2.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax2.set_xlabel('')
ax2.set_ylabel("Carbon (mg m$^{-3}$)", fontsize=labsize)
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b) Carbon", fontsize=titlesize, color = 'darkorange')
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Nitrogen subplot ---
sns.boxplot(
    x='Measurement',
    y='Nitrogen',
    data=df_nitrogen,
    palette=["r", "royalblue", "orchid"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax3,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax3.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax3.set_xlabel('')
ax3.set_ylabel("Nitrogen (mg m$^{-3}$)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c) Nitrogen", fontsize=titlesize, color = 'm')
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Phosphorus subplot ---
sns.boxplot(
    x='Measurement',
    y='Phosphorus',
    data=df_phos,
    palette=["r", "royalblue", "orchid"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax4.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax4.set_xlabel('')
ax4.set_ylabel("Phosphorus (mg m$^{-3}$)", fontsize=labsize)
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d) Phosphorus", fontsize=titlesize, color = 'c')
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
# Limit y-axis ticks to a maximum of 5
ax4.yaxis.set_major_locator(MaxNLocator(nbins=4))
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# Final layout adjustments and saving
#plt.tight_layout()
plt.savefig('plots/HOT_Boxplots_Conc_Chl_Carbon_Nitrogen_Phos_paper.jpeg',
            format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### BOXPLOTS - INTEGRATED RESULTS

#print(bottle_prof.columns.tolist())

# Plot styling variables
labsize = 20
titlesize = 25
ticksize = 16
legendsize = 12
line_thickness = 1.5
box_width = 0.36
show_outliers = True

# Redfield C:N ratio
redfield_ratio = 106 / 16

# Melt data for C:Chl and C:N
df_chl      = bottle_prof[['Chla_data_Int','Chla_mod_Int', 'Chla_mod_surf_Int', 'Chla_mod_sub_Int']].melt(var_name='Measurement', value_name='Chla')
df_carbon   = bottle_prof[['POC_data_int','POC_mod_total_int','PhytoC_int_discreet','POC_mod_surf_int', 'POC_mod_sub_int']].melt(var_name='Measurement', value_name='Carbon')
df_nitrogen = bottle_prof[['PON_data_int','PON_mod_total_int','PhytoN_int_discreet','PON_mod_surf_int', 'PON_mod_sub_int']].melt(var_name='Measurement', value_name='Nitrogen')
df_phos     = bottle_prof[['POP_data_int','POP_mod_total_int','PhytoP_int_discreet','POP_mod_surf_int', 'POP_mod_sub_int']].melt(var_name='Measurement', value_name='Phosphorus')

# Create figure with 3 subplots
fig, ([ax1, ax2], [ax3,ax4]) = plt.subplots(ncols=2,nrows=2, figsize=(16, 12))#, gridspec_kw={'width_ratios': [1, 1, 1,1]}
fig.subplots_adjust(wspace=0.3,hspace=0.3)
# --- Chl subplot ---
sns.boxplot(
    x='Measurement',
    y='Chla',
    data=df_chl,
    palette=['gray',"g","r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=0.3,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax1.set_xticklabels(['Chl$_{data}$','Chl$_{model}$','Surface', 'Subsurface'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylabel("Chl-a (mg m$^{-2}$)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a) Chl-a", fontsize=titlesize, color = 'g')
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Carbon subplot ---
sns.boxplot(
    x='Measurement',
    y='Carbon',
    data=df_carbon,
    palette=['gray',"darkorange","g","r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax2,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax2.set_xticklabels(['PC$_{data}$','PC$_{model}$','C$_{phyto}$','Surface', 'Subsurface'], fontsize=ticksize)
ax2.set_xlabel('')
ax2.set_ylabel("Carbon (g m$^{-2}$)", fontsize=labsize)
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b) Carbon", fontsize=titlesize, color = 'darkorange')
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Nitrogen subplot ---
sns.boxplot(
    x='Measurement',
    y='Nitrogen',
    data=df_nitrogen,
    palette=["gray","m","g","r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax3,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax3.set_xticklabels(['PN$_{data}$','PN$_{model}$','N$_{phyto}$','Surface', 'Subsurface'], fontsize=ticksize)
ax3.set_xlabel('')
ax3.set_ylabel("Nitrogen (g m$^{-2}$)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c) Nitrogen", fontsize=titlesize, color = 'm')
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Phosphorus subplot ---
sns.boxplot(
    x='Measurement',
    y='Phosphorus',
    data=df_phos,
    palette=["gray","c","g","r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax4.set_xticklabels(['PP$_{data}$','PP$_{model}$','P$_{phyto}$','Surface', 'Subsurface'], fontsize=ticksize)
ax4.set_xlabel('')
ax4.set_ylabel("Phosphorus (mg m$^{-2}$)", fontsize=labsize)
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d) Phosphorus", fontsize=titlesize, color = 'c')
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
#ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# Final layout adjustments and saving
#plt.tight_layout()
plt.savefig('plots/HOT_Boxplots_Int_Chl_Carbon_Nitrogen_Phos_paper.jpeg',
            format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### SUMMARY STATS TABLE - RATIOS ###

bottle_prof.info()

# Compute the ratio column
bottle_prof["C_Chl_ratio_C1_C2"] = bottle_prof["C_Chl_ratio_C1"] / bottle_prof["C_Chl_ratio_C2"]
bottle_prof["C_N_molar_C1_C2"] = bottle_prof["C_N_molar_C1"] / bottle_prof["C_N_molar_C2"]
# Compute the ratio column
bottle_prof["C_P_molar_C1_C2"] = bottle_prof["C_P_molar_C1"] / bottle_prof["C_P_molar_C2"]
# Compute the ratio column
bottle_prof["N_P_molar_C1_C2"] = bottle_prof["N_P_molar_C1"] / bottle_prof["N_P_molar_C2"]

#print(bottle_prof.columns.tolist())

ratio_list = ['Temp_ML', 'Temp_DCM', 'Nitrate_ML_median', 'Nitrate_DCM_conc', 'PAR_MLD', 'PAR_DCM',
              'C_Chl_ratio_C1', 'C_Chl_ratio_C2','C_N_molar_C1', 'C_N_molar_C2','C_N_molar_C1_C2',
              'C_P_molar_C1', 'C_P_molar_C2', 'N_P_molar_C1', 'N_P_molar_C2']
#biomass_list

# Compute summary statistics
summary_stats = bottle_prof[ratio_list].describe()

# Compute MAD for each column and add as a new row
mad_values = bottle_prof[ratio_list].apply(lambda x: MAD(x, nan_policy='omit'))

# Convert MAD to a DataFrame row
mad_df = pd.DataFrame([mad_values], index=['MAD'])

# Insert MAD as the fourth row
summary_stats = pd.concat([summary_stats.iloc[:3], mad_df, summary_stats.iloc[3:]])

# Round all values to 2 decimal places
summary_stats = summary_stats.round(2)

# Save to CSV
summary_stats.to_csv("Tables/HOT_Env_Ratio_Summary_Stats.csv")

# Print summary stats
print(summary_stats)

#%%


# 1. Define your rows → (surface, subsurface, combined) columns
params = {
    'Temp':               ('Temp_ML',            'Temp_DCM',           None),
    'Nitrate':            ('Nitrate_ML_median',  'Nitrate_DCM_conc',   None),
    'PAR':                ('PAR_MLD',            'PAR_DCM',            None),
    'C:Chl ratio':        ('C_Chl_ratio_C1',     'C_Chl_ratio_C2',     'C_Chl_ratio_C1_C2'),
    'C:N ratio':          ('C_N_molar_C1',       'C_N_molar_C2',       'C_N_molar_C1_C2'),
    'C:P ratio':          ('C_P_molar_C1',       'C_P_molar_C2',       'C_P_molar_C1_C2'),
    'N:P ratio':          ('N_P_molar_C1',       'N_P_molar_C2',       'N_P_molar_C1_C2'),
}

# 2. Prepare an empty DataFrame
table = pd.DataFrame(index=params.keys(), columns=['Surface', 'Subsurface', 'Surface/Subsurface'])

# 3. Fill in each row
for name, (surf_col, sub_col, comb_col) in params.items():
    # Surface
    m_s = bottle_prof[surf_col].median()
    d_s = MAD(bottle_prof[surf_col], nan_policy='omit')
    table.at[name, 'Surface'] = f"{m_s:.2f} ± {d_s:.2f}"

    # Subsurface
    m_sub = bottle_prof[sub_col].median()
    d_sub = MAD(bottle_prof[sub_col], nan_policy='omit')
    table.at[name, 'Subsurface'] = f"{m_sub:.2f} ± {d_sub:.2f}"

    # Surface/Subsurface
    if comb_col is not None:
        # use the combined column for ratio‐rows
        m_c = bottle_prof[comb_col].median()
        d_c = MAD(bottle_prof[comb_col], nan_policy='omit')
        table.at[name, 'Surface/Subsurface'] = f"{m_c:.2f} ± {d_c:.2f}"
    else:
        # leave blank for non‐ratio rows
        table.at[name, 'Surface/Subsurface'] = ""

# 4. Save or display
table.to_csv("Tables/HOT_Env_Surface_Subsurface_Ratios.csv", index=True)
table.to_excel("Tables/HOT_Env_Surface_Subsurface_Ratios.xlsx", index=True)
print(table)

#%%

### PAPER FIGURE 6 - C:Chl and C:N

print(bottle_prof.columns.tolist())

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plot styling variables
labsize = 19
titlesize = 23
ticksize = 17
legendsize = 11
line_thickness = 1.5
box_width = 0.3
show_outliers = False

# Redfield C:N ratio
redfield_ratio = 106 / 16
cn_karl_2021_0_45 = 7.03
cn_karl_2021_75_125 = 6.37

# Arteaga 2016 modelled median Chl:C ratio for the subtropical surface ocean (HOT region) Figure 4a
arteaga_ratio = 1/0.008  # g Chl per g C
arteaga_FlowCyto_ratio = 1/0.005  # g Chl per g C

bottle_prof['Data_Part_C_N_molar_ML_DCM_ratio'] = bottle_prof['Data_Part_C_N_molar_ML']/bottle_prof['Data_Part_C_N_molar_DCM']
cn_particulate_ML = bottle_prof['Data_Part_C_N_molar_ML'].median()
cn_particulate_DCM = bottle_prof['Data_Part_C_N_molar_DCM'].median()
cn_particulate_ML_DCM_ratio = bottle_prof['Data_Part_C_N_molar_ML_DCM_ratio'].median()

# Melt data for C:Chl and C:N
df_chl = bottle_prof[['C_Chl_ratio_C1', 'C_Chl_ratio_C2']].melt(var_name='Measurement', value_name='C_Chl_ratio')
df_cn = bottle_prof[['C_N_molar_C1', 'C_N_molar_C2', 'C_N_molar_bk']].melt(var_name='Measurement', value_name='C_N_ratio')
df_cn_data = bottle_prof[['Data_Part_C_N_molar_ML', 'Data_Part_C_N_molar_DCM']].melt(var_name='Measurement', value_name='Data_C_N_ratio')
# Filter df_cn_data to only surface + subsurface (drop anything else just in case)
df_cn_data = df_cn_data[df_cn_data['Measurement'].isin(
    ['Data_Part_C_N_molar_ML', 'Data_Part_C_N_molar_DCM']
)]

# Create figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 3, 1]})

# --- C:Chl subplot ---
# Add Arteaga modelled median line
ax1.axhline(
    arteaga_ratio,
    color='navy',
    linestyle='--',
    linewidth=line_thickness,
    label=f'Arteaga et al. 2016 Model ({arteaga_ratio:.0f})', zorder=0
)
ax1.axhline(
    arteaga_FlowCyto_ratio,
    color='darkorange',
    linestyle='--',
    linewidth=line_thickness,
    label=f'Arteaga et al. 2016 FlowCyto ({arteaga_FlowCyto_ratio:.0f})', zorder=0
)
sns.boxplot(
    x='Measurement',
    y='C_Chl_ratio',
    data=df_chl,
    palette=["r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax1.set_xticklabels(['Surface', 'Subsurface'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylim(-2,605)
ax1.set_ylabel("C:Chl", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a)", fontsize=titlesize)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- C:N subplot ---
ax2.axhline(redfield_ratio, color='red', linestyle='--', linewidth=line_thickness, label=f'Redfield C:N ({redfield_ratio:.2f})', zorder=0)
#ax2.axhline(cn_karl_2021_0_45, color='navy', linestyle='--', linewidth=line_thickness, label=f'Karl et al. 2021 0-45m ({cn_karl_2021_0_45:.2f})', zorder=0)
#ax2.axhline(cn_karl_2021_75_125, color='k', linestyle='--', linewidth=line_thickness, label=f'Karl et al. 2021 75-125m ({cn_karl_2021_75_125:.2f})', zorder=0)
#ax2.axhline(cn_particulate_ML, color='navy', linestyle='-.', linewidth=line_thickness, label=f'Particulate ML ({cn_particulate_ML:.2f})', zorder=0)
#ax2.axhline(cn_particulate_DCM, color='k', linestyle='-.', linewidth=line_thickness, label=f'Particulate DCM ({cn_particulate_DCM:.2f})', zorder=0)
# Plot diamonds for particulate medians at the Surface (0) and Subsurface (1) categories
ax2.plot(0, cn_particulate_ML, 'd', color='orange', markersize=7,
         label=f'Particulate ML ({cn_particulate_ML:.2f})', zorder=5)
ax2.plot(1, cn_particulate_DCM, 'X', color='orange', markersize=9,
         label=f'Particulate DCM ({cn_particulate_DCM:.2f})', zorder=5)
# =============================================================================
# ax2.plot(0, cn_karl_2021_0_45, 's', color='navy', markersize=9,
#          label=f'Karl et al. 2021 0-45m ({cn_karl_2021_0_45:.2f})', zorder=5)
# ax2.plot(1, cn_karl_2021_75_125, 's', color='k', markersize=9,
#          label=f'Karl et al. 2021 75-125m ({cn_karl_2021_75_125:.2f})', zorder=5)
# =============================================================================

sns.boxplot(
    x='Measurement',
    y='C_N_ratio',
    data=df_cn,
    order=['C_N_molar_C1', 'C_N_molar_C2', 'C_N_molar_bk'],
    palette=["r", "royalblue", "orchid"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax2,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
# =============================================================================
# # second set of boxplots for only surface and subsurface category
# sns.boxplot(
#     x='Measurement',
#     y='Data_C_N_ratio',
#     data=df_cn_data,
#     order=['Data_Part_C_N_molar_ML', 'Data_Part_C_N_molar_DCM'],
#     palette=["r", "royalblue"],
#     showfliers=show_outliers,
#     linewidth=line_thickness,
#     width=box_width*0.5,
#     ax=ax2,
#     boxprops=dict(edgecolor='black'),
#     whiskerprops=dict(color='black', linewidth=line_thickness),
#     capprops=dict(color='black', linewidth=line_thickness),
#     medianprops=dict(color='black', linewidth=line_thickness),
# )
# # Restore correct tick positions and labels
# ax2.set_xticks([0, 1, 2])
# #Ensure the full last box is visible
# ax2.set_xlim(-0.5, 2.5)
# =============================================================================
ax2.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax2.set_xlabel('')
ax2.set_ylim(3.4,10.5)
ax2.set_ylabel("C:N (mol:mol)", fontsize=labsize)
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b)", fontsize=titlesize)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax2.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- C1/C2 subplot ---
ax3.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness, label='Surface=Subsurface', zorder=0)
ax3.plot(0, cn_particulate_ML_DCM_ratio, 'o', color='darkgreen', markersize=8,
         label='Particulate ML:DCM', zorder=5)
sns.boxplot(
    y=bottle_prof["C_N_molar_C1_C2"],
    color="palegreen",
    showfliers=show_outliers,
    width=box_width,
    ax=ax3,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax3.set_title("(c)", fontsize=titlesize)
ax3.set_xlabel('')
ax3.set_ylim(0.75,1.85)
ax3.set_ylabel("Ratio", fontsize=labsize, color="darkgreen")
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_xticklabels(['Surface:Subsurface'], fontsize=ticksize)
ax3.legend(loc='best', fontsize=10.7, frameon=True)

# Final layout adjustments and saving
plt.tight_layout()
plt.savefig('plots/HOT_Combined_Boxplots_CChl_CN_SurfSub_paper.jpeg',
            format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### Chl:C ratio for Arteaga comparison


# Plot styling variables
labsize        = 16
titlesize      = 25
ticksize       = 15
legendsize     = 12
line_thickness = 1.5
box_width      = 0.3
show_outliers  = False

# Arteaga 2016 modelled median Chl:C ratio for the subtropical surface ocean (HOT region)
arteaga_ratio = 0.008  # g Chl per g C
arteaga_FlowCyto_ratio = 0.005  # g Chl per g C

# Melt your two C:Chl measurements into long form
df_chl = (
    bottle_prof[['C_Chl_ratio_C1', 'C_Chl_ratio_C2']]
    .melt(var_name='Depth', value_name='C_to_Chl')
)

# Invert to get Chl:C
df_chl['Chl_to_C'] = 1.0 / df_chl['C_to_Chl']

# Compute median and MAD for each depth
stats = df_chl.groupby('Depth')['Chl_to_C'].agg(
    median='median',
    mad=lambda x: MAD(x, nan_policy='omit')  # returns the median absolute deviation
).reset_index()

# Create single boxplot of Chl:C
fig, ax1 = plt.subplots(figsize=(6, 6))

sns.boxplot(
    x='Depth', y='Chl_to_C',
    data=df_chl,
    palette=["r", "royalblue"],
    showfliers=show_outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)

# Add Arteaga modelled median line
ax1.axhline(
    arteaga_ratio,
    color='navy',
    linestyle='--',
    linewidth=line_thickness,
    label=f'Arteaga et al. 2016 Chl:C = {arteaga_ratio:.3f}',zorder=0
)
ax1.axhline(
    arteaga_FlowCyto_ratio,
    color='darkorange',
    linestyle='--',
    linewidth=line_thickness,
    label=f'Arteaga et al. 2016 FlowCyto Chl:C = {arteaga_FlowCyto_ratio:.3f}',zorder=0
)

# Annotate each box with median ± MAD to the right
for i, row in stats.iterrows():
    x = i                      # category index (0, 1)
    y = row['median']
    txt = f"{row['median']:.3f} ± {row['mad']:.3f}"
    ax1.text(
        x + box_width/2 + 0.05,  # shift right by half box_width plus a little padding
        y,
        txt,
        ha='left',               # left‐align text so it grows rightward
        va='center',             # vertically centered on the median
        fontsize=legendsize,
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='black'
        )
    )

# Final styling
ax1.set_xticklabels(['Surface', 'Subsurface'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylabel("Chl:C", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_ylim(-0.002,0.052)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax1.legend(loc='upper left', fontsize=legendsize)

plt.tight_layout()
plt.savefig(
    'plots/HOT_ChltoCarbon_Ratios_paper.jpeg',
    format='jpeg', dpi=300, bbox_inches="tight"
)
plt.show()


#%%

### BOXPLOTS C:P — REFINED FORMAT

# --- Define styling parameters ---
line_thickness = 1.5
box_width = 0.28
ticksize = 14
labsize = 14
titlesize = 18
legendsize = 12
outliers = False

# --- Convert Date column to datetime format ---
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'], format='%Y-%m-%d')

# --- Split data into before and after November 2011 ---
before_nov_2011 = bottle_prof[bottle_prof['Date'] < '2011-11-01']
after_nov_2011 = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

# --- Function to melt data for plotting ---
def melt_data(df):
    return df[['C_P_molar_C1', 'C_P_molar_C2', 'C_P_molar_bk']].melt(
        var_name='Measurement',
        value_name='C_P_ratio'
    )

# Melt both datasets
df_melted_before = melt_data(before_nov_2011)
df_melted_after = melt_data(after_nov_2011)

# --- Compute Redfield ratio ---
redfield_ratio = 106 / 1

# --- Create the figure with unpacked axis handles ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(12, 12), gridspec_kw={'width_ratios': [3, 1]})
fig.subplots_adjust(wspace=0.28,hspace=0.28)

# --- Plot (a) Before Nov 2011 C:P boxplot ---
ax1.axhline(redfield_ratio, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield C:P ({redfield_ratio:.1f})', zorder=0)
sns.boxplot(
    x='Measurement',
    y='C_P_ratio',
    data=df_melted_before,
    palette=["r", "royalblue", "orchid"],
    showfliers=outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax1.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylabel("C:P (mol:mol)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a) Before Nov 2011", fontsize=titlesize)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Plot (b) Before Nov 2011 Surface/Subsurface ratio ---
ax2.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface', zorder=0)
sns.boxplot(
    y=before_nov_2011["C_P_molar_C1_C2"],
    color="palegreen",
    showfliers=outliers,
    width=box_width,
    linewidth=line_thickness,
    ax=ax2,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax2.set_ylabel("C:P Surface/Subsurface", fontsize=labsize, color="darkgreen")
ax2.set_xticklabels(['Surface/Subsurface'], fontsize=ticksize)
ax2.set_xlabel('')
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b) Before Nov 2011", fontsize=titlesize)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Plot (c) From Nov 2011 C:P boxplot ---
ax3.axhline(redfield_ratio, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield C:P ({redfield_ratio:.1f})', zorder=0)
sns.boxplot(
    x='Measurement',
    y='C_P_ratio',
    data=df_melted_after,
    palette=["r", "royalblue", "orchid"],
    showfliers=outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax3,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax3.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax3.set_xlabel('')
ax3.set_ylabel("C:P (mol:mol)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c) From Nov 2011", fontsize=titlesize)
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax3.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Plot (d) From Nov 2011 Surface/Subsurface ratio ---
ax4.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface', zorder=0)
sns.boxplot(
    y=after_nov_2011["C_P_molar_C1_C2"],
    color="palegreen",
    showfliers=outliers,
    width=box_width,
    linewidth=line_thickness,
    ax=ax4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax4.set_ylabel("C:P Surface/Subsurface", fontsize=labsize, color="darkgreen")
ax4.set_xticklabels(['Surface/Subsurface'], fontsize=ticksize)
ax4.set_xlabel('')
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d) From Nov 2011", fontsize=titlesize)
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Final layout adjustments and saving ---
#plt.tight_layout()
plt.savefig('plots/Boxplots_CP_before_after_Nov2011.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Styling parameters ---
line_thickness = 1.5
box_width = 0.25
delta = box_width*0.7
ticksize = 14
labsize = 14
titlesize = 18
legendsize = 10
outliers = False

# --- Prepare data ---
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'])
before = bottle_prof[bottle_prof['Date'] < '2011-11-01']
after  = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

# --- Category setup ---
cats     = ['C_P_molar_C1','C_P_molar_C2','C_P_molar_bk']
labels   = ['Surface','Subsurface','Non-Algal']
colors   = ['r','royalblue','orchid']
redfield = 106.0

#plt.rcdefaults() 
# --- Make figure with two subplots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False, gridspec_kw={'width_ratios': [3, 1]})

# --- Left: C:P boxplots ---
ax1.axhline(redfield, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield C:P ({redfield:.0f}:1)')

for i, (cat, lbl, col) in enumerate(zip(cats, labels, colors)):
    # Pre-Nov
    data_pre = before[cat].dropna()
    ax1.boxplot(
        data_pre,
        positions=[i - delta],
        widths=box_width,
        patch_artist=True,
        showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness),
    )
    # Post-Nov (same color + hatch)
    data_post = after[cat].dropna()
    ax1.boxplot(
        data_post,
        positions=[i + delta],
        widths=box_width,
        patch_artist=True,
        showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black',
                      linewidth=line_thickness, hatch='///'),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness),
    )

ax1.set_xticks(range(len(cats)))
ax1.set_xticklabels(labels, fontsize=ticksize)
ax1.set_ylim(-7,430)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax1.grid(True, axis='y', linestyle='--', color='grey', linewidth=0.7, alpha = 0.5)
ax1.set_ylabel("C:P (mol:mol)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title('(a)', fontsize=titlesize)

# --- Right: Surface/Subsurface ratio ---
ax2.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface')

# Pre‑Nov ratio
ratio_pre = before['C_P_molar_C1_C2'].dropna()
ax2.boxplot(
    ratio_pre,
    positions=[0 - delta],
    widths=box_width,
    patch_artist=True,
    showfliers=outliers,
    boxprops=dict(facecolor='palegreen', edgecolor='black', linewidth=line_thickness),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
)

# Post‑Nov ratio (hatched)
ratio_post = after['C_P_molar_C1_C2'].dropna()
ax2.boxplot(
    ratio_post,
    positions=[0 + delta],
    widths=box_width,
    patch_artist=True,
    showfliers=outliers,
    boxprops=dict(facecolor='palegreen', edgecolor='black',
                  linewidth=line_thickness, hatch='///'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness),
)

ax2.set_xticks([0])
ax2.set_xticklabels(['Surface/Subsurface'], fontsize=ticksize)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax2.grid(True, axis='y', linestyle='--', color='grey', linewidth=0.7, alpha = 0.5)
ax2.set_ylabel("Ratio", fontsize=labsize, color='darkgreen')
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title('(b)', fontsize=titlesize)

# --- Shared legend ---
legend_elems = [
    Patch(facecolor='white', edgecolor='black', label='Pre-Nov 2011'),
    Patch(facecolor='white', edgecolor='black', hatch='///', label='Post-Nov 2011'),
    plt.Line2D([0],[0], color='red', ls='--', lw=line_thickness, label=f'Redfield C:P ({redfield:.0f})'),
    plt.Line2D([0],[0], color='darkgreen', ls='--', lw=line_thickness, label='Surface = Subsurface')
]
ax1.legend(handles=legend_elems, fontsize=legendsize, loc='upper right')

plt.savefig('plots/Boxplots_CP_before_after_Nov2011_sidebyside.jpeg',
            dpi=300, bbox_inches='tight')
plt.show()


#%%

### BOXPLOTS N:P — REFINED FORMAT

# --- Define styling parameters ---
line_thickness = 1.5
box_width = 0.28
ticksize = 14
labsize = 14
titlesize = 18
legendsize = 12
outliers = False

# --- Convert Date column to datetime format ---
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'], format='%Y-%m-%d')

# --- Split data into before and after November 2011 ---
before_nov_2011 = bottle_prof[bottle_prof['Date'] < '2011-11-01']
after_nov_2011 = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

# --- Function to melt N:P data for plotting ---
def melt_data_np(df):
    return df[['N_P_molar_C1', 'N_P_molar_C2', 'N_P_molar_bk']].melt(
        var_name='Measurement',
        value_name='N_P_ratio'
    )

# Melt both datasets
df_np_melted_before = melt_data_np(before_nov_2011)
df_np_melted_after = melt_data_np(after_nov_2011)

# --- Compute Redfield ratio ---
redfield_ratio_np = 16 / 1

# --- Create the figure with unpacked axis handles ---
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows=2, ncols=2, figsize=(12, 12), gridspec_kw={'width_ratios': [3, 1]}
)
fig.subplots_adjust(wspace=0.28, hspace=0.28)

# --- Plot (a) Before Nov 2011 N:P boxplot ---
ax1.axhline(redfield_ratio_np, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield N:P ({redfield_ratio_np:.1f})', zorder=0)
sns.boxplot(
    x='Measurement',
    y='N_P_ratio',
    data=df_np_melted_before,
    palette=["r", "royalblue", "orchid"],
    showfliers=outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax1,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax1.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax1.set_xlabel('')
ax1.set_ylabel("N:P (mol:mol)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a) Before Nov 2011", fontsize=titlesize)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Plot (b) Before Nov 2011 Surface/Subsurface N:P ratio ---
ax2.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface', zorder=0)
sns.boxplot(
    y=before_nov_2011["N_P_molar_C1_C2"],
    color="palegreen",
    showfliers=outliers,
    width=box_width,
    linewidth=line_thickness,
    ax=ax2,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax2.set_ylabel("N:P Surface/Subsurface", fontsize=labsize, color="darkgreen")
ax2.set_xticklabels(['Surface/Subsurface'], fontsize=ticksize)
ax2.set_xlabel('')
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b) Before Nov 2011", fontsize=titlesize)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Plot (c) From Nov 2011 N:P boxplot ---
ax3.axhline(redfield_ratio_np, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield N:P ({redfield_ratio_np:.1f})', zorder=0)
sns.boxplot(
    x='Measurement',
    y='N_P_ratio',
    data=df_np_melted_after,
    palette=["r", "royalblue", "orchid"],
    showfliers=outliers,
    linewidth=line_thickness,
    width=box_width,
    ax=ax3,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax3.set_xticklabels(['Surface', 'Subsurface', 'Non-Algal'], fontsize=ticksize)
ax3.set_xlabel('')
ax3.set_ylabel("N:P (mol:mol)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c) From Nov 2011", fontsize=titlesize)
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)
ax3.legend(loc='upper right', fontsize=legendsize, frameon=True)

# --- Plot (d) From Nov 2011 Surface/Subsurface N:P ratio ---
ax4.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface', zorder=0)
sns.boxplot(
    y=after_nov_2011["N_P_molar_C1_C2"],
    color="palegreen",
    showfliers=outliers,
    width=box_width,
    linewidth=line_thickness,
    ax=ax4,
    boxprops=dict(edgecolor='black'),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)
ax4.set_ylabel("N:P Surface/Subsurface", fontsize=labsize, color="darkgreen")
ax4.set_xticklabels(['Surface/Subsurface'], fontsize=ticksize)
ax4.set_xlabel('')
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d) From Nov 2011", fontsize=titlesize)
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.5)

# --- Final layout adjustments and saving ---
# plt.tight_layout()
plt.savefig('plots/Boxplots_NP_before_after_Nov2011.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### PAPER FIGURE 7 - C:P and N:P BOXPLOT for paper -  Pre Post #???

# --- Styling parameters ---
line_thickness = 1.5
box_width = 0.25
delta = box_width * 0.7
ticksize = 17
labsize = 19
titlesize = 21
legendsize = 11
outliers = False

# --- Prepare data ---
bottle_prof['Data_Part_C_P_molar_ML_DCM_ratio'] = bottle_prof['Data_Part_C_P_molar_ML']/bottle_prof['Data_Part_C_P_molar_DCM']
bottle_prof['Data_Part_N_P_molar_ML_DCM_ratio'] = bottle_prof['Data_Part_N_P_molar_ML']/bottle_prof['Data_Part_N_P_molar_DCM']

bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'])
before = bottle_prof[bottle_prof['Date'] < '2011-11-01']
after  = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

cp_particulate_ML_pre            = before['Data_Part_C_P_molar_ML'].median()
cp_particulate_DCM_pre           = before['Data_Part_C_P_molar_DCM'].median()
cp_particulate_ML_DCM_ratio_pre  = before['Data_Part_C_P_molar_ML_DCM_ratio'].median()
cp_particulate_ML_post           = after['Data_Part_C_P_molar_ML'].median()
cp_particulate_DCM_post          = after['Data_Part_C_P_molar_DCM'].median()
cp_particulate_ML_DCM_ratio_post = after['Data_Part_C_P_molar_ML_DCM_ratio'].median()

np_particulate_ML_pre            = before['Data_Part_N_P_molar_ML'].median()
np_particulate_DCM_pre           = before['Data_Part_N_P_molar_DCM'].median()
np_particulate_ML_DCM_ratio_pre  = before['Data_Part_N_P_molar_ML_DCM_ratio'].median()
np_particulate_ML_post           = after['Data_Part_N_P_molar_ML'].median()
np_particulate_DCM_post          = after['Data_Part_N_P_molar_DCM'].median()
np_particulate_ML_DCM_ratio_post = after['Data_Part_N_P_molar_ML_DCM_ratio'].median()

# --- Figure with 2×2 subplots ---
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(15, 12),
    gridspec_kw={'width_ratios': [3, 1]}
)
#fig.subplots_adjust(wspace=0.3,hspace=0.3)
ax1, ax2, ax3, ax4 = axes.flatten()

# --- Common settings ---
# Category definitions
cats_cp    = ['C_P_molar_C1', 'C_P_molar_C2', 'C_P_molar_bk']
labels_cp  = ['Surface', 'Subsurface', 'Non-Algal']
colors_cp  = ['r', 'royalblue', 'orchid']
redfield_cp = 106.0


cats_np    = ['N_P_molar_C1', 'N_P_molar_C2', 'N_P_molar_bk']
labels_np  = labels_cp  # same labels
colors_np  = colors_cp  # same colors
redfield_np = 16.0


# --- Row 1: C:P ---
# (a) C:P boxplots
ax1.axhline(redfield_cp, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield C:P ({redfield_cp:.0f}:1)')
for i, (cat, col) in enumerate(zip(cats_cp, colors_cp)):
    # Pre
    data_pre = before[cat].dropna()
    ax1.boxplot(
        data_pre,
        positions=[i - delta], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )
    
    # Post
    data_post = after[cat].dropna()
    ax1.boxplot(
        data_post,
        positions=[i + delta], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness, hatch='///'),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )
    
ax1.plot(0- delta, cp_particulate_ML_pre, 'd', color='orange', markersize=7,
         label='Particulate ML', zorder=5)
ax1.plot(1- delta, cp_particulate_DCM_pre, 'X', color='orange', markersize=9,
         label='Particulate DCM', zorder=5)
ax1.plot(0+ delta, cp_particulate_ML_post, 'd', color='orange', markersize=7,
         label='Particulate ML', zorder=5)
ax1.plot(1+ delta, cp_particulate_DCM_post, 'X', color='orange', markersize=9,
         label='Particulate DCM', zorder=5)    
    
ax1.set_xticks(range(len(cats_cp)))
ax1.set_xticklabels(labels_cp, fontsize=ticksize)
ax1.set_ylim(-7, 430)
ax1.set_xlim(-0.8, 2.8)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax1.set_ylabel("C:P (mol:mol)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a)", fontsize=titlesize)

# (b) C:P Surface/Subsurface ratio
ax2.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface')
for period, offset, hatch in [('Pre', -delta, ''), ('Post', +delta, '///')]:
    vals = (before if period=='Pre' else after)['C_P_molar_C1_C2'].dropna()
    ax2.boxplot(
        vals,
        positions=[0 + offset], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor='palegreen', edgecolor='black', linewidth=line_thickness, hatch=hatch),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )

ax2.plot(0- delta, cp_particulate_ML_DCM_ratio_pre, 'o', color='darkgreen', markersize=8,
         label='Particulate ML:DCM Pre', zorder=5)
ax2.plot(0+ delta, cp_particulate_ML_DCM_ratio_post, 'o', color='darkgreen', markersize=8,
         label='Particulate ML:DCM Post', zorder=5)

ax2.set_xticks([0])
ax2.set_xticklabels(['Surface:Subsurface'], fontsize=ticksize)
ax2.set_ylim(0.2,3.4)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax2.set_ylabel("Ratio", fontsize=labsize, color='darkgreen')
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b)", fontsize=titlesize)

# --- Row 2: N:P ---
# (c) N:P boxplots
ax3.axhline(redfield_np, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield N:P ({redfield_np:.0f}:1)')
for i, (cat, col) in enumerate(zip(cats_np, colors_np)):
    data_pre = before[cat].dropna()
    ax3.boxplot(
        data_pre,
        positions=[i - delta], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )
    data_post = after[cat].dropna()
    ax3.boxplot(
        data_post,
        positions=[i + delta], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness, hatch='///'),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )
ax3.plot(0- delta, np_particulate_ML_pre, 'd', color='orange', markersize=7,
         label='Particulate ML', zorder=5)
ax3.plot(1- delta, np_particulate_DCM_pre, 'X', color='orange', markersize=9,
         label='Particulate DCM', zorder=5)
ax3.plot(0+ delta, np_particulate_ML_post, 'd', color='orange', markersize=7,
         label='Particulate ML', zorder=5)
ax3.plot(1+ delta, np_particulate_DCM_post, 'X', color='orange', markersize=9,
         label='Particulate DCM', zorder=5)
  
    
ax3.set_xticks(range(len(cats_np)))
ax3.set_xticklabels(labels_np, fontsize=ticksize)
#ax3.set_ylim(-7, 430)
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax3.set_ylabel("N:P (mol:mol)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c)", fontsize=titlesize)

# (d) N:P Surface/Subsurface ratio
ax4.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness,
            label='Surface = Subsurface')
for period, offset, hatch in [('Pre', -delta, ''), ('Post', +delta, '///')]:
    vals = (before if period=='Pre' else after)['N_P_molar_C1_C2'].dropna()
    ax4.boxplot(
        vals,
        positions=[0 + offset], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor='palegreen', edgecolor='black', linewidth=line_thickness, hatch=hatch),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )
    
ax4.plot(0- delta, np_particulate_ML_DCM_ratio_pre, 'o', color='darkgreen', markersize=8,
         label='Particulate ML:DCM Pre', zorder=5)
ax4.plot(0+ delta, np_particulate_ML_DCM_ratio_post, 'o', color='darkgreen', markersize=8,
         label='Particulate ML:DCM Post', zorder=5)
    
ax4.set_xticks([0])
ax4.set_xticklabels(['Surface:Subsurface'], fontsize=ticksize)
#ax4.set_ylim(-7, 430)
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax4.set_ylabel("Ratio", fontsize=labsize, color='darkgreen')
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d)", fontsize=titlesize)

# --- Legend on top left of first subplot ---
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
# =============================================================================
# legend_elems = [
#     Patch(facecolor='white', edgecolor='black', label='Pre-Nov 2011'),
#     Patch(facecolor='white', edgecolor='black', hatch='///', label='Post-Nov 2011'),
#     plt.Line2D([0],[0], color='red', ls='--', lw=line_thickness, label='Redfield'),
#     plt.Line2D([0],[0], color='darkgreen', ls='--', lw=line_thickness, label='Surface=Subsurface')
# ]
# =============================================================================
legend_elems = [
    Patch(facecolor='white', edgecolor='black', label='Pre-Nov 2011'),
    Patch(facecolor='white', edgecolor='black', hatch='///', label='Post-Nov 2011'),
    plt.Line2D([0], [0], color='red', ls='--', lw=line_thickness, label='Redfield'),
    plt.Line2D([0], [0], marker='d', color='orange', markersize=7, linestyle='None', label='Particulate ML'),
    plt.Line2D([0], [0], marker='X', color='orange', markersize=9, linestyle='None', label='Particulate DCM')
]

legend_elems2 = [
    plt.Line2D([0],[0], color='darkgreen', ls='--', lw=line_thickness, label='Surface=Subsurface'),
    plt.Line2D([0], [0], marker='o', color='darkgreen', markersize=8, linestyle='None', label='Particulate ML:DCM')
]

ax1.legend(handles=legend_elems, fontsize=legendsize, loc='upper right')
ax2.legend(handles=legend_elems2, fontsize=legendsize, loc='upper left')

#plt.tight_layout()
plt.savefig('plots/Boxplots_CP_NP_pre_post.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#%%

### PAPER FIGURE 7 - C:P and N:P BOXPLOT for paper (POST-2011 ONLY)

# --- Styling parameters ---
line_thickness = 1.5
box_width = 0.25
ticksize = 17
labsize = 19
titlesize = 21
outliers = False

# --- Prepare data ---
bottle_prof['Data_Part_C_P_molar_ML_DCM_ratio'] = (
    bottle_prof['Data_Part_C_P_molar_ML'] / bottle_prof['Data_Part_C_P_molar_DCM']
)
bottle_prof['Data_Part_N_P_molar_ML_DCM_ratio'] = (
    bottle_prof['Data_Part_N_P_molar_ML'] / bottle_prof['Data_Part_N_P_molar_DCM']
)

bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'])
after = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

# Extract post medians for particulate annotations
cp_particulate_ML_post           = after['Data_Part_C_P_molar_ML'].median()
cp_particulate_DCM_post          = after['Data_Part_C_P_molar_DCM'].median()
cp_particulate_ML_DCM_ratio_post = after['Data_Part_C_P_molar_ML_DCM_ratio'].median()

np_particulate_ML_post           = after['Data_Part_N_P_molar_ML'].median()
np_particulate_DCM_post          = after['Data_Part_N_P_molar_DCM'].median()
np_particulate_ML_DCM_ratio_post = after['Data_Part_N_P_molar_ML_DCM_ratio'].median()

# --- Figure with 2×2 subplots ---
fig, axes = plt.subplots(
    nrows=2, ncols=2,
    figsize=(14, 12),
    gridspec_kw={'width_ratios': [3, 1]}
)
ax1, ax2, ax3, ax4 = axes.flatten()

# --- Common settings ---
cats_cp   = ['C_P_molar_C1', 'C_P_molar_C2', 'C_P_molar_bk']
labels_cp = ['Surface', 'Subsurface', 'Non-Algal']
colors_cp = ['r', 'royalblue', 'orchid']
redfield_cp = 106.0

cats_np   = ['N_P_molar_C1', 'N_P_molar_C2', 'N_P_molar_bk']
labels_np = labels_cp
colors_np = colors_cp
redfield_np = 16.0

# ===============================================================
# (a) C:P boxplots (POST ONLY)
# ===============================================================
ax1.axhline(redfield_cp, color='red', linestyle='--', linewidth=line_thickness,
            label=f'Redfield C:P ({redfield_cp:.0f}:1)')

for i, (cat, col) in enumerate(zip(cats_cp, colors_cp)):
    data_post = after[cat].dropna()

    ax1.boxplot(
        data_post,
        positions=[i], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )

# Post–2011 particulate medians
ax1.plot(0, cp_particulate_ML_post, 'd', color='orange', markersize=7, zorder=5)
ax1.plot(1, cp_particulate_DCM_post, 'X', color='orange', markersize=9, zorder=5)

ax1.set_xticks(range(len(cats_cp)))
ax1.set_xticklabels(labels_cp, fontsize=ticksize)
ax1.set_ylim(-7, 330)
#ax1.set_xlim(-0.8, 2.8)
ax1.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax1.set_ylabel("C:P (mol:mol)", fontsize=labsize)
ax1.tick_params(axis='y', labelsize=ticksize)
ax1.set_title("(a)", fontsize=titlesize)

# ===============================================================
# (b) C:P Surface/Subsurface ratio (POST ONLY)
# ===============================================================
ax2.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness)

vals = after['C_P_molar_C1_C2'].dropna()
ax2.boxplot(
    vals,
    positions=[0], widths=box_width,
    patch_artist=True, showfliers=outliers,
    boxprops=dict(facecolor='palegreen', edgecolor='black', linewidth=line_thickness),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)

ax2.plot(0, cp_particulate_ML_DCM_ratio_post, 'o', color='darkgreen', markersize=8, zorder=5)

ax2.set_xticks([0])
ax2.set_xticklabels(['Surface:Subsurface'], fontsize=ticksize)
ax2.set_ylim(0.2, 3.4)
ax2.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax2.set_ylabel("Ratio", fontsize=labsize, color='darkgreen')
ax2.tick_params(axis='y', labelsize=ticksize)
ax2.set_title("(b)", fontsize=titlesize)

# ===============================================================
# (c) N:P boxplots (POST ONLY)
# ===============================================================
ax3.axhline(redfield_np, color='red', linestyle='--', linewidth=line_thickness)

for i, (cat, col) in enumerate(zip(cats_np, colors_np)):
    data_post = after[cat].dropna()

    ax3.boxplot(
        data_post,
        positions=[i], widths=box_width,
        patch_artist=True, showfliers=outliers,
        boxprops=dict(facecolor=col, edgecolor='black', linewidth=line_thickness),
        whiskerprops=dict(color='black', linewidth=line_thickness),
        capprops=dict(color='black', linewidth=line_thickness),
        medianprops=dict(color='black', linewidth=line_thickness)
    )

ax3.plot(0, np_particulate_ML_post, 'd', color='orange', markersize=7, zorder=5)
ax3.plot(1, np_particulate_DCM_post, 'X', color='orange', markersize=9, zorder=5)

ax3.set_xticks(range(len(cats_np)))
ax3.set_xticklabels(labels_np, fontsize=ticksize)
ax3.set_ylim(-1, 44)
ax3.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax3.set_ylabel("N:P (mol:mol)", fontsize=labsize)
ax3.tick_params(axis='y', labelsize=ticksize)
ax3.set_title("(c)", fontsize=titlesize)

# ===============================================================
# (d) N:P Surface/Subsurface ratio (POST ONLY)
# ===============================================================
ax4.axhline(1, color='darkgreen', linestyle='--', linewidth=line_thickness)

vals = after['N_P_molar_C1_C2'].dropna()
ax4.boxplot(
    vals,
    positions=[0], widths=box_width,
    patch_artist=True, showfliers=outliers,
    boxprops=dict(facecolor='palegreen', edgecolor='black', linewidth=line_thickness),
    whiskerprops=dict(color='black', linewidth=line_thickness),
    capprops=dict(color='black', linewidth=line_thickness),
    medianprops=dict(color='black', linewidth=line_thickness)
)

ax4.plot(0, np_particulate_ML_DCM_ratio_post, 'o', color='darkgreen', markersize=8, zorder=5)

ax4.set_xticks([0])
ax4.set_xticklabels(['Surface:Subsurface'], fontsize=ticksize)
ax4.set_ylim(0.2, 3.4)
ax4.grid(True, axis='x', linestyle='--', color='grey', linewidth=0.9)
ax4.set_ylabel("Ratio", fontsize=labsize, color='darkgreen')
ax4.tick_params(axis='y', labelsize=ticksize)
ax4.set_title("(d)", fontsize=titlesize)

# --- Legend ---
from matplotlib.patches import Patch
legend_elems = [
    plt.Line2D([0],[0], color='red', ls='--', lw=line_thickness, label='Redfield'),
    plt.Line2D([0], [0], marker='d', color='orange', markersize=7, linestyle='None', label='Particulate ML'),
    plt.Line2D([0], [0], marker='X', color='orange', markersize=9, linestyle='None', label='Particulate DCM')
]

legend_elems2 = [
    plt.Line2D([0],[0], color='darkgreen', ls='--', lw=line_thickness, label='Surface=Subsurface'),
    plt.Line2D([0], [0], marker='o', color='darkgreen', markersize=8, linestyle='None', label='Particulate ML:DCM')
]

ax1.legend(handles=legend_elems, fontsize=12, loc='upper right')
ax2.legend(handles=legend_elems2, fontsize=12, loc='upper left')

plt.savefig('plots/Boxplots_CP_NP_post_only.jpeg', dpi=300, bbox_inches='tight')
plt.show()


#%%

### PAPER TABLE 1 - Ratio summary table

print(bottle_prof.columns.tolist())

import pandas as pd
from scipy.stats import median_abs_deviation as MAD  # robust MAD function

# Ensure Date is datetime
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'])

# Split dataset pre/post Nov 2011
before = bottle_prof[bottle_prof['Date'] < '2011-11-01']
after  = bottle_prof[bottle_prof['Date'] >= '2011-11-01']

# 1. Define your parameters
params = {
    'Temp':               ('Temp_ML',                   'Temp_DCM',           None, None),
    'Nitrate':            ('Nitrate_ML_median',         'Nitrate_DCM_conc',   None, None),
    'PAR':                ('PAR_MLD',                   'PAR_DCM',            None, None),
    'C:Chl ratio':        ('C_Chl_ratio_C1',            'C_Chl_ratio_C2',            None,          'C_Chl_ratio_C1_C2'),
    'C:N ratio':          ('C_N_molar_C1',              'C_N_molar_C2',             'C_N_molar_bk', 'C_N_molar_C1_C2'),
    'Particulate C:N ratio': ('Data_Part_C_N_molar_ML', 'Data_Part_C_N_molar_DCM',   None,          'Data_Part_C_N_molar_ML_DCM_ratio'),
}

special_params = {
    'C:P ratio': ('C_P_molar_C1','C_P_molar_C2','C_P_molar_bk','C_P_molar_C1_C2'),
    'Particulate C:P ratio': ('Data_Part_C_P_molar_ML','Data_Part_C_P_molar_DCM',None,'Data_Part_C_P_molar_ML_DCM_ratio'),
    'N:P ratio': ('N_P_molar_C1','N_P_molar_C2','N_P_molar_bk','N_P_molar_C1_C2'),
    'Particulate N:P ratio': ('Data_Part_N_P_molar_ML','Data_Part_N_P_molar_DCM',None,'Data_Part_N_P_molar_ML_DCM_ratio'),
}

# 2. Prepare empty table
table = pd.DataFrame(columns=['Surface', 'Subsurface', 'Non-Algal', 'Surface/Subsurface'])

# 3. Fill "normal" parameters
for name, (surf_col, sub_col, bk_col, comb_col) in params.items():
    def safe_val(col):
        if col is not None and col in bottle_prof:
            m = bottle_prof[col].median()
            d = MAD(bottle_prof[col], nan_policy='omit')
            return f"{m:.2f} ± {d:.2f}"
        else:
            return ""
    
    surface_val = safe_val(surf_col)
    subsurface_val = safe_val(sub_col)
    non_algal_val = safe_val(bk_col)
    combined_val = safe_val(comb_col)
    
    table.loc[name] = [surface_val, subsurface_val, non_algal_val, combined_val]

# 4. Fill special parameters with pre/post split
for name, (surf_col, sub_col, bk_col, comb_col) in special_params.items():
    for label, subset in zip(['Pre-Nov 2011', 'Post-Nov 2011'], [before, after]):
        rowname = f"{name} ({label})"

        def safe_val_subset(col):
            if col is not None and col in subset:
                m = subset[col].median()
                d = MAD(subset[col], nan_policy='omit')
                return f"{m:.2f} ± {d:.2f}"
            else:
                return ""
        
        surface_val = safe_val_subset(surf_col)
        subsurface_val = safe_val_subset(sub_col)
        non_algal_val = safe_val_subset(bk_col)
        combined_val = safe_val_subset(comb_col)

        table.loc[rowname] = [surface_val, subsurface_val, non_algal_val, combined_val]

# 5. Save table
table.to_csv("Tables/HOT_Env_Surface_Subsurface_Ratios_paper.csv", index=True)
table.to_excel("Tables/HOT_Env_Surface_Subsurface_Ratios_paper.xlsx", index=True)

print(table)

"END"
