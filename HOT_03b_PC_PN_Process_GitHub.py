"""
HOT: Inspect & Clean POC/PON data & setup data frame

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
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.colors import LinearSegmentedColormap
# Import specific modules from packages
from dateutil import relativedelta
#from math import nan
from scipy.stats import zscore
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore');

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    rd = relativedelta.relativedelta( pd.to_datetime( end ), pd.to_datetime( start ) )
    date_len  = '{}y{}m{}d'.format(rd.years,rd.months,rd.days)
    return date_len
    
def remove_outliers_local_minima(data, depths, window_size=3, threshold=3, replace_with_nans=True):
    
    """
    Remove outliers based on z-scores and local minima detection.
    
    Parameters:
        data (array): Input data (e.g., POC values).
        depths (array): Depths corresponding to the data.
        window_size (int): Rolling window size for smoothing (moving average/median).
        threshold (float): Z-score threshold to flag outliers.
        replace_with_nans (bool): Replace outliers with NaNs.
        
    Returns:
        filtered_data (array): Data with outliers removed/replaced.
        depths (array): Corresponding depths (unaltered).
    """
    # Copy input data
    filtered_data = np.array(data, dtype=float)

    # Calculate rolling median (robust to outliers)
    rolling_median = pd.Series(data).rolling(window=window_size, center=True, min_periods=1).median()

    # Calculate deviations from the rolling median
    deviations = data - rolling_median

    # Calculate Z-scores for deviations
    deviation_zscores = np.abs(zscore(deviations, nan_policy='omit'))
    
    # Identify Z-score outliers
    outlier_mask = deviation_zscores > threshold
    
    # Identify local minima: Points lower than neighbors within the window
    local_minima_mask = (data < rolling_median) & (deviations < -threshold)
    #local_minima_mask = (data < rolling_median) & (deviations < 0)
    
    # Combine all masks
    combined_mask = outlier_mask | local_minima_mask

    # Replace outliers with NaNs if required
    if replace_with_nans:
        filtered_data[combined_mask] = np.nan

    return filtered_data, depths

#%%

### READ & EXTRACT CTD DATA ###

### Read/Import cleaned CTD data from CSV
# CSV filename
filename_1 = 'data/HOT_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'DateTime'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
time_year     = ctd.loc[:,'yyyy'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.to_datetime(ctd['DateTime'].values)

### Cruise ID list for CTD ###
# Extract cruise_ID & Removes Duplicates
ID_list_ctd = pd.unique(ID_ctd) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

### Read CTD prof data ###
# CSV filename
filename_2 = 'data/HOT_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2,index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()

# Extract required data from df
try:
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['DateTime'])
except: 
    ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof.index)

ctd_prof.set_index(ctd_DateTime_prof, inplace= True)
ctd_prof.info()

ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
ctd_DecYear_prof  = ctd_prof.loc[:,'DecYear'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD_boyer'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_prof['Date']), max(ctd_prof['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(ctd_prof['Date'])))
print("Max Date: "+str(max(ctd_prof['Date'])))

#%%

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE PIGMENT BOTTLE ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Pigments_Chla_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6 = pd.read_csv(filename_1, index_col = 0)
# Inspect df
bottle_6.info()

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'DateTime'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_chla     = bottle_6.loc[:,'Chla'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()
# Bottle DateTime data
b_DateTime  = pd.to_datetime(bottle_6['DateTime'].values)

### Cruise ID list for Chla ###
# Extract cruise_ID & Removes Duplicates
ID_list_6 = pd.unique(b_ID)

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_6['Date']), max(bottle_6['Date']))
print("Timespan: "+str(b_date_length)) # 32y5m14d

#%%

######
### READ & CLEAN ORIGINAL POC BOTTLE DATA ###
######

#Filename
file_pigment1 = 'data/HOT_Bottle_Cleaned2.csv' #name file location and csv file name

df = pd.read_csv(file_pigment1, index_col = 0)
df.info()

#For HPLC Chla Dataset
bot = df[["Cruise_ID","CRN","CASTNO", "Cruise_ID_o","DateTime",'DecYear','Date','yyyy','mm','time','press', 'depth',
             'pc', 'pn']]
# low level nitrate and phosphate might be usefull need separate df for each and then later match per cruise because sampled on different casts.

bot.head()
bot.info()

# Rename Columns
bot.rename(columns={"pc": "POC",
                                "pn": "PON"},inplace=True)

# Remove rows with NaN values
bot_poc = bot.dropna(subset=['POC'])
bot_pon = bot.dropna(subset=['PON'])
bot = bot.dropna(subset=['POC','PON'])
# Remove rows with NaN depth values
bot = bot.dropna(subset=['depth'])
bot.info()

bot = bot.drop_duplicates(subset=["Cruise_ID", "depth"])

# Remove rows with depths below 1000m
bot = bot[bot["depth"]<1010]

# Convert nutrient units umol/kg to mmol/m3 then to mg/m3
bot['POC'] = bot['POC']*12.011*1025/1000 # replaced original column with converted values
# Convert nutrient units umol/kg to mmol/m3 then to mg/m3
bot['PON'] = bot['PON']*14.007*1025/1000

bot.info()
# sort
bot = bot.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bot = bot.reset_index(drop=True)

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bot['Date'] = pd.to_datetime(bot['DateTime']).dt.date

### Timespan of bottle data ###

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bot['Date']))+" to "+str(max(bot['Date'])))

# Print period timespan of bottle data using base date subtraction - only days
print("Bottle Date Length: "+str(max(bot['Date'])-min(bot['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bot['Date']), max(bot['Date']))
print("Timespan: "+str(b_date_length))

#%%

### EXPLORE POC DATA ###

# Summary stats Table for POC
print(bot[["POC"]].describe())

bot[["POC"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bot[["Cruise_ID", "POC"]].groupby("Cruise_ID").count()
print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POC")["POC"].count())

# Bar plot of profiles per number per number of depths sampled
bottle_poc_profcount.groupby("POC")["POC"].count().plot.bar()
plt.show()

### COUNT POC PROFILES PER YEAR ###

# Create new df with number of CTD profiles (cruises) per year
bottle_poc_y = bot[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_poc_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_poc_y.to_markdown())

#%%

### RENAME DF ###

# Remove rows with NaN POC values
bottle_poc = bot.dropna(subset=['POC'], inplace=False)
print(len(bot))
print(len(bottle_poc))
# Sort new df by time and depth again
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])
# Reset bottle df index removing old index with missing numbers after slice
bottle_poc = bottle_poc.reset_index(drop=True)

#%%

### REMOVE PROFILES WITH NO MATCHING PIGMENT profiles ###

### Cruise_ID list
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
ID_list_poc = pd.unique(b2_ID)

print(len(ID_list_poc))
print(len(ID_list_6))

# Create new df containing only data for profiles also in pigment bottle list
bottle_poc =  bottle_poc[bottle_poc.Cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

### Cruise_ID list
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/HOT_Bottle_POC_AllDepths.csv')

#%%

### REMOVE SPIKES FROM POC PROFILES ###

# Remove duplicate measurements
bottle_poc.drop_duplicates(subset=['Cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'DateTime'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_pon      = bottle_poc.loc[:,'PON'].to_numpy()
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()
b2_Decimal_year = bottle_poc.loc[:,'DecYear'].to_numpy()

### Cruise_ID list & Removes Duplicates
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))

# New arrays to store cleaned POC and PON data
POC_f = np.full(len(b2_poc), np.nan)  # Initialize with NaNs
PON_f = np.full(len(b2_pon), np.nan)  # Initialize with NaNs

# Loop through each profile
for ID in ID_list_poc:
    # Indices for the current profile
    prof_poc_idx = np.where(bottle_poc.Cruise_ID == ID)
    
    # Extract POC data
    prof_poc = b2_poc[prof_poc_idx]
    prof_poc_depth = b2_depth[prof_poc_idx]
    
    # Extract PON data for the same profile
    prof_pon = b2_pon[prof_poc_idx]
    
    # Z-SCORE OUTLIER REMOVAL FOR POC
    filtered_poc, filtered_depths = remove_outliers_local_minima(prof_poc, prof_poc_depth, window_size=3, threshold=3, replace_with_nans=True)
    
    # Z-SCORE OUTLIER REMOVAL FOR PON#remove_outliers_local_minima_gradient
    filtered_pon, _ = remove_outliers_local_minima(prof_pon, prof_poc_depth, window_size=3, threshold=2.7, replace_with_nans=True)
    
    # Convert the filtered data to arrays
    filtered_poc = np.array(filtered_poc)
    filtered_pon = np.array(filtered_pon)
    
    # Save the filtered data back into the initialized arrays
    POC_f[prof_poc_idx] = filtered_poc
    PON_f[prof_poc_idx] = filtered_pon

# Count NaN values for POC and PON
nan_count_poc = np.isnan(POC_f).sum()
nan_count_pon = np.isnan(PON_f).sum()
print("Number of NaN values in POC:", nan_count_poc)
print("Number of NaN values in PON:", nan_count_pon)

# Add the cleaned POC and PON data to the DataFrame
bottle_poc['POC'] = POC_f
bottle_poc['PON'] = PON_f

# Remove rows with NaN POC values
bottle_poc = bottle_poc.dropna(subset=['POC', 'PON'], inplace=False)

# Reset bottle df index removing old index with missing numbers after slice
bottle_poc = bottle_poc.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/HOT_Bottle_POC_AllDepths.csv')

print(116 in bottle_poc['Cruise_ID'].values)

#%%

# Manual quality control

# Remove rows for Cruise_ID 219
msk = (bottle_poc['Cruise_ID'] == 219)
idx_to_drop = bottle_poc.index[msk]
bottle_poc = bottle_poc.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_poc['Cruise_ID'] == 35) #& (bottle_poc['press'].isin([46.6,120.6, 122.6,128.0]))
idx_to_drop = bottle_poc.index[msk]
bottle_poc = bottle_poc.drop(idx_to_drop)

# Remove rows for Cruise_ID 34
msk = (bottle_poc['Cruise_ID'] == 34) #& (bottle_poc['press'].isin([46.6,120.6, 122.6,128.0]))
idx_to_drop = bottle_poc.index[msk]
bottle_poc = bottle_poc.drop(idx_to_drop)

# Remove rows for Cruise_ID 227
msk = (bottle_poc['Cruise_ID'] == 227) & (bottle_poc['press'].isin([4.8,150.5]))
idx_to_drop = bottle_poc.index[msk]
bottle_poc = bottle_poc.drop(idx_to_drop)

# Reset bottle df index removing old index with missing numbers after slice
bottle_poc = bottle_poc.reset_index(drop=True)

#%%

### COUNT POC MEASUREMENTS PER PROFILE < 6 MEASUREMENTS ###

bottle_poc.info()

poc_no_prof = np.zeros(len(bottle_poc))  # Initialize array

for i in range(len(bottle_poc)):
    mask = (bottle_poc['Cruise_ID'] == bottle_poc.at[i, 'Cruise_ID'])
    poc = bottle_poc.loc[mask, 'POC'].values
    depth = bottle_poc.loc[mask, 'depth'].values

# If the shallowest measurement is > 30 OR the deepest measurement is not > 150,
    # then mark the profile as poor quality.
    if depth.min() > 30 or depth.max() <= 170:
        poc_no_prof[i] = 0
    else:
        depth_mask = depth <= 300
        poc = poc[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(poc)
        poc = poc[non_nan_mask]

        poc_no_prof[i] = np.count_nonzero(poc)

# Add Chla count per profile as column to bottle df
bottle_poc['poc_no_prof'] = poc_no_prof

# Remove profiles without more than 6 measurments
bottle_poc = bottle_poc[bottle_poc["poc_no_prof"] > 5]

# Remove rows where depth >= 
bottle_poc = bottle_poc[bottle_poc["depth"] < 1010]

# Sort new df by time and depth again
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

bottle_poc.info()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_poc[["CRN", "POC"]].groupby("CRN").count()
#print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POC")["POC"].count())

# Quick pandas boxplots
bottle_poc.boxplot(by='mm',column='POC')

#%%

### Inspect & Save Final POC Data ###

# Create new df containing only data for profiles also in pigment bottle list
bottle_poc.info()

bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

print(len(bottle_poc))

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_poc[["Cruise_ID", "POC"]].groupby("Cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POC")["POC"].count())

### COUNT BOTTLE PROFILES PER YEAR AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()
# Write to csv
#bottle_y.to_csv('data/BATS_Bottle_profiles_per_year.csv')

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

# Extract all bottle df rows of this profile into new df to print
#bot_prof = bottle_poc[bottle_poc['cruise_ID'] == 30042002]
#print(bot_prof.head(5))

# Write Cleaned bottle df to csv
bottle_poc.to_csv('data/HOT_Bottle_POC.csv')

#%%

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new dataset ###
b2_time     = bottle_poc.loc[:,'DateTime'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_pon      = bottle_poc.loc[:,'PON'].to_numpy()
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()
b2_Decimal_year = bottle_poc.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.DatetimeIndex(b2_time, dtype='datetime64[ns]', name='date_time', freq=None)

### Cruise_ID list
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))
# 303 profiles with 6 or more POC measurements

### CREATE POC PROF DF

# Create POC df with ID list to save single profile values
bottle_poc_prof = bottle_poc.drop_duplicates(subset=['CRN'])
# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

print(len(bottle_poc_prof))
print(len(ID_list_poc))

# Slice df to only have ID, time and date columns
bottle_poc_prof = bottle_poc_prof[['Cruise_ID','DateTime','Date','DecYear','yyyy','mm' ]]
print(len(bottle_poc_prof))

bottle_poc_prof.info()

# Write Cleaned bottle df to csv
bottle_poc_prof.to_csv('data/HOT_Bottle_POC_profData.csv')

#%%

### SCATTER PLOT of ALL POC PROFILES ###

# Create the plot
#my_cmap = plt.get_cmap('cmo.phase_r',12) seaborn colors
# Use a perceptually uniform colormap
my_cmap = plt.get_cmap('viridis', 12)

# Define a custom colormap where warmer colors are in the middle (June and July)
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
          '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026', '#313695']
nodes = [0.0, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 1.0]
cmap = LinearSegmentedColormap.from_list("custom_diverging", list(zip(nodes, colors)), N=12)

fig, ax = plt.subplots(figsize=(8, 10))

# Scatter plot with colormap
scatter = ax.scatter(b2_poc, b2_depth, c=b2_month, alpha=0.7, cmap=cmap)

# Invert y-axis to show depth increasing downwards
ax.set_ylim([260, 0])
ax.set_ylabel('Depth (m)', fontsize=18)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('POC (mg m$^{-3}$)', fontsize=18, labelpad=12)  # Move xlabel slightly further from tick labels
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labelsize=14)

# Add gridlines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend
cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('Month', fontsize=16)
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
cbar.ax.tick_params(labelsize=14)

# Adjust layout for better spacing
plt.tight_layout()

# Save the figure with high resolution
fig.savefig('plots/Bottle-POC_depths.png', format='png', dpi=300, bbox_inches="tight")

plt.show()

#%%

### COMBINED SCATTER PLOT OF CHLA AND POC ###

# Define a custom colormap where warmer colors are in the middle (June and July)
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
          '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026', '#313695']
nodes = [0.0, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 1.0]
cmap = LinearSegmentedColormap.from_list("custom_diverging", list(zip(nodes, colors)), N=12)

# Create the figure and two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

# Scatter plot for Chla with the custom colormap
scatter_chla = axs[0].scatter(b_chla, b_depth, c=b_month, alpha=0.8, cmap=cmap)
axs[0].set_ylim([260, 0])
axs[0].set_ylabel('Depth (m)', fontsize=18)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].xaxis.tick_top()
axs[0].set_xlabel('HPLC Chla (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[0].xaxis.set_label_position('top')
axs[0].xaxis.set_tick_params(labelsize=14)
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# =============================================================================
# # Add colorbar for Chla subplot
# cbar_chla = plt.colorbar(scatter_chla, ax=axs[0], orientation='vertical', pad=0.02)
# cbar_chla.set_label('Month', fontsize=16)
# cbar_chla.set_ticks(np.arange(1, 13))
# cbar_chla.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# cbar_chla.ax.tick_params(labelsize=14)
# =============================================================================

# Scatter plot for POC with the custom colormap
scatter_poc = axs[1].scatter(b2_poc, b2_depth, c=b2_month, alpha=0.8, cmap=cmap)
axs[1].set_ylim([260, 0])
axs[1].yaxis.set_tick_params(labelsize=14)
axs[1].xaxis.tick_top()
axs[1].set_xlabel('POC (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[1].xaxis.set_label_position('top')
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# =============================================================================
# # Add colorbar for POC subplot
# cbar_poc = plt.colorbar(scatter_poc, ax=axs[1], orientation='vertical', pad=0.02)
# cbar_poc.set_label('Month', fontsize=16)
# cbar_poc.set_ticks(np.arange(1, 13))
# cbar_poc.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# cbar_poc.ax.tick_params(labelsize=14)
# =============================================================================

# Add a single shared colorbar
cbar = fig.colorbar(scatter_chla, ax=axs, orientation='vertical', pad=0.02, location='right')
cbar.set_label('Month', fontsize=16)
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
cbar.ax.tick_params(labelsize=14)

# Save the figure with high resolution
fig.savefig('plots/Scatter_Chla_POC_depths_all.png', format='png', dpi=300, bbox_inches="tight")

plt.show()

#%%

### COMBINED SCATTER PLOT OF CHLA, POC & PON ###

# Define a custom colormap where warmer colors are in the middle (June and July)
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
          '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026', '#313695']
nodes = [0.0, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 1.0]
cmap = LinearSegmentedColormap.from_list("custom_diverging", list(zip(nodes, colors)), N=12)

# Create the figure and two subplots
fig, axs = plt.subplots(1, 3, figsize=(16, 8), sharey=True)

# Scatter plot for Chla with the custom colormap
scatter_chla = axs[0].scatter(b_chla, b_depth, c=b_month, alpha=0.8, cmap=cmap)
axs[0].set_ylim([260, 0])
axs[0].set_ylabel('Depth (m)', fontsize=18)
axs[0].yaxis.set_tick_params(labelsize=14)
axs[0].xaxis.tick_top()
axs[0].set_xlabel('HPLC Chla (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[0].xaxis.set_label_position('top')
axs[0].xaxis.set_tick_params(labelsize=14)
axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# =============================================================================
# # Add colorbar for Chla subplot
# cbar_chla = plt.colorbar(scatter_chla, ax=axs[0], orientation='vertical', pad=0.02)
# cbar_chla.set_label('Month', fontsize=16)
# cbar_chla.set_ticks(np.arange(1, 13))
# cbar_chla.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# cbar_chla.ax.tick_params(labelsize=14)
# =============================================================================

# Scatter plot for POC with the custom colormap
scatter_poc = axs[1].scatter(b2_poc, b2_depth, c=b2_month, alpha=0.8, cmap=cmap)
axs[1].set_ylim([260, 0])
axs[1].yaxis.set_tick_params(labelsize=14)
axs[1].xaxis.tick_top()
axs[1].set_xlabel('POC (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[1].xaxis.set_label_position('top')
axs[1].xaxis.set_tick_params(labelsize=14)
axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

# =============================================================================
# # Add colorbar for POC subplot
# cbar_poc = plt.colorbar(scatter_poc, ax=axs[1], orientation='vertical', pad=0.02)
# cbar_poc.set_label('Month', fontsize=16)
# cbar_poc.set_ticks(np.arange(1, 13))
# cbar_poc.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# cbar_poc.ax.tick_params(labelsize=14)
# =============================================================================

# Scatter plot for POC with the custom colormap
scatter_poc = axs[2].scatter(b2_pon, b2_depth, c=b2_month, alpha=0.8, cmap=cmap)
axs[2].set_ylim([260, 0])
axs[2].yaxis.set_tick_params(labelsize=14)
axs[2].xaxis.tick_top()
axs[2].set_xlabel('PON (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[2].xaxis.set_label_position('top')
axs[2].xaxis.set_tick_params(labelsize=14)
axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a single shared colorbar
cbar = fig.colorbar(scatter_chla, ax=axs, orientation='vertical', pad=0.02, location='right')
cbar.set_label('Month', fontsize=16)
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
cbar.ax.tick_params(labelsize=14)

# Adjust layout for better spacing and colorbar positioning
#plt.subplots_adjust(right=0.85)  # Adjust the right side of the subplots to make room for the colorbar

# Adjust layout for better spacing
#plt.tight_layout()

# Save the figure with high resolution
fig.savefig('plots/Scatter_Chla_POC_PON_depths_all.png', format='png', dpi=300, bbox_inches="tight")

plt.show()

#%%

### POC SINGLE PROFILE PLOT ###

ID_1 = 	328#56#296#17#69#116#69

poc_max_ID = b2_ID[np.where(b2_poc == np.max(b2_poc))]
print(poc_max_ID)

# Find POC profile data

# Get MLD from CTD
prof_MLD_idx = np.where(ID_list_ctd == ID_1)
prof_MLD     = MLD[prof_MLD_idx]

# =============================================================================
# # Chla data from Bottle
# prof_bottle_idx = np.where(bottle_6.Cruise_ID == ID_1)
# prof_chla    = b_chla[prof_bottle_idx]
# prof_chla_depth = b_depth[prof_bottle_idx]
# =============================================================================

ID_where = (bottle_poc['CRN'] == ID_1)
bot_ID = bottle_poc['Cruise_ID'].loc[ID_where]
bot_ID = int(np.unique(bot_ID))


# Bottle data for Profile = ID_1
AS = np.where(bottle_poc.CRN == ID_1) # Index for bottle data

# PP data
prof_poc_idx     = np.where(bottle_poc.CRN == ID_1)
prof_poc_1       = b2_poc[prof_poc_idx]
prof_pon_1       = b2_pon[prof_poc_idx]
prof_poc_depth_1 = b2_depth[prof_poc_idx]

print(prof_poc_1)

b2_DateTime_1 = b2_DateTime.date[prof_poc_idx]
print(b2_DateTime_1[0])


print("MLD: "+str(prof_MLD))

#Define the figure window including 5 subplots orientated horizontally
fig, (ax3,ax4) = plt.subplots(1,2, sharey=True, figsize=(10,6), \
gridspec_kw={'hspace': 0.2})

ax3.axhline(y=prof_MLD, color = 'k',linestyle = '--', label= 'MLD')
ax3.plot(prof_poc_1,prof_poc_depth_1, \
         color = 'g',  marker = 'o', linestyle = '-',label= 'POC')
ax3.set_ylabel('Depth (m)', fontsize=20)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_ylim([270,0]) 
ax3.set_xlabel('POC (ug/kg)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xlim(xmin=0.00, xmax=np.max(prof_poc_1)+1.5)
ax3.legend(loc="lower right", fontsize=10,title= ID_1)
ax3.text(np.min(prof_poc_1)+0.005, 268, " Date: "+str(b2_DateTime_1[0]), color='k', fontsize=12)
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax3.set_xscale('log')

ax4.axhline(y=prof_MLD, color = 'k',linestyle = '--', label= 'MLD')
ax4.plot(prof_pon_1,prof_poc_depth_1, \
         color = 'm',  marker = 'o', linestyle = '-',label= 'PON')
ax4.set_ylabel('Depth (m)', fontsize=20)
ax4.yaxis.set_tick_params(labelsize=15)
ax4.set_ylim([270,0]) 
ax4.set_xlabel('PON (ug/kg)', fontsize=15, color = 'k')
ax4.xaxis.set_tick_params(labelsize=15)
ax4.set_xlim(xmin=0.00, xmax=np.max(prof_pon_1)*1.1)
ax4.legend(loc="lower right", fontsize=10,title= ID_1)
ax4.text(np.min(prof_poc_1)+0.005, 268, " Date: "+str(b2_DateTime_1[0]), color='k', fontsize=12)
ax4.xaxis.set_major_locator(plt.MaxNLocator(5))

plt.show()
