"""
HOT - Inspect & Clean Particulate Phosphorous Data & setup dataframe

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.colors import LinearSegmentedColormap
# Import specific modules from packages
from dateutil import relativedelta
from scipy.stats import zscore
# Supress
import warnings
warnings.filterwarnings("ignore")

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    rd = relativedelta.relativedelta( pd.to_datetime( end ), pd.to_datetime( start ) )
    #date_len = str(rd.years)+"yrs"+str(rd.months)+"m"+str(rd.days+"d")
    date_len  = '{}y{}m{}d'.format(rd.years,rd.months,rd.days)
    #return rd.years, rd.months, rd.days
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
    #local_maxima_mask = (data > rolling_median) & (deviations > threshold)
    
    # Combine all masks
    combined_mask = outlier_mask | local_minima_mask #| local_maxima_mask

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

ID_list_ctd       = ctd_prof['Cruise_ID'].values
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
print(len(ID_list_6))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_6['Date']), max(bottle_6['Date']))
print("Timespan: "+str(b_date_length)) # 32y5m14d

#%%

### IMPORT CLEANED POC DATA & MAKE BOTTLE ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Bottle_POC.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)

bottle_poc.info()

### Extract required data from df ###
b2_time     = bottle_poc.loc[:,'DateTime'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_pon      = bottle_poc.loc[:,'PON'].to_numpy()
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_poc['DateTime'].values)

### Cruise_ID list
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))

#%%

######
### READ & CLEAN ORIGINAL BOTTLE DATA ###
######

#Filename
file_1 = 'data/HOT_Bottle_Cleaned2.csv' #name file location and csv file name
#file_pigment2 = 'data/niskin_BCO_BMO.csv' #name file location and csv file name

# Step 1: Read the first two rows to get column names and units
df = pd.read_csv(file_1, index_col = 0)
df.info()

#Select smaller set of columns
bot = df[["Cruise_ID","CRN","CASTNO", "Cruise_ID_o","DateTime",'DecYear','Date','yyyy','mm','time','press', 'depth',
             'pp']]
# low level nitrate and phosphate might be usefull need separate df for each and then later match per cruise because sampled on different casts.

bot.head()
bot.info()

# Rename Columns
bot.rename(columns={"pp": "POP"},inplace=True)

#For HPLC Chla Dataset
# Remove rows with NaN HPLC chla values
bot = bot.dropna(subset=['POP'])
# Remove rows with NaN depth values
bot = bot.dropna(subset=['depth'])
bot.info()

#bot = bot.drop_duplicates(subset=["Cruise_ID", "depth"]) sometime multiple measurements per depth. use outlier check to remove rather than drop duplicate as this just keep first combo and drops rest

# Remove rows with depths below 300m
bot = bot[bot["depth"]<1010]
# POP original units = nmol/kg, convert to both nmol/L and mg/m3
bot['POP_nmolL'] = bot['POP']*1025/1000#
# Convert nmol/kg to mg/m³ using seawater density of 1025 kg/m³
bot['POP'] = bot['POP']*30.97376*1025/1000/1000#

# Rename Columns
bot.info()
# sort by time
bot = bot.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bot = bot.reset_index(drop=True)

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bot['Date'] = pd.to_datetime(bot['DateTime']).dt.date
#print(bottle['Date'])

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
print(bot[["POP"]].describe())

# Boxplot of Turner, HPLC and Avg Chla
bot[["POP"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bot[["Cruise_ID", "POP"]].groupby("Cruise_ID").count()
print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POP")["POP"].count())

# Bar plot of profiles per number per number of depths sampled
bottle_poc_profcount.groupby("POP")["POP"].count().plot.bar()
plt.show()

### COUNT POC PROFILES PER YEAR ###

# Create new df with number of CTD profiles (cruises) per year
bottle_poc_y = bot[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Write to csv
#PPpd_y.to_csv('data/BATS_PP_profiles_per_year.csv')

# Bar plot of profiles per year
bottle_poc_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_poc_y.to_markdown())


#%%

### RENAME DF ###

# Remove rows with NaN POC values
bottle_pop = bot.dropna(subset=['POP'], inplace=False)
print(len(bot))
print(len(bottle_pop))
# Sort new df by time and depth again
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])
# Reset bottle df index removing old index with missing numbers after slice
bottle_pop = bottle_pop.reset_index(drop=True)

#%%

### REMOVE PROFILES WITH NO MATCHING PIGMENT profiles ###

### Cruise_ID list
b2_ID       = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
ID_list_pop = pd.unique(b2_ID)

print(len(ID_list_pop))
print(len(ID_list_6))

# Create new df containing only data for profiles also in pigment bottle list
bottle_pop =  bottle_pop[bottle_pop.Cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

### Cruise_ID list
b2_ID       = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
ID_list_pop = pd.unique(b2_ID)
print(len(ID_list_pop))

print(len(bottle_pop))

# Write Cleaned bottle df to csv
bottle_pop.to_csv('data/HOT_Bottle_POP_AllDepths.csv')

cast_counts = bottle_pop.groupby(["CRN", "CASTNO"]).size().reset_index(name='count')
 
# Identify duplicated 'CRN' entries
cast_duplicates = cast_counts[cast_counts['CRN'].duplicated(keep=False)]

#%%

# Remove duplicate measurements
bottle_pop.drop_duplicates(subset=['Cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by ID and depth
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

#%%

### REMOVE SPIKES FROM POP PROFILES ###

bottle_pop.info()

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b3_depth    = bottle_pop.loc[:,'depth'].to_numpy()
b3_pop      = bottle_pop.loc[:,'POP'].to_numpy()
b3_ID       = bottle_pop.loc[:,'Cruise_ID'].to_numpy()

### Cruise_ID list & Removes Duplicates
ID_list_pop = pd.unique(b3_ID)
print(len(ID_list_pop))

# New array to store cleaned POP data
POP_f = np.full(len(b3_pop), np.nan)  # Initialize with NaNs

# Loop through each profile
for ID in ID_list_pop:
    # Indices for the current profile
    prof_pop_idx = np.where(bottle_pop.Cruise_ID == ID)
    
    # Extract POP data
    prof_pop = b3_pop[prof_pop_idx]
    prof_pop_depth = b3_depth[prof_pop_idx]
    
    # OUTLIER REMOVAL FOR POP
    filtered_pop, filtered_depths = remove_outliers_local_minima(prof_pop, prof_pop_depth, window_size=4, threshold=2.3, replace_with_nans=True)
    
    # Convert the filtered data to arrays
    filtered_pop = np.array(filtered_pop)
    
    # Save the filtered data back into the initialized arrays
    POP_f[prof_pop_idx] = filtered_pop

# Count NaN values for POP
nan_count_pop = np.isnan(POP_f).sum()
print("Number of NaN values in POP:", nan_count_pop)

# Add the cleaned POP data to the DataFrame
bottle_pop['POP'] = POP_f

# Remove rows with NaN POC values
bottle_pop = bottle_pop.dropna(subset=['POP'], inplace=False)

# Reset bottle df index removing old index with missing numbers after slice
bottle_pop = bottle_pop.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_pop.to_csv('data/HOT_Bottle_POP_AllDepths.csv')

#%%

# Manual quality control

#ID 220 & press > 170.6
msk = (bottle_pop['Cruise_ID'] == 220) & (bottle_pop['press'] >300)
# count how many points will be removed
n_removed = msk.sum()
print(f"Removing {n_removed} rows (Cruise_ID=220, press > 300 dbar)")
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

#ID 161
msk = (bottle_pop['Cruise_ID'] == 161)
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 303) & (bottle_pop['press'].isin([201.4,251.2]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 215) & (bottle_pop['press'].isin([46.1,174.4]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 323) & (bottle_pop['press'].isin([148.7,349.3]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 324) & (bottle_pop['press'].isin([351.2]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 319) & (bottle_pop['press'].isin([349.6]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 42) & (bottle_pop['press'].isin([200,220]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 164) & (bottle_pop['press'].isin([4]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 177) & (bottle_pop['press'].isin([9.1,45.8]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 35
msk = (bottle_pop['Cruise_ID'] == 215) & (bottle_pop['press'].isin([25.2]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 217
msk = (bottle_pop['Cruise_ID'] == 217) & (bottle_pop['press'].isin([4.9]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Remove rows for Cruise_ID 227
msk = (bottle_pop['Cruise_ID'] == 227) & (bottle_pop['press'].isin([4.8]))
idx_to_drop = bottle_pop.index[msk]
bottle_pop = bottle_pop.drop(idx_to_drop)

# Reset bottle df index removing old index with missing numbers after slice
bottle_pop = bottle_pop.reset_index(drop=True)

"Total of 1x profile and 18 additional outliers data points manually removed"


#%%

### COUNT POP MEASUREMENTS PER PROFILE < 6 MEASUREMENTS ###

bottle_pop.info()

pop_no_prof = np.zeros(len(bottle_pop))  # Initialize array

for i in range(len(bottle_pop)):
    mask = (bottle_pop['Cruise_ID'] == bottle_pop.at[i, 'Cruise_ID'])
    pop = bottle_pop.loc[mask, 'POP'].values
    depth = bottle_pop.loc[mask, 'depth'].values

# If the shallowest measurement is > 30 OR the deepest measurement is not > 150,
    # then mark the profile as poor quality.
    if depth.min() > 30 or depth.max() <= 170:
        pop_no_prof[i] = 0
    else:
        depth_mask = depth <= 300
        pop = pop[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(pop)
        pop = pop[non_nan_mask]

        pop_no_prof[i] = np.count_nonzero(pop)

# Add Chla count per profile as column to bottle df
bottle_pop['pop_no_prof'] = pop_no_prof

# Remove profiles without more than 6 Chl measurments
bottle_pop = bottle_pop[bottle_pop["pop_no_prof"] > 5]

# Remove rows where depth >= 
bottle_pop = bottle_pop[bottle_pop["depth"] < 1010]

# Sort new df by time and depth again
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])
#bottle_poc = bottle_poc.sort_values(by=['time','cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

bottle_pop.info()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_pop[["CRN", "POP"]].groupby("CRN").count()
#print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POP")["POP"].count())

# Quick pandas boxplots
bottle_pop.boxplot(by='mm',column='POP')

# Finding indices where 'pop_no_prof' equals 6
indices = np.where(bottle_pop["pop_no_prof"] == 23)

# Extracting the corresponding values from the 'CRN' column
corresponding_crn_values = bottle_pop.loc[indices, "CRN"]

# Display the result
print(corresponding_crn_values)

#%%

### Inspect & Save Final POP Data ###

bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

print(len(bottle_pop))

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_pop[["Cruise_ID", "POP"]].groupby("Cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("POP")["POP"].count())

### COUNT BOTTLE PROFILES PER YEAR AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

# Write Cleaned bottle df to csv
bottle_pop.to_csv('data/HOT_Bottle_POP.csv')

#%%

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new dataset ###
b3_time     = bottle_pop.loc[:,'DateTime'].to_numpy()
b3_date     = bottle_pop.loc[:,'Date'].to_numpy()
b3_depth    = bottle_pop.loc[:,'depth'].to_numpy()
b3_pop      = bottle_pop.loc[:,'POP'].to_numpy()
b3_ID       = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
b3_year     = bottle_pop.loc[:,'yyyy'].to_numpy()
b3_month    = bottle_pop.loc[:,'mm'].to_numpy()
b3_Decimal_year = bottle_pop.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b3_DateTime = pd.DatetimeIndex(b3_time, dtype='datetime64[ns]', name='date_time', freq=None)

### Cruise_ID list
ID_list_pop = pd.unique(b3_ID)
print(len(ID_list_pop))

### CREATE POC PROF DF

# Create POC df with ID list to save single profile values
bottle_pop_prof = bottle_pop.drop_duplicates(subset=['CRN'])

# Reset bottle df index replacing old index column
bottle_pop_prof = bottle_pop_prof.reset_index(drop=True)

# Slice df to only have ID, time and date columns
bottle_pop_prof = bottle_pop_prof[['Cruise_ID','DateTime','Date','DecYear','yyyy','mm' ]]
print(len(bottle_pop_prof))

bottle_pop_prof.info()

# Write Cleaned bottle df to csv
bottle_pop_prof.to_csv('data/HOT_Bottle_POP_profData.csv')

#%%

### SCATTER PLOT of ALL POP PROFILES ###

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
scatter = ax.scatter(b3_pop, b3_depth, c=b3_month, alpha=0.7, cmap=cmap)

# Invert y-axis to show depth increasing downwards
ax.set_ylim([260, 0])
ax.set_ylabel('Depth (m)', fontsize=18)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('POP (mg m$^{-3}$)', fontsize=18, labelpad=12)  # Move xlabel slightly further from tick labels
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
fig.savefig('plots/Bottle-POP_depths.png', format='png', dpi=300, bbox_inches="tight")

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
fig, axs = plt.subplots(1, 4, figsize=(25, 8), sharey=True)

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

# Scatter plot for PON with the custom colormap
scatter_poc = axs[2].scatter(b2_pon, b2_depth, c=b2_month, alpha=0.8, cmap=cmap)
axs[2].set_ylim([260, 0])
axs[2].yaxis.set_tick_params(labelsize=14)
axs[2].xaxis.tick_top()
axs[2].set_xlabel('PON (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[2].xaxis.set_label_position('top')
axs[2].xaxis.set_tick_params(labelsize=14)
axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

# Scatter plot for PON with the custom colormap
scatter_poc = axs[3].scatter(b3_pop, b3_depth, c=b3_month, alpha=0.8, cmap=cmap)
axs[3].set_ylim([260, 0])
axs[3].yaxis.set_tick_params(labelsize=14)
axs[3].xaxis.tick_top()
axs[3].set_xlabel('POP (mg m$^{-3}$)', fontsize=18, labelpad=12)
axs[3].xaxis.set_label_position('top')
axs[3].xaxis.set_tick_params(labelsize=14)
axs[3].grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a single shared colorbar
cbar = fig.colorbar(scatter_chla, ax=axs, orientation='vertical', pad=0.02, location='right')
cbar.set_label('Month', fontsize=16)
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
cbar.ax.tick_params(labelsize=14)

# Save the figure with high resolution
fig.savefig('plots/Scatter_Chla_POC_PON_POP_depths_all.png', format='png', dpi=300, bbox_inches="tight")

plt.show()

#%%

### POP SINGLE PROFILE PLOT ###

ID_1 = 	328

pop_max_ID = b3_ID[np.where(b3_pop == np.max(b3_pop))]
print(pop_max_ID)

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

ID_where = (bottle_pop['CRN'] == ID_1)
bot_ID = bottle_pop['Cruise_ID'].loc[ID_where]
bot_ID = int(np.unique(bot_ID))


# Bottle data for Profile = ID_1
#AS = np.where(bottle_poc.Cruise_ID == bot_ID) # Index for bottle data
AS = np.where(bottle_pop.CRN == ID_1) # Index for bottle data

# PP data
prof_pop_idx     = np.where(bottle_pop.CRN == ID_1)
#prof_poc_idx     = np.where(bottle_poc.cruise_ID == ID_1)
prof_pop_1       = b3_pop[prof_pop_idx]
prof_pop_depth_1 = b3_depth[prof_pop_idx]

print(prof_pop_1)

b3_DateTime_1 = b3_DateTime.date[prof_pop_idx]
print(b3_DateTime_1[0])


print("MLD: "+str(prof_MLD))

#Define the figure window including 5 subplots orientated horizontally
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(6,6), \
gridspec_kw={'hspace': 0.2})

ax3.plot([np.min(prof_pop_1)-0.25,np.max(prof_pop_1)+0.05],[prof_MLD,prof_MLD], \
         color = 'k', marker = 'None', linestyle = '--', label= 'MLD')
ax3.plot(prof_pop_1,prof_pop_depth_1, \
         color = 'k',  marker = 'o', linestyle = '-',label= 'POP')
ax3.set_ylabel('Depth (m)', fontsize=20)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_ylim([270,0]) 
ax3.set_xlabel('POP (mg m$^{-3}$)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xlim(xmin=0.00, xmax=np.max(prof_pop_1)+0.05)
ax3.legend(loc="lower right", fontsize=10,title= ID_1)
ax3.text(np.min(prof_pop_1)+0.005, 268, " Date: "+str(b3_DateTime_1[0]), color='k', fontsize=12)
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax3.set_xscale('log')
plt.show()
