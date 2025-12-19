"""
HOT: Inspect & Clean HPLC CHL profiles & setup into data frame

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
import matplotlib as mpl
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# Import specific modules from packages
from dateutil import relativedelta
from matplotlib import pyplot as plt
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

#%%
######
### READ, CLEAN & FILTER ORIGINAL BOTTLE PIGMENT DATA ###
######

#Filename
file_pigment1 = 'data/HOT_Bottle_Cleaned2.csv' #name file location and csv file name

df = pd.read_csv(file_pigment1, index_col = 0)
df.info()

#Select columns For HPLC Chla Dataset
bottle = df[["Cruise_ID","CRN","CASTNO", "Cruise_ID_o","DateTime",'DecYear','Date','yyyy','mm','time','press', 'depth',
             'chl', 'hplc']]
# Rename Columns
bottle.rename(columns={"chl": "Fchla","hplc": "Chla"},inplace=True)
bottle.head()
bottle.info()


#%%

# Inspect HPLC Chl vs Fluo Chl

# Remove rows with NaN HPLC chla values
bottle_chl_hplc_F = bottle.dropna(subset=['Fchla'])
# Remove rows with NaN depth values
bottle_chl_hplc_F = bottle_chl_hplc_F.dropna(subset=['depth'])
bottle_chl_hplc_F = bottle_chl_hplc_F.reset_index(drop=True)
bottle_chl_hplc_F.info()

from scipy.stats import spearmanr

# Compute Spearman correlation between Chla (x) and Fchla (y)
x = bottle_chl_hplc_F['Chla']/1000
y = bottle_chl_hplc_F['Fchla']
rho, pval = spearmanr(x, y)

# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.7, edgecolor='k')

# Label axes
plt.xlabel('HPLC-Chla (µg/L)')
plt.ylabel('Fchla (µg/L)')
plt.title('Scatter: Fchla vs Chla')

# Annotate with Spearman r and p‐value
text_str = f"Spearman $r$ = {rho:.2f}\n$p$ = {pval:.2e}"
# Place annotation in upper left corner with a white box behind text
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))

plt.savefig('plots/HPLCvsTurner_Scatter_All.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

plt.tight_layout()
plt.show()

#%%

# Remove rows with NaN HPLC chla values
bottle = bottle.dropna(subset=['Chla'])
# Remove rows with NaN depth values
bottle = bottle.dropna(subset=['depth'])
bottle = bottle.reset_index(drop=True)
bottle.info()

bottle = bottle.drop_duplicates(subset=["Cruise_ID", "depth"])

bottle['Cruise_ID'].head()
bottle['Chla'].head()

# Remove rows with depths below 300m
bottle = bottle[bottle["depth"]<405]

# Convert HPLC Chla to ug/L # Most HPLC concentrations in ng/kg, convert to ug/kg
bottle['Chla'] = bottle['Chla']/1000 # replaced original column with converted values

# sort
bottle = bottle.sort_values(by=['Date','depth'])
# Reset bottle df index removing old index with missing numbers after slice
bottle = bottle.reset_index(drop=True)

#%%

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract year into new column called "yyyy"
bottle['yyyy'] = pd.to_datetime(bottle['DateTime']).dt.year 
#print(bottle['yyyy'])

# convert to datetime format & extract mopnth into new column called "mm"
bottle['mm'] = pd.to_datetime(bottle['DateTime']).dt.month
#print(bottle['mm'])

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bottle['Date'] = pd.to_datetime(bottle['DateTime']).dt.date
#print(bottle['Date'])

### Timespan of bottle data ###

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle['Date']))+" to "+str(max(bottle['Date'])))

# Print period timespan of bottle data using base date subtraction - only days
print("Bottle Date Length: "+str(max(bottle['Date'])-min(bottle['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle['Date']), max(bottle['Date']))
print("Timespan: "+str(b_date_length))

#%%
### FIND MAX CHLA PROFILE ###

#Now only using HPLC and checking for outliers in HPLC only
bottle.info()
b_chla  = bottle.loc[:,'Chla'].to_numpy()
b_Fchla = bottle.loc[:,'Fchla'].to_numpy()

print(np.nanmax(b_Fchla))
print(np.nanmax(b_chla))

bot_prof_max = bottle[bottle['Chla'] == np.nanmax(b_chla)]
print(bot_prof_max) # Only the one row containing max HPLC chla

# Extract Cruise ID of max chla profile
ID_max = bot_prof_max['Cruise_ID'].values
print(ID_max)

# Extract Date of max chla profile
b_date_max = bot_prof_max['Date']
print(b_date_max)

# No outlier profiles in HPLC data

#%%
### SUMMARY & DESCRIPTIVE STATS on CHL Data ###

# Inspect first few rows of current dataframe
print(bottle.head(5))
#bottle.info()

# Summary stats Table for Turner, HPLC and Avg Chla
print(bottle[["Chla","Fchla"]].describe())

# Boxplot of Turner, HPLC and Avg Chla
bottle[["Chla","Fchla"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_2 = bottle[["Cruise_ID", "Chla", "Fchla"]].groupby("Cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_2.groupby("Fchla")["Fchla"].count())
print(bottle_2.groupby("Chla")["Chla"].count())

# Bar plot of profiles per number of depths sampled
bottle_2.groupby("Fchla")["Fchla"].count().plot.bar()
plt.show()
bottle_2.groupby("Chla")["Chla"].count().plot.bar()
plt.show()
#%%

### COUNT HPLC CHL PROFILES PER YEAR ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Print nice Table for Notebook
print(bottle_y.to_markdown())

#%%
### REMOVE DUPLICATE CASTS PER CRUISE ###

# Drop rows where Chl = zero
bottle = bottle.drop(bottle[(bottle.Chla == 0) & (bottle.depth < 101)].index)

#bottle = bottle.sort_values(by=['Date','depth'])
bottle = bottle.sort_values(by=["CRN","CASTNO",'depth'])

bottle.info()

### Remove casts with least measurements where more than one cast per cruise ###
# Assume 'bottle' is a DataFrame that has been previously defined
data = bottle.copy()

# Group by 'CRN' and 'CASTNO' and count number of samples per profile
cast_counts = data.groupby(["CRN", "CASTNO"]).size().reset_index(name='count')

# Identify duplicated 'CRN' entries
cast_duplicates = cast_counts[cast_counts['CRN'].duplicated(keep=False)]
print(cast_duplicates.to_markdown())

# Find indices of rows with the minimum count of 'CASTNO' for each 'CRN'
cast_duplicates_min_idx = cast_duplicates.loc[cast_duplicates.groupby('CRN')['count'].idxmin()]

# Merge data with the minimum duplicate casts to identify non-matching rows
merged_df = data.merge(cast_duplicates_min_idx, on=['CRN', 'CASTNO'], how='left', indicator=True)

# Filter out rows that have a match in the merge (keep only 'left_only' entries)
data_filtered = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['count', '_merge'])

# Sort the data by 'Date' and 'depth'
data_sorted = data_filtered.sort_values(by=['Date', 'depth']).reset_index(drop=True)

# Copy the cleaned and sorted DataFrame back to 'bottle'
bottle = data_sorted.copy()

#%%

### COUNT CHL MEASUREMENTS PER PROFILE < 6 MEASUREMENTS ###

# Also include only profiles that have at least one measurement above 30m
        
No_CHL_Prof = np.zeros(len(bottle))  # Initialize array

for i in range(len(bottle)):
    mask = (bottle['Cruise_ID'] == bottle.at[i, 'Cruise_ID'])
    chla = bottle.loc[mask, 'Chla'].values
    depth = bottle.loc[mask, 'depth'].values

    if depth.min() > 30:
        No_CHL_Prof[i] = 0
    else:
        depth_mask = depth <= 300
        chla = chla[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(chla)
        chla = chla[non_nan_mask]

        No_CHL_Prof[i] = np.count_nonzero(chla)



# Add Chla count per profile as column to bottle df
bottle['No_CHL_Prof'] = No_CHL_Prof

# Remove profiles without more than 6 Chl measurments
bottle_6 = bottle[bottle["No_CHL_Prof"] > 5]

# Sort
bottle_6 = bottle_6.sort_values(by=['Date','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Test if new df count only includes 6 or more per Cruise ID
bottle_6[["Cruise_ID","Chla"]].groupby("Cruise_ID").count()

# Quick pandas boxplots
bottle_6.boxplot(by='mm',column='Chla')

#%%

### COUNT BOTTLE PROFILES PER YEAR AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

#%%

### EXTRACT CLEANED DATA & MAKE BOTTLE ID LIST ###

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'DateTime'].to_numpy()
b_time_2   = pd.to_datetime(bottle_6['DateTime'])
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_Fchla    = bottle_6.loc[:,'Fchla'].to_numpy()
b_chla     = bottle_6.loc[:,'Chla'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()

#Convert bottle float time to Datetimeindex type
b_DateTime = pd.DatetimeIndex(b_time, dtype='datetime64[ns]', name='date_time', freq=None)

### Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(b_ID)

print(len(ID_list_6))

#%%
### SUMMARY PLOTS FOR NEW FILTERED BOTTLE DATA ##

# Line plot of HPLC chla vs Time
fig, ax = plt.subplots(1, figsize=(8,5))
plt.plot(b_time_2,b_chla)
#fig.savefig('plots/Bottle-Chla_HPLC_6depths.png', format='png', dpi=300, bbox_inches="tight")

# Scatter plot of all HPLC chla vs depth with month as colour
fig, ax = plt.subplots(1, figsize=(6,7), \
gridspec_kw={'hspace': 0.4})   
scatter = ax.scatter(b_chla,b_depth, c = b_month, alpha = 0.6, cmap = mpl.colormaps['viridis_r'])
ax.set_ylim([260,0])
ax.set_ylabel('Depth (m)', fontsize=15)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('HPLC Chla', fontsize=15)   
ax.xaxis.set_label_position('top')
ax.xaxis.set_tick_params(labelsize=14)
# add legend to the plot with names
ax.legend(loc="best", fontsize=11, handles=scatter.legend_elements()[0], 
           title="Month", labels = scatter.legend_elements()[1])
fig.savefig('plots/Bottle-HPLC_chla_depths.png', format='png', dpi=300, bbox_inches="tight")

# Scatter plot HPLC vs Turner Chla
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(8,8))
scatter = ax3.scatter(b_Fchla,b_chla, alpha=0.5, c = 'g')
ax3.set_ylabel('HPLC Chl-a (ug/L)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Fluorescence Chl-a (ug/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
# =============================================================================
# # add legend to the plot with names
# ax3.legend(loc="best", fontsize=11, handles=scatter.legend_elements()[0], 
#            title="Year", labels = scatter.legend_elements()[1])
# =============================================================================
fig.savefig('plots/HPLCvsTurner_Scatter.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

# Scatter plot HPLC vs Turner Chla
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(8,8), \
gridspec_kw={'hspace': 0.2})   
ax3.scatter(b_Fchla,b_chla, alpha=0.5, c = 'g')
ax3.set_ylabel('HPLC Chl-a (ug/L)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Fluorescence Chl-a (ug/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xscale('log')
ax3.set_yscale('log')
#fig.savefig('plots/HPLCvsTurner_Scatter.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

#%%

# Box plots per month all HPLC, Turner and Avg Chla data (Seaborn package)
fig, (ax1,ax2) = plt.subplots(2, figsize=(8,10),gridspec_kw={'hspace': 0.3})
#fig, ax = plt.subplots()
#fig.set_size_inches((12,4))
sns.boxplot(x='mm',y='Chla',data=bottle_6,ax=ax1)
ax1.set_title('(a) HPLC Chlorophyll-a (µg/l)', fontsize = 20, color='k')
ax1.set_ylabel('Chl-a (µg/L)', fontsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Month', fontsize=15, color = 'k')
ax1.xaxis.set_tick_params(labelsize=15)

# Box plot of all HPLC chla vs depth
sns.boxplot(x='mm',y='Fchla',data=bottle_6,ax=ax2)
ax2.set_title('(b) Turner Chlorophyll-a (µg/l)', fontsize = 20, color='k')
ax2.set_ylabel('Chl-a (µg/L)', fontsize=15)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Month', fontsize=15, color = 'k')
ax2.xaxis.set_tick_params(labelsize=15)


plt.show()
fig.savefig('plots/Monthly_Boxplot_Chla.png', format='png', dpi=300, bbox_inches="tight")

#%%

### EXTRACT CTD DATA ###

### READ CLEANED CTD DATA FROM CSV ###

# CSV filename
filename_1 = 'data/HOT_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

### Extract required data from CTD dataframe into numpy arrays ###
ctd_time      = ctd.loc[:,'DateTime'].to_numpy()
ctd_date      = ctd.loc[:,'Date'].to_numpy()
time_year     = ctd.loc[:,'yyyy'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
temperature   = ctd.loc[:,'temperature'].to_numpy()
salinity      = ctd.loc[:,'salinity'].to_numpy()
density       = ctd.loc[:,'density'].to_numpy()
bvf           = ctd.loc[:,'BVF'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
doxy          = ctd.loc[:,'dissolved_oxygen'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
time_2        = pd.to_datetime(ctd['DateTime']) # panda series
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.DatetimeIndex(time_2)

### Cruise ID list for CTD ###
# Extract cruise_ID
ID_list_ctd = ctd['cruise_ID'].values

# Converts to pandas timeseries array
ID_list_ctd = pd.Series(ID_list_ctd)

# Removes Duplicates
ID_list_ctd = pd.unique(ID_list_ctd) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))

### Read CTD prof data ###

# CSV filename
filename_2 = 'data/HOT_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2, index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()

# Extract required data from df
ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['DateTime'])
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD_boyer'].to_numpy()

#%%
### REMOVE BOTTLE PROFILES WITH NO MATCHING CTD MLD ###

#Create new bottle df containing only rows matching Cruise_ID in the ID_list_6
# Test
ctdmatch = bottle_6.Cruise_ID.isin(ID_list_ctd)

# Extract new df that only has matching CTD cruise IDs
bottle_6 = bottle_6[bottle_6.Cruise_ID.isin(ID_list_ctd)]

# Sort df again
bottle_6 = bottle_6.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Test if new df count only includes 6 or more per Cruise ID
bottle_2 = bottle_6[["Cruise_ID","Chla"]].groupby("Cruise_ID").count()
print(bottle_2.groupby("Chla")["Chla"].count())

print(bottle_6.head())
bottle_6.info()

### New Cruise_ID list for new df is ID_list_6
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
### Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(b_ID)

print(len(ID_list_6))

#%%

### Write Cleaned bottle data to CSV ###

# Write Cleaned bottle df to intermediate csv
bottle_6.to_csv('data/HOT_Pigments_Chla_Cleaned.csv')

#%%
### Get MLDs for bottle profiles ###

# Inspect CTD single profile data
ctd_prof.head()

# Create new single CTD data df containing only data for profiles also in Bottle list
ctdprof_in_bottle =  ctd_prof[ctd_prof.Cruise_ID.isin(bottle_6['CRN'])]
#ctdprof_in_bottle =  ctd_prof[ctd_prof.ID_list.isin(ID_list_6)]
ctdprof_in_bottle.head()
# Sort df again
ctdprof_in_bottle = ctdprof_in_bottle.sort_values(by=['Cruise_ID','DateTime'])

# Reset bottle df index replacing old index column
ctdprof_in_bottle = ctdprof_in_bottle.reset_index(drop=True)
ctdprof_in_bottle.info()

### Extract required single ctd data from new df ###
b_time_mld             = ctdprof_in_bottle.loc[:,'DateTime'].to_numpy()
b_MLD_prof_temp        = ctdprof_in_bottle.loc[:,'MLD_temp'].to_numpy()
b_MLD_prof_dens        = ctdprof_in_bottle.loc[:,'MLD_dens'].to_numpy()
b_MLD_prof_dens_boyer  = ctdprof_in_bottle.loc[:,'MLD_boyer'].to_numpy()
b_MLD_prof_ID          = ctdprof_in_bottle.loc[:,'Cruise_ID'].to_numpy()

#%%

### CREATE BOTTLE PROFILE SINGLE DATA DF ###

# MLD df with time to extract bottle MLD later for each profile ID, some profiles on same time/date so can't use np.unique
bottle_6.info()
bottle_prof = bottle_6[['DateTime','Cruise_ID','CRN','CASTNO','DecYear','Date','yyyy','mm',]]
bottle_prof = bottle_prof.drop_duplicates(subset=['Cruise_ID'])
print(len(bottle_prof))

# Sort df again first on time & then ID to account to profiles on same date
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID','Date'])    

bottle_prof['MLD_temp'] = b_MLD_prof_temp
bottle_prof['MLD_dens'] = b_MLD_prof_dens
bottle_prof['MLD_dens_boyer'] = b_MLD_prof_dens_boyer
bottle_prof['CTD_time'] = b_time_mld
bottle_prof['CTD_ID']   = b_MLD_prof_ID

# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)
bottle_prof.info()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
bottle_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(bottle_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

bottle_date_length = date_span('1989-8-25', max(bottle_prof['Date']))

# Save to CSV
bottle_prof.to_csv('data/HOT_Bottle_Pigments_profData.csv')

#%%

### EXTRACT CLEANED DATA & MAKE BOTTLE ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Pigments_Chla_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'DateTime'].to_numpy()
b_time_2   = pd.to_datetime(bottle_6['DateTime'])
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_Fchla    = bottle_6.loc[:,'Fchla'].to_numpy()
b_chla     = bottle_6.loc[:,'Chla'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()

# Bottle DateTime data
b_DateTime     = pd.DatetimeIndex(b_time_2)

### Cruise_ID list for new df is ID_list_6

### Cruise_ID list for new df is ID_list_6
# Removes Duplicates
ID_list_6 = pd.unique(b_ID)
print(len(ID_list_6))

#%%

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
scatter = ax.scatter(b_chla, b_depth, c=b_month, alpha=0.7, cmap=cmap)

# Invert y-axis to show depth increasing downwards
ax.set_ylim([260, 0])
ax.set_ylabel('Depth (m)', fontsize=18)
ax.yaxis.set_tick_params(labelsize=14)
ax.xaxis.tick_top()
ax.set_xlabel('HPLC Chl-a (mg m$^{-3}$)', fontsize=18, labelpad=12)  # Move xlabel slightly further from tick labels
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
fig.savefig('plots/Bottle-HPLC_chla_depths.png', format='png', dpi=300, bbox_inches="tight")

plt.show()

#%%

### COUNT BOTTLE PROFILES PER month AGAIN ###

# Create new df with number of CTD profiles (cruises) per year
bottle_y = bottle_6[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()
# Write to csv
#bottle_y.to_csv('data/BATS_Bottle_profiles_per_year.csv')

# Bar plot of profiles per year
bottle_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_y.to_markdown())

#%%

### EXAMPLE PROFILE PLOT ###
# Example profile of bottle data overlain with matched CTD fluoresence & MLD

# Define profile / Cruise ID number
ID_1 =  328#16#296#117#116#290 
ID_2 =  ID_1

# Extract all bottle df rows of this profile into new df to print
bot_prof = bottle_6[bottle_6['CRN'] == ID_1]
#bot_prof = bottle_6[bottle_6['Cruise_ID'] == ID_1]
print(bot_prof.head(5))

### Define and extract variables for specific profile ###

# CTD data for Profile = ID_1
x = np.where(ctd.cruise_ID == ID_2) # Index for ctd data
depth_1        = depth[x]
temperature_1  = temperature[x]
fluorescence_1 = fluorescence[x]
bvf_1          = bvf[x]

ID_where = (bottle_6['CRN'] == ID_1)
bot_ID = bottle_6['Cruise_ID'].loc[ID_where]
bot_ID = int(np.unique(bot_ID))


# Bottle data for Profile = ID_1
AS = np.where(bottle_6.CRN == ID_1) # Index for bottle data
b_Fchla_1      = b_Fchla[AS]
b_chla_1       = b_chla[AS]
#b_chla_avg_1   = b_chla_avg[AS]
b_depth_1      = b_depth[AS]
b_DateTime_1   = b_DateTime.date[AS]

# MLD of profile
ASD = np.where(ID_list_ctd == ID_2) # Index for calculated MLD
MLD_1 = MLD[ASD]

b_DateTime_1 = b_DateTime_1[1]
print(b_DateTime_1)

# Print maximum HPLC chla of profile
#print (max(b_chla_1))
# Print profile MLD
print(float(MLD_1))

#Plot size
XSIZE = 6 #Define the xsize of the figure window
YSIZE = 6 #Define the ysize of the figure window

##Define the figure window with 1 subplot
fig, (ax3) = plt.subplots(1, figsize=(XSIZE,YSIZE), \
gridspec_kw={'hspace': 0.2})
# MLD Line 
ax3.plot([np.min(b_chla)-0.02,np.max(b_chla)+0.05],[MLD_1,MLD_1], \
         color = 'r', marker = 'None', linestyle = '--', label= 'MLD')
# CTD Fluoresence
ax3.plot(fluorescence_1,depth_1, \
         color = 'g', marker = 'o', linestyle = 'None', label= 'CTD Fluorescence')
# HPLC Chl-a
ax3.plot(b_chla_1,b_depth_1, \
         color = 'b', marker = 'o', linestyle = 'None', label= 'HPLC Chl-a')
# Turner Chl-a
ax3.plot(b_Fchla_1,b_depth_1, \
         color = 'r', marker = 'o', linestyle = 'None', label= 'Turner Chl-a')
# Set axis info and titles
ax3.set_ylabel('Depth (m)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=13)
ax3.set_ylim([300,0]) 
ax3.set_xlabel('Chl-a (µg/L)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=13)
if np.max(fluorescence_1) > np.max(b_chla_1):
    ax3.set_xlim(xmin=np.min(b_chla_1)-0.025, xmax=np.max(fluorescence_1)+0.005)
else:
    ax3.set_xlim(xmin=np.min(b_chla_1)-0.025, xmax=np.max(b_chla_1)+0.005)
ax3.xaxis.set_major_locator(plt.MaxNLocator(4))
ax3.legend(loc="lower right", fontsize=13,title= ID_1)
ax3.text(np.min(b_chla_1)+0.01, 298, "Date: "+str(b_DateTime_1), color='k', fontsize=12)
fig.savefig('plots/Bottle_Profile_'+str(ID_1)+'.png', format='png', dpi=300, bbox_inches="tight")

#Complete the plot
plt.show()

print(len(ID_list_6))
