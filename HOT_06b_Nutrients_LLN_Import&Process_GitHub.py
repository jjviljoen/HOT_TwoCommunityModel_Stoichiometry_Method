"""
HOT - Inspect & Clean Bottle Low Level Nutrient Data & setup dataframe

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
plt.rcParams["font.family"] = "sans-serif"
from dateutil import relativedelta
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore');

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    """
    Calculate the difference in years, months, and days between two dates.
    """
    # Calculate the relative delta between the start and end dates
    rd = relativedelta.relativedelta(pd.to_datetime(end), pd.to_datetime(start))
    # Construct the string representing the duration in years, months, and days
    date_len = '{}y{}m{}d'.format(rd.years, rd.months, rd.days)
    # Return the formatted date length string
    return date_len

#%%

### READ & EXTRACT CTD DATA ###

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

ID_list_ctd       = ctd_prof.loc[:,'Cruise_ID'].to_numpy()
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
ctd_DecYear_prof  = ctd_prof.loc[:,'DecYear'].to_numpy()
MLD_boyer         = ctd_prof.loc[:,'MLD_boyer'].to_numpy()

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
print("Timespan: "+str(b_date_length)) # 
print("Min Date: "+str(min(bottle_6['Date'])))
print("Max Date: "+str(max(bottle_6['Date'])))

#%%

# IMPORT BOTTLE PROF df before integration to filter nutrient profiles for those deeper than DCM depth (modelled)

### Bottle Single MLD & DateTime ###
# CSV filename
filename_1 = 'data/HOT_Bottle_profData.csv'#HOT_Bottle_Pigments_profData

# Load data from CSV. "index_col = 0" makes the first column the index.
bottle_prof = pd.read_csv(filename_1, index_col=0)

# Inspect the dataframe
bottle_prof.info()
print(len(ID_list_6))           # Print the length of the Cruise_ID list
print(len(bottle_prof))         # Print the number of rows in the dataframe

# Sort dataframe by Cruise_ID
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])

# Reset dataframe index, replacing the old index column
bottle_prof = bottle_prof.reset_index(drop=True)

### Extract bottle MLD with corresponding time ###
b_DateTime_prof = pd.to_datetime(bottle_prof['DateTime'].values)   # Convert 'DateTime' column to pandas datetime series
b_dec_year_prof = bottle_prof['DecYear'].to_numpy()         # Extract 'DecYear' column as numpy array
b_DCM_depth     = bottle_prof.loc[:,'DCM_depth'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

#%%

######
### READ & CLEAN ORIGINAL BOTTLE DATA ###
######

# Original txt Bottle data from the BATS BIOS site Dropbox at https://bats.bios.asu.edu/bats-data/
# Data downloaded on 18 July 2023 and converted to CSV

#Filename
file_pigment1 = 'data/HOT_Bottle_Cleaned2.csv' #name file location and csv file name
#file_pigment2 = 'data/niskin_BCO_BMO.csv' #name file location and csv file name

# Step 1: Read the first two rows to get column names and units
df = pd.read_csv(file_pigment1, index_col = 0)
df.info()

#For Nutrient Dataset
bot = df[["Cruise_ID","CRN","CASTNO", "Cruise_ID_o","DateTime",'DecYear','Date','yyyy','mm','time','press', 'depth',
             'nit', 'phos', 'sil','lln','llp','no2']]
# low level nitrate and phosphate might be usefull need separate df for each and then later match per cruise because sampled on different casts.

# Rename Columns
#bot.rename(columns={"chl": "Fchla","hplc": "Chla"},inplace=True)
bot.head()
bot.info()

#('phos', 'umol/kg'),
#('nit', 'umol/kg'),
#('sil', 'umol/kg'),
#('lln', 'nmol/kg'),
#('llp', 'nmol/kg')

# Inspect Low-level nutrients - see: https://hahana.soest.hawaii.edu/hot/methods/llnuts.html
# Remove rows with NaN values
bottle_nut_lln = bot.dropna(subset=['lln'])
bottle_nut_llp = bot.dropna(subset=['llp'])

print(np.nanmax(bottle_nut_lln['depth']))

# Remove rows with NaN values
#bottle_nut = bot.dropna(subset=['nit','phos','sil'])
#bottle_nut_sil = bot.dropna(subset=['sil'])
# Remove rows with NaN depth values
bottle_nut = bottle_nut_lln.dropna(subset=['depth']).copy()

# Filter for only needed columns
bottle_nut = bottle_nut[["Cruise_ID","CRN","CASTNO", "Cruise_ID_o","DateTime",'DecYear','Date','yyyy','mm','time','press', 'depth',
                             'lln']]
bottle_nut = bottle_nut.reset_index(drop=True)
bottle_nut.info()
bottle_nut.head()

bottle_nut['lln_o'] = bottle_nut['lln']

# Convert LLN units nmol/kg to nmol/L
bottle_nut['lln_nmol'] = bottle_nut['lln']*1025/1000 #(10^6)

bottle_nut['lln_umol'] = bottle_nut['lln']*1025/1000000 #(10^6)

#bottle_nut = bottle_nut.drop_duplicates(subset=["Cruise_ID", "depth"])

# Remove rows with depths below 400m
bottle_nut = bottle_nut[bottle_nut["depth"]<500] # 7611 rows
# Remove rows with negative values
bottle_nut = bottle_nut[bottle_nut["lln"]>=0] # 7611 rows

bottle_nut.head()
bottle_nut.info()

# sort by time
#bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','Date','depth'])
bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','depth'])
# Reset bottle df index replacing old index column
bottle_nut = bottle_nut.reset_index(drop=True)

# convert to datetime format & extract Date yyyy-mm-dd into Date column again
bottle_nut['Date'] = pd.to_datetime(bottle_nut['DateTime']).dt.date
#print(bottle['Date'])

### Timespan of Nutrient data ###

# Print start and end dates of Nutrient data
print("Bottle Dates: "+str(min(bottle_nut['Date']))+" to "+str(max(bottle_nut['Date'])))

# Print period timespan of Nutrient data using base date subtraction - only days
print("Bottle Date Length: "+str(max(bottle_nut['Date'])-min(bottle_nut['Date'])))

# Print timespan of Nutrient data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_nut['Date']), max(bottle_nut['Date']))
print("Timespan: "+str(b_date_length))

#%%

### EXPLORE NITRATE DATA ###

# Summary stats Table for POC
print(bottle_nut[["lln"]].describe())

# Boxplot
bottle_nut[["lln"]].plot.box()
plt.show()

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_nut[["Cruise_ID", "lln"]].groupby("Cruise_ID").count()
print(bottle_poc_profcount)

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("lln")["lln"].count())

# Bar plot of profiles per number per number of depths sampled
bottle_poc_profcount.groupby("lln")["lln"].count().plot.bar()
plt.show()

### COUNT PROFILES PER YEAR ###

# Create new df with number of CTD profiles (cruises) per year
bottle_nut_y = bottle_nut[["Cruise_ID", "yyyy"]].groupby("yyyy").nunique()

# Bar plot of profiles per year
bottle_nut_y.plot.bar(color = "darkgreen")
plt.show()

# Nice Table for Notebook
print(bottle_nut_y.to_markdown())

#%%

### REMOVE PROFILES WITH NO MATCHING PIGMENT profiles ###

### Cruise_ID list
b2_ID       = bottle_nut.loc[:,'Cruise_ID'].to_numpy()
ID_list_nut = pd.unique(b2_ID)

print(len(ID_list_nut))
print(len(ID_list_6))

# Create new df containing only data for profiles also in pigment bottle list
bottle_nut =  bottle_nut[bottle_nut.Cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_nut = bottle_nut.reset_index(drop=True)

### Cruise_ID list
b2_ID       = bottle_nut.loc[:,'Cruise_ID'].to_numpy()
ID_list_nut = pd.unique(b2_ID)
print(len(ID_list_nut))
print(len(ID_list_6))

print(min(ID_list_nut))
print(min(ID_list_6))

# Find cruises in ID_list_6 but not in ID_list_nut
missing_cruises = [int(cruise) for cruise in ID_list_6 if cruise not in ID_list_nut]
print(missing_cruises)

# Write Cleaned bottle df to csv
bottle_nut.to_csv('data/HOT_Bottle_Nutrients_LLN_AllDepths.csv')

#%%

### REMOVE PROFILES SHALLOWER THAN DCM ###

bottle_nut.info() # df with vertical profile nutrient data per profile
bottle_prof.info() # df containing one single value of DCM depth per profile among others, per profile (i.e., much shorter)

ID_list_nut_dcm = []   # list to collect valid profile IDs

for ID in ID_list_nut:
    # Nutrient profile depths (multiple values)
    nit_depth = bottle_nut.loc[bottle_nut['Cruise_ID'] == ID, 'depth'].values
    if nit_depth.size == 0:
        continue

    nit_depth_max = np.nanmax(nit_depth)

    # DCM depth from bottle_prof (single value expected)
    dcm_depth = bottle_prof.loc[bottle_prof['Cruise_ID'] == ID, 'DCM_depth'].values

    if dcm_depth.size == 0:
        continue

    dcm_depth = float(dcm_depth)   # convert array → scalar

    # add to list only if nutrient profile is deeper than DCM
    if dcm_depth < nit_depth_max:
        ID_list_nut_dcm.append(int(ID))   # optional int() to keep list neat
        
print(len(ID_list_nut))
print(len(ID_list_6))
print(len(ID_list_nut_dcm))


# Filter nutrient df to only valid profiles
bottle_nut = bottle_nut[bottle_nut['Cruise_ID'].isin(ID_list_nut_dcm)]

before = len(ID_list_nut)   # number of rows before filtering
after  = len(ID_list_nut_dcm)   # number of rows after filtering

print("Original number of nutrient profiles:", before)
print(f"Profiles removed: {before - after}")
print("Valid number nutrient profiles:", after)

#%%

### MULTI-CAST PROFILE MATCHING & DEPTH AUGMENTATION ###

# ---------------------------------------------------------------------
# Handle cruises with multiple nutrient casts (CASTNO):
# For some Cruise_IDs, more than one nutrient cast exists. We first
# identify the "primary" cast as the one with sampling date closest to
# the bottle_prof reference date (best temporal match).
#
# Nutrient profiles are taken from this primary cast, but if it does not
# extend deeper than the DCM depth, we selectively augment the profile
# using measurements from other casts. Only depths deeper than the
# deepest value of the primary cast are appended to gain deeper coverage.
#
# Final result per Cruise_ID = primary cast + deeper points from other
# casts where useful, producing one cleaned nutrient profile per cruise.
# ---------------------------------------------------------------------

# Ensure Date columns are datetime objects (not strings)
bottle_nut['Date']  = pd.to_datetime(bottle_nut['Date']).dt.date
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date']).dt.date

filtered_profiles = []   # store final cleaned nutrient subsets

for ID in ID_list_nut_dcm:
    # All nutrient casts for this cruise
    df_nut = bottle_nut[bottle_nut['Cruise_ID'] == ID].copy()

    if df_nut.empty:
        continue

    # Profile reference date + DCM depth (single row expected)
    prof_row = bottle_prof[bottle_prof['Cruise_ID'] == ID].iloc[0]
    prof_date = prof_row['Date']
    dcm_depth = prof_row['DCM_depth']

    # === 1) Find CAST with date closest to bottle_prof date ===
    df_nut['date_diff'] = abs(df_nut['Date'] - prof_date)
    best_cast = df_nut.loc[df_nut['date_diff'].idxmin(), 'CASTNO']

    # Split best cast and others
    df_best = df_nut[df_nut['CASTNO'] == best_cast].copy()
    df_other = df_nut[df_nut['CASTNO'] != best_cast].copy()

    primary_max_depth = df_best['depth'].max()

    # === 2) If primary does not exceed DCM, extend deeper using other casts ===
    if primary_max_depth < dcm_depth and not df_other.empty:
        # keep only depths deeper than primary cast
        extras = df_other[df_other['depth'] > primary_max_depth].copy()

        # optional: also require deeper than DCM
        # extras = extras[extras['depth'] > dcm_depth]

        df_final = pd.concat([df_best, extras], ignore_index=True)
    else:
        df_final = df_best.copy()

    # store with preserved Cruise_ID tag
    filtered_profiles.append(df_final)

# === FINAL CLEANED NUTRIENT DATAFRAME ===
bottle_nut_clean = pd.concat(filtered_profiles, ignore_index=True)

print("Original rows:", len(bottle_nut))
print("Cleaned rows:", len(bottle_nut_clean))
print("Profiles retained:", bottle_nut_clean['Cruise_ID'].nunique()) #258

bottle_nut = bottle_nut_clean.copy()

#%%

### REMOVE PROFILES WITH <6 MEASUREMENTS ###

# Extract required variables
bottle_nut = bottle_nut.reset_index(drop=True)  # Resetting the index for easier access

nut_no_prof = np.zeros(len(bottle_nut))  # Initialize array

#Create count function that count all non-nan values including zeros as nutrients often undetected (0) at surface
def count_values(array):
    # Convert the array to a NumPy array
    array = np.array(array)
    
    # Exclude NaNs and count values including zeros
    count = np.sum(~np.isnan(array) | (array == 0))
    
    return count

for i in range(len(bottle_nut)):
    mask = (bottle_nut['Cruise_ID'] == bottle_nut.at[i, 'Cruise_ID'])
    nit = bottle_nut.loc[mask, 'lln_nmol'].values
    depth = bottle_nut.loc[mask, 'depth'].values

    if depth.min() > 30: # Ensure profiles have measurements shallower than 30m
        nut_no_prof[i] = 0
    else:
        depth_mask = depth <= 350 # Consider only upper 350m to count measurements
        nit = nit[depth_mask]
        depth = depth[depth_mask]
        non_nan_mask = ~np.isnan(nit)
        nit = nit[non_nan_mask]

        nut_no_prof[i] = count_values(nit)#np.count_nonzero(nit)

bottle_nut['nut_no_prof'] = nut_no_prof

bottle_nut = bottle_nut[bottle_nut['nut_no_prof'] > 5] #4530 #Keep only profile with 6 or more measurements

################

# Remove rows where depth >= 
bottle_nut = bottle_nut[bottle_nut["depth"] < 510]

# Remove duplicate measurements
bottle_nut.drop_duplicates(subset=['Cruise_ID', 'depth'], keep='last', inplace=True)

# Sort new df by Cruise_ID and depth again
bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_nut = bottle_nut.reset_index(drop=True)

bottle_nut.info()

print("Profiles retained:", bottle_nut['Cruise_ID'].nunique()) #258

# Group by Cruise ID profile and count number of samples per profile
bottle_poc_profcount = bottle_nut[["Cruise_ID", "lln_nmol"]].groupby("Cruise_ID").count()

# Count number of profiles per number of depths sampled
print(bottle_poc_profcount.groupby("lln_nmol")["lln_nmol"].count())

print("Final LLN Profiles retained:", bottle_nut['Cruise_ID'].nunique()) #258

#%%

# Save Cleaned Nutrient data

# Write Cleaned bottle df to csv
bottle_nut.to_csv('data/HOT_Bottle_Nutrients_LLN_Cleaned.csv')

#%%

### EXTRACT CLEANED DATA & MAKE NEW ID LIST ###
# CSV filename
filename_1 = 'data/HOT_Bottle_Nutrients_LLN_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_nut = pd.read_csv(filename_1, index_col = 0)

# Sort new df by ID and depth
bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_nut = bottle_nut.reset_index(drop=True)

# Ensure Date columns are datetime objects (not strings)
bottle_nut['Date']  = pd.to_datetime(bottle_nut['Date']).dt.date

### EXTRACT CLEANED DATA & MAKE NEW ID LIST ###
bottle_nut.info()
### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_nut.loc[:,'DateTime'].to_numpy()
b2_date     = bottle_nut.loc[:,'Date'].to_numpy()
b2_depth    = bottle_nut.loc[:,'depth'].to_numpy()
b2_lln_nmol = bottle_nut.loc[:,'lln_nmol'].to_numpy()
b2_lln_umol = bottle_nut.loc[:,'lln_umol'].to_numpy()
b2_ID       = bottle_nut.loc[:,'Cruise_ID'].to_numpy()
b2_year     = bottle_nut.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_nut.loc[:,'mm'].to_numpy()
b2_Decimal_year = bottle_nut.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_nut['DateTime'].values)

### Cruise_ID list
ID_list_nut = pd.unique(b2_ID)
print(len(ID_list_nut))
# 254 profiles

# Print start and end dates of Nutrient data
print("Bottle Dates: "+str(min(bottle_nut['Date']))+" to "+str(max(bottle_nut['Date'])))

# Print timespan of Nutrient data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_nut['Date']), max(bottle_nut['Date']))
print("Timespan: "+str(b_date_length))

#%%

### CREATE PROF DF

# Create df with ID list to save single profile values
bottle_nut_prof = bottle_nut.drop_duplicates(subset=['Cruise_ID'])
# Reset bottle df index replacing old index column
bottle_nut_prof = bottle_nut_prof.reset_index(drop=True)

print(len(bottle_nut_prof))
print(len(ID_list_nut))

# Slice df to only have ID, time and date columns
bottle_nut_prof = bottle_nut_prof[['Cruise_ID','DateTime','Date','DecYear','yyyy','mm' ]]
bottle_nut_prof.info()

# Write Cleaned bottle df to csv
bottle_nut_prof.to_csv('data/HOT_Bottle_Nutrients_LLN_Nitracline.csv')

#%%

###########################
# NUTRICLINE DEPTH FUNCTION Test
###########################

def calculate_nutricline_depth(depths, nitrate_concentration,
                               threshold=0.5,
                               min_depth=50,
                               mld=None,
                               delta=0.05):
    """
    Calculate the nitracline depth (Z_NO3) from a profile of nitrate data.

    Two modes:
    1. Standard fixed threshold (default 0.5 µmol/kg).
    2. Delta above the mixed-layer mean, if `mld` and `delta` are provided.
    Default delta set as 0.05 µmol/L similar to Xu et al. 2025 - https://doi.org/10.1038/s41467-025-65710-2

    Args:
    - depths (list or 1D array): Increasing depths in meters.
    - nitrate_concentration (list or 1D array): Nitrate concentrations (µmol/kg).
    - threshold (float, default=0.5): The concentration threshold (µmol/kg) for fixed threshold mode.
    - min_depth (float or None): Ignore crossings shallower than this depth.
    - mld (float or None): Mixed-layer depth in meters. Required if using delta mode.
    - delta (float or None): Concentration increment above the mixed-layer mean for delta mode.

    Returns:
    - nitracline_depth (float or None): Interpolated depth where the profile crosses the threshold,
      or None if no crossing is found (or crossing is shallower than min_depth).
    """
    import numpy as np

    depths = np.array(depths)
    nitrate_concentration = np.array(nitrate_concentration)

    # Determine threshold to use
    if mld is not None and delta is not None:
        # Use delta above mixed-layer mean
        ml_mask = depths <= mld
        if np.any(ml_mask):
            ml_mean = np.nanmean(nitrate_concentration[ml_mask])
            threshold_use = ml_mean + delta
        else:
            # If no points in MLD, fall back to fixed threshold
            threshold_use = threshold
    else:
        threshold_use = threshold

    # Loop through profile pairs to find first crossing
    for i in range(1, len(depths)):
        d0, d1 = depths[i-1], depths[i]
        n0, n1 = nitrate_concentration[i-1], nitrate_concentration[i]

        # Check if threshold is bracketed
        if n0 < threshold_use <= n1:
            # Linear interpolation
            frac = (threshold_use - n0) / (n1 - n0)
            depth_cross = d0 + frac * (d1 - d0)

            # Apply minimum depth filter
            if min_depth is not None and depth_cross < min_depth:
                return None
            return depth_cross

    # No crossing found
    return None


#%%

### Nutrient SINGLE PROFILE PLOT ###

ID_1 = 	33#117
# Inly surface: 10208005
# Supplemental Example Profile: 10195003,10195003

# Extract all bottle df rows of this profile into new df to print
bot_prof = bottle_nut[bottle_nut['CRN'] == ID_1]
#bot_prof = bottle_6[bottle_6['Cruise_ID'] == ID_1]
print(bot_prof[['CASTNO','DateTime','depth','lln_nmol']].head(20))

# Find POC profile data

# Get MLD from CTD
prof_MLD_idx = np.where(ctd_prof.Cruise_ID == ID_1)
prof_MLD     = MLD_boyer[prof_MLD_idx]

# Extract profile data
prof_nit_idx     = np.where(bottle_nut.Cruise_ID == ID_1)
prof_lln      = b2_lln_nmol[prof_nit_idx]
prof_lln_umol      = b2_lln_umol[prof_nit_idx]
#prof_phosphate   = b2_phosphate[prof_nit_idx]
prof_nit_depth   = b2_depth[prof_nit_idx]

print(prof_lln)

b2_DateTime_1 = b2_DateTime.date[prof_nit_idx]
print(b2_DateTime_1[0])
prof_date = b2_DateTime_1[0]

### FIND NUTRICLINE ###
depths = prof_nit_depth#prof_lln_depth           # e.g. [0, 10, 20, …] in meters
nitrate_concentration = prof_lln_umol  # in µmol/kg

z_no3 = calculate_nutricline_depth(depths,
                                   nitrate_concentration, mld = prof_MLD, delta=0.05,
                                   min_depth=30)
if z_no3 is not None:
    print(f"Nitracline (Z_NO3) at {z_no3:.2f} m")
else:
    print("No nitracline depth found (threshold not crossed or above min_depth).")


print("MLD: "+str(prof_MLD))

#Define the figure window including 5 subplots orientated horizontally
fig, (ax3) = plt.subplots(1, sharey=True, figsize=(6,6), \
gridspec_kw={'hspace': 0.2})

ax3.axhline(y=prof_MLD, color = 'k',linestyle = '--', label= 'MLD')
if z_no3 is not None:
    ax3.axhline(y=z_no3, color = 'royalblue',linestyle = '--', label= 'nitracline')
ax3.plot(prof_lln,prof_nit_depth, \
         color = 'blue',  marker = 'o', linestyle = 'None',label= 'LLN')
ax3.set_ylabel('Depth (m)', fontsize=15)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_ylim([300,0]) 
ax3.set_xlabel('Nitrate (nmol l$^{-1}$)', fontsize=15, color = 'k')
ax3.xaxis.set_tick_params(labelsize=15)
ax3.set_xlim(xmin=0.00, xmax=np.nanmax(prof_lln)*1.05)
ax3.legend(loc="lower right", fontsize=10,title= ID_1)
ax3.text(np.nanmin(prof_lln)+0.005, 268, " Date: "+str(b2_DateTime_1[0]), color='k', fontsize=12)
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
#ax3.set_xscale('log')
plt.show()

#%%

########
# COMPUTE NUTRICLINES
########

bottle_nut.info()

# Create empty arrays to store calculated MLD and single time/date per profile
num_profiles = len(ID_list_nut)
z_nitrate    = np.full(num_profiles, np.nan)
#z_phosphate  = np.full(num_profiles, np.nan)

nut_DateTime_prof = np.empty(num_profiles, dtype='datetime64[ns]')
nut_DecYear_prof  = np.full(num_profiles, np.nan)

ID_list_nut = np.array(ID_list_nut)

# Vectorized loop for nitracline and date per profile
for count, i in enumerate(ID_list_nut):
    asx = np.where(b2_ID == i)[0]
    if len(asx) == 0:
        continue

    prof_depth     = b2_depth[asx]
    prof_nitrate   = b2_lln_umol[asx]
    #prof_phosphate = b2_phosphate[asx]
    # Get MLD from CTD
    prof_MLD_idx = np.where(ctd_prof.Cruise_ID == i)
    prof_MLD     = MLD_boyer[prof_MLD_idx]
    
    # Get the first instance of ctd_DateTime and ctd_Decimal_year for this profile
    nut_DateTime_prof[count] = b2_DateTime[asx][0]
    nut_DecYear_prof[count]  = b2_Decimal_year[asx][0]

    if len(prof_depth) >= 6 and np.min(prof_depth) <= 30:
        z_nitrate[count]   = calculate_nutricline_depth(prof_depth,prof_nitrate,min_depth=30, mld = prof_MLD, delta=0.05)
        
###


  
plt.plot(nut_DateTime_prof,z_nitrate, label = "Nitracline")
#plt.plot(nut_DateTime_prof,z_phosphate, label = "Phosphacline")
plt.legend()
plt.show()

#%%

nut_DateTime_prof = pd.DatetimeIndex(nut_DateTime_prof, dtype='datetime64[ns]', name='date_time', freq=None)

bottle_nut_prof.info()
#units
bottle_nut_prof['Z_nitrate']   = z_nitrate
#bottle_nut_prof['Z_phosphate'] = z_phosphate
bottle_nut_prof['DateCheck'] = nut_DateTime_prof

bottle_nut_prof.info()

# Write Cleaned bottle df to csv
bottle_nut_prof.to_csv('data/HOT_Bottle_Nutrients_LLN_profData.csv')

"END"