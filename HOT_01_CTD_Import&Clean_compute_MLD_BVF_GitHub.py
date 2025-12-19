"""
HOT: CTD Import & Clean Original CTD data

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

# might have to downgrade numpy<1.24 for holteandtalley function that works on older numpy.
Or just remove HolteTalley MLD method and just use Boyer.

Updated: 18 Dec 2025

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
import seawater as sw # Import CSIRO seawater package
import datetime
import gsw
# Import specific modules from packages
from datetime import timedelta, date
from dateutil import relativedelta
from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from holteandtalley import HolteAndTalley # Import MLD code available here https://github.com/garrettdreyfus/python-holteandtalley
from math import nan
from scipy import interpolate # used to interpolate profiles for contour plots
from scipy.stats import spearmanr

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

def serial_date_to_string(srl_no):
    new_date = datetime.datetime(1988,10,1,0,0) + datetime.timedelta(srl_no - 1)
    return new_date.strftime("%Y-%m-%d")

#%%
### READ & CLEAN - HOT CTD data ###

# name file location and csv file name
filename3 = 'data/hot_ctd_direct_28Aug2024.xlsx' #name file location and csv file name #

# Original CTD data from HOT-DOGS site at https://hahana.soest.hawaii.edu/hot/hot-dogs/cextraction.html
# Methods and original data here: https://hahana.soest.hawaii.edu/hot/protocols/protocols.html
# Data downloaded on 27 Aug 2024 and copied into an Excel xlsx file
# Data acknowledgement: "Data obtained via the Hawaii Ocean Time-series HOT-DOGS application; University of Hawai'i at Mānoa. National Science Foundation Award # 1756517".

# Load the Excel file, skipping the first 3 rows
ctd = pd.read_excel(filename3,skiprows=2)
ctd.info()

### Convert comma delimited data into dataframe 
# Split the data in the first column into separate columns based on commas
split_columns = ctd.iloc[:, 0].str.split(', ', expand=True)
# Replace empty strings and spaces with NaN
split_columns.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
# Drop any columns in the split columns that contain only NaN values
split_columns = split_columns.dropna(axis=1, how='all')

# Drop the original first column
ctd = ctd.drop(ctd.columns[0], axis=1)

# Concatenate the split columns back to the dataframe
ctd = pd.concat([split_columns, ctd], axis=1)

# Set the first row as column headers
ctd.columns = ctd.iloc[0]  # Set the first row as header
ctd = ctd[1:]  # Remove the first row from the dataframe

# Extract column names and units from the first row
column_names = ctd.columns.tolist()
units = ctd.iloc[0].tolist()

# Create a list of tuples (column_name, unit)
columns_with_units = list(zip(column_names, units))
print("Columns with Units:")
print(columns_with_units)

# Strip leading spaces from column names
cleaned_column_names = [col.lstrip() for col in column_names]

# Remove the units by using the cleaned column names
ctd.columns = cleaned_column_names
ctd = ctd[1:]  # Remove the first row that contains the units

# Reset index after removing the first row
ctd.reset_index(drop=True, inplace=True)

# Strip leading spaces from all cells in the dataframe
ctd = ctd.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

ctd.info()
ctd.head()

# Save the modified dataframe to CSV
ctd.to_csv('data/hot_ctd_direct_28Aug2024_split.csv', index=False)

#%%

filename = 'data/hot_ctd_direct_28Aug2024_split.csv'

ctd = pd.read_csv(filename)
# Inspect CTD df
ctd.info()
ctd.head()

#Extract DecYear and Compute DateTime and Date
ctd_julian   = ctd.loc[:,'julian'].to_numpy(dtype='float64')
    
# Vectorize the function
vectorized_serial_date_to_string = np.vectorize(serial_date_to_string)
# Apply the vectorized function to the entire array
ctd_DateTime = vectorized_serial_date_to_string(ctd_julian).astype('datetime64[s]')
    
ctd_DateTime = pd.DatetimeIndex(ctd_DateTime)

ctd['DateTime'] = ctd_DateTime
ctd[['temp','theta']].head(20)
#ctd.info()

print(np.max(ctd['DateTime']))
print(np.max(ctd['crn']))

ctd.replace(-9, np.nan, inplace = True)
ctd.head()

# Compute Depth from presure using new gsw package
ctd["depth"] = -1*(gsw.z_from_p(np.array(ctd["press"]), 22.75))

ctd = ctd[["crn","julian", "DateTime",'press','depth','temp',
             'sal', 'sigma','oxy','fluor']]

ctd.info()

# Rename Columns
ctd.rename(columns={"crn": "cruise_ID",
                                "temp": "temperature",
                                "sal": "salinity",
                                "sigma": "pot_density",
                                "oxy": "dissolved_oxygen",
                                "fluor": "fluorescence"},inplace=True)

ctd['density'] = ctd['pot_density']

#ctd.head()
#ctd.info()

### Remove NaN values ###
 
# Remove rows with NaN temperature values
ctd = ctd.dropna(subset=['temperature'], inplace=False)
#ctd = ctd.drop(ctd[ctd.temperature == -999].index)

# Removes rows with NaN temperature values (~1200 rows)
ctd = ctd.dropna(subset=['salinity'], inplace=False)

# Remove rows with depths below 400m
ctd = ctd[ctd["salinity"]<36] #405
# Reset bottle df index removing old index with missing numbers after slice
ctd = ctd.reset_index(drop=True)

#Convert LonW to LonE
#ctd['lon'] = ctd['lon']*-1
#ctd['lon'].head()
    
ctd_DateTime = pd.DatetimeIndex(ctd['DateTime'])

ctd_date = pd.to_datetime(ctd_DateTime).date

# Remove rows with depths below 500m
ctd = ctd[ctd["depth"]<501] #405
# Reset bottle df index removing old index with missing numbers after slice
ctd = ctd.reset_index(drop=True)

ctd.sort_values(by=['cruise_ID','depth'], inplace=True)
# Reset bottle df index removing old index with missing numbers after slice
ctd = ctd.reset_index(drop=True)

ctd.info()

# Print start and end dates of bottle data
print("CTD Dates: "+str(min(ctd_date))+" to "+str(max(ctd_date)))

# Print period timespan of bottle data using base date subtraction - only days
print("CTD Date Length: "+str(max(ctd_date)-min(ctd_date)))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_date), max(ctd_date))
print("Timespan: "+str(ctd_date_length))

print("Max Cruise: "+str(max(ctd['cruise_ID'])))
print("Max Julian Day: "+str(max(ctd['julian'])))

#%%

#Compute DecYear from DateTime
ctd_DateTime = pd.to_datetime(ctd['DateTime'])

# Use vectorized operations to compute decimal years
ctd_DecYear = ctd_DateTime.apply(pyasl.decimalYear).to_numpy()

# Assign the computed decimal years back to the DataFrame
ctd['Dec_Year'] = ctd_DecYear
ctd.info()

#%%

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
ctd['Date'] = pd.to_datetime(ctd['DateTime']).dt.date
#print(ctd['Date'])

# convert to datetime format & extract year into new column called "yyyy"
ctd['yyyy'] = pd.to_datetime(ctd['DateTime']).dt.year 
#print(ctd['yyyy'])

# convert to datetime format & extract month into new column called "mm"
ctd['mm'] = pd.to_datetime(ctd['DateTime']).dt.month
#print(ctd['mm'])


#%%

### FILTER AFTER 1990 ###

ctd.info()
ctd.head()

#Create copy of df to calculate measurements lost/removed
ctd_x = ctd.copy()

# Remove profiles before 1989
ctd = ctd[ctd["Date"]>date(1988,12,31)]

# Sort df again
ctd = ctd.sort_values(by=['cruise_ID','depth'])

# Reset bottle df index replacing old index column
ctd = ctd.reset_index(drop=True)

#%%

### REMOVE CTD PROFILES WITH NO SURFACE MESUREMENTS ###

# Remove CTD profiles with no surface measurements
grouped = ctd.groupby('cruise_ID')
ctd = grouped.filter(lambda x: (x['depth'].min() <= 11) and (len(x) >= 11))

# Sort df again
ctd = ctd.sort_values(by=['DateTime','depth'])

# Reset bottle df index replacing old index column
ctd = ctd.reset_index(drop=True)

# Test and inspect new df 
ctd[["cruise_ID","temperature"]].groupby("cruise_ID").count()
print(ctd.head()) # print first few lines of ctd df
print(ctd.info()) # print list of column names and count of non zero values

ctd.to_csv('data/HOT_CTD_01.csv')

#%%

### Extract required data from CTD dataframe into numpy arrays ###
ctd.info()

ctd_date      = ctd.loc[:,'Date'].to_numpy()
depth         = ctd.loc[:,'depth'].to_numpy()
pressure      = ctd.loc[:,'press'].to_numpy()
temperature   = ctd.loc[:,'temperature'].to_numpy()
salinity      = ctd.loc[:,'salinity'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
ID_ctd        = ctd.loc[:,'cruise_ID'].to_numpy()
ctd_DateTime  = pd.to_datetime(ctd['DateTime'].values)

ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()

### Cruise ID list for CTD ###
ID_list_ctd = pd.unique(ID_ctd) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

#%%
### COUNT CTD PROFILES PER YEAR ###

#Create new df with number of CTD profiles (cruises) per year
ctd_y = ctd[["cruise_ID", "yyyy"]].groupby("yyyy").nunique()
print(ctd_y)

# Nice Table for Notebook
print(ctd_y.to_markdown())

#%%

### COMPUTE VARIABLES NEEDED FROM CTD DATA ###
# Variables: MLD and BVF

# Station ALOHA coordinates (22.75° N, 158.00° W)
lat, lon = 22.75, -158.00

# Convert PSS‑78 Practical Salinity to Absolute Salinity (g/kg)
SA = gsw.SA_from_SP(salinity, pressure, lon, lat)

# Convert ITS‑90 in‑situ temperature to Conservative Temperature (°C)
CT = gsw.CT_from_t(SA, temperature, pressure)

# Compute potential density (kg/m³) referenced to the surface (0 dbar)
pot_rho = gsw.pot_rho_t_exact(SA, CT, pressure, p_ref=0) #used for Boyer or other MLD criteria

# 'Old' density
#density = sw.dens0(salinity, temperature)


# Add to the existing CTD DataFrame
ctd["density"] = pot_rho#density
ctd["salinity"] = SA
ctd["temperature"] = CT
#ctd["pot_rho"] = pot_rho
ctd.info()

#%%
### COMPUTE Mixed Layer Depth & CTD Prof Meta Data###

def boyer_mld(depth, density, ref_depth=10.0, delta_rho=0.03):
    """
    Boyer (2004) MLD: first depth where density differs from
    its value at ref_depth by delta_rho.
    """
    depth = np.asarray(depth)
    density = np.asarray(density)
    i_ref = np.argmin(np.abs(depth - ref_depth))
    rho_ref = density[i_ref]
    mask = density >= (rho_ref + delta_rho)
    return float(depth[mask][0]) if np.any(mask) else np.nan

def compare_mld(depth, temperature, salinity, density,
                smoothing_window=5, min_ref_depth=10):
    """
    Compare Boyer and Holt & Talley MLDs on the same profile,
    including Holt & Talley temperature, salinity, and density algorithms.
    - Applies median smoothing to the top `min_ref_depth` meters
      for temperature, salinity, and density to mitigate surface noise.
    - Returns a dict with:
        * boyer_mld
        * holt_talley_density_mld
        * holt_talley_temp_mld
        * holt_talley_salinity_mld
        * density_difference (density MLD difference)
    """
    from scipy.signal import medfilt
    # Ensure numpy arrays
    depth = np.asarray(depth)
    temperature = np.asarray(temperature)
    salinity = np.asarray(salinity)
    density = np.asarray(density)

    # Identify surface layer mask
    surf_mask = depth <= min_ref_depth

    # Smooth surface layer for all variables
    temp_sm = temperature.copy()
    sal_sm = salinity.copy()
    dens_sm = density.copy()
    temp_sm[surf_mask] = medfilt(temperature[surf_mask], kernel_size=smoothing_window)
    sal_sm[surf_mask] = medfilt(salinity[surf_mask], kernel_size=smoothing_window)
    dens_sm[surf_mask] = medfilt(density[surf_mask], kernel_size=smoothing_window)

    # 1) Compute Boyer MLD on smoothed density
    mld_boyer = boyer_mld(depth, dens_sm)

    # 2) Run Holt & Talley on smoothed profile
    ht = HolteAndTalley(depth, temp_sm, sal_sm, dens_sm)
    mld_ht_density  = float(ht.densityMLD)
    mld_ht_temp     = float(ht.tempMLD)
    mld_ht_salinity = float(ht.salinityMLD)

    # 3) Collect results
    return {
        "boyer_mld":                mld_boyer,
        "holt_talley_density_mld":  mld_ht_density,
        "holt_talley_temp_mld":     mld_ht_temp,
        "holt_talley_salinity_mld": mld_ht_salinity,
        "density_difference":       mld_ht_density - mld_boyer,
    }

# Add to the existing CTD DataFrame
temperature   = ctd['temperature'].values
salinity      = ctd['salinity'].values
density       = ctd['density'].values
    
### Test more efficient loop
# Create empty arrays to store calculated MLD and single time/date per profile
num_profiles = len(ID_list_ctd)
MLD = np.full(num_profiles, np.nan)
MLD_d = np.full(num_profiles, np.nan)
MLD_s = np.full(num_profiles, np.nan)
MLD_boyer = np.full(num_profiles, np.nan)
ctd_DateTime_prof = np.empty(num_profiles, dtype='datetime64[ns]')
ctd_DecYear_prof = np.full(num_profiles, np.nan)

# Convert ID_list_ctd to a numpy array for efficient comparisons
ID_list_ctd = np.array(ID_list_ctd)

# Vectorized loop for MLD, time, Decimal year, and lat/lon per profile
for count, i in enumerate(ID_list_ctd):
    asx = np.where(ID_ctd == i)[0]
    if len(asx) == 0:
        continue

    A = depth[asx]
    B = temperature[asx]
    S = salinity[asx]
    D = density[asx]
    
    # Get the first instance of ctd_DateTime and ctd_Decimal_year for this profile
    ctd_DateTime_prof[count] = ctd_DateTime[asx][0]
    ctd_DecYear_prof[count] = ctd_Decimal_year[asx][0]

    if len(A) >= 10 and np.min(A) <= 20:
        h = HolteAndTalley(A, B, S, D)
        comp = compare_mld(A,B,S,D)
        MLD[count] = comp['holt_talley_temp_mld']#h.tempMLD
        MLD_d[count] = comp['holt_talley_density_mld']
        MLD_s[count] = comp['holt_talley_salinity_mld']#h.salinityMLD
        MLD_boyer[count] = comp['boyer_mld'] #compare_mld(depth, temperature, salinity, density)
        
        
###

# Compare various MLDs
  
plt.plot(ctd_DateTime_prof,MLD, label = "Temp")
plt.plot(ctd_DateTime_prof,MLD_d, label = "Dens")
plt.plot(ctd_DateTime_prof,MLD_s, label = "Sal")
plt.legend()
plt.show()

ctd_DateTime_prof = pd.DatetimeIndex(ctd_DateTime_prof, dtype='datetime64[ns]', name='date_time', freq=None)

#Plot rough scatter map of profile locations
plt.scatter(MLD,MLD_d)
plt.scatter(MLD,MLD_boyer)
plt.show()


plt.plot(ctd_DateTime_prof,MLD_d, label = "Dens HT")
plt.plot(ctd_DateTime_prof,MLD_boyer, label = "Dens Boyer")
plt.legend()
plt.show()

"Boyer MLD best - matches HPLC chla profiles and same method used for previous HOT studies Karl et al. 2021, 2022"

#%%

### SCATTER OF TEMP & DENSITY MLD ###

ctd_month_prof = pd.to_datetime(ctd_DateTime_prof).month

# Define a custom colormap where warmer colors are in the middle (June and July)
colors = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', 
          '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026', '#313695']
nodes = [0.0, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 1.0]
cmap = LinearSegmentedColormap.from_list("custom_diverging", list(zip(nodes, colors)), N=12)

# Create a scatter plot with colors based on the month
plt.figure(figsize=(10, 6))
scatter = plt.scatter(MLD, MLD_d, c=ctd_month_prof, cmap=cmap, alpha=0.7, edgecolors='k')

# Add color bar to show which color corresponds to which month
cbar = plt.colorbar(scatter, ticks=np.arange(1, 13))
cbar.set_label('Month')
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Add axis labels
plt.xlabel('Temperature MLD (m)', fontsize=14)
plt.ylabel('Density MLD (m)', fontsize=14)

# Optional: add grid for better readability
#plt.grid(True)

# Display the plot
plt.show()

#%%
### COMPUTE BVF ###
# Compute Brunt–Väisälä (buoyancy) frequency (BVF)
bvf =  np.empty(len(temperature))+nan #Consider renaming to BVF
for i in ID_list_ctd:
    A = depth[ctd.cruise_ID == i]
    B = temperature[ctd.cruise_ID == i]
    C = salinity[ctd.cruise_ID == i]
    #D = lat[ctd.cruise_ID == i]
    if len(A) >= 10: 
        if np.min(A) <= 20:    
            #BRUNT_T = sw.bfrq(C, B, A, D[0])
            BRUNT_T = sw.bfrq(C, B, A)
            BRUNT_T1 = BRUNT_T[0]
            BRUNT_T1 = np.resize(BRUNT_T1, len(BRUNT_T1))
            BRUNT_T3 = BRUNT_T[2]
            BRUNT_T3 = np.resize(BRUNT_T3, len(BRUNT_T3))
            interpfunc = interpolate.interp1d(BRUNT_T3, BRUNT_T1, kind='linear', fill_value="extrapolate")
            xxx = interpfunc(A)
            bvf[ctd.cruise_ID == i] = xxx
    
plt.plot(bvf) #Why some negative spikes?
plt.show()

# Add BVF to existing CTD data frame
ctd["BVF"] = bvf
ctd.head()

#%%

### SAVE DFs TO CSV ###

# Save CTD data df to csv
ctd.to_csv('data/HOT_CTD_Cleaned.csv')

#%%
### SAVE MLD AND single CTD values to DF ###

# df with all single CTD calculations
CTDpd = pd.DataFrame()
CTDpd["Cruise_ID"] = ID_list_ctd
CTDpd["DateTime"]  = ctd_DateTime_prof
CTDpd["MLD_temp"]  = MLD
CTDpd["MLD_dens"]  = MLD_d
CTDpd["MLD_sal"]   = MLD_s
CTDpd["MLD_boyer"] = MLD_boyer
CTDpd['Date']      = ctd_DateTime_prof.date # extract Date yyyy-mm-dd from DatetimeIndex
CTDpd['DecYear']   = ctd_DecYear_prof

CTDpd.head()
CTDpd.info()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(CTDpd['Date']), max(CTDpd['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(CTDpd['Date'])))
print("Max Date: "+str(max(CTDpd['Date'])))

CTDpd.to_csv('data/HOT_CTD_profData.csv')

#%%

### READ CLEANED CTD DATA FROM CSV ###

# CSV filename
filename_1 = 'data/HOT_CTD_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd        = pd.read_csv(filename_1, index_col = 0)

ctd.info()

### Extract required data from CTD dataframe into numpy arrays ###
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
ctd_Decimal_year = ctd.loc[:,'Dec_Year'].to_numpy()
ctd_DateTime  = pd.to_datetime(ctd['DateTime'].values)
ctd_month_prof = pd.to_datetime(ctd_DateTime_prof).month

### Cruise ID list for CTD ###
# Extract cruise_ID
ID_list_ctd = ctd['cruise_ID'].values

# Removes Duplicates
ID_list_ctd = pd.unique(pd.Series(ID_list_ctd)) # ID_list_ctd now = x1 ID cruise number per CTD profile

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ID_ctd))

### Read CTD prof data ###

# CSV filename
filename_2 = 'data/HOT_CTD_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
ctd_prof = pd.read_csv(filename_2, index_col = 0)
# Inspect ctd_prof df
ctd_prof.info()
#ctd_prof.head()

# Extract required data from df
ctd_DateTime_prof = pd.DatetimeIndex(ctd_prof['DateTime'])
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
MLD               = ctd_prof.loc[:,'MLD_temp'].to_numpy()
MLD_d             = ctd_prof.loc[:,'MLD_dens'].to_numpy()
MLD_s             = ctd_prof.loc[:,'MLD_sal'].to_numpy()
MLD_boyer         = ctd_prof.loc[:,'MLD_boyer'].to_numpy()
ctd_DecYear_prof  = ctd_prof.loc[:,'DecYear'].to_numpy()

plt.plot(ctd_DateTime_prof,MLD)
plt.plot(ctd_DateTime_prof,MLD_d)
plt.show()

#%%

### PLOT SCATTER OF MLDs ###

# Remove NaNs from data
# MLDtemp
ads = np.where(~np.isnan(MLD))
MLD_1 = MLD[ads]
MLD_d_1 = MLD_d[ads]
ctd_DecYear_prof_1 = ctd_DecYear_prof[ads]

STATS_REG = spearmanr(MLD_1, MLD_d_1)
# R value
R_mld = ("{0:.2f}".format(STATS_REG[0]))
# P value
P_mld = ("{0:.3f}".format(STATS_REG[1]))
print([R_mld, P_mld])

# Set plot up
fig, (ax1) = plt.subplots(1, figsize=(8, 6))
fig.subplots_adjust(wspace=0.26, hspace=0.2)
fig.patch.set_facecolor('White')

# Subplot
im1 = ax1.scatter(MLD_1, MLD_d_1, c=ctd_month_prof, alpha=0.7, cmap=cmap,
                  label='R = '+str(R_mld)+'; $p$ = '+str(P_mld))
#ax1.set_title('(a) Chl-a', fontsize=18, color='k')
ax1.set_ylabel('MLDdens (m)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('MLDtemp (m)', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.legend(loc="upper left", fontsize=12,
           title='Spearman Correlation', title_fontsize=14)
#ax1.set_xlim([-0.05, 1.21])
#ax1.set_ylim([-0.05, 1.21])

# Add color bar to show which color corresponds to which month
cbar = fig.colorbar(im1, ax=ax1,ticks=np.arange(1, 13))
cbar.set_label('Month')
cbar.set_ticks(np.arange(1, 13))
cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

fig.savefig('plots/HOT_Scatter_MLD_Spearman.png',
            format='png', dpi=300, bbox_inches="tight")
plt.show()

#%%

### COMPARE TEMPERATURE & DENSITY MLD ###

#MLD timeseries plot

# Set plot up
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(18, 6.5))
fig.subplots_adjust(wspace=0.26, hspace=0)
fig.patch.set_facecolor('White')

# Chla Subplot
ax1.plot(ctd_DateTime_prof,MLD, color='b', alpha=0.9,
                  label='MLD-temp')
ax1.plot(ctd_DateTime_prof,MLD_d, color='orange', alpha=0.9,
                  label='MLD-dens')
ax1.set_title('(a)', fontsize=18, color='k')
ax1.set_ylabel('MLD (m)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Year', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.legend(loc="upper right", fontsize=14)
#ax1.set_xlim([-0.05, 1.21])
#ax1.set_ylim([-0.05, 1.21])
ax1.locator_params(nbins=7)

im2 = ax2.scatter(MLD_1, MLD_d_1, c=ctd_DecYear_prof_1, alpha=0.5, cmap='viridis_r',
                  label='R = '+str(R_mld)+'; $p$ = '+str(P_mld))
ax2.set_title('(b)', fontsize=18, color='k')
ax2.set_ylabel('MLD-dens (m)', fontsize=16)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('MLD-temp (m)', fontsize=16)
ax2.xaxis.set_tick_params(labelsize=15)
ax2.legend(loc="upper left", fontsize=14,
           title='Spearman Correlation', title_fontsize=14)
#ax1.set_xlim([-0.05, 1.21])
#ax1.set_ylim([-0.05, 1.21])
ax2.locator_params(nbins=7)
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.ax.locator_params(nbins=6)
cbar2.set_label("Year", size=16)
cbar2.ax.tick_params(labelsize=15)

fig.savefig('plots/HOT_MLD_Compare_timeseries.png',
            format='png', dpi=300, bbox_inches="tight")
plt.show()

#%% 

### PLOT EXAMPLE PROFILE CTD DATA ###

PROFILE_ID = 328

# Extract all CTD rows of this profile into new df to print
prof_check = ctd[ctd['cruise_ID'] == PROFILE_ID]
print(prof_check)

# find indexes of ctd ID for the selected profile
AS = np.where(ID_ctd == PROFILE_ID)
# Extract variables of specific profile
depth_1        = depth[AS]
temperature_1  = temperature[AS]
salinity_1     = salinity[AS]
fluorescence_1 = fluorescence[AS]
density_1      = density[AS]
bvf_1          = bvf[AS]
time_date_1    = ctd_DateTime[AS]
doxy_1         = doxy[AS]

# Find index of single value variables of specific variables
ASD = np.where(ID_list_ctd == PROFILE_ID)
# Extract single value variables
MLD_TEMP = MLD[ASD]
MLD_dens = MLD_d[ASD]
MLD_sal  = MLD_s[ASD]
MLD_boy  = MLD_boyer[ASD]

#Figure parameters
XSIZE           = 18    #Define the xsize of the figure window
YSIZE           = 12    #Define the ysize of the figure window
Title_font_size = 18    #Define the font size of the titles
Label_font_size = 15    #Define the font size of the labels
Title           = "Figure 2"      #Plot title
TEMP_COLOUR     = 'r'   #Temperature colours see https://matplotlib.org/2.0.2/api/colors_api.html
PSAL_COLOUR     = 'b'   #Salinity colours see https://matplotlib.org/2.0.2/api/colors_api.html
DOXY_COLOUR     = 'c'   #Dissolved Oxy colours see https://matplotlib.org/2.0.2/api/colors_api.html
FLUO_COLOUR     = 'g'   #Chla colours see https://matplotlib.org/2.0.2/api/colors_api.html
DENSITY_COLOUR  = 'k' 
BRUNT_COLOUR    = 'orange'

# Define the figure window including 5 subplots orientated horizontally
fig, ([ax1, ax2, ax3, ax4], [ax5, ax7, ax8]) = plt.subplots(2,4, sharey=True, figsize=(XSIZE,YSIZE), \
gridspec_kw={'hspace': 0.2})
    
fig.tight_layout()
    
st = fig.suptitle("CTD-ID: "+str(PROFILE_ID)+" DateTime: "+str(time_date_1[0]),\
          fontsize=20,
          color="k")
st.set_y(0.92)

# Temperature subplot
ax1.plot(temperature_1,depth_1, \
         color = TEMP_COLOUR, marker = 'o', linestyle = 'None')
ax1.plot([np.min(temperature_1)-0.5,np.max(temperature_1)+0.5],[MLD_TEMP,MLD_TEMP],
         color = 'g', marker = 'None', linestyle = '--', label = 'MLD-Temp')
ax1.plot([np.min(temperature_1)-0.5,np.max(temperature_1)+0.5],[MLD_dens,MLD_dens],
         color = 'm', marker = 'None', linestyle = '--', label = 'MLD-Dens')
ax1.axhline(y=MLD_sal, color='b', linestyle='--', label='MLD')
ax1.set_ylabel('Depth (m)', fontsize=Label_font_size)
ax1.yaxis.set_tick_params(labelsize=Label_font_size)
ax1.set_ylim([20,0]) 
ax1.set_xlabel('Temperature ($^o$C)', fontsize=Title_font_size, color = TEMP_COLOUR)
ax1.xaxis.set_tick_params(labelsize=Label_font_size)
ax1.set_xlim(xmin=np.min(temperature_1)-0.5, xmax=np.max(temperature_1)+0.5)
ax1.xaxis.set_major_locator(plt.MaxNLocator(2))
ax1.legend(loc="lower right", fontsize=12)

# Salinity subplot
ax2.plot(salinity_1,depth_1, \
         color = PSAL_COLOUR, marker = 'o', linestyle = 'None')
ax2.yaxis.set_tick_params(labelsize=Label_font_size)
ax2.set_ylim([20,0]) 
ax2.set_xlabel('Salinity (PSU)', fontsize=Title_font_size, color = PSAL_COLOUR)
ax2.xaxis.set_tick_params(labelsize=Label_font_size)
ax2.set_xlim(xmin=np.min(salinity_1)-0.05, xmax=np.max(salinity_1)+0.05)
ax2.xaxis.set_major_locator(plt.MaxNLocator(2))

# Density subplot
ax3.plot(density_1,depth_1, \
         color = DENSITY_COLOUR, marker = 'o', linestyle = 'None')
ax3.yaxis.set_tick_params(labelsize=Label_font_size)
ax3.set_ylim([20,0])
ax3.set_xlabel('Density', fontsize=Title_font_size, color = DENSITY_COLOUR)
ax3.xaxis.set_tick_params(labelsize=Label_font_size)
ax3.set_xlim(xmin=np.min(density_1)-0.05, xmax=np.max(density_1)+0.05)
ax3.xaxis.set_major_locator(plt.MaxNLocator(2))

# Fluorescence subplot
ax4.plot(fluorescence_1,depth_1, \
         color = FLUO_COLOUR, marker = 'o', linestyle = 'None')
ax4.yaxis.set_tick_params(labelsize=Label_font_size)
ax4.set_ylim([20,0]) 
ax4.set_xlabel('Fluorescence (RFU)', fontsize=Title_font_size, color = FLUO_COLOUR)
ax4.xaxis.set_tick_params(labelsize=Label_font_size)
ax4.set_xlim(xmin=np.nanmin(fluorescence_1)-0.05, xmax=np.nanmax(fluorescence_1)+0.05)
ax4.xaxis.set_major_locator(plt.MaxNLocator(2))

# Dissolved Oxygen subplot
ax5.plot(doxy_1,depth_1, \
         color = DOXY_COLOUR, marker = 'o', linestyle = 'None') 
ax5.set_xlabel('DOXY (micro mol kg$^{-3}$)', fontsize=Title_font_size, color= DOXY_COLOUR)
ax5.xaxis.set_tick_params(labelsize=Label_font_size)
ax5.set_ylabel('Depth (m)', fontsize=Label_font_size)
ax5.yaxis.set_tick_params(labelsize=Label_font_size)
ax5.xaxis.set_major_locator(plt.MaxNLocator(2))
#ax5.legend(loc="lower right", fontsize=12)

# BVF subplot  
ax7.plot(bvf_1,depth_1, \
         color = BRUNT_COLOUR, marker = 'o', linestyle = 'None')
ax7.plot([np.min(bvf_1)-0.0005,np.max(bvf_1)+0.0005],[MLD_TEMP,MLD_TEMP],color = 'g', marker = 'None', linestyle = '--', label = 'MLD(T) Holt')
# ax6.set_ylabel('Depth (m)', fontsize=15)
ax7.yaxis.set_tick_params(labelsize=Label_font_size)
ax7.set_ylim([20,0])
ax7.set_xlabel('BVF', fontsize=Title_font_size, color = BRUNT_COLOUR)
ax7.xaxis.set_tick_params(labelsize=Label_font_size)
ax7.set_xlim(xmin=np.min(bvf_1)-0.000005, xmax=np.max(bvf_1)+0.000005)
ax7.xaxis.set_major_locator(plt.MaxNLocator(2))
ax7.legend(loc="lower right", fontsize=12)

fig.savefig('plots/HOT-CTD_panel_'+str(PROFILE_ID)+'.png', format='png', dpi=300, bbox_inches="tight")
    
#Complete the plot
plt.show()

compare_mld(depth_1,temperature_1, salinity_1,density_1,min_ref_depth=10)
