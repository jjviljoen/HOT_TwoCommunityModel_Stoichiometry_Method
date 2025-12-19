"""
HOT - Import In-Situ PAR data at HOT and progate to ML and DCM levels

Matched above water PAR values also converted to below water within functions used to
calculate the average mixed layer irradiance and irradiance levels at the DCM by
propagating surface satellite PAR to the DCM depth using the Beer-lambert law and in situ estimates of Kd.

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
import math

# Import specific modules from packages
from datetime import datetime, timedelta
from dateutil import relativedelta
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial" # acutally Arial font in background

# Supress
import warnings
warnings.filterwarnings("ignore") # Added to remove the warning "UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray." on 2nd to last cell of code
#
# Supresses outputs when trying to "divide by zero" or "divide by NaN" or "treatment for floating-point overflow"
#np.seterr(divide='ignore', over='ignore', invalid='ignore');
np.seterr(all='ignore');

#%%

### DEFINE ALL FUNCTIONS ###

def date_span(start, end):
    """
    Calculate the difference in years, months, and days between two dates.
    """
    # Calculate the relative delta between the start and end dates
    rd = relativedelta.relativedelta(
        pd.to_datetime(end), pd.to_datetime(start))
    # Construct the string representing the duration in years, months, and days
    date_len = '{}y{}m{}d'.format(rd.years, rd.months, rd.days)
    return date_len

def calculate_PAR_Z_dcm(PAR_Surf, Kd, Z_dcm):
    PAR_Surf -= PAR_Surf*0.02 # deduct 2% from satellite PAR at above water to account for reflectance
    PAR_Z_dcm = PAR_Surf * math.exp(-Kd * Z_dcm)
    return PAR_Z_dcm

def calculate_avgPAR_mld(PAR_Surf, Kd, Zm):
    PAR_Surf -= PAR_Surf*0.02 # deduct 2% from satellite PAR at above water to account for reflectance
    E = (PAR_Surf/(Kd*Zm)) * (1-math.exp(-Kd * Zm))
    return E
    
#%%

### IMPORT INTEGRATED DAILY PAR

#Import Int PAR from HOT PI - units LiCOR daily integrals (E/m^2/d)
# Data provided by Angelique White 

# For tab-separated values
df = pd.read_csv('data/licor_par_int_angel.txt', skiprows=2, delim_whitespace=True, index_col=False)
df.info()

# Convert 'jday' and 'year' to datetime and extract month and day
df['date'] = df.apply(lambda row: datetime(int(row['year']), 1, 1) + timedelta(days=int(row['jday']) - 1), axis=1)
df['mm'] = df['date'].dt.month
df['dd'] = df['date'].dt.day
df['date'] = df['date'].dt.date

df.rename(columns={'int': 'PAR_int'}, inplace=True)
df.info()

# Save the processed DataFrame to CSV
df.to_csv('data/HOT_licor_par_int_cleaned.csv', index=False)

par_df = df.copy()

#%%

### IMPORT BOTTLE PROF DATA

# CSV filename
filename_1 = 'data/HOT_Bottle_profData_Int.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)

#Print list of columns
print(bottle_prof.columns.tolist())

# Sort dataframe by Cruise_ID
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])

# Reset dataframe index, replacing the old index column
bottle_prof = bottle_prof.reset_index(drop=True)

### Extract data needed
ID_list_bottle  = bottle_prof['Cruise_ID'].to_numpy()
b_DateTime_prof = pd.to_datetime(bottle_prof['DateTime'].values)   # Convert 'DateTime' column to pandas datetime series
b_date = bottle_prof['Date'].to_numpy()

#bottle_prof.plot.scatter('MLD_dens_boyer','MLD_used')

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

#%%

### MATCH PAR DATA WITH BOTTLE PROFILES ###

# Ensure date columns are in datetime format
bottle_prof['Date'] = pd.to_datetime(bottle_prof['Date'])
par_df['date'] = pd.to_datetime(par_df['date'])

# Extract date-only values for comparison
bottle_dates = bottle_prof['Date'].dt.date.to_numpy()
par_dates = par_df['date'].dt.date.to_numpy()

# Prepare arrays for matched values
matched_par_int = np.full(len(bottle_prof), np.nan)
matched_par_id = np.full(len(bottle_prof), np.nan, dtype=object)
matched_par_date = pd.Series([pd.NaT] * len(bottle_prof), dtype='datetime64[ns]')


# Match PAR data to bottle profiles
for i, prof_date in enumerate(bottle_dates):
    match_idx = np.where(par_dates == prof_date)[0]
    
    if match_idx.size > 0:
        # Exact match found
        matched_par_int[i] = par_df.loc[match_idx[0], 'PAR_int']
        matched_par_id[i] = par_df.loc[match_idx[0], 'crn']
        matched_par_date[i] = par_df.loc[match_idx[0], 'date']
    else:
        # Try ±2 day window
        window_matches = np.where(
            (par_dates >= prof_date - timedelta(days=2)) &
            (par_dates <= prof_date + timedelta(days=2))
        )[0]
        
        if window_matches.size > 0:
            matched_par_int[i] = par_df.loc[window_matches[0], 'PAR_int']
            matched_par_id[i] = par_df.loc[window_matches[0], 'crn']
            matched_par_date[i] = par_df.loc[window_matches[0], 'date']
        else:
            print(f"No match within ±2 days for profile date: {prof_date}")

# Add matched data to the bottle profile DataFrame
bottle_prof['PAR_int'] = matched_par_int
bottle_prof['PAR_ID'] = matched_par_id
bottle_prof['PAR_date'] = matched_par_date.dt.date

# Report number of matches
print(f"Number of matched profiles (including ±2 day window): {np.count_nonzero(~np.isnan(matched_par_int))}")

# Find bottle Cruise ID where PAR is NaN
missing_par_cruise_ids = bottle_prof.loc[bottle_prof['PAR_int'].isna(), 'Cruise_ID'].tolist()
# Display the result
print("Cruises Int PAR not available")
print(missing_par_cruise_ids)


#%%

### COMPUTE MIXED LAYER & DCM PAR

bottle_prof.info()

# Initialize arrays to store results
par_DCM = []
par_MLD = []

# Loop through each row in the DataFrame
for index, row in bottle_prof.iterrows():
    par_surface = row['PAR_int']
    prof_Kd = row['Kd']
    prof_dcm_depth = row['DCM_depth']
    mld = row['MLD_used'] # MLD boyer used during model fitting
    
    # Calculate PAR at DCM depth
    result_dcm = calculate_PAR_Z_dcm(par_surface, prof_Kd, prof_dcm_depth)
    par_DCM.append(result_dcm)
    
    # Calculate average PAR in MLD
    result_mld = calculate_avgPAR_mld(par_surface, prof_Kd, mld)
    par_MLD.append(result_mld)

# Add results to the DataFrame
bottle_prof['PAR_DCM'] = par_DCM
bottle_prof['PAR_MLD'] = par_MLD

bottle_prof.info()

#Save df to csv
bottle_prof.to_csv('data/HOT_Bottle_profData_Int.csv')


"END"

