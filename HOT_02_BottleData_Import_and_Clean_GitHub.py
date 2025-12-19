"""
HOT: Import & Clean Bottle data from Station ALOHA downloaded from HOT-DOGS

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 18 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk

"""
#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
import gsw
# Import specific modules from packages
from datetime import date
from PyAstronomy import pyasl # used to compute decimal year from DateTime & back. Info: https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/decimalYear.html
from dateutil import relativedelta
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

#%%

#############################
### READ, CLEAN & FILTER ORIGINAL BOTTLE DATA ###
#############################

filename_bottle = 'data/hot_bottle_direct_23April2025.xlsx'
# Data last downloaded on 23 April 2025 and copied into an Excel xlsx file
# Data acknowledgement: "Data obtained via the Hawaii Ocean Time-series HOT-DOGS application; University of Hawai'i at Mānoa. National Science Foundation Award # 1756517".

# Load the Excel file, skipping the first 2 rows
df = pd.read_excel(filename_bottle, skiprows=2)

# REMOVE TRAILING COMMAS FROM THE RAW FIRST-COLUMN TEXT
df.iloc[:, 0] = (df.iloc[:, 0].astype(str)        # ensure it’s stringy
      .str.replace(r',\s*$', '', regex=True)      # drop any comma + spaces at end
)

# Split the data in the first column into separate columns based on commas
split_columns = df.iloc[:, 0].str.split(', ', expand=True)

# Replace empty strings (or cells with only spaces) with NaN
split_columns.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
# Drop columns that are entirely NaN
split_columns = split_columns.dropna(axis=1, how='all')

# Replace the original DataFrame with the cleaned (split) DataFrame
df = split_columns.copy()

# Set the first row as column headers then drop that row
df.columns = df.iloc[0]
df = df[1:]

# Optionally, extract units (if needed for reference) from the next row
column_names = df.columns.tolist()  # original column names
units = df.iloc[0].tolist()           # units row (if desired)
print("Columns with Units:")
print(list(zip(column_names, units)))

# Clean column names: remove any leading spaces
cleaned_column_names = [col.lstrip() for col in column_names]
df.columns = cleaned_column_names
df.info()

# Remove the row containing the units
df = df[1:]

# Remove any leading spaces from all string cells in the DataFrame
df = df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

bottle = df.copy()
bottle.info()

bottle["date"].values[-1]

# Replace empty cells with Nan
# first, replace any exact numeric -9 (this covers int and float -9.0)
bottle.replace(-9, np.nan, inplace=True)

bottle.replace(
    to_replace=r'^-0*9(?:\.0+)?$',
    value=np.nan,
    regex=True,
    inplace=True
)
bottle.head()

#Inspect Chla
bottle['hplc'].head()

# Removes last 2 digits from ID to make all results in profile the same ID
bottle['Cruise_ID'] = bottle['botid'].astype(str).str[:-2].astype(np.int64)
# Extract Cruise number
bottle['CRN'] = bottle['Cruise_ID'].astype(str).str[:-5].astype(np.int64)
# Extract cast no
bottle['CASTNO'] = bottle['Cruise_ID'].astype(str).str[-2:].astype(np.int64)

#Many Cruise data without date and time, find and add dates manually from HOT cruise summaries
bottle_NaN_dates = bottle[bottle['date'].isnull()]

bottle_NaN_dates = bottle_NaN_dates[['botid', 'CRN', 'CASTNO']]

#Find unique CRN and cast missing dates
bottle_NaN_dates = bottle_NaN_dates.drop_duplicates(subset=["CRN", "CASTNO"])
# Reset bottle df index removing old index with missing numbers after slice
bottle_NaN_dates = bottle_NaN_dates.reset_index(drop=True)

#Save smaller dataset
# Write csv
bottle_NaN_dates.to_csv('data/HOT_Bottle_NaN_dates.csv')

#Import Cruise summary data
# Read CSV
cruise_summary   = pd.read_csv('data/HOT_CruiseSummaries.csv', index_col = 0) #Read CSV as panda dataframe
cruise_summary.info()

print("Max Cruise No. = ",max(cruise_summary['Cruise']))
#Loop to add missing dates to bottle df from NaN dates df
df =bottle.copy()
df2 = cruise_summary.copy()

for i in range(len(df)):
    if pd.isna(df['date'].iloc[i]):
        crn =  df['CRN'].iloc[[i]].values
        cast = df['CASTNO'].iloc[[i]].values
        date_where = (df2['Cruise'] == crn[0]) & (df2['Cast'] == cast[0])
        date_1 = df2['Date'].loc[date_where]
        if len(date_1) > 1:
            date_1 = int(date_1.iloc[1])
        else:
            date_1 = int(date_1)
        df['date'].iloc[[i]] = date_1
    if pd.isna(df['time'].iloc[i]):
        crn =  df['CRN'].iloc[[i]].values
        cast = df['CASTNO'].iloc[[i]].values
        if cast[0] > 0:
            date_where = (df2['Cruise'] == crn[0]) & (df2['Cast'] == cast[0])
            time_1 = df2['Time'].loc[date_where]
            if len(time_1) > 1:
                time_1 = int(time_1.iloc[1])
            else:
                time_1 = int(time_1)
            df['time'].iloc[[i]] = time_1
        
#df[df.isnull().any(1)]
bottle_NaN_dates = df[df['time'].isnull()]

bottle_NaN_dates = bottle_NaN_dates[['botid', 'CRN', 'CASTNO']]

#Find unique CRN and cast missing dates
bottle_NaN_dates = bottle_NaN_dates.drop_duplicates(subset=["CRN", "CASTNO"])
# Reset bottle df index removing old index with missing numbers after slice
bottle_NaN_dates = bottle_NaN_dates.reset_index(drop=True)

#Replace bottle with updated df        
bottle = df        

# Replace nan time values with 000000 values instead of removing, thus gaining some early years bottle profiles
bottle['time'] = bottle['time'].fillna('000000')

# Remove rows with NaN depth values
bottle = bottle.dropna(subset=['date'])
# Reset bottle df index removing old index with missing numbers after slice
bottle = bottle.reset_index(drop=True)

bottle['Cruise_ID_o'] = bottle['Cruise_ID']
bottle['Cruise_ID'] = bottle['CRN']

# Reformat date
# convert date column to datetime format
bottle['date_0'] = bottle['date'].astype(np.int64).astype(str).str.pad(width=6, fillchar='0')
bottle['Date'] = pd.to_datetime(bottle['date_0'], format='%m%d%y')
#convert datetime to date
bottle['Date'] = pd.to_datetime(bottle['Date']).dt.date.astype(str)
bottle.info()

#convert datetime to date
bottle['Date'] = pd.to_datetime(bottle['Date']).dt.date

# Ensure all times are strings
bottle['time'] = bottle['time'].astype(str).str.strip()

# Normalize time strings to hhmmss format
def normalize_time_str(t):
    t = t.strip()
    if len(t) == 3:
        # e.g. '202' -> pad to 4 '0202', then to 6 '020200' => 02:02:00
        t = t.zfill(4).ljust(6, '0')
    elif len(t) == 4:
        # e.g. '1610' -> '161000' => 16:10:00
        t = t.ljust(6, '0')
    elif len(t) == 5:
        # e.g. '15304' -> zfill to 6 '015304' => 01:53:04
        t = t.zfill(6)
    # if len == 6, leave as-is
    return t

bottle['time_n'] = bottle['time'].apply(normalize_time_str)

# Finally parse into a proper time object
bottle['time_f'] = (pd.to_datetime(bottle['time_n'], format='%H%M%S'
                                   , errors='coerce').dt.time)
# After above format fix, still some NaT time result after coerce 
#as some original time given value has less than 3 characters which 
#results in NaT when trying to format

#replace any NaT/NaN in time_f with the next non-missing time below.
bottle['time_f'] = bottle['time_f'].bfill()

check_df = bottle[['botid', 'CRN','CASTNO','time','time_n', 'time_f' ]]

bottle_NaN_dates = bottle[bottle['time_f'].isnull()]

bottle_NaN_dates = bottle_NaN_dates[['botid', 'CRN', 'CASTNO']]

# Combine date and time columns into datetimeindex object
bottle['DateTime'] = pd.to_datetime(bottle["Date"].astype(str) + " " + bottle["time_f"].astype(str))

bottle['DateTime'] = pd.DatetimeIndex(bottle['DateTime'])

bottle['time'] = bottle['time_f']

bottle.info()

#%%
#Compute DecYear from DateTime
b_DateTime = pd.to_datetime(bottle['DateTime'])

# Use vectorized operations to compute decimal years
b_DecYear = b_DateTime.apply(pyasl.decimalYear).to_numpy()

# Assign the computed decimal years back to the DataFrame
bottle['DecYear'] = b_DecYear

# Display DataFrame information
bottle.info()
#%%

# Compute Depth from presure using new gsw package
bottle["depth"] = -1*(gsw.z_from_p(np.array(bottle["press"]), 22.75))

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract year into new column called "yyyy"
bottle['yyyy'] = pd.to_datetime(bottle['DateTime']).dt.year 

# convert to datetime format & extract mopnth into new column called "mm"
bottle['mm'] = pd.to_datetime(bottle['DateTime']).dt.month

# convert to datetime format & extract Date yyyy-mm-dd into new column called "Date"
bottle['Date'] = pd.to_datetime(bottle['DateTime']).dt.date

# Drop the specified columns
bottle = bottle.drop(columns=['botid', 'date', 'time_n', 'time_f', 'date_0'])
bottle.info()

# Specify the desired order of columns
first_columns = ["Cruise_ID", "CRN", "CASTNO", "Cruise_ID_o", "DateTime", 'DecYear','Date', 'yyyy', 'mm','time', 'press', 'depth']
remaining_columns = [col for col in bottle.columns if col not in first_columns]
new_column_order = first_columns + remaining_columns

# Rearrange the columns
bottle = bottle[new_column_order]
bottle.info()
bottle.head()

#%%

# Check HPLC Chla date range
# Remove rows with NaN HPLC chla values
bottle_chla = bottle.dropna(subset=['hplc'])

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
bottle_date_length = date_span(min(bottle_chla['Date']), max(bottle_chla['Date']))
print("Timespan: "+str(bottle_date_length))
print("Min Date: "+str(min(bottle_chla['Date'])))
print("Max Date: "+str(max(bottle_chla['Date'])))

bottle_chla_prof = bottle_chla.drop_duplicates(subset=['Cruise_ID'])
print(len(bottle_chla_prof))

### EXTRACT ONLY PROFILES AFTER 1990 ###

bottle.info()

#Create copy of df to calculate measurements lost/removed
bottle_x = bottle.copy()

# Remove profiles before 1989.
# To get full year with end date of 2022-09-03, need to have date from 1989-08-25
bottle = bottle[bottle["Date"]>date(1989,8,1)]

# Sort df again
bottle = bottle.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle = bottle.reset_index(drop=True)

#Calculate measurements lost
prof_lost = len(bottle_x) - len(bottle)
print("Bottle Measurements Lost = "+str(prof_lost))

#Save dataset
#bottle.to_csv('data/HOT_Bottle_Cleaned.csv')
bottle.to_csv('data/HOT_Bottle_Cleaned2.csv')