"""
HOT - Fit Two-Community Particulate and Stoichiometry Model to HPLC Chla, PC, PN & PP data

Model developed and modified based on Brewin et al. (2022) Two-Community model and
the extended version used in Viljoen et al. (2024).
Brewin et al. (2022) - https://doi.org/10.1029/2021JC018195
Viljoen et al. (2024) - https://www.nature.com/articles/s41558-024-02136-6

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

### PLOT EVERY PROFILE MODEL FITTED TO?

model_plot_1 = False # If true will plot both Chla & POC model fits for each profile loop

#%%
### LOAD PACKAGES ##
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial" # Set font for all plots
import cmocean
# Import specific modules from packages
from datetime import date
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate # used to interpolate profiles for contour plots
from scipy.stats import spearmanr
from lmfit import report_fit
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore');

#%%

### IMPORT TWO-COMMUNITY MODEL & CUSTOM FUNCTIONS ###

#Import Three Community Module and Functions
from MODULE_2community_particulate_model_functions_Dec2025 import two_community_model, particulate_model, date_span, integrate_sections

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
temperature   = ctd.loc[:,'temperature'].to_numpy()
salinity      = ctd.loc[:,'salinity'].to_numpy()
density       = ctd.loc[:,'density'].to_numpy()
bvf           = ctd.loc[:,'BVF'].to_numpy()
fluorescence  = ctd.loc[:,'fluorescence'].to_numpy()
doxy          = ctd.loc[:,'dissolved_oxygen'].to_numpy()
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

ID_list_ctd       = ctd_prof.loc[:,'Cruise_ID'].to_numpy()
ctd_date_prof     = ctd_prof.loc[:,'Date'].to_numpy()
ctd_DecYear_prof  = ctd_prof.loc[:,'DecYear'].to_numpy()
MLD_temp          = ctd_prof.loc[:,'MLD_temp'].to_numpy()
MLD_dens          = ctd_prof.loc[:,'MLD_dens'].to_numpy()
MLD_sal           = ctd_prof.loc[:,'MLD_sal'].to_numpy()
MLD_boyer         = ctd_prof.loc[:,'MLD_boyer'].to_numpy()

# Compare length of ID_list to all cells containing cruise/profile IDs
print(len(ID_list_ctd))
print(len(ctd_prof))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_prof['Date']), max(ctd_prof['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(ctd_prof['Date'])))
print("Max Date: "+str(max(ctd_prof['Date'])))

#%%

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE CHLA ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Pigments_Chla_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6 = pd.read_csv(filename_1, index_col = 0)
# Inspect df
bottle_6.info()

bottle_6 = bottle_6.sort_values(by=['Cruise_ID','depth'])

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
print("Timespan: "+str(b_date_length))

### Bottle Single MLD & DateTimeIndex ###

### Read/Import cleaned Bottle data from CSV
# CSV filename
filename_1 = 'data/HOT_Bottle_Pigments_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)
bottle_prof.info()
print(len(ID_list_6))
print(len(bottle_prof))

bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])

# Extract bottle MLD with corresponding time ###
b_DateTime_prof  = pd.to_datetime(bottle_prof['DateTime'].values)

### Timespan of bottle data ###

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_prof['Date']))+" to "+str(max(bottle_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))

# Inspect first few rows of the bottle df
print(bottle_prof.head())

#%%

### EXTRACT CLEANED POC DATA & ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Bottle_POC.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)

# Remove rows where depth >= 
bottle_poc = bottle_poc[bottle_poc["depth"] < 470]

bottle_poc.info()

# Sort new df by time, ID and depth
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
b2_DecYear  = bottle_poc.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(bottle_poc['DateTime'].values)

### Cruise_ID list
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))
# 295 profiles with 6 or more POC measurements that matches bottle pigment list

### Import POC PROF Data

# CSV filename
filename_1 = 'data/HOT_Bottle_POC_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc_prof = pd.read_csv(filename_1, index_col = 0)

bottle_poc_prof.info()

# Sort new df by time, ID and depth
bottle_poc_prof = bottle_poc_prof.sort_values(by=['Cruise_ID'])

# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

print(len(bottle_poc_prof))
print(len(ID_list_poc))

idcheck = bottle_poc_prof['Cruise_ID'].values
idcheck = pd.unique(idcheck)
len(idcheck)

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_poc_prof['Date']))+" to "+str(max(bottle_poc_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_poc_prof['Date']), max(bottle_poc_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%
#########################################
### FILTER CHLA FOR ONLY POC PROFILES ###
#########################################

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE CHLA ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Pigments_Chla_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()
bottle_6['Cruise_ID'].head()
bottle_poc['Cruise_ID'].head()

# Create new df containing only data for profiles with matching POC profiles
cruise_check = bottle_6.Cruise_ID.isin(ID_list_poc)
bottle_6 =  bottle_6[bottle_6.Cruise_ID.isin(ID_list_poc)]

# Sort new df by ID and depth
bottle_6 = bottle_6.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_6.to_csv('data/HOT_Pigments_Chla_Cleaned.csv')

bottle_6.info()

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_chla     = bottle_6.loc[:,'Chla'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()

# Bottle DateTime data
b_DateTime     = pd.DatetimeIndex(bottle_6['DateTime'])

### Cruise_ID list for new df is ID_list_6
ID_list_6 = pd.Series(b_ID)
ID_list_6 = pd.unique(ID_list_6)
print(len(ID_list_6))

### Bottle Single MLD & DateTimeIndex ###

### Read/Import cleaned Bottle data from CSV
# CSV filename
filename_1 = 'data/HOT_Bottle_Pigments_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof   = pd.read_csv(filename_1, index_col = 0)
bottle_prof.info()
print(len(ID_list_6))
print(len(bottle_prof))

# Create new df containing only data for profiles also in pigment bottle list
bottle_prof =  bottle_prof[bottle_prof.Cruise_ID.isin(ID_list_6)]

# Sort new df by ID and depth
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])

# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)

# Write Cleaned bottle df to csv
bottle_prof.to_csv('data/HOT_Bottle_profData.csv')

# Extract bottle MLD with corresponding time ###
b_DateTime_prof  = pd.to_datetime(bottle_prof['DateTime'])
b_MLD_prof_temp  = bottle_prof.loc[:,'MLD_temp'].to_numpy()
b_MLD_prof_dens  = bottle_prof.loc[:,'MLD_dens'].to_numpy()

print(len(ID_list_6))
print(len(bottle_prof))

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_prof['Date']))+" to "+str(max(bottle_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%

### EXTRACT CLEANED POP DATA & ID LIST ###

# CSV filename
filename_1 = 'data/HOT_Bottle_POP.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_pop   = pd.read_csv(filename_1, index_col = 0)

# Remove rows where depth >=  470
bottle_pop = bottle_pop[bottle_pop["depth"] < 470]

bottle_pop.info()

# Create new df containing only data for profiles with matching Chla & POC profiles
bottle_pop =  bottle_pop[bottle_pop.Cruise_ID.isin(ID_list_6)]

# Sort new df by time, ID and depth
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new dataset ###
b3_time     = bottle_pop.loc[:,'DateTime'].to_numpy()
#b2_time_2   = pd.to_datetime(bottle_poc['DateTime'])
b3_date      = bottle_pop.loc[:,'Date'].to_numpy()
b3_pop_depth = bottle_pop.loc[:,'depth'].to_numpy()
b3_pop       = bottle_pop.loc[:,'POP'].to_numpy()
b3_ID        = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
b3_year      = bottle_pop.loc[:,'yyyy'].to_numpy()
b3_month     = bottle_pop.loc[:,'mm'].to_numpy()
b3_Decimal_year = bottle_pop.loc[:,'DecYear'].to_numpy()

print(np.nanmax(b3_pop_depth))

#Convert array object to Datetimeindex type
b3_DateTime = pd.DatetimeIndex(b3_time, dtype='datetime64[ns]', name='date_time', freq=None)

### Cruise_ID list
ID_list_pop = pd.unique(b3_ID)
print(len(ID_list_pop))

# Using set difference
values_not_in_pop = list(set(ID_list_6) - set(ID_list_pop))

# Print the result
print(values_not_in_pop)

### Import POP PROF Data

# CSV filename
filename_1 = 'data/HOT_Bottle_POP_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_pop_prof = pd.read_csv(filename_1, index_col = 0)

bottle_pop_prof.info()

# Create new df containing only data for profiles also in pigment bottle list
bottle_pop_prof =  bottle_pop_prof[bottle_pop_prof.Cruise_ID.isin(ID_list_6)]

# Sort new df by time, ID and depth
bottle_pop_prof = bottle_pop_prof.sort_values(by=['Cruise_ID'])

# Reset bottle df index replacing old index column
bottle_pop_prof = bottle_pop_prof.reset_index(drop=True)

print(len(bottle_pop_prof))
print(len(ID_list_pop))

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_pop_prof['Date']))+" to "+str(max(bottle_pop_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_pop_prof['Date']), max(bottle_pop_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%

################
### IMPORT EUPHOTIC DEPTH DATA
################

### Bottle Single MLD & DateTime ###
# CSV filename
filename_1 = 'data/HOT_PAR_Zp_Kd_final.csv'

# Load data from CSV. "index_col = 0" makes the first column the index.
Zp_Kd_df = pd.read_csv(filename_1)

# Inspect the dataframe
Zp_Kd_df.info()
print(len(ID_list_6))           # Print the length of the Cruise_ID list
print(len(Zp_Kd_df))         # Print the number of rows in the dataframe

# Sort dataframe by Cruise_ID
Zp_Kd_df = Zp_Kd_df.sort_values(by=['Cruise_ID'])

# Reset dataframe index, replacing the old index column
Zp_Kd_df = Zp_Kd_df.reset_index(drop=True)

### Cruise_ID list
ID_list_Zp = Zp_Kd_df['Cruise_ID'].to_numpy()
print(len(ID_list_Zp))

### Extract Zp and Kd from HOT ###
Kd_prof = Zp_Kd_df['Kd_Morel_tuned_sw'].to_numpy()
Zp_prof = Zp_Kd_df['Zeu_Morel_tuned_sw'].to_numpy()

#%%

### APPLY MODEL TO SINGLE PROFILE ###
ID_1 = 328

# Make copy of bottle Chla array
m_chla = b_chla

# CTD MLD for profile
prof_MLD_idx = np.where(ID_list_ctd == ID_1)
prof_MLD_boyer   = MLD_boyer[prof_MLD_idx]

prof_MLD = prof_MLD_boyer
# Chla Bottle data
prof_bottle_idx = np.where(bottle_6.CRN == ID_1)
b_DateTime_1 = b_DateTime.date[prof_bottle_idx]
print(b_DateTime_1[0])

prof_depth   = b_depth[prof_bottle_idx]
prof_chla    = m_chla[prof_bottle_idx]

# Extract Zp
where_zp_idx = np.where(Zp_Kd_df.Cruise_ID == ID_1)
Zp = Zp_prof[where_zp_idx]
Kd = Kd_prof[where_zp_idx]

# POC data
prof_poc_idx    = np.where(bottle_poc.CRN == ID_1)
prof_poc        = b2_poc[prof_poc_idx]
prof_pon        = b2_pon[prof_poc_idx]
prof_poc_depth  = b2_depth[prof_poc_idx]

# Extract all bottle df rows of this profile into new df to print
bot_prof = bottle_poc[bottle_poc['Cruise_ID'] == ID_1]
print(bot_prof.head(5))

# POP data
prof_pop_idx    = np.where(bottle_pop.CRN == ID_1)
prof_pop        = b3_pop[prof_pop_idx]
prof_pop_depth  = b3_pop_depth[prof_pop_idx]

b3_DateTime_1   = b3_DateTime.date[prof_pop_idx]
prof_pop_date = b3_DateTime_1[0] 

b2_DateTime_1   = b2_DateTime.date[prof_poc_idx]
prof_poc_date = b2_DateTime_1[0] 

# Remove nan from POC data
ads             = np.where(~np.isnan(prof_poc))
prof_poc        = prof_poc[ads]
prof_poc_depth  = prof_poc_depth[ads]

# Fit model
fit_result = two_community_model(prof_chla, prof_depth, prof_MLD, prof_poc, prof_poc_depth,
                                   data_type = 'bottle', Kd = Kd)
# Show all variables available in fit result tuple
#print(list(fit_result.keys()))
################# CHL-A Fit Results ##################
# Model Parameters
P1_final    = fit_result['P1_final']
TAU1_final  = fit_result['TAU1_final']
BM2_final   = fit_result['BM2_final']
TAU2_final  = fit_result['TAU2_final']
SIG2_final  = fit_result['SIG2_final']

Chl_FitType  = fit_result['fit_type']
# Step 1 fit statistics
#report_fit(fit_result['Chl_FitReport_1'])
#Chl_FitStats_chisq_1   = fit_result['Chl_FitReport_chisq_1']
Chl_FitStats_redchi_1  = fit_result['Chl_FitReport_redchi_1']
Chl_FitStats_aic_1     = fit_result['Chl_FitReport_aic_1']
Chl_FitStats_bic_1     = fit_result['Chl_FitReport_bic_1']
# Step 1 parameter errors
Chl_FitErr_P2_1        = fit_result['Chl_FitReport_P2_err_1'] # Only p2 error extract as P1 has fixed relationship with P2: P1 = 10**(0.08 * P2 + 0.66) # Red Sea P1 Tau1 Relationship
# Step 2 fit statistics
report_fit(fit_result['Chl_FitReport_2'])
Chl_FitStats_chisq_2   = fit_result['Chl_FitReport_chisq_2']
Chl_FitStats_redchi_2  = fit_result['Chl_FitReport_redchi_2']
Chl_FitStats_aic_2     = fit_result['Chl_FitReport_aic_2']
Chl_FitStats_bic_2     = fit_result['Chl_FitReport_bic_2']
Chl_FitStats_P4        = fit_result['Chl_FitReport_P4'] # Extract separately as final Tau2 = P4_FIT + P5_FIT * 3.0
# Step 2 parameter errors
Chl_FitErr_P1_2        = fit_result['Chl_FitReport_P1_err_2'] # Zero in a two step fit for two communities
Chl_FitErr_P2_2        = fit_result['Chl_FitReport_P2_err_2'] # Zero in a two step fit for two communities
Chl_FitErr_P3_2        = fit_result['Chl_FitReport_P3_err_2'] # Zero for 1 community fit
Chl_FitErr_P4_2        = fit_result['Chl_FitReport_P4_err_2'] # Zero for 1 community fit
Chl_FitErr_P5_2        = fit_result['Chl_FitReport_P5_err_2'] # Zero for 1 community fit
Chl_FitErr_Tau2        = fit_result['Chl_FitReport_Tau2_err'] # Zero for 1 community fit

# Model Chl Arrays
MLD_pop_FIT    = fit_result['Chl_C1_fit']
DCM_pop_FIT    = fit_result['Chl_C2_fit']
CHL_model_fit  = fit_result['Chl_Total_fit']
# Model Chl Arrays High Res
MLD_pop_FIT2    = fit_result['Chl_C1_fit_HiRes']
DCM_pop_FIT2    = fit_result['Chl_C2_fit_HiRes']
CHL_model_fit2  = fit_result['Chl_Total_fit_HiRes']
prof_depth2     = fit_result['prof_depth_HiRes']
prof_chla_surf  = fit_result['prof_chla_surf']
mod_dcm_depth   = fit_result['DCM1_depth']
mod_dcm_peak    = fit_result['DCM1_peak']
mod_dcm_width   = fit_result['DCM1_width']
Kd              = fit_result['Kd']
#Zp              = fit_result['Zp']

mld_od = prof_MLD*Kd
print("MLD_OD: "+str(mld_od))
print("TAU1_final: "+str(TAU1_final))
print("TAU2_final: "+str(TAU2_final))
print("SIG2_final: "+str(SIG2_final))

print("Zp: ",Zp)
#print("Zp_int: ",Zp_2)
#print(MLD_OD)
print("MLD: "+str(prof_MLD))
print("DCM1_depth: "+str(mod_dcm_depth))
print("DCM1_peak: "+str(mod_dcm_peak))
print("DCM1_width: "+str(mod_dcm_width))

################# POC fit Results ##################

# POC Fit Stats & Errors
Chl_FitType  = fit_result['fit_type']
# fit statistics
#report_fit(fit_result['POC_FitReport'])
POC_FitStats_chisq  = fit_result['POC_FitReport_chisq']
POC_FitStats_redchi = fit_result['POC_FitReport_redchi']
POC_FitStats_aic    = fit_result['POC_FitReport_aic']
POC_FitStats_bic    = fit_result['POC_FitReport_bic']
# Fit Parameter errors
POC_FitErr_P1        = fit_result['POC_FitReport_P1_err'] # C1 ratio error
POC_FitErr_P2        = fit_result['POC_FitReport_P2_err'] # C2 ratio error
POC_FitErr_P3        = fit_result['POC_FitReport_P3_err'] # Background error

SURF_POC  = fit_result['prof_poc_surf']
C_Chl_ratio_C1 = fit_result['C_Chl_ratio_C1']
C_Chl_ratio_C2 = fit_result['C_Chl_ratio_C2']
# Model POC Arrays
MLD_pop_FIT_POC = fit_result['POC_C1_fit']
DCM_pop_FIT_POC = fit_result['POC_C2_fit']
prof_phytoC = fit_result['POC_Phyto_fit']
prof_phytoC     = MLD_pop_FIT_POC + DCM_pop_FIT_POC
BACKGROUND_POC = fit_result['POC_background']
TOTAL_POC_MODEL = fit_result['POC_Total_fit']
livingC_percent = prof_phytoC/TOTAL_POC_MODEL*100
# Model POC Arrays High Res
MLD_pop_FIT2_POC  = fit_result['POC_C1_fit_HiRes']
DCM_pop_FIT2_POC  = fit_result['POC_C2_fit_HiRes']
prof_phytoC_2     = MLD_pop_FIT2_POC + DCM_pop_FIT2_POC
BACKGROUND2_POC   = fit_result['POC_background_HiRes']
POC_model_fit2    = fit_result['POC_Total_fit_HiRes']
livingC_percent2 = prof_phytoC_2/POC_model_fit2*100

print("C:Chl_1 = "+str(C_Chl_ratio_C1))
print("C:Chl_2 = "+str(C_Chl_ratio_C2))

### PON Fit ###

#from model_function_module_nutrients import three_community_model, particulate_model 
pon_fit = particulate_model(prof_pon, prof_poc_depth, prof_chla_surf, Kd, P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final)

# Save PON Fit Stats & Errors
# POC Fit Stats & Errors
Part_FitType  = pon_fit['fit_type']
# fit statistics
#report_fit(pon_fit['FitReport'])
PON_FitStats_chisq  = pon_fit['FitReport_chisq']
PON_FitStats_redchi = pon_fit['FitReport_redchi']
PON_FitStats_aic    = pon_fit['FitReport_aic']
PON_FitStats_bic    = pon_fit['FitReport_bic']
# Fit Parameter errors
PON_FitErr_P1        = pon_fit['FitReport_P1_err'] # C1 ratio error
PON_FitErr_P2        = pon_fit['FitReport_P2_err'] # C2 ratio error
PON_FitErr_P3        = pon_fit['FitReport_P3_err'] # Background error

N_Chl_ratio_C1        = pon_fit['P_Chl_ratio_C1']
N_Chl_ratio_C2        = pon_fit['P_Chl_ratio_C2']
surface_N             = pon_fit['C1_fit']
subsurface_N          = pon_fit['C2_fit']
background_N          = pon_fit['background']
PON_model_total       = pon_fit['Total_fit']
surface_N_HiRes       = pon_fit['C1_fit_HiRes']
subsurface_N_HiRes    = pon_fit['C2_fit_HiRes']
background_N_HiRes    = pon_fit['background_HiRes']
PON_model_total_HiRes = pon_fit['Total_fit_HiRes']

print("N:Chl_1 = "+str(N_Chl_ratio_C1))
print("N:Chl_2 = "+str(N_Chl_ratio_C2))

# Calculate weight-to-weight C:N ratio
C_N_ratio_C1 = (C_Chl_ratio_C1)/(N_Chl_ratio_C1)
C_N_ratio_C2 = (C_Chl_ratio_C2)/(N_Chl_ratio_C2)
print("C:N_1 weight = "+str(C_N_ratio_C1))
print("C:N_2 weight = "+str(C_N_ratio_C2))

bk_ratio_CN = BACKGROUND_POC[0]/background_N[0]
print(bk_ratio_CN)

# Calculate molar C:N ratio
# Define the atomic mass ratio
C_N_ATOMIC_RATIO = 12.01 / 14.01  # Approximately 0.857
# Calculate molar C:N ratios by multiplying weight ratios by atomic ratio
C_N_molar_C1 = C_N_ratio_C1 / C_N_ATOMIC_RATIO
C_N_molar_C2 = C_N_ratio_C2 / C_N_ATOMIC_RATIO
bk_ratio_CN_molar = bk_ratio_CN / C_N_ATOMIC_RATIO
print("C:N_1 (molar) =", C_N_molar_C1)
print("C:N_2 (molar) =", C_N_molar_C2)
print("Background C:N (molar) =", bk_ratio_CN_molar)

### POP Fit ###
pop_fit = particulate_model(prof_pop, prof_pop_depth, prof_chla_surf, Kd, P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final)

# Save PON Fit Stats & Errors
# POC Fit Stats & Errors
Part_FitType  = pon_fit['fit_type']
# fit statistics
#report_fit(pop_fit['FitReport'])
POP_FitStats_chisq  = pop_fit['FitReport_chisq']
POP_FitStats_redchi = pop_fit['FitReport_redchi']
POP_FitStats_aic    = pop_fit['FitReport_aic']
POP_FitStats_bic    = pop_fit['FitReport_bic']
# Fit Parameter errors
POP_FitErr_P1        = pop_fit['FitReport_P1_err'] # C1 ratio error
POP_FitErr_P2        = pop_fit['FitReport_P2_err'] # C2 ratio error
POP_FitErr_P3        = pop_fit['FitReport_P3_err'] # Background error

P_Chl_ratio_C1        = pop_fit['P_Chl_ratio_C1']
P_Chl_ratio_C2        = pop_fit['P_Chl_ratio_C2']
surface_P             = pop_fit['C1_fit']
subsurface_P          = pop_fit['C2_fit']
background_P          = pop_fit['background']
POP_model_total       = pop_fit['Total_fit']
surface_P_HiRes       = pop_fit['C1_fit_HiRes']
subsurface_P_HiRes    = pop_fit['C2_fit_HiRes']
background_P_HiRes    = pop_fit['background_HiRes']
POP_model_total_HiRes = pop_fit['Total_fit_HiRes']

print("P:Chl_1 = "+str(P_Chl_ratio_C1))
print("P:Chl_2 = "+str(P_Chl_ratio_C2))

C_P_ratio_C1 = (C_Chl_ratio_C1)/(P_Chl_ratio_C1)
C_P_ratio_C2 = (C_Chl_ratio_C2)/(P_Chl_ratio_C2)
print("C:P_1 = "+str(C_P_ratio_C1))
print("C:P_2 = "+str(C_P_ratio_C2))
N_P_ratio_C1 = (N_Chl_ratio_C1)/(P_Chl_ratio_C1)
N_P_ratio_C2 = (N_Chl_ratio_C2)/(P_Chl_ratio_C2)
print("N:P_1 = "+str(N_P_ratio_C1))
print("N:P_2 = "+str(N_P_ratio_C2))

bk_ratio_CP = BACKGROUND_POC[0]/background_P[0]
print(bk_ratio_CP)
bk_ratio_NP = background_N[0]/background_P[0]
print(bk_ratio_NP)

# Define atomic mass ratios for molar conversions
C_P_ATOMIC_RATIO = 12.01 / 30.97  # Approximately 0.388
N_P_ATOMIC_RATIO = 14.01 / 30.97  # Approximately 0.452

C_P_molar_C1 = C_P_ratio_C1/C_P_ATOMIC_RATIO
C_P_molar_C2 = C_P_ratio_C2/C_P_ATOMIC_RATIO
print("C:P_1 molar = "+str(C_P_molar_C1))
print("C:P_2 molar = "+str(C_P_molar_C2))
N_P_molar_C1 = N_P_ratio_C1/N_P_ATOMIC_RATIO
N_P_molar_C2 = N_P_ratio_C2/N_P_ATOMIC_RATIO
print("N:P_1 molar = "+str(N_P_molar_C1))
print("N:P_2 molar = "+str(N_P_molar_C2))

bk_molar_CP = bk_ratio_CP/C_P_ATOMIC_RATIO
print("Background C:P (molar) =", bk_molar_CP)
bk_molar_NP = bk_ratio_NP/N_P_ATOMIC_RATIO
print("Background N:P (molar) =", bk_molar_NP)


# PLOT MODEL FIT TO BOTH CHLA, PC, PN and PP

import matplotlib.pyplot as plt
import numpy as np

# === USER-ADJUSTABLE PARAMETERS ===
fig_size = (15, 5)
wspace = 0.1
hspace = 0.3
facecolor = 'white'
marker_size = 10
line_width = 3
# Font sizes
title_fs = 15
xlabel_fs = 15
ylabel_fs = 15
tick_fs = 12
legend_fs = 10

# PLOT MODEL FIT TO CHLA, POC, PON, POP
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=fig_size)
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.patch.set_facecolor(facecolor)

# --- (a) Chl-a ---
chl_max = max(np.nanmax(prof_chla), np.nanmax(CHL_model_fit))
chl_min = -chl_max * 0.05
chl_max *= 1.2
ax1.axhline(y=prof_MLD, color='k', linestyle='--')
ax1.plot(prof_chla, prof_depth, color='g', marker='X', markersize=marker_size, linestyle='None', label='Chl-a Data')
ax1.plot(MLD_pop_FIT2, prof_depth2, color='r', linestyle='-', linewidth=line_width, label='Surface')
ax1.plot(DCM_pop_FIT2, prof_depth2, color='b', linestyle='-', linewidth=line_width, label='Subsurface')
ax1.plot(CHL_model_fit2, prof_depth2, color='k', linestyle='-', linewidth=line_width, label='Total')
ax1.set_ylabel('Depth (m)', fontsize=ylabel_fs)
ax1.set_title('(a) Chl-a', fontsize=title_fs, color='g')
ax1.set_ylim([200, 0])
ax1.set_xlabel('Chl-a (mg m$^{-3}$)', fontsize=xlabel_fs)
ax1.set_xlim(xmin=chl_min, xmax=chl_max)
ax1.legend(loc='lower right', fontsize=legend_fs)
ax1.tick_params(axis='both', labelsize=tick_fs)
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))

# --- (b) POC ---
poc_max = max(np.nanmax(prof_poc), np.nanmax(TOTAL_POC_MODEL))
poc_min = -poc_max * 0.05
poc_max *= 1.2
ax2.axhline(y=prof_MLD, color='k', linestyle='--')
ax2.plot(prof_poc, prof_poc_depth, color='orange', marker='X', markersize=marker_size, linestyle='None', label='PC Data')
ax2.plot(MLD_pop_FIT2_POC, prof_depth2, color='r', linestyle='-', linewidth=line_width, label='Surface')
ax2.plot(DCM_pop_FIT2_POC, prof_depth2, color='b', linestyle='-', linewidth=line_width, label='Subsurface')
ax2.axvline(x=BACKGROUND_POC[0], color='m', linestyle='--', linewidth=line_width, label='Non-Algal Bk')
ax2.plot(POC_model_fit2, prof_depth2, color='k', linestyle='-', linewidth=line_width, label='Total')
ax2.set_title('(b) PC', fontsize=title_fs, color='orange')
ax2.set_ylim([200, 0])
ax2.set(yticklabels=[])
ax2.set_xlabel('PC (mg m$^{-3}$)', fontsize=xlabel_fs)
ax2.set_xlim(xmin=poc_min, xmax=poc_max)
ax2.legend(loc='lower right', fontsize=legend_fs)
ax2.tick_params(axis='both', labelsize=tick_fs)
ax2.xaxis.set_major_locator(plt.MaxNLocator(5))

# --- (c) PON ---
pon_max = max(np.nanmax(prof_pon), np.nanmax(PON_model_total))
pon_min = -pon_max * 0.05
pon_max *= 1.2
ax3.axhline(y=prof_MLD, color='k', linestyle='--')
ax3.plot(prof_pon, prof_poc_depth, color='purple', marker='X', markersize=marker_size, linestyle='None', label='PN Data')
ax3.plot(surface_N_HiRes, prof_depth2, color='r', linestyle='-', linewidth=line_width, label='Surface')
ax3.plot(subsurface_N_HiRes, prof_depth2, color='b', linestyle='-', linewidth=line_width, label='Subsurface')
ax3.axvline(x=background_N[0], color='m', linestyle='--', linewidth=line_width, label='Non-Algal Bk')
ax3.plot(PON_model_total_HiRes, prof_depth2, color='k', linestyle='-', linewidth=line_width, label='Total')
ax3.set_title('(c) PN', fontsize=title_fs, color='purple')
ax3.set_ylim([200, 0])
ax3.set(yticklabels=[])
ax3.set_xlabel('PN (mg m$^{-3}$)', fontsize=xlabel_fs)
ax3.set_xlim(xmin=pon_min, xmax=pon_max)
ax3.legend(loc='lower right', fontsize=legend_fs)
ax3.tick_params(axis='both', labelsize=tick_fs)
ax3.xaxis.set_major_locator(plt.MaxNLocator(5))

# --- (d) POP ---
pop_max = max(np.nanmax(prof_pop), np.nanmax(POP_model_total))
pop_min = -pop_max * 0.05
pop_max *= 1.2
ax4.axhline(y=prof_MLD, color='k', linestyle='--')
ax4.plot(prof_pop, prof_pop_depth, color='c', marker='X', markersize=marker_size, linestyle='None', label='PP Data')
ax4.plot(surface_P_HiRes, prof_depth2, color='r', linestyle='-', linewidth=line_width, label='Surface')
ax4.plot(subsurface_P_HiRes, prof_depth2, color='b', linestyle='-', linewidth=line_width, label='Subsurface')
ax4.axvline(x=background_P[0], color='m', linestyle='--', linewidth=line_width, label='Non-Algal Bk')
ax4.plot(POP_model_total_HiRes, prof_depth2, color='k', linestyle='-', linewidth=line_width, label='Total')
ax4.set_title('(d) PP', fontsize=title_fs, color='c')
ax4.set_ylim([200, 0])
ax4.set(yticklabels=[])
ax4.set_xlabel('PP (mg m$^{-3}$)', fontsize=xlabel_fs)
ax4.set_xlim(xmin=pop_min, xmax=pop_max)
ax4.legend(loc='lower right', fontsize=legend_fs)
ax4.tick_params(axis='both', labelsize=tick_fs)
ax4.xaxis.set_major_locator(plt.MaxNLocator(5))

# Save the figure
fig.savefig(f'plots/HOT_ModelFit_Chl_POC_PON_POP_{ID_1}_methods.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

# === USER-ADJUSTABLE PARAMETERS ===
IDs = [328, 298]  # list of profile IDs to plot
fig_size = (17, 10)
wspace = 0.1
hspace = 0.3
facecolor = 'white'
marker_size = 8
line_width = 3
# Font sizes
title_fs = 14
xlabel_fs = 12
ylabel_fs = 12
tick_fs = 10
legend_fs = 9


def fit_profile(ID_1):
    # Extract profile-specific data using boolean masks
    prof_MLD = MLD_boyer[ID_list_ctd == ID_1]
    prof_depth = b_depth[bottle_6.CRN == ID_1]
    prof_chla = b_chla[bottle_6.CRN == ID_1]

    # Light attenuation
    Kd = Kd_prof[Zp_Kd_df.Cruise_ID == ID_1]

    # POC
    mask_poc = bottle_poc.CRN == ID_1
    prof_poc = b2_poc[mask_poc]
    prof_poc_depth = b2_depth[mask_poc]
    valid = ~np.isnan(prof_poc)
    prof_poc = prof_poc[valid]
    prof_poc_depth = prof_poc_depth[valid]

    # PON
    prof_pon = b2_pon[mask_poc]

    # POP
    mask_pop = bottle_pop.CRN == ID_1
    prof_pop = b3_pop[mask_pop]
    prof_pop_depth = b3_pop_depth[mask_pop]

    # Fit three-community model
    fit = two_community_model(
        prof_chla, prof_depth, prof_MLD,
        prof_poc, prof_poc_depth,
        data_type='bottle', Kd=Kd
    )

    # High-res depth
    prof_depth2 = fit['prof_depth_HiRes']

    # Organize results
    chla = {
        'surface': fit['Chl_C1_fit_HiRes'],
        'subsurf': fit['Chl_C2_fit_HiRes'],
        'total': fit['Chl_Total_fit_HiRes'],
        'background': None
    }
    poc = {
        'surface': fit['POC_C1_fit_HiRes'],
        'subsurf': fit['POC_C2_fit_HiRes'],
        'total': fit['POC_Total_fit_HiRes'],
        'background': fit['POC_background_np']
    }

    # PON model
    pon_fit = particulate_model(
        prof_pon, prof_poc_depth, fit['prof_chla_surf'], Kd,
        fit['P1_final'], fit['TAU1_final'], fit['BM2_final'], fit['TAU2_final'], fit['SIG2_final']
    )
    pon = {
        'surface': pon_fit['C1_fit_HiRes'],
        'subsurf': pon_fit['C2_fit_HiRes'],
        'total': pon_fit['Total_fit_HiRes'],
        'background': pon_fit['background_np']
    }

    # POP model
    pop_fit = particulate_model(
        prof_pop, prof_pop_depth, fit['prof_chla_surf'], Kd,
        fit['P1_final'], fit['TAU1_final'], fit['BM2_final'], fit['TAU2_final'], fit['SIG2_final']
    )
    pop = {
        'surface': pop_fit['C1_fit_HiRes'],
        'subsurf': pop_fit['C2_fit_HiRes'],
        'total': pop_fit['Total_fit_HiRes'],
        'background': pop_fit['background_np']
    }

    return {
        'ID': ID_1,
        'MLD': prof_MLD,
        'prof_depth': prof_depth,
        'prof_depth2': prof_depth2,
        'prof_chla': prof_chla,
        'prof_poc': prof_poc,
        'prof_poc_depth': prof_poc_depth,
        'prof_pon': prof_pon,
        'prof_pop': prof_pop,
        'prof_pop_depth': prof_pop_depth,
        'chla': chla,
        'poc': poc,
        'pon': pon,
        'pop': pop
    }

# Fit all profiles
results = [fit_profile(ID) for ID in IDs]

# Plotting
fig, axes = plt.subplots(2, 4, figsize=fig_size, sharey='row')
fig.subplots_adjust(wspace=wspace, hspace=hspace)
fig.patch.set_facecolor(facecolor)

# Define top and bottom row labels
top_labels = ['(a) Chl-a', '(b) PC', '(c) PN', '(d) PP']
bot_labels = ['(e) Chl-a', '(f) PC', '(g) PN', '(h) PP']
colors = ['g', 'orange', 'purple', 'c']

for row, res in enumerate(results):
    for col, var in enumerate(['chla', 'poc', 'pon', 'pop']):
        ax = axes[row, col]
        d = res[var]
        if var == 'chla':
            ax.plot(res['prof_chla'], res['prof_depth'], color='g', marker='X', markersize=marker_size, linestyle='None')
        elif var == 'poc':
            ax.plot(res['prof_poc'], res['prof_poc_depth'], color='orange', marker='X', markersize=marker_size, linestyle='None')
        elif var == 'pon':
            ax.plot(res['prof_pon'], res['prof_poc_depth'], color='purple', marker='X', markersize=marker_size, linestyle='None')
        else:
            ax.plot(res['prof_pop'], res['prof_pop_depth'], color='c', marker='X', markersize=marker_size, linestyle='None')

        ax.plot(d['surface'], res['prof_depth2'], color='r', linewidth=line_width)
        ax.plot(d['subsurf'], res['prof_depth2'], color='b', linewidth=line_width)
        ax.plot(d['total'], res['prof_depth2'], color='k', linewidth=line_width)
        ax.axhline(y=res['MLD'], color='k', linestyle='--')
        if d['background'] is not None:
            ax.axvline(x=d['background'], color='m', linestyle='--', linewidth=line_width)

        ax.set_ylim([200, 0])
        label = top_labels[col] if row == 0 else bot_labels[col]
        ax.set_title(label, fontsize=title_fs, color=colors[col])
        if col == 0:
            ax.set_ylabel('Depth (m)', fontsize=ylabel_fs)
        else:
            ax.set_ylabel(None)
            ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xlabel(f"{label.split(') ')[1]} (mg m${{-3}}$)", fontsize=xlabel_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.legend(['Data', 'Surface', 'Subsurface', 'Total', 'Bk'], fontsize=legend_fs, loc='lower right')

fig.savefig(f'plots/HOT_ModelFit_multiIDs_{"_".join(map(str, IDs))}.jpeg', dpi=300, bbox_inches='tight')
plt.show()

#%%

### LOOP MODEL FIT ###

#Include plotting of every profile model fitted?
model_plot = False  # If True will plot both Chla & POC model fits for each profile loop

#Fit Model to which Chla parameter?
m_chla = b_chla # HPLC Chla

# Initialize arrays for storing results
ID_list_bottle = ID_list_6
chl_mod_surface    = np.full(len(m_chla), np.nan)
chl_mod_subsurface = np.full(len(m_chla), np.nan)
chl_mod_total      = np.full(len(m_chla), np.nan)
chl_data_input     = np.full(len(m_chla), np.nan)
chl_mod_data_diff  = np.full(len(m_chla), np.nan)

poc_mod_surface    = np.full(len(b2_poc), np.nan)
poc_mod_subsurface = np.full(len(b2_poc), np.nan)
poc_mod_total      = np.full(len(b2_poc), np.nan)
poc_mod_data_diff  = np.full(len(b2_poc), np.nan)
poc_mod_bk         = np.full(len(b2_poc), np.nan)
poc_data_input     = np.full(len(b2_poc), np.nan)

C_Chl_ratio_C1 = np.full(len(ID_list_poc), np.nan)
C_Chl_ratio_C2 = np.full(len(ID_list_poc), np.nan)

POC_bk_np     = np.full(len(ID_list_poc), np.nan)

MLD_used_poc = np.full(len(ID_list_poc), np.nan)
df_poc_ID    = np.full(len(ID_list_poc), np.nan)
df_poc_date  = np.full(len(ID_list_poc), np.nan, dtype='datetime64[s]')
df_poc_DateTime = np.full(len(ID_list_poc), np.nan, dtype='datetime64[s]')
df_poc_DecYear = np.full(len(ID_list_poc), np.nan)

P1_model_fit = np.full(len(ID_list_bottle), np.nan)
P2_model_fit = np.full(len(ID_list_bottle), np.nan)
P3_model_fit = np.full(len(ID_list_bottle), np.nan)
P4_model_fit = np.full(len(ID_list_bottle), np.nan)
P5_model_fit = np.full(len(ID_list_bottle), np.nan)
P4_array     = np.full(len(ID_list_bottle), np.nan)
MLD_used     = np.full(len(ID_list_bottle), np.nan)
KD_used      = np.full(len(ID_list_bottle), np.nan)
Zp_used      = np.full(len(ID_list_bottle), np.nan)

chl_data_surf = np.full(len(ID_list_bottle), np.nan)
# Save Chl Fit Report results
# Step 1 fit statistics
Chl_FitStats_chisq_1_array  = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_redchi_1_array = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_aic_1_array    = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_bic_1_array    = np.full(len(ID_list_bottle), np.nan)
# Step 1 parameter errors
Chl_FitErr_P2_1_array       = np.full(len(ID_list_bottle), np.nan)
# Step 2 fit statistics
Chl_FitStats_chisq_2_array   = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_redchi_2_array  = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_aic_2_array     = np.full(len(ID_list_bottle), np.nan)
Chl_FitStats_bic_2_array     = np.full(len(ID_list_bottle), np.nan)
# Step 2 parameter errors
Chl_FitErr_P1_2_array        = np.full(len(ID_list_bottle), np.nan)
Chl_FitErr_P2_2_array        = np.full(len(ID_list_bottle), np.nan)
Chl_FitErr_P3_2_array        = np.full(len(ID_list_bottle), np.nan)
Chl_FitErr_P4_2_array        = np.full(len(ID_list_bottle), np.nan)
Chl_FitErr_P5_2_array        = np.full(len(ID_list_bottle), np.nan)
Chl_FitErr_Tau2_array        = np.full(len(ID_list_bottle), np.nan)

# POC Fit Stats & Errors
POC_Fit_ratio_C1  = np.full(len(ID_list_bottle), np.nan)
POC_Fit_ratio_C2  = np.full(len(ID_list_bottle), np.nan)
POC_Fit_bk        = np.full(len(ID_list_bottle), np.nan)
POC_FitStats_chisq_array  = np.full(len(ID_list_bottle), np.nan)
POC_FitStats_redchi_array = np.full(len(ID_list_bottle), np.nan)
POC_FitStats_aic_array   = np.full(len(ID_list_bottle), np.nan)
POC_FitStats_bic_array    = np.full(len(ID_list_bottle), np.nan)
# POC Fit Parameter errors
POC_FitErr_P1_array       = np.full(len(ID_list_bottle), np.nan)
POC_FitErr_P2_array       = np.full(len(ID_list_bottle), np.nan)
POC_FitErr_P3_array       = np.full(len(ID_list_bottle), np.nan)

# PON Fit Stats & Errors
PON_Fit_ratio_C1  = np.full(len(ID_list_bottle), np.nan)
PON_Fit_ratio_C2  = np.full(len(ID_list_bottle), np.nan)
PON_Fit_bk        = np.full(len(ID_list_bottle), np.nan)
PON_FitStats_chisq_array  = np.full(len(ID_list_bottle), np.nan)
PON_FitStats_redchi_array = np.full(len(ID_list_bottle), np.nan)
PON_FitStats_aic_array   = np.full(len(ID_list_bottle), np.nan)
PON_FitStats_bic_array    = np.full(len(ID_list_bottle), np.nan)
# PON Fit Parameter errors
PON_FitErr_P1_array       = np.full(len(ID_list_bottle), np.nan)
PON_FitErr_P2_array       = np.full(len(ID_list_bottle), np.nan)
PON_FitErr_P3_array       = np.full(len(ID_list_bottle), np.nan)

# POP Fit Stats & Errors
POP_Fit_ratio_C1  = np.full(len(ID_list_bottle), np.nan)
POP_Fit_ratio_C2  = np.full(len(ID_list_bottle), np.nan)
POP_Fit_bk        = np.full(len(ID_list_bottle), np.nan)
POP_FitStats_chisq_array  = np.full(len(ID_list_bottle), np.nan)
POP_FitStats_redchi_array = np.full(len(ID_list_bottle), np.nan)
POP_FitStats_aic_array    = np.full(len(ID_list_bottle), np.nan)
POP_FitStats_bic_array    = np.full(len(ID_list_bottle), np.nan)
# POP Fit Parameter errors
POP_FitErr_P1_array       = np.full(len(ID_list_bottle), np.nan)
POP_FitErr_P2_array       = np.full(len(ID_list_bottle), np.nan)
POP_FitErr_P3_array       = np.full(len(ID_list_bottle), np.nan)

# Integration arrays for modelled hi-res results
chla_mod_total_int = np.full(len(ID_list_bottle), np.nan)
chla_mod_surf_int  = np.full(len(ID_list_bottle), np.nan)
chla_mod_surf_conc = np.full(len(ID_list_bottle), np.nan)
chla_mod_sub_int   = np.full(len(ID_list_bottle), np.nan)
chla_mod_sub_conc  = np.full(len(ID_list_bottle), np.nan)
chla_mod_dcm_conc  = np.full(len(ID_list_bottle), np.nan)

poc_mod_total_int = np.full(len(ID_list_bottle), np.nan)
poc_mod_surf_int  = np.full(len(ID_list_bottle), np.nan)
poc_mod_sub_int   = np.full(len(ID_list_bottle), np.nan)
poc_mod_dcm_conc  = np.full(len(ID_list_bottle), np.nan)
poc_mod_surf_conc = np.full(len(ID_list_bottle), np.nan)
poc_mod_sub_conc  = np.full(len(ID_list_bottle), np.nan)
phytoC_dcm_conc   = np.full(len(ID_list_bottle), np.nan)
phytoC_int        = np.full(len(ID_list_bottle), np.nan)
phytoC_ratio_median_mld = np.full(len(ID_list_bottle), np.nan)
phytoC_ratio_dcm        = np.full(len(ID_list_bottle), np.nan)

pon_mod_total_int = np.full(len(ID_list_bottle), np.nan)
pon_mod_surf_int  = np.full(len(ID_list_bottle), np.nan)
pon_mod_sub_int   = np.full(len(ID_list_bottle), np.nan)
pon_mod_dcm_conc  = np.full(len(ID_list_bottle), np.nan)
pon_mod_surf_conc = np.full(len(ID_list_bottle), np.nan)
pon_mod_sub_conc  = np.full(len(ID_list_bottle), np.nan)
phytoN_dcm_conc   = np.full(len(ID_list_bottle), np.nan)
phytoN_int        = np.full(len(ID_list_bottle), np.nan)
phytoN_ratio_median_mld = np.full(len(ID_list_bottle), np.nan)
phytoN_ratio_dcm = np.full(len(ID_list_bottle), np.nan)

pop_mod_total_int = np.full(len(ID_list_bottle), np.nan)
pop_mod_surf_int  = np.full(len(ID_list_bottle), np.nan)
pop_mod_sub_int   = np.full(len(ID_list_bottle), np.nan)
pop_mod_surf_conc = np.full(len(ID_list_bottle), np.nan)
pop_mod_sub_conc  = np.full(len(ID_list_bottle), np.nan)
pop_mod_dcm_conc  = np.full(len(ID_list_bottle), np.nan)
phytoP_dcm_conc   = np.full(len(ID_list_bottle), np.nan)
phytoP_int        = np.full(len(ID_list_bottle), np.nan)
phytoP_ratio_median_mld = np.full(len(ID_list_bottle), np.nan)
phytoP_ratio_dcm = np.full(len(ID_list_bottle), np.nan)

# PON Fit arrays
pon_mod_surface    = np.full(len(b2_poc), np.nan)
pon_mod_subsurface = np.full(len(b2_poc), np.nan)
pon_mod_bk         = np.full(len(b2_poc), np.nan)
pon_mod_total      = np.full(len(b2_poc), np.nan)
pon_data_input     = np.full(len(b2_poc), np.nan)
pon_mod_data_diff  = np.full(len(b2_poc), np.nan)

N_Chl_ratio_C1 = np.full(len(ID_list_poc), np.nan)
N_Chl_ratio_C2 = np.full(len(ID_list_poc), np.nan)
C_N_ratio_C1   = np.full(len(ID_list_poc), np.nan)
C_N_ratio_C2   = np.full(len(ID_list_poc), np.nan)
bk_ratio_CN    = np.full(len(ID_list_poc), np.nan)

# POP Fit arrays
pop_mod_surface    = np.full(len(b3_pop), np.nan)
pop_mod_subsurface = np.full(len(b3_pop), np.nan)
pop_mod_bk         = np.full(len(b3_pop), np.nan)
pop_mod_total      = np.full(len(b3_pop), np.nan)
pop_data_input     = np.full(len(b3_pop), np.nan)
pop_mod_data_diff  = np.full(len(b3_pop), np.nan)

P_Chl_ratio_C1 = np.full(len(ID_list_pop), np.nan)
P_Chl_ratio_C2 = np.full(len(ID_list_pop), np.nan)
C_P_ratio_C1   = np.full(len(ID_list_pop), np.nan)
C_P_ratio_C2   = np.full(len(ID_list_pop), np.nan)
bk_ratio_CP    = np.full(len(ID_list_pop), np.nan)
N_P_ratio_C1   = np.full(len(ID_list_pop), np.nan)
N_P_ratio_C2   = np.full(len(ID_list_pop), np.nan)
bk_ratio_NP    = np.full(len(ID_list_pop), np.nan)

df_pop_ID      = np.full(len(ID_list_pop), np.nan)
df_pop_date    = np.full(len(ID_list_pop), np.nan, dtype='datetime64[s]')
df_pop_DecYear = np.full(len(ID_list_pop), np.nan)
MLD_used_pop   = np.full(len(ID_list_pop), np.nan)

# Loop through all profiles
count = 0
pp_count = 0
poc_count = 0  # count to save for POC single profile values
pop_count = 0

for i in ID_list_bottle:
    #i = 162
    print(f"--- Profile ID: {i} ---")
    where_bottle_idx = np.where(bottle_6.Cruise_ID == i)
    prof_date = np.unique(b_date[where_bottle_idx])
    prof_MLD_idx   = np.where(ID_list_ctd == i)
    prof_MLD_t     = MLD_temp[prof_MLD_idx]
    prof_MLD_d     = MLD_dens[prof_MLD_idx]
    prof_MLD_s     = MLD_sal[prof_MLD_idx]
    prof_MLD_boyer = MLD_boyer[prof_MLD_idx]
    
    prof_MLD = prof_MLD_boyer#prof_MLD_d#min(prof_MLD_t,prof_MLD_d,prof_MLD_s)
    
    # Extract Zp
    where_zp_idx = np.where(Zp_Kd_df.Cruise_ID == i)
    Zp = Zp_prof[where_zp_idx]
    Kd = Kd_prof[where_zp_idx]
    
    if prof_MLD >= 0:
        MLD_used[count] = prof_MLD
        
        prof_depth = b_depth[where_bottle_idx]
        prof_chla = m_chla[where_bottle_idx]
        b_DateTime_1 = b_DateTime.date[where_bottle_idx]
        prof_date = b_DateTime_1[0]

        # POC data
        prof_poc_idx = np.where(bottle_poc.CRN == i)
        prof_poc = b2_poc[prof_poc_idx]
        prof_poc_depth = b2_depth[prof_poc_idx]
        b2_DateTime_1   = b2_DateTime.date[prof_poc_idx]
        prof_poc_date = b2_DateTime_1[0]
        prof_poc_DateTime = b2_DateTime[prof_poc_idx][0]
        prof_poc_DecYear = b2_DecYear[prof_poc_idx]
        
        # CHL used in the model
        chl_data_input[where_bottle_idx] = prof_chla

        # Fit model to Chla & POC data
        print("Chl-a & POC Fit:")
        fit_result = two_community_model(prof_chla, prof_depth, prof_MLD, prof_poc, prof_poc_depth,
                                           data_type='bottle',Kd = Kd)

        # Extract Kd estimate
        #Kd = fit_result['Kd']
        #Zp = fit_result['Zp']
        #Kd_2,Zp_2 = calculate_Kd_Zp_Chl_int(prof_chla,prof_depth, use_interpolation=False)
        # Model Parameters
        P1_final    = fit_result['P1_final']
        TAU1_final  = fit_result['TAU1_final']
        BM2_final   = fit_result['BM2_final']
        TAU2_final  = fit_result['TAU2_final']
        SIG2_final  = fit_result['SIG2_final']
        mod_dcm_depth = fit_result['DCM1_depth']
        
        if i in ID_list_poc: #and np.min(prof_poc_depth) <= 1/Kd:
            
            # Store model results for Chla
            P1_model_fit[count] = fit_result['P1_final']
            P2_model_fit[count] = fit_result['TAU1_final']
            P3_model_fit[count] = fit_result['BM2_final']
            P4_model_fit[count] = fit_result['TAU2_final']
            P5_model_fit[count] = fit_result['SIG2_final']
            P4_array[count]     = fit_result['Chl_FitReport_P4']
            
            # Save Chl Fit Report results
            # Step 1 fit statistics
            Chl_FitStats_chisq_1_array[count]   = fit_result['Chl_FitReport_chisq_1']
            Chl_FitStats_redchi_1_array[count]  = fit_result['Chl_FitReport_redchi_1']
            Chl_FitStats_aic_1_array[count]     = fit_result['Chl_FitReport_aic_1']
            Chl_FitStats_bic_1_array[count]     = fit_result['Chl_FitReport_bic_1']
            # Step 1 parameter errors
            Chl_FitErr_P2_1_array[count]        = fit_result['Chl_FitReport_P2_err_1'] # Only p2 error extract as P1 has fixed relationship with P2: P1 = 10**(0.08 * P2 + 0.66) # Red Sea P1 Tau1 Relationship
            # Step 2 fit statistics
            Chl_FitStats_chisq_2_array[count]   = fit_result['Chl_FitReport_chisq_2']
            Chl_FitStats_redchi_2_array[count]  = fit_result['Chl_FitReport_redchi_2']
            Chl_FitStats_aic_2_array[count]     = fit_result['Chl_FitReport_aic_2']
            Chl_FitStats_bic_2_array[count]     = fit_result['Chl_FitReport_bic_2']
            # Step 2 parameter errors
            Chl_FitErr_P1_2_array[count]       = fit_result['Chl_FitReport_P1_err_2'] # Zero in a two step fit for two communities
            Chl_FitErr_P2_2_array[count]       = fit_result['Chl_FitReport_P2_err_2'] # Zero in a two step fit for two communities
            Chl_FitErr_P3_2_array[count]       = fit_result['Chl_FitReport_P3_err_2'] # Zero for 1 community fit
            Chl_FitErr_P4_2_array[count]       = fit_result['Chl_FitReport_P4_err_2'] # Zero for 1 community fit
            Chl_FitErr_P5_2_array[count]       = fit_result['Chl_FitReport_P5_err_2'] # Zero for 1 community fit
            Chl_FitErr_Tau2_array[count]       = fit_result['Chl_FitReport_Tau2_err']

            chl_mod_surface[where_bottle_idx]    = fit_result['Chl_C1_fit']
            chl_mod_subsurface[where_bottle_idx] = fit_result['Chl_C2_fit']
            chl_mod_total[where_bottle_idx]      = fit_result['Chl_Total_fit']
            chl_mod_data_diff[where_bottle_idx]  = fit_result['Chl_Total_fit'] - prof_chla
            
            # Integrate Model Chla
            chla_int_dict             = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['Chl_Total_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            chla_mod_total_int[count] = chla_int_dict['above_zp']
            chla_mod_dcm_conc[count]  = chla_int_dict['conc_at_dcm']
            chla_mod_surf_dict        = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['Chl_C1_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            chla_mod_surf_int[count]  = chla_mod_surf_dict['above_zp']
            chla_mod_surf_conc[count] = chla_mod_surf_dict['median_above_mld']
            chla_mod_sub_dict         = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['Chl_C2_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            chla_mod_sub_int[count]   = chla_mod_sub_dict['above_zp']
            chla_mod_sub_conc[count]  = chla_mod_sub_dict['conc_at_dcm']

            prof_chla_surf = fit_result['prof_chla_surf']
            chl_data_surf[count] = prof_chla_surf
            KD_used[count] = Kd
            Zp_used[count] = Zp

            # Store model results for POC
            
            # Save POC Fit Report Results
            POC_Fit_ratio_C1[count] = fit_result['C_Chl_ratio_C1']
            POC_Fit_ratio_C2[count] = fit_result['C_Chl_ratio_C2']
            POC_Fit_bk[count]       = fit_result['POC_background_np']  
            # POC Fit Stats & Errors
            POC_FitStats_chisq_array[count]  = fit_result['POC_FitReport_chisq']
            POC_FitStats_redchi_array[count] = fit_result['POC_FitReport_redchi']
            POC_FitStats_aic_array[count]    = fit_result['POC_FitReport_aic']
            POC_FitStats_bic_array[count]    = fit_result['POC_FitReport_bic']
            # Fit Parameter errors
            POC_FitErr_P1_array[count]       = fit_result['POC_FitReport_P1_err'] # C1 ratio error
            POC_FitErr_P2_array[count]       = fit_result['POC_FitReport_P2_err'] # C2 ratio error
            POC_FitErr_P3_array[count]       = fit_result['POC_FitReport_P3_err'] # Background error
            
            poc_data_input[prof_poc_idx]     = prof_poc
            poc_mod_surface[prof_poc_idx]    = fit_result['POC_C1_fit']
            poc_mod_subsurface[prof_poc_idx] = fit_result['POC_C2_fit']
            poc_mod_bk[prof_poc_idx]         = fit_result['POC_background']
            poc_mod_total[prof_poc_idx]      = fit_result['POC_Total_fit']
            poc_mod_data_diff[prof_poc_idx]  = fit_result['POC_Total_fit'] - prof_poc
            
            # Integrate Model POC
            poc_int_dict             = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['POC_Total_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            poc_mod_dcm_conc[count]  = poc_int_dict['conc_at_dcm']
            poc_mod_total_int[count] = poc_int_dict['above_zp']
            poc_mod_surf_dict        = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['POC_C1_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            poc_mod_surf_int[count]  = poc_mod_surf_dict['above_zp']
            poc_mod_surf_conc[count] = poc_mod_surf_dict['median_above_mld']
            poc_mod_sub_dict         = integrate_sections(fit_result['prof_depth_HiRes'],fit_result['POC_C2_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            poc_mod_sub_int[count]   = poc_mod_sub_dict['above_zp']
            poc_mod_sub_conc[count]  = poc_mod_sub_dict['conc_at_dcm']
            
            phytoC                         = fit_result['POC_C1_fit_HiRes'] + fit_result['POC_C2_fit_HiRes']
            phytoC_ratio                   = phytoC/fit_result['POC_Total_fit_HiRes']
            phytoC_ratio_dict              = integrate_sections(fit_result['prof_depth_HiRes'],phytoC_ratio,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            phytoC_ratio_median_mld[count] = phytoC_ratio_dict['median_above_mld']
            phytoC_ratio_dcm[count]        = phytoC_ratio_dict['conc_at_dcm']
            poc_int_dict               = integrate_sections(fit_result['prof_depth_HiRes'],phytoC,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            phytoC_dcm_conc[count]     = poc_int_dict['conc_at_dcm']
            phytoC_int[count]          = poc_int_dict['above_zp']

            # Store additional parameters
            # POC scaling factors and background
            POC_bk_np[poc_count] = fit_result['POC_background_np']
            # C:Chl Ratios
            C_Chl_ratio_C1[poc_count] = fit_result['C_Chl_ratio_C1']
            C_Chl_ratio_C2[poc_count] = fit_result['C_Chl_ratio_C2']
            MLD_used_poc[poc_count]   = prof_MLD
            df_poc_ID[poc_count] = i
            df_poc_date[poc_count] = prof_poc_date
            df_poc_DateTime[poc_count] = prof_poc_DateTime
            df_poc_DecYear[poc_count] = prof_poc_DecYear[0]
            
            ### PON Fit ###
            # PON Sampled and analysed on same samples as POC
            prof_pon = b2_pon[prof_poc_idx]
            print("PON Fit:")
            pon_fit = particulate_model(prof_pon, prof_poc_depth, prof_chla_surf, Kd, P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final)
            
            # Save PON Fit Report Results
            PON_Fit_ratio_C1[count] = pon_fit['P_Chl_ratio_C1']
            PON_Fit_ratio_C2[count] = pon_fit['P_Chl_ratio_C2']
            PON_Fit_bk[count]       = pon_fit['background_np']
            # PON Fit Stats & Errors
            PON_FitStats_chisq_array[count]  = pon_fit['FitReport_chisq']
            PON_FitStats_redchi_array[count] = pon_fit['FitReport_redchi']
            PON_FitStats_aic_array[count]    = pon_fit['FitReport_aic']
            PON_FitStats_bic_array[count]    = pon_fit['FitReport_bic']
            # Fit Parameter errors
            PON_FitErr_P1_array[count]       = pon_fit['FitReport_P1_err'] # C1 ratio error
            PON_FitErr_P2_array[count]       = pon_fit['FitReport_P2_err'] # C2 ratio error
            PON_FitErr_P3_array[count]       = pon_fit['FitReport_P3_err'] # Background error
                       
            pon_data_input[prof_poc_idx]     = prof_pon
            pon_mod_surface[prof_poc_idx]    = pon_fit['C1_fit']
            pon_mod_subsurface[prof_poc_idx] = pon_fit['C2_fit']
            pon_mod_bk[prof_poc_idx]         = pon_fit['background']
            pon_mod_total[prof_poc_idx]      = pon_fit['Total_fit']
            pon_mod_data_diff[prof_poc_idx]  = pon_fit['Total_fit'] - prof_pon
            
            # Integrate Model PON
            pon_int_dict             = integrate_sections(pon_fit['prof_depth_HiRes'],pon_fit['Total_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            pon_mod_dcm_conc[count]  = pon_int_dict['conc_at_dcm']
            pon_mod_total_int[count] = pon_int_dict['above_zp']
            
            pon_mod_surf_dict        = integrate_sections(pon_fit['prof_depth_HiRes'],pon_fit['C1_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            pon_mod_surf_int[count]  = pon_mod_surf_dict['above_zp']
            pon_mod_surf_conc[count] = pon_mod_surf_dict['median_above_mld']
            
            pon_mod_sub_dict         = integrate_sections(pon_fit['prof_depth_HiRes'],pon_fit['C2_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            pon_mod_sub_int[count]   = pon_mod_sub_dict['above_zp']
            pon_mod_sub_conc[count]  = pon_mod_sub_dict['conc_at_dcm']
            
            phytoN                         = pon_fit['C1_fit_HiRes'] + pon_fit['C2_fit_HiRes']
            phytoN_ratio                   = phytoN/pon_fit['Total_fit_HiRes']
            phytoN_ratio_dict              = integrate_sections(pon_fit['prof_depth_HiRes'],phytoN_ratio,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            phytoN_ratio_median_mld[count] = phytoN_ratio_dict['median_above_mld']
            phytoN_ratio_dcm[count]        = phytoN_ratio_dict['conc_at_dcm']
            pon_int_dict               = integrate_sections(pon_fit['prof_depth_HiRes'],phytoN,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
            phytoN_dcm_conc[count]     = pon_int_dict['conc_at_dcm']
            phytoN_int[count]          = pon_int_dict['above_zp']
            
            N_Chl_ratio_C1[poc_count] = pon_fit['P_Chl_ratio_C1']
            N_Chl_ratio_C2[poc_count] = pon_fit['P_Chl_ratio_C2']
            C_N_ratio_C1[poc_count]   = fit_result['C_Chl_ratio_C1']/pon_fit['P_Chl_ratio_C1']#(C_Chl_ratio_C1)/(N_Chl_ratio_C1)
            C_N_ratio_C2[poc_count]   = fit_result['C_Chl_ratio_C2']/pon_fit['P_Chl_ratio_C2']#(C_Chl_ratio_C2)/(N_Chl_ratio_C2)
            bk_ratio_CN[poc_count]    = fit_result['POC_background'][0]/pon_fit['background'][0]#BACKGROUND_POC[0]/background_N[0]

            # POP data
            prof_pop_idx    = np.where(bottle_pop.CRN == i)
            prof_pop        = b3_pop[prof_pop_idx]
            prof_pop_depth  = b3_pop_depth[prof_pop_idx]
            if i in ID_list_pop: #and np.min(prof_pop_depth) <= 1/Kd:
                ### POP Fit ###
                prof_pop_date    = b3_DateTime.date[prof_pop_idx]
                prof_pop_date    = prof_pop_date[0]
                prof_pop_DecYear = b3_Decimal_year[prof_pop_idx]
                prof_pop_DecYear = prof_pop_DecYear[0]
                print("POP Fit:")
                # Fit POP
                pop_fit = particulate_model(prof_pop, prof_pop_depth, prof_chla_surf, Kd, P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final)
                
                # Save POP Fit Report Results
                POP_Fit_ratio_C1[count] = pop_fit['P_Chl_ratio_C1']
                POP_Fit_ratio_C2[count] = pop_fit['P_Chl_ratio_C2']
                POP_Fit_bk[count]       = pop_fit['background_np']
                # POP Fit Stats & Errors
                POP_FitStats_chisq_array[count]  = pop_fit['FitReport_chisq']
                POP_FitStats_redchi_array[count] = pop_fit['FitReport_redchi']
                POP_FitStats_aic_array[count]    = pop_fit['FitReport_aic']
                POP_FitStats_bic_array[count]    = pop_fit['FitReport_bic']
                # Fit Parameter errors
                POP_FitErr_P1_array[count]       = pop_fit['FitReport_P1_err'] # C1 ratio error
                POP_FitErr_P2_array[count]       = pop_fit['FitReport_P2_err'] # C2 ratio error
                POP_FitErr_P3_array[count]       = pop_fit['FitReport_P3_err'] # Background error
                
                pop_data_input[prof_pop_idx]     = prof_pop
                pop_mod_surface[prof_pop_idx]    = pop_fit['C1_fit']
                pop_mod_subsurface[prof_pop_idx] = pop_fit['C2_fit']
                pop_mod_bk[prof_pop_idx]         = pop_fit['background']
                pop_mod_total[prof_pop_idx]      = pop_fit['Total_fit']
                pop_mod_data_diff[prof_pop_idx]  = pop_fit['Total_fit'] - prof_pop
                
                # Integrate Model PON
                pop_int_dict             = integrate_sections(pop_fit['prof_depth_HiRes'],pop_fit['Total_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
                pop_mod_dcm_conc[count]  = pop_int_dict['conc_at_dcm']
                pop_mod_total_int[count] = pop_int_dict['above_zp']
                pop_mod_surf_dict  = integrate_sections(pop_fit['prof_depth_HiRes'],pop_fit['C1_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
                pop_mod_surf_int[count]  = pop_mod_surf_dict['above_zp']
                pop_mod_surf_conc[count] = pop_mod_surf_dict['median_above_mld']
                pop_mod_sub_dict         = integrate_sections(pop_fit['prof_depth_HiRes'],pop_fit['C2_fit_HiRes'],prof_MLD,Zp,dcm_depth=mod_dcm_depth)
                pop_mod_sub_int[count]   = pop_mod_sub_dict['above_zp']
                pop_mod_sub_conc[count]  = pop_mod_sub_dict['conc_at_dcm']
                
                phytoP                         = pop_fit['C1_fit_HiRes'] + pop_fit['C2_fit_HiRes']
                phytoP_ratio                   = phytoP/pop_fit['Total_fit_HiRes']
                phytoP_ratio_dict              = integrate_sections(pop_fit['prof_depth_HiRes'],phytoP_ratio,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
                phytoP_ratio_median_mld[count] = phytoP_ratio_dict['median_above_mld']
                phytoP_ratio_dcm[count]        = phytoP_ratio_dict['conc_at_dcm']
                pop_int_dict               = integrate_sections(pop_fit['prof_depth_HiRes'],phytoP,prof_MLD,Zp,dcm_depth=mod_dcm_depth)
                phytoP_dcm_conc[count]     = pop_int_dict['conc_at_dcm']
                phytoP_int[count]          = pop_int_dict['above_zp']
                
                P_Chl_ratio_C1[pop_count] = pop_fit['P_Chl_ratio_C1']
                P_Chl_ratio_C2[pop_count] = pop_fit['P_Chl_ratio_C2']
    
                C_P_ratio_C1[pop_count] = fit_result['C_Chl_ratio_C1']/pop_fit['P_Chl_ratio_C1']
                C_P_ratio_C2[pop_count] = fit_result['C_Chl_ratio_C2']/pop_fit['P_Chl_ratio_C2']
                
                N_P_ratio_C1[pop_count] = pon_fit['P_Chl_ratio_C1']/pop_fit['P_Chl_ratio_C1']
                N_P_ratio_C2[pop_count] = pon_fit['P_Chl_ratio_C2']/pop_fit['P_Chl_ratio_C2']
    
                bk_ratio_CP[pop_count] = fit_result['POC_background'][0]/pop_fit['background'][0]
                bk_ratio_NP[pop_count] = pon_fit['background'][0]/pop_fit['background'][0]
                
                df_pop_ID[pop_count]      = i
                df_pop_date[pop_count]    = prof_pop_date
                df_pop_DecYear[pop_count] = prof_pop_DecYear
                MLD_used_pop[pop_count]   = prof_MLD
                
                pop_count += 1

            # Optional: Plot results
            if model_plot:
                
                chla_total = fit_result['CHL_model_fit']
                chla_surf = fit_result['MLD_pop_FIT']
                chla_sub = fit_result['DCM_pop_FIT']
                #prof_date = <prof_date>
                prof_depth2 = fit_result['prof_depth2']
                chla_total2 = fit_result['CHL_model_fit2']
                chla_surf2 = fit_result['MLD_pop_FIT2']
                chla_sub2 = fit_result['DCM_pop_FIT2']
                
                poc_total = fit_result['TOTAL_POC_MODEL']
                poc_surf = fit_result['MLD_pop_FIT_POC']
                poc_sub = fit_result['DCM_pop_FIT_POC']
                poc_bk = fit_result['BACKGROUND_POC']

                poc_total2 = fit_result['POC_model_fit2']
                poc_surf2 = fit_result['MLD_pop_FIT2_POC']
                poc_sub2 = fit_result['DCM_pop_FIT2_POC']
                
                pon_total = pon_fit['Total_fit']
                pon_surf = pon_fit['C1_fit']
                pon_sub = pon_fit['C2_fit']
                pon_bk = pon_fit['background']
                pon_total2 = pon_fit['Total_fit_HiRes']
                pon_surf2 = pon_fit['C1_fit_HiRes']
                pon_sub2 = pon_fit['C2_fit_HiRes']
                
                pop_total = pop_fit['Total_fit']
                pop_surf = pop_fit['C1_fit']
                pop_sub = pop_fit['C2_fit']
                pop_bk = pop_fit['background']
                pop_total2 = pop_fit['Total_fit_HiRes']
                pop_surf2 = pop_fit['C1_fit_HiRes']
                pop_sub2 = pop_fit['C2_fit_HiRes']
                
                
                fig, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2, 2, figsize=(12, 14), constrained_layout=False)
                fig.subplots_adjust(wspace=0.1, hspace=0.3)  # Adjust spacing between plots
                fig.patch.set_facecolor('White')

                # Plot Chl-a profile
                ax1.set_title('(a) Chl-a', fontsize=19, color='g')
                ax1.axhline(y=prof_MLD, color='k', linestyle='--', label='MLD')
                ax1.plot(prof_chla, prof_depth, color='g', marker='X', markersize=8, label='Data')
                ax1.plot(chla_total, prof_depth, color='k', marker='o', markersize=6, linestyle='None', label='Total')
                ax1.plot(chla_surf, prof_depth, color='r', marker='o', markersize=5, linestyle='None', label='Surface')
                ax1.plot(chla_sub, prof_depth, color='b', marker='o', markersize=4, linestyle='None', label='Subsurface')
                ax1.plot(chla_total2, prof_depth2, color='k', linestyle='-', label=None)
                ax1.plot(chla_surf2, prof_depth2, color='r', linestyle='-', label=None)
                ax1.plot(chla_sub2, prof_depth2, color='b', linestyle='-', label=None)
                ax1.set_ylabel('Depth (m)', fontsize=16)
                ax1.set_ylim([270, 0])
                ax1.yaxis.set_tick_params(labelsize=14)
                ax1.xaxis.set_tick_params(labelsize=14)
                ax1.set_xlabel('Chl-a (mg m$^{-3}$)', fontsize=16)
                ax1.legend(loc="lower right", fontsize=10, title=f'ID: {i}')
                ax1.text(np.min(chla_sub*1.4) + 0.01, 265, f"Date: {prof_date}", color='k', fontsize=12)

                # Plot POC profile
                ax2.set_title('(b) POC', fontsize=19, color='orange')
                ax2.axhline(y=prof_MLD, color='k', linestyle='--', label='MLD')
                ax2.plot(prof_poc,prof_poc_depth, color = 'orange',  marker = 'X', markersize = 8, linestyle = '-',label= 'Data')
                ax2.axvline(x= poc_bk[0], color = 'm', linestyle = '-.', linewidth=2, label= 'Background')
                ax2.plot(poc_surf, prof_poc_depth, color='r', marker='o', linestyle='None', label='Surface')
                ax2.plot(poc_sub, prof_poc_depth, color='b', marker='o', linestyle='None', label='Subsurface')
                ax2.plot(poc_total,prof_poc_depth, \
                         color = 'k', marker = 'o', linestyle='None', label= 'Total')
                ax2.plot(poc_total2, prof_depth2, color='k', linestyle='-', label=None)
                ax2.plot(poc_surf2, prof_depth2, color='r', linestyle='-', label=None)
                ax2.plot(poc_sub2, prof_depth2, color='b', linestyle='-', label=None)
                #ax3.set_ylabel('Depth (m)', fontsize=16)
                #ax3.set(yticklabels=[])
                ax2.yaxis.set_tick_params(labelsize=14)
                ax2.xaxis.set_tick_params(labelsize=14)
                ax2.set_ylim([270, 0])
                ax2.set_xlabel('POC (mg m$^{-3}$)', fontsize=16)
                ax2.legend(loc="lower right", fontsize=10, title=f'ID: {i}')
                ax2.text(np.min(prof_poc*1.4), 265, f"Date: {prof_poc_date}", color='k', fontsize=12)                
                # Plot PON profile
                ax3.set_title('(c) PON', fontsize=19, color='purple')
                ax3.axhline(y=prof_MLD, color='k', linestyle='--', label='MLD')
                ax3.plot(prof_pon,prof_poc_depth, color = 'purple',  marker = 'X', markersize = 8, linestyle = '-',label= 'Data')
                ax3.axvline(x= pon_bk[0], color = 'm', linestyle = '-.', linewidth=2, label= 'Background')
                ax3.plot(pon_surf, prof_poc_depth, color='r', marker='o', linestyle='None', label='Surface')
                ax3.plot(pon_sub, prof_poc_depth, color='b', marker='o', linestyle='None', label='Subsurface')
                ax3.plot(pon_total,prof_poc_depth, \
                         color = 'k', marker = 'o', linestyle='None', label= 'Total')
                ax3.plot(pon_total2, prof_depth2, color='k', linestyle='-', label=None)
                ax3.plot(pon_surf2, prof_depth2, color='r', linestyle='-', label=None)
                ax3.plot(pon_sub2, prof_depth2, color='b', linestyle='-', label=None)
                #ax3.set_ylabel('Depth (m)', fontsize=16)
                #ax3.set(yticklabels=[])
                ax3.yaxis.set_tick_params(labelsize=14)
                ax3.xaxis.set_tick_params(labelsize=14)
                ax3.set_ylim([270, 0])
                ax3.set_xlabel('PON (mg m$^{-3}$)', fontsize=16)
                ax3.legend(loc="lower right", fontsize=10, title=f'ID: {i}')
                ax3.text(np.min(prof_pon*1.4), 265, f"Date: {prof_poc_date}", color='k', fontsize=12)
                if i in ID_list_pop and np.min(prof_pop_depth) <= 1/Kd:
                    # Plot POP profile
                    ax4.set_title('(d) POP', fontsize=19, color='c')
                    ax4.axhline(y=prof_MLD, color='k', linestyle='--', label='MLD')
                    ax4.plot(prof_pop,prof_pop_depth, color = 'c',  marker = 'X', markersize = 8, linestyle = '-',label= 'Data')
                    ax4.axvline(x= pop_bk[0], color = 'm', linestyle = '-.', linewidth=2, label= 'Background')
                    ax4.plot(pop_surf, prof_pop_depth, color='r', marker='o', linestyle='None', label='Surface')
                    ax4.plot(pop_sub, prof_pop_depth, color='b', marker='o', linestyle='None', label='Subsurface')
                    ax4.plot(pop_total,prof_pop_depth, \
                             color = 'k', marker = 'o', linestyle='None', label= 'Total')
                    ax4.plot(pop_total2, prof_depth2, color='k', linestyle='-', label=None)
                    ax4.plot(pop_surf2, prof_depth2, color='r', linestyle='-', label=None)
                    ax4.plot(pop_sub2, prof_depth2, color='b', linestyle='-', label=None)
                    #ax3.set_ylabel('Depth (m)', fontsize=16)
                    #ax3.set(yticklabels=[])
                    ax4.yaxis.set_tick_params(labelsize=14)
                    ax4.xaxis.set_tick_params(labelsize=14)
                    ax4.set_ylim([270, 0])
                    ax4.set_xlabel('POP (mg m$^{-3}$)', fontsize=16)
                    ax4.legend(loc="lower right", fontsize=10, title=f'ID: {i}')
                    ax4.text(np.min(prof_pop*1.4), 265, f"Date: {prof_pop_date}", color='k', fontsize=12)
                
                fig.savefig('plots/LoopFit/Two_Community_Particulates_9May2025/HOT_ModelFit_Loop_ID_'+str(i)+'.png', format='png', dpi=300, bbox_inches="tight")
                
                plt.show()

            poc_count += 1
        count += 1
        nonzero = np.count_nonzero(~np.isnan(P1_model_fit))
        POP_nonzero = np.count_nonzero(~np.isnan(P_Chl_ratio_C1))
        print("CHLA Fitted: "+ str(np.count_nonzero(~np.isnan(P1_model_fit))))
        print(f"Profile ID: {i}, Count: {count}, Fitted: {nonzero}, POP Fitted: {POP_nonzero}")
        not_fitted = count-nonzero
        #count += 1

#%%

### PLOT TEST SCATTER OF RAW POC VS MODEL POC ###

print(len(chl_mod_total))
print(len(b_chla))

print(len(poc_mod_total))
print(len(b2_poc))

print(len(pon_mod_total))
print(len(b2_pon))

print(len(pop_mod_total))
print(len(pop_data_input))

# Remove NaNs from data
#Chla
ads          = np.where(~np.isnan(chl_mod_total))
CHL_model_1  = chl_mod_total[ads]
CHL_raw_1    = chl_data_input[ads]
b_depth_1    = b_depth[ads]
#POC
ads          = np.where(~np.isnan(poc_mod_total))
POC_model_1  = poc_mod_total[ads]
POC_raw_1    = poc_data_input[ads]
b2_depth_1   = b2_depth[ads]

#PON
ads          = np.where(~np.isnan(pon_mod_total))
PON_model_1  = pon_mod_total[ads]
PON_raw_1    = pon_data_input[ads]
b2_depth_PON_1   = b2_depth[ads]

#POP
ads          = np.where(~np.isnan(pop_mod_total))
POP_model_1  = pop_mod_total[ads]
POP_raw_1    = pop_data_input[ads]
b3_depth_1   = b3_pop_depth[ads]

print(len(b3_pop_depth))
print(len(b3_depth_1))

np.nanmax(POC_raw_1)
np.nanmax(CHL_raw_1)

# Correlation of Raw Chla vs Model Chla
STATS_REG  = spearmanr(CHL_raw_1, CHL_model_1)
#R value
R_chla     = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_chla     = ("{0:.3f}".format(STATS_REG[1]))
print([R_chla,P_chla])

# Correlation of Raw POC vs Model POC
STATS_REG  = spearmanr(POC_raw_1, POC_model_1)
#R value
R_poc      = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_poc      = ("{0:.3f}".format(STATS_REG[1]))
print([R_poc,P_poc])

# Correlation PON
STATS_REG  = spearmanr(PON_raw_1, PON_model_1)
#R value
R_pon      = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_pon      = ("{0:.3f}".format(STATS_REG[1]))
print([R_pon,P_pon])

# Correlation POP
STATS_REG  = spearmanr(POP_raw_1, POP_model_1)
#R value
R_pop      = ("{0:.2f}".format(STATS_REG[0]))
#P value
P_pop      = ("{0:.3f}".format(STATS_REG[1]))
print([R_pop,P_pop])

#Set plot up
fig, ([ax1,ax2],[ax3,ax4]) = plt.subplots(2,2, figsize=(15,12))
fig.subplots_adjust(wspace=0.26,hspace=0.3)
fig.patch.set_facecolor('White')

#Chla Subplot
im1 = ax1.scatter(CHL_raw_1,CHL_model_1, c = b_depth_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_chla)+'; $p$ = '+str(P_chla))
ax1.set_title('(a) Chl-a', fontsize = 18, color='k')
ax1.set_ylabel('Model Chl-a (mg m$^{-3}$)', fontsize=16)
ax1.yaxis.set_tick_params(labelsize=15)
ax1.set_xlabel('Data Chl-a  (mg m$^{-3}$)', fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.legend(loc="upper left", fontsize=10,title= 'Spearman Correlation')
ax1.set_xlim([-0.05,0.6]) 
ax1.set_ylim([-0.05,0.6]) 
ax1.locator_params(nbins=7)
cbar1 = fig.colorbar(im1,ax=ax1)
cbar1.ax.locator_params(nbins=6)
cbar1.set_label("Depth (m)", size  = 16)
cbar1.ax.tick_params(labelsize = 15)
#cbar1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.3f'))

#POC Subplot
im2 = ax2.scatter(POC_raw_1,POC_model_1, c = b2_depth_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_poc)+'; $p$ = '+str(P_poc))
ax2.set_title('(b) POC', fontsize = 18, color='k')
ax2.set_ylabel('Model POC (mg m$^{-3}$)', fontsize=16)
ax2.yaxis.set_tick_params(labelsize=15)
ax2.set_xlabel('Data POC (mg m$^{-3}$)', fontsize=16)
ax2.xaxis.set_tick_params(labelsize=15)
ax2.legend(loc="upper left", fontsize=10,title= 'Spearman Correlation')
ax2.set_xlim([-2,60]) 
ax2.set_ylim([-2,60]) 
ax2.locator_params(nbins=8)
cbar2 = fig.colorbar(im2,ax=ax2)
cbar2.ax.locator_params(nbins=6)
cbar2.set_label("Depth (m)", size  = 16)
cbar2.ax.tick_params(labelsize = 15)

#PON Subplot
im3 = ax3.scatter(PON_raw_1,PON_model_1, c = b2_depth_PON_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_pon)+'; $p$ = '+str(P_pon))
ax3.set_title('(c) PON', fontsize = 18, color='k')
ax3.set_ylabel('Model PON (mg m$^{-3}$)', fontsize=16)
ax3.yaxis.set_tick_params(labelsize=15)
ax3.set_xlabel('Data PON (mg m$^{-3}$)', fontsize=16)
ax3.xaxis.set_tick_params(labelsize=15)
ax3.legend(loc="upper left", fontsize=10,title= 'Spearman Correlation')
#ax3.set_xlim([-2,60]) 
#ax3.set_ylim([-2,60]) 
ax3.locator_params(nbins=8)
cbar3 = fig.colorbar(im3,ax=ax3)
cbar3.ax.locator_params(nbins=6)
cbar3.set_label("Depth (m)", size  = 16)
cbar3.ax.tick_params(labelsize = 15)

#POP Subplot
im4 = ax4.scatter(POP_raw_1,POP_model_1, c = b3_depth_1,alpha = 0.5,cmap = 'viridis_r', 
            label = 'R = '+str(R_pop)+'; $p$ = '+str(P_pop))
ax4.set_title('(d) POP', fontsize = 18, color='k')
ax4.set_ylabel('Model POP (mg m$^{-3}$)', fontsize=16)
ax4.yaxis.set_tick_params(labelsize=15)
ax4.set_xlabel('Data POP (mg m$^{-3}$)', fontsize=16)
ax4.xaxis.set_tick_params(labelsize=15)
ax4.legend(loc="upper left", fontsize=10,title= 'Spearman Correlation')
#ax3.set_xlim([-2,60]) 
#ax3.set_ylim([-2,60]) 
ax4.locator_params(nbins=8)
cbar4 = fig.colorbar(im4,ax=ax4)
cbar4.ax.locator_params(nbins=6)
cbar4.set_label("Depth (m)", size  = 16)
cbar4.ax.tick_params(labelsize = 15)

fig.savefig('plots/HOT_Scatter_Spearman_Raw_vs_ModelResults_CHL_POC_PON_POP2.png', format='png', dpi=300, bbox_inches="tight")
plt.show()

#%%

### SAVE MODEL FIT RESULTS TO DATAFRAMES & EXPORT TO CSV ###

### SAVE CHL MODEL RESULTS ###

# add Chla model fit results to bottle pigment df
bottle_6['CHL_mod_surface']    = chl_mod_surface
bottle_6['CHL_mod_subsurface'] = chl_mod_subsurface
bottle_6['CHL_mod_total']      = chl_mod_total
bottle_6['CHL_used']           = chl_data_input
bottle_6['CHL_mod_data_diff']  = chl_mod_data_diff

## Filter for profiles not fitted ##
bottle_6 = bottle_6.dropna(subset=['CHL_mod_surface'])

# Summary stats for surface & subsurface Chla
bottle_6[['CHL_mod_surface','CHL_mod_subsurface']].describe()

# Save dataframe with Chla Model output to CSV
bottle_6.to_csv('data/HOT_Bottle_Pigments_ModelResults.csv')

model_params_errors = bottle_prof.copy()

# Save model fitting stats and parameters and their errors
model_params_errors['Chl_Fit_P1']    = P1_model_fit #P1_TEMP
model_params_errors['Chl_Fit_P2']    = P2_model_fit #TAU1_TEMP
model_params_errors['Chl_Fit_P3']    = P3_model_fit #BM2_TEMP = DCM max
model_params_errors['Chl_Fit_P4']    = P4_array
model_params_errors['Chl_Fit_P5']    = P5_model_fit #SIG2_TEMP
model_params_errors['Chl_Fit_Tau2']  = P4_model_fit #TAU2_TEMP = Depth of DCM max = P4 + P5 x3

# Save Chl Fit Report results
# Step 1 fit statistics
model_params_errors['Chl_Fit_chisq_1']  = Chl_FitStats_chisq_1_array
model_params_errors['Chl_Fit_redchi_1'] = Chl_FitStats_redchi_1_array
model_params_errors['Chl_Fit_aic_1']    = Chl_FitStats_aic_1_array
model_params_errors['Chl_Fit_bic_1']    = Chl_FitStats_bic_1_array
# Step 1 parameter errors
model_params_errors['Chl_Fit_P2_1_error'] = Chl_FitErr_P2_1_array
# Step 2 fit statistics
model_params_errors['Chl_Fit_chisq_2'] = Chl_FitStats_chisq_2_array
model_params_errors['Chl_Fit_redchi_2'] = Chl_FitStats_redchi_2_array
model_params_errors['Chl_Fit_aic_2'] = Chl_FitStats_aic_2_array
model_params_errors['Chl_Fit_bic_2'] = Chl_FitStats_bic_2_array
# Step 2 parameter errors
model_params_errors['Chl_Fit_P1_2_error'] = Chl_FitErr_P1_2_array
model_params_errors['Chl_Fit_P2_2_error'] = Chl_FitErr_P2_2_array
model_params_errors['Chl_Fit_P2_1_error_combined'] = model_params_errors['Chl_Fit_P2_1_error'].fillna(model_params_errors['Chl_Fit_P2_2_error'])
model_params_errors['Chl_Fit_P3_2_error'] = Chl_FitErr_P3_2_array
model_params_errors['Chl_Fit_P4_2_error'] = Chl_FitErr_P4_2_array
model_params_errors['Chl_Fit_P5_2_error'] = Chl_FitErr_P5_2_array
model_params_errors['Chl_FitErr_Tau2']    = Chl_FitErr_Tau2_array

# POC Fit Stats & Errors
model_params_errors['POC_Fit_P1']  = POC_Fit_ratio_C1
model_params_errors['POC_Fit_P2']  = POC_Fit_ratio_C2
model_params_errors['POC_Fit_P3']  = POC_Fit_bk
model_params_errors['POC_Fit_chisq']  = POC_FitStats_chisq_array
model_params_errors['POC_Fit_redchi'] = POC_FitStats_redchi_array
model_params_errors['POC_Fit_aic']    = POC_FitStats_aic_array
model_params_errors['POC_Fit_bic']    = POC_FitStats_bic_array
# POC Fit Parameter errors
model_params_errors['POC_Fit_P1_error'] = POC_FitErr_P1_array
model_params_errors['POC_Fit_P2_error'] = POC_FitErr_P2_array
model_params_errors['POC_Fit_P3_error'] = POC_FitErr_P3_array

# PON Fit Stats & Errors
model_params_errors['PON_Fit_P1']  = PON_Fit_ratio_C1
model_params_errors['PON_Fit_P2']  = PON_Fit_ratio_C2
model_params_errors['PON_Fit_P3']  = PON_Fit_bk
model_params_errors['PON_Fit_chisq']  = PON_FitStats_chisq_array
model_params_errors['PON_Fit_redchi'] = PON_FitStats_redchi_array
model_params_errors['PON_Fit_aic']    = PON_FitStats_aic_array
model_params_errors['PON_Fit_bic']    = PON_FitStats_bic_array
# PON Fit Parameter errors
model_params_errors['PON_Fit_P1_error'] = PON_FitErr_P1_array
model_params_errors['PON_Fit_P2_error'] = PON_FitErr_P2_array
model_params_errors['PON_Fit_P3_error'] = PON_FitErr_P3_array

# POP Fit Stats & Errors
model_params_errors['POP_Fit_P1']  = POP_Fit_ratio_C1
model_params_errors['POP_Fit_P2']  = POP_Fit_ratio_C2
model_params_errors['POP_Fit_P3']  = POP_Fit_bk
model_params_errors['POP_Fit_chisq']  = POP_FitStats_chisq_array
model_params_errors['POP_Fit_redchi'] = POP_FitStats_redchi_array
model_params_errors['POP_Fit_aic']    = POP_FitStats_aic_array
model_params_errors['POP_Fit_bic']    = POP_FitStats_bic_array
# POP Fit Parameter errors
model_params_errors['POP_Fit_P1_error'] = POP_FitErr_P1_array
model_params_errors['POP_Fit_P2_error'] = POP_FitErr_P2_array
model_params_errors['POP_Fit_P3_error'] = POP_FitErr_P3_array

print(model_params_errors.columns.tolist())

model_params_errors = model_params_errors.drop(
    columns=['CRN', 'MLD_temp', 'MLD_dens', 'MLD_dens_boyer', 'CTD_time', 'CTD_ID'],
    errors='ignore'
)

# Save bottle single data to CSV
model_params_errors.to_csv('data/HOT_ModelFit_StatsErrors.csv')

# Add single bottle profile data to separate bottle prof df
print(len(bottle_prof))
print(len(P3_model_fit))
bottle_prof['Kd']           = KD_used
bottle_prof['Zp']           = Zp_used
bottle_prof['MLD_used']     = MLD_used
bottle_prof['Chla_data_surf_conc'] = chl_data_surf
bottle_prof['P1']           = P1_model_fit #P1_TEMP
bottle_prof['TAU1']         = P2_model_fit #TAU1_TEMP
bottle_prof['BM2']          = P3_model_fit #BM2_TEMP = DCM max
bottle_prof['TAU2']         = P4_model_fit #TAU2_TEMP = Depth of DCM max
bottle_prof['SIG2']         = P5_model_fit #SIG2_TEMP
bottle_prof['P4']           = P4_array
bottle_prof['DCM_peak']     = bottle_prof['BM2']*bottle_prof['Chla_data_surf_conc']
bottle_prof['DCM_depth']    = bottle_prof['TAU2']/bottle_prof['Kd']
bottle_prof['DCM_width']    = bottle_prof['SIG2']/bottle_prof['Kd']

# Add integrated CHL
bottle_prof['Chla_mod_Int']       = chla_mod_total_int
bottle_prof['Chla_mod_surf_Int']  = chla_mod_surf_int
bottle_prof['Chla_mod_sub_Int']   = chla_mod_sub_int
bottle_prof['Chla_mod_dcm_conc']  = chla_mod_dcm_conc
bottle_prof['chla_mod_surf_conc'] = chla_mod_surf_conc
bottle_prof['chla_mod_sub_conc']  = chla_mod_sub_conc

# Add integrated POC
bottle_prof['POC_mod_surf_int']  = poc_mod_surf_int/1000
bottle_prof['POC_mod_sub_int']   = poc_mod_sub_int/1000
bottle_prof['POC_mod_total_int'] = poc_mod_total_int/1000
bottle_prof['POC_mod_dcm_conc']  = poc_mod_dcm_conc
bottle_prof['POC_mod_surf_conc'] = poc_mod_surf_conc
bottle_prof['POC_mod_sub_conc']  = poc_mod_sub_conc

bottle_prof['PhytoC_int_discreet']     = phytoC_int/1000
bottle_prof['PhytoC_ratio_median_mld'] = phytoC_ratio_median_mld
bottle_prof['PhytoC_ratio_dcm']        = phytoC_ratio_dcm
bottle_prof['PhytoC_dcm_conc']         = phytoC_dcm_conc

# Add integrated PON
bottle_prof['PON_mod_surf_int']   = pon_mod_surf_int/1000
bottle_prof['PON_mod_sub_int']    = pon_mod_sub_int/1000
bottle_prof['PON_mod_total_int']  = pon_mod_total_int/1000
bottle_prof['PON_mod_dcm_conc']   = pon_mod_dcm_conc
bottle_prof['PON_mod_surf_conc']  = pon_mod_surf_conc
bottle_prof['PON_mod_sub_conc']   = pon_mod_sub_conc

bottle_prof['PhytoN_int_discreet']     = phytoN_int/1000
bottle_prof['PhytoN_ratio_median_mld'] = phytoN_ratio_median_mld
bottle_prof['PhytoN_ratio_dcm']        = phytoN_ratio_dcm
bottle_prof['PhytoN_dcm_conc']         = phytoN_dcm_conc

# Add integrated POP
bottle_prof['POP_mod_surf_int']   = pop_mod_surf_int
bottle_prof['POP_mod_sub_int']    = pop_mod_sub_int
bottle_prof['POP_mod_total_int']  = pop_mod_total_int
bottle_prof['POP_mod_dcm_conc']   = pop_mod_dcm_conc
bottle_prof['POP_mod_surf_conc']  = pop_mod_surf_conc
bottle_prof['POP_mod_sub_conc']   = pop_mod_sub_conc

bottle_prof['PhytoP_int_discreet']     = phytoP_int
bottle_prof['PhytoP_ratio_median_mld'] = phytoP_ratio_median_mld
bottle_prof['PhytoP_ratio_dcm']        = phytoP_ratio_dcm
bottle_prof['PhytoP_dcm_conc']         = phytoP_dcm_conc

## Filter for profiles not fitted ##
# Drop POC profiles not fitted - NaN
bottle_prof = bottle_prof.dropna(subset=['P1'])

# Inspect DCM depth
bottle_prof['DCM_depth'].describe()

#Create new ID list for Chla profiles fitted
ID_list_6 = bottle_prof.loc[:,'Cruise_ID'].to_numpy()

# Save bottle single data to CSV
bottle_prof.to_csv('data/HOT_Bottle_profData.csv')

#%%

### SAVE POC & PON MODEL RESULTS ###

# Add POC model results to bottle POC df
bottle_poc['POC_mod_surface']    = poc_mod_surface
bottle_poc['POC_mod_subsurface'] = poc_mod_subsurface
bottle_poc['POC_mod_bk']         = poc_mod_bk
bottle_poc['POC_mod_total']      = poc_mod_total
bottle_poc['POC_used']           = poc_data_input
bottle_poc['POC_mod_data_diff']  = poc_mod_data_diff

# Calculate Phyto Carbon and Living Carbon Ratio directly in DF
bottle_poc['Phyto_Carbon'] = bottle_poc['POC_mod_surface'] + bottle_poc['POC_mod_subsurface']
bottle_poc['PhytoC_Ratio'] = bottle_poc['Phyto_Carbon'] / bottle_poc['POC_mod_total']

# Add PON model results to bottle POC df
bottle_poc['PON_mod_surface']    = pon_mod_surface
bottle_poc['PON_mod_subsurface'] = pon_mod_subsurface
bottle_poc['PON_mod_bk']         = pon_mod_bk
bottle_poc['PON_mod_total']      = pon_mod_total
bottle_poc['PON_used']           = pon_data_input
bottle_poc['PON_mod_data_diff']  = pon_mod_data_diff

# Calculate Phyto Nitrogen and Living Nitrogen Ratio directly in DF
bottle_poc['Phyto_Nitrogen'] = bottle_poc['PON_mod_surface'] + bottle_poc['PON_mod_subsurface']
bottle_poc['PhytoN_Ratio'] = bottle_poc['Phyto_Nitrogen'] / bottle_poc['PON_mod_total']

## Filter for profiles not fitted ##
bottle_poc = bottle_poc.dropna(subset=['POC_mod_surface'])

#Create new ID list for POC&PON profiles fitted
poc_cruise_ID = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
# Removes Duplicates
ID_list_poc = pd.unique(poc_cruise_ID)
print(len(ID_list_poc))

# Save POC & PON Model results to CSV
bottle_poc.to_csv('data/HOT_Bottle_POC_ModelResults.csv')

bottle_poc_prof.info()

print(len(bottle_poc_prof))
print(len(C_Chl_ratio_C1))
np.count_nonzero(np.isnan(C_Chl_ratio_C1))

# Setup new DF for Model POC prof data (single values per profile)
bottle_poc_prof_x = pd.DataFrame()
bottle_poc_prof_x['Cruise_ID']  = df_poc_ID.astype('int')
bottle_poc_prof_x['DateTime']   = df_poc_DateTime
bottle_poc_prof_x['Date']       = df_poc_date
bottle_poc_prof_x['DecYear']    = df_poc_DecYear#df_Decimal_year
bottle_poc_prof_x['yyyy']       = pd.to_datetime(bottle_poc_prof_x['Date']).dt.year 
bottle_poc_prof_x['mm']         = pd.to_datetime(bottle_poc_prof_x['Date']).dt.month

# Add POC ratios and scaling factors to DF
bottle_poc_prof_x['C_Chl_ratio_C1'] = C_Chl_ratio_C1
bottle_poc_prof_x['C_Chl_ratio_C2'] = C_Chl_ratio_C2
bottle_poc_prof_x['MLD_used']       = MLD_used_poc
bottle_poc_prof_x['POC_bk_conc']    = POC_bk_np

# Add PON ratios and scaling factors to DF
bottle_poc_prof_x['N_Chl_ratio_C1']  = N_Chl_ratio_C1
bottle_poc_prof_x['N_Chl_ratio_C2']  = N_Chl_ratio_C2
bottle_poc_prof_x['C_N_ratio_C1']    = C_N_ratio_C1
bottle_poc_prof_x['C_N_ratio_C2']    = C_N_ratio_C2
bottle_poc_prof_x['bk_ratio_CN']     = bk_ratio_CN

# Calculate molar C:N ratio
C_N_ATOMIC_RATIO = 12.01 / 14.01  # Define the atomic mass ratio = Approximately 0.857
# Calculate molar C:N ratios by dividing weight ratios by atomic ratio
bottle_poc_prof_x['C_N_molar_C1']      = C_N_ratio_C1 / C_N_ATOMIC_RATIO
bottle_poc_prof_x['C_N_molar_C2']      = C_N_ratio_C2 / C_N_ATOMIC_RATIO
bottle_poc_prof_x['bk_ratio_CN_molar'] = bk_ratio_CN / C_N_ATOMIC_RATIO

bottle_poc_prof_x.info()

print(len(bottle_poc_prof_x))

bottle_poc_prof_x.info()

# Remove prof where model could not be fitted and replace old prof df
bottle_poc_prof = bottle_poc_prof_x.dropna(subset=['C_Chl_ratio_C1'])

# Sort new df by ID and depth
bottle_poc_prof = bottle_poc_prof.sort_values(by=['Cruise_ID'])

# Reset df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)

bottle_poc_prof.info()

# Save new dataframe with Model output to CSV
bottle_poc_prof.to_csv('data/HOT_Bottle_POC_ProfData.csv')

print(np.nanmax(C_Chl_ratio_C1))

plt.plot(bottle_poc_prof['DecYear'],bottle_poc_prof['C_Chl_ratio_C1'],linestyle = '-', label = 'C_Chl_ratio_C1')
plt.plot(bottle_poc_prof['DecYear'],bottle_poc_prof['C_Chl_ratio_C2'],linestyle = '-', label = 'C_Chl_ratio_C2')
plt.legend(loc="upper left", fontsize=10)
plt.show

#%%
### C:N Histograms

# Helper function to add vertical lines for median and IQR
def add_stat_lines(ax, data, color):
    median = data.median()
    iqr_low, iqr_high = data.quantile(0.25), data.quantile(0.75)
    ax.axvline(median, color=color, linestyle='-', linewidth=1.8, label=f'Median: {median:.1f}')
    ax.axvline(iqr_low, color=color, linestyle='--', linewidth=1.2, label=f'IQR Low: {iqr_low:.1f}')
    ax.axvline(iqr_high, color=color, linestyle='--', linewidth=1.2, label=f'IQR High: {iqr_high:.1f}')

# Redfield C:N ratio
redfield_ratio = 106 / 16  # approximately 6.625

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the figure size as needed

# Plot histograms for each column
bottle_poc_prof['C_N_molar_C1'].plot.hist(ax=axes[0], bins=30, color='lightcoral', edgecolor='black')
axes[0].set_title('Surface C:N Molar Ratio', color = 'r')
axes[0].set_xlabel('C:N Molar Ratio')
axes[0].set_ylabel('Frequency')
axes[0].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:N')
add_stat_lines(axes[0], bottle_poc_prof['C_N_molar_C1'], color='green')
axes[0].legend()

bottle_poc_prof['C_N_molar_C2'].plot.hist(ax=axes[1], bins=30, color='skyblue', edgecolor='black')
axes[1].set_title('Subsurface C:N Molar Ratio', color = 'b')
axes[1].set_xlabel('C:N Molar Ratio')
axes[1].set_ylabel('Frequency')
axes[1].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:N')
add_stat_lines(axes[1], bottle_poc_prof['C_N_molar_C2'], color='green')
axes[1].legend()

bottle_poc_prof['bk_ratio_CN_molar'].plot.hist(ax=axes[2], bins=30, color='thistle', edgecolor='black')
axes[2].set_title('Bk C:N Molar Ratio', color = 'purple')
axes[2].set_xlabel('Bk Ratio C:N Molar')
axes[2].set_ylabel('Frequency')
axes[2].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:N')
add_stat_lines(axes[2], bottle_poc_prof['bk_ratio_CN_molar'], color='green')
axes[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
# Save the figure
fig.savefig('plots/Histograms_CN_with_Stats.png', format='png', dpi=300, bbox_inches="tight")
# Show the plot
plt.show()

plt.plot(bottle_poc_prof['DecYear'],bottle_poc_prof['C_N_molar_C1'],linestyle = '-', label = 'C_N_molar_C1')
plt.plot(bottle_poc_prof['DecYear'],bottle_poc_prof['C_N_molar_C2'],linestyle = '-', label = 'C_N_molar_C2')
plt.legend(loc="upper left", fontsize=10)
plt.ylim(0,25)
plt.show

#%%

### SAVE POP MODEL RESULTS ###

# Add POC model results to bottle POC df
bottle_pop['POP_mod_surface']    = pop_mod_surface
bottle_pop['POP_mod_subsurface'] = pop_mod_subsurface
bottle_pop['POP_mod_bk']         = pop_mod_bk
bottle_pop['POP_mod_total']      = pop_mod_total
bottle_pop['POP_used']           = pop_data_input
bottle_pop['POP_mod_data_diff']  = pop_mod_data_diff

# Calculate Phyto Nitrogen and Living Nitrogen Ratio directly in DF
bottle_pop['Phyto_Phos'] = bottle_pop['POP_mod_surface'] + bottle_pop['POP_mod_subsurface']
bottle_pop['PhytoP_Ratio'] = bottle_pop['Phyto_Phos'] / bottle_pop['POP_mod_total']

bottle_pop.info()

# Filter profiles not fitted
bottle_pop = bottle_pop.dropna(subset=['POP_mod_surface'])

#Create new ID list for POC&PON profiles fitted
pop_cruise_ID = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
# Removes Duplicates
ID_list_pop = pd.unique(pop_cruise_ID)
print(len(ID_list_pop))

# Save POP Model results to CSV
bottle_pop.to_csv('data/HOT_Bottle_POP_ModelResults.csv')

print(len(bottle_pop_prof))
print(len(C_P_ratio_C1))

# Setup new DF for Model POC prof data (single values per profile)
bottle_pop_prof_x = pd.DataFrame()
bottle_pop_prof_x['Cruise_ID']  = df_pop_ID.astype('int')
bottle_pop_prof_x['Date']       = df_pop_date
bottle_pop_prof_x['DecYear']    = df_pop_DecYear#df_Decimal_year
bottle_pop_prof_x['yyyy']       = pd.to_datetime(bottle_pop_prof_x['Date']).dt.year 
bottle_pop_prof_x['mm']         = pd.to_datetime(bottle_pop_prof_x['Date']).dt.month

# Add POP ratios and scaling factors to DF
bottle_pop_prof_x['P_Chl_ratio_C1']  = P_Chl_ratio_C1
bottle_pop_prof_x['P_Chl_ratio_C2']  = P_Chl_ratio_C2
bottle_pop_prof_x['C_P_ratio_C1']    = C_P_ratio_C1
bottle_pop_prof_x['C_P_ratio_C2']    = C_P_ratio_C2
bottle_pop_prof_x['bk_ratio_CP']     = bk_ratio_CP
bottle_pop_prof_x['N_P_ratio_C1']    = N_P_ratio_C1
bottle_pop_prof_x['N_P_ratio_C2']    = N_P_ratio_C2
bottle_pop_prof_x['bk_ratio_NP']     = bk_ratio_NP

# Define atomic mass ratios for molar conversions
C_P_ATOMIC_RATIO = 12.01 / 30.97  # Approximately 0.388
N_P_ATOMIC_RATIO = 14.01 / 30.97  # Approximately 0.452
# Calculate molar C:P ratios by dividing weight ratios by atomic ratio
bottle_pop_prof_x['C_P_molar_C1']      = C_P_ratio_C1 / C_P_ATOMIC_RATIO
bottle_pop_prof_x['C_P_molar_C2']      = C_P_ratio_C2 / C_P_ATOMIC_RATIO
bottle_pop_prof_x['bk_ratio_CP_molar'] = bk_ratio_CP / C_P_ATOMIC_RATIO
# Calculate molar N:P ratios by dividing weight ratios by atomic ratio
bottle_pop_prof_x['N_P_molar_C1']      = N_P_ratio_C1 / N_P_ATOMIC_RATIO
bottle_pop_prof_x['N_P_molar_C2']      = N_P_ratio_C2 / N_P_ATOMIC_RATIO
bottle_pop_prof_x['bk_ratio_NP_molar'] = bk_ratio_NP / N_P_ATOMIC_RATIO

bottle_pop_prof_x.info()

print(len(bottle_pop_prof_x))

# Remove prof where model could not be fitted and replace old prof df
bottle_pop_prof = bottle_pop_prof_x.dropna(subset=['P_Chl_ratio_C1'])

# Sort new df by ID and depth
bottle_pop_prof = bottle_pop_prof.sort_values(by=['Cruise_ID'])

# Reset df index replacing old index column
bottle_pop_prof = bottle_pop_prof.reset_index(drop=True)

bottle_pop_prof.info()

# Save new dataframe with Model output to CSV
bottle_pop_prof.to_csv('data/HOT_Bottle_POP_ProfData.csv')

#print(np.nanmax(C_P_molar_C1))

plt.plot(bottle_pop_prof['DecYear'],bottle_pop_prof['C_P_molar_C1'],linestyle = '-', label = 'C_P_molar_C1')
plt.plot(bottle_pop_prof['DecYear'],bottle_pop_prof['C_P_molar_C2'],linestyle = '-', label = 'C_P_molar_C2')
plt.legend(loc="upper left", fontsize=10)
plt.ylim(0,1000)
plt.show

#%%
### C:P Histograms

# Redfield C:P ratio
redfield_ratio = 106 / 1  # Redfield ratio for C:P is 106:1

# Create a figure with 3 subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust the figure size as needed

# Plot histograms for each column
bottle_pop_prof['C_P_molar_C1'].plot.hist(ax=axes[0], bins=50, color='lightcoral', edgecolor='black')
axes[0].set_title('Surface C:P Molar Ratio', color='r')
axes[0].set_xlabel('C:P Molar Ratio')
axes[0].set_ylabel('Frequency')
axes[0].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:P')
add_stat_lines(axes[0], bottle_pop_prof['C_P_molar_C1'], color='green')
axes[0].legend()

bottle_pop_prof['C_P_molar_C2'].plot.hist(ax=axes[1], bins=50, color='skyblue', edgecolor='black')
axes[1].set_title('Subsurface C:P Molar Ratio', color='b')
axes[1].set_xlabel('C:P Molar Ratio')
axes[1].set_ylabel('Frequency')
axes[1].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:P')
add_stat_lines(axes[1], bottle_pop_prof['C_P_molar_C2'], color='green')
axes[1].legend()

bottle_pop_prof['bk_ratio_CP_molar'].plot.hist(ax=axes[2], bins=50, color='thistle', edgecolor='black')
axes[2].set_title('Bk C:P Molar Ratio', color='purple')
axes[2].set_xlabel('Bk Ratio C:P Molar')
axes[2].set_ylabel('Frequency')
axes[2].axvline(redfield_ratio, color='red', linestyle='--', linewidth=1.8, label='Redfield C:P')
add_stat_lines(axes[2], bottle_pop_prof['bk_ratio_CP_molar'], color='green')
axes[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
fig.savefig('plots/Histograms_CP_with_Stats.png', format='png', dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

# Time-series plot
plt.plot(bottle_pop_prof['DecYear'], bottle_pop_prof['C_P_molar_C1'], linestyle='-', label='C_P_molar_C1')
plt.plot(bottle_pop_prof['DecYear'], bottle_pop_prof['C_P_molar_C2'], linestyle='-', label='C_P_molar_C2')
plt.legend(loc="upper left", fontsize=10)
plt.ylim(0, 500)  # Adjusted based on typical C:P molar ratio ranges
plt.xlabel('Year')
plt.ylabel('C:P Molar Ratio')
plt.title('Time Series of C:P Molar Ratios')
plt.show()

#%%

### FILTER CHLA MODEL RESULTS FOR ONLY FITTED

# CSV filename
filename_1 = 'data/HOT_Bottle_Pigments_ModelResults.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_6   = pd.read_csv(filename_1, index_col = 0)

bottle_6.info()

# Sort new df by ID and depth
bottle_6 = bottle_6.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_6 = bottle_6.reset_index(drop=True)

bottle_6.info()

# Save new dataframe with Model output to CSV
bottle_6.to_csv('data/HOT_Bottle_Pigments_ModelResults.csv')

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b_time     = bottle_6.loc[:,'time'].to_numpy()
b_date     = bottle_6.loc[:,'Date'].to_numpy()
b_DateTime = bottle_6.loc[:,'DateTime'].to_numpy()
b_depth    = bottle_6.loc[:,'depth'].to_numpy()
b_chla     = bottle_6.loc[:,'Chla'].to_numpy()
b_ID       = bottle_6.loc[:,'Cruise_ID'].to_numpy()
b_year     = bottle_6.loc[:,'yyyy'].to_numpy()
b_month    = bottle_6.loc[:,'mm'].to_numpy()
b_Decimal_year = bottle_6.loc[:,'DecYear'].to_numpy()

# Bottle DateTime data
b_DateTime = pd.to_datetime(b_DateTime)
#m_chla = b_chla
### Cruise_ID list for new df is ID_list_6

# Converts to pandas timeseries array
ID_list_6 = pd.Series(b_ID)
# Removes Duplicates
ID_list_6 = pd.unique(ID_list_6)
print(len(ID_list_6))
# 275 profiles

# Extract Filtered Chl Model Results
chl_mod_surface      = bottle_6.loc[:,'CHL_mod_surface'].to_numpy()
chl_mod_subsurface   = bottle_6.loc[:,'CHL_mod_subsurface'].to_numpy()
chl_mod_total        = bottle_6.loc[:,'CHL_mod_total'].to_numpy()
chl_data_input       = bottle_6.loc[:,'CHL_used'].to_numpy()
chl_mod_data_diff    = bottle_6.loc[:,'CHL_mod_data_diff'].to_numpy()

# Import Chl prof Data
# CSV filename
filename_1 = 'data/HOT_Bottle_profData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_prof = pd.read_csv(filename_1, index_col = 0)

bottle_prof.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
bottle_prof = bottle_prof.dropna(subset=['P1'])

# Sort new df by ID and depth
bottle_prof = bottle_prof.sort_values(by=['Cruise_ID'])
# Reset bottle df index replacing old index column
bottle_prof = bottle_prof.reset_index(drop=True)

print(len(bottle_prof))

# Save new dataframe with Model output to CSV
bottle_prof.to_csv('data/HOT_Bottle_profData.csv')

#%%
######
### CONTOUR PLOTS OF MODEL RESULTS ###
#####

### Interpolate data ###

ID_list_bottle = ID_list_6

#m_chla = b_chla

# Set xy (area) of contour plot
y = 220
x2 = len(ID_list_bottle)

print(len(bottle_6))

#Depth
New_depth = np.array(range(0, y))

CHLA_m_surf   = np.empty([x2,y])#+nan
CHLA_m_subb   = np.empty([x2,y])#+nan
CHLA_m_totl   = np.empty([x2,y])#+nan
CHLA_inter    = np.empty([x2,y])#+nan

Depth_bot     = np.empty([x2,y])#+nan
Time_bot      = (np.empty([x2,y], dtype='datetime64[s]'))

# Chlorophyll-a data interpolation
count = 0    
for i in ID_list_bottle:
    a = b_chla[bottle_6.Cruise_ID == i]
    b = b_depth[bottle_6.Cruise_ID == i]
    valid1 = ~np.isnan(b)
    valid2 = ~np.isnan(a)
    a      = a[valid2]
    c      = b[valid2]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(c,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_inter[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    if len(b_depth[bottle_6.Cruise_ID == i]) > 1:
        a = chl_mod_surface[bottle_6.Cruise_ID == i]
        b = b_depth[bottle_6.Cruise_ID == i]
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_surf[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    if len(b_depth[bottle_6.Cruise_ID == i]) > 1:
        a = chl_mod_subsurface[bottle_6.Cruise_ID == i]
        b = b_depth[bottle_6.Cruise_ID == i]
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_subb[count,:]  = xxx
    count=count+1

count = 0    
for i in ID_list_bottle:
    a = chl_mod_total[bottle_6.Cruise_ID == i]
    b = b_depth[bottle_6.Cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        CHLA_m_totl[count,:]  = xxx
        AD               = b_DateTime[bottle_6.Cruise_ID == i]
        Time_bot[count,:]  = AD[0]    
        Depth_bot[count,:] = New_depth
    count=count+1

#%%
### Plot
#Figure parameters that can be changed 
CHLA_COL = cmocean.cm.algae
POC_COL  = cmocean.cm.tempo
BBP_COL           = mpl.cm.cividis#bbp colour scale (see https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html)
DIFF_COL          = cmocean.cm.balance
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 16           #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17           #Define the font size of the Colourbar labels
pad_width         = 0.02
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1  
#Define the figure window including 5 subplots orientated vertically
#plt.style.use('default')
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)
    
#2022-12-16
#x axistest#
#Bottle Dates: 1989-07-31 to 2022-09-03
xaxi = [date(1989,8,25),date(2022,9,3)]

#SUBPLOT 1: Raw CHL TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_inter.copy()
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)

valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

# Build modified colormap
custom_cmap = CHLA_COL.copy()
custom_cmap.set_under('white')

im1            = ax1.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = custom_cmap, extend = 'min')
##Set axis info and titles
ax1.set_ylim([200,0]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) Chl-a Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width)
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 2: Model of total chlA
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_totl.copy()

PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)

valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2

##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im2            = ax2.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = custom_cmap, extend = 'min')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_xlim([xaxi[0],xaxi[-1]]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total Chl-a', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 3: Model data difference
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_totl - CHLA_inter

PCT_1          = np.nanpercentile(chl_mod_data_diff, Percentiles_lower)
#PCT_2          = np.nanpercentile(chl_mod_data_diff, Percentiles_upper)
PCT_1 = -0.25
PCT_2 = 0.25
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 100)
im3            = ax3.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = mpl.cm.bwr, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0]) 
#ax3.set_xlim([xaxi[0],xaxi[-1]]) 
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width)
#cbar3.set_ticks(np.linspace(PCT_1, PCT_2, 5))   # evenly spaced ticks incl. 0
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 4: CHL A surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_surf.copy()
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)

valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im4            = ax4.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = custom_cmap,alpha =1, extend = 'min')
##Set axis info and titles
ax4.set_ylim([200,0]) 
#ax4.set_xlim([xaxi[0],xaxi[-1]]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Chl-a', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

#SUBPLOT 5: CHL A subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = CHLA_m_subb.copy()
PCT_1          = np.nanpercentile(b_chla, Percentiles_lower)
PCT_2          = np.nanpercentile(b_chla, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im5          = ax5.contourf(Time_bot, Depth_bot, IN_DATA, levels,cmap = custom_cmap,alpha =1, extend = 'min')
##Set axis info and titles
ax5.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Chl-a', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax5.set_xlabel('Year', fontsize=Cbar_title_size, color='k')
# Save Plot
fig.savefig('plots/HOT_Contour_Model_Chla.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

# CSV filename
filename_1 = 'data/HOT_Bottle_POC_ModelResults.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc   = pd.read_csv(filename_1, index_col = 0)

bottle_poc.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
#bottle_poc = bottle_poc.dropna(subset=['POC_mod_surface'])

# Sort new df by ID and depth
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_poc = bottle_poc.reset_index(drop=True)

# Save new dataframe with Model output to CSV
bottle_poc.to_csv('data/HOT_Bottle_POC_ModelResults.csv')

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b2_time     = bottle_poc.loc[:,'time'].to_numpy()
b2_DateTime = bottle_poc.loc[:,'DateTime'].to_numpy()
b2_DecYear     = bottle_poc.loc[:,'DecYear'].to_numpy()
b2_date     = bottle_poc.loc[:,'Date'].to_numpy()
b2_depth    = bottle_poc.loc[:,'depth'].to_numpy()
b2_poc      = bottle_poc.loc[:,'POC'].to_numpy()
b2_pon      = bottle_poc.loc[:,'PON'].to_numpy()
b2_ID       = bottle_poc.loc[:,'Cruise_ID'].to_numpy()
b2_year     = bottle_poc.loc[:,'yyyy'].to_numpy()
b2_month    = bottle_poc.loc[:,'mm'].to_numpy()

#Convert array object to Datetimeindex type
b2_DateTime = pd.to_datetime(b2_DateTime)

### Cruise_ID list
# Removes Duplicates
ID_list_poc = pd.unique(b2_ID)
print(len(ID_list_poc))
# 275 profiles

# Extract POC Model Results
poc_mod_surface    = bottle_poc.loc[:,'POC_mod_surface'].to_numpy()
poc_mod_subsurface = bottle_poc.loc[:,'POC_mod_subsurface'].to_numpy()
poc_mod_total      = bottle_poc.loc[:,'POC_mod_total'].to_numpy()
poc_mod_bk         = bottle_poc.loc[:,'POC_mod_bk'].to_numpy()
poc_mod_data_diff  = bottle_poc.loc[:,'POC_mod_data_diff'].to_numpy()
# Extract PON Model Results
pon_mod_surface    = bottle_poc.loc[:,'PON_mod_surface'].to_numpy()
pon_mod_subsurface = bottle_poc.loc[:,'PON_mod_subsurface'].to_numpy()
pon_mod_total      = bottle_poc.loc[:,'PON_mod_total'].to_numpy()
pon_mod_bk         = bottle_poc.loc[:,'PON_mod_bk'].to_numpy()
pon_mod_data_diff  = bottle_poc.loc[:,'PON_mod_data_diff'].to_numpy()

# Import POC prof Data
# CSV filename
filename_1 = 'data/HOT_Bottle_POC_ProfData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_poc_prof   = pd.read_csv(filename_1, index_col = 0)

bottle_poc_prof.info()

# Remove NaN fit profiles
#bottle_poc_prof = bottle_poc_prof.dropna(subset=['C_Chl_ratio_C1'])

# Sort new df by ID and depth
bottle_poc_prof = bottle_poc_prof.sort_values(by=['Cruise_ID'])
# Reset bottle df index replacing old index column
bottle_poc_prof = bottle_poc_prof.reset_index(drop=True)
print(len(bottle_poc_prof))

# Save new dataframe with Model output to CSV
bottle_poc_prof.to_csv('data/HOT_Bottle_POC_ProfData.csv')

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_poc_prof['Date']))+" to "+str(max(bottle_poc_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_poc_prof['Date']), max(bottle_poc_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%
######
### CONTOUR PLOTS OF POC MODEL RESULTS ###
#####

### Interpolate data ###

ID_list_bottle = ID_list_poc

#m_chla = b_chla

# Set xy (area) of contour plot
y = 300
x2 = len(ID_list_bottle)

#Depth
New_depth = np.array(range(0, y))

POC_m_surf   = np.empty([x2,y])#+nan
POC_m_subb   = np.empty([x2,y])#+nan
POC_m_totl   = np.empty([x2,y])#+nan
POC          = np.empty([x2,y])#+nan
POC_m_bk     = np.empty([x2,y])#+nan

Depth_int_poc     = np.empty([x2,y])#+nan
Time_int_poc      = (np.empty([x2,y], dtype='datetime64[s]'))

# Raw POC interpolation
count = 0    
for i in ID_list_bottle:
    a = b2_poc[bottle_poc.Cruise_ID == i]
    b = b2_depth[bottle_poc.Cruise_ID == i]
    valid1 = ~np.isnan(b)
    valid2 = ~np.isnan(a)
    a      = a[valid2]
    c      = b[valid2]
    if len(b) > 1:
        interpfunc    = interpolate.interp1d(c,a, kind='linear',fill_value="extrapolate")
        xxx           = interpfunc(New_depth)
        POC[count,:]  = xxx
        AD                 = b2_DateTime[bottle_poc.Cruise_ID == i]
        Time_int_poc[count,:]  = AD[0]    
        Depth_int_poc[count,:] = New_depth
    count=count+1

# POC surface interpolation
count = 0    
for i in ID_list_bottle:
    a = poc_mod_surface[bottle_poc.Cruise_ID == i]
    b = b2_depth[bottle_poc.Cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_surf[count,:]  = xxx
    count=count+1

# POC subsurface interpolation
count = 0    
for i in ID_list_bottle:
    a = poc_mod_subsurface[bottle_poc.Cruise_ID == i]
    b = b2_depth[bottle_poc.Cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_subb[count,:]  = xxx
    count=count+1

# POC total model interpolation
count = 0    
for i in ID_list_bottle:
    a = poc_mod_total[bottle_poc.Cruise_ID == i]
    b = b2_depth[bottle_poc.Cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_totl[count,:]  = xxx
    count=count+1
    
# POC background interpolation
count = 0    
for i in ID_list_bottle:
    a = poc_mod_bk[bottle_poc.Cruise_ID == i]
    b = b2_depth[bottle_poc.Cruise_ID == i]
    if len(b) > 1:
        interpfunc       = interpolate.interp1d(b,a, kind='linear',fill_value="extrapolate")
        xxx              = interpfunc(New_depth)
        POC_m_bk[count,:]  = xxx
    count=count+1

#%%
### Plot
#Figure parameters that can be changed 
POC_COL           = cmocean.cm.tempo
DIFF_COL          = mpl.cm.bwr
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 18           #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17           #Define the font size of the Colourbar labels
pad_width         = 0.02
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1  
#Define the figure window including 6 subplots orientated vertically
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)

#PC Date Range: 1989-08-25 to 2022-09-02
xaxi = [date(1989,8,25),date(2022,9,2)]

#SUBPLOT 1: POC time series TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC.copy()
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2

##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

# Build modified colormap
custom_cmap = POC_COL.copy()
custom_cmap.set_under('white')

im1            = ax1.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap, alpha =1,extend = 'both')
##Set axis info and titles
ax1.set_ylim([200,0]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) PC Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width,extend='min')
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 2: POC total model time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_totl.copy()
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2

##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im2            = ax2.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_xlim([xaxi[0],xaxi[-1]]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total PC', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 3: POC total model Difference time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_totl - POC
#PCT_1          = np.nanpercentile(poc_mod_data_diff, Percentiles_lower)
#PCT_2          = np.nanpercentile(poc_mod_data_diff, Percentiles_upper)
PCT_1 = -25
PCT_2 = 25
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 100)
im3            = ax3.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = DIFF_COL, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0]) 
#ax3.set_xlim([xaxi[0],xaxi[-1]]) 
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width, extend='both')
cbar3.set_ticks(np.linspace(PCT_1, PCT_2, 5))   # evenly spaced ticks incl. 0
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 4: POC surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_surf.copy()
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)

valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im4            = ax4.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax4.set_ylim([200,0]) 
#ax4.set_xlim([xaxi[0],xaxi[-1]]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Phytoplankton Carbon', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 5: POC subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_subb.copy()

PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im5          = ax5.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax5.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Phytoplankton Carbon', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 6: POC background TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POC_m_bk.copy()
PCT_1          = np.nanpercentile(b2_poc, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_poc, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im6          = ax6.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap)
##Set axis info and titles
ax6.set_ylim([200,0]) 
#ax5.set_xlim([xaxi[0],xaxi[-1]]) 
ax6.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax6.set_title('(f) Model PC Background', fontsize = Title_font_size, color='m')
ax6.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax6.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar6 = fig.colorbar(im6, ax=ax6, pad = pad_width)
cbar6.ax.locator_params(nbins=5)
cbar6.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar6.ax.tick_params(labelsize = Cbar_label_size)
cbar6.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
ax6.set_xlabel('Year', fontsize=Title_font_size, color='k')

# Save Plot
fig.savefig('plots/HOT_Contour_Model_POC.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%
######
### PON INTERPOLATION FOR CONTOUR PLOTS ###
#####

# Constants
y = 300  # Depth range
New_depth = np.arange(0, y)  # New depth grid
x2 = len(ID_list_poc)  # Number of time series

# Initialize output arrays
PON = np.empty((x2, y))  # Raw PON
PON_m_surf = np.empty((x2, y))  # Surface PON model
PON_m_subb = np.empty((x2, y))  # Subsurface PON model
PON_m_totl = np.empty((x2, y))  # Total PON model
PON_m_bk = np.empty((x2, y))  # Background PON model

Depth_int_poc = np.empty((x2, y))  # Depth matrix
Time_int_poc = np.empty((x2, y), dtype='datetime64[s]')  # Time matrix

# Function to perform interpolation
def interpolate_poc(data_array, depth_array, new_depth, output_array):
    """
    Interpolates POC data to a uniform depth grid.
    
    Args:
    data_array (ndarray): The POC data to interpolate.
    depth_array (ndarray): Corresponding depth values.
    new_depth (ndarray): The target depth grid.
    output_array (ndarray): The array to store interpolated results.
    """
    count = 0
    for i in ID_list_poc:
        data = data_array[bottle_poc.Cruise_ID == i]
        depth = depth_array[bottle_poc.Cruise_ID == i]
        valid = ~np.isnan(data) & ~np.isnan(depth)
        data, depth = data[valid], depth[valid]
        if len(depth) > 1:
            interp_func = interpolate.interp1d(depth, data, kind='linear', fill_value="extrapolate")
            output_array[count, :] = interp_func(new_depth)
        count += 1

# Perform interpolations
interpolate_poc(b2_pon, b2_depth, New_depth, PON)  # Raw PON
interpolate_poc(pon_mod_surface, b2_depth, New_depth, PON_m_surf)  # Surface model PON
interpolate_poc(pon_mod_subsurface, b2_depth, New_depth, PON_m_subb)  # Subsurface modelPON
interpolate_poc(pon_mod_total, b2_depth, New_depth, PON_m_totl)  # Total model PON
interpolate_poc(pon_mod_bk, b2_depth, New_depth, PON_m_bk)  # Background model PON

# Fill Depth_bot and Time_bot
count = 0
for i in ID_list_bottle:
    if len(b2_depth[bottle_poc.Cruise_ID == i]) > 1:
        Depth_int_poc[count, :] = New_depth
        Time_int_poc[count, :] = b2_DateTime[bottle_poc.Cruise_ID == i][0]  # Use first timestamp
    count += 1
    
#%%
### PON CONTOUR PLOTS ###
#Figure parameters that can be changed 
PON_COL           = cmocean.cm.turbid
DIFF_COL          = mpl.cm.bwr
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 18            #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17            #Define the font size of the Colourbar labels
pad_width         = 0.02          #cbar offset from figure
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1  

#Define the figure window including 5 subplots orientated vertically
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)

#Bottle Dates: 1989-07-30 to 2022-09-02
xaxi = [date(1989,8,25),date(2022,9,2)]

# Build modified colormap
custom_cmap = PON_COL.copy()
custom_cmap.set_under('white')

#SUBPLOT 1: POC time series TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON.copy()
PCT_1          = np.nanpercentile(b2_pon, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_pon, Percentiles_upper)
#PCT_1 = 0.0
#PCT_2 = 50
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2

##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im1            = ax1.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap, alpha =1,extend = 'min')
##Set axis info and titles
ax1.set_ylim([200,0]) 
#ax1.set_xlim([xaxi[0],xaxi[-1]]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) PN Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width)
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 2: POC total model time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON_m_totl.copy()
PCT_1          = np.nanpercentile(b2_pon, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_pon, Percentiles_upper)
#PCT_1 = 0.0
#PCT_2 = 0.5
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im2            = ax2.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total PN', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 3: POC total model Difference time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON_m_totl - PON
#PCT_1          = np.nanpercentile(POC_diff, Percentiles_lower)
#PCT_2          = np.nanpercentile(POC_diff, Percentiles_upper)
PCT_1 = -4
PCT_2 = 4
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 100)
im3            = ax3.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = DIFF_COL, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0]) 
#ax3.set_xlim([xaxi[0],xaxi[-1]]) 
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width)
cbar3.set_ticks(np.linspace(PCT_1, PCT_2, 5))   # evenly spaced ticks incl. 0
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 4: POC surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON_m_surf.copy()
PCT_1          = np.nanpercentile(b2_pon, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_pon, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im4            = ax4.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax4.set_ylim([200,0]) 
#ax4.set_xlim([xaxi[0],xaxi[-1]]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Phytoplankton Nitrogen', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 5: POC subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON_m_subb.copy()
PCT_1          = np.nanpercentile(b2_pon, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_pon, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
#Plot
im5          = ax5.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax5.set_ylim([200,0])  
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Phytoplankton Nitrogen', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 6: POC background TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = PON_m_bk
PCT_1          = np.nanpercentile(b2_pon, Percentiles_lower)
PCT_2          = np.nanpercentile(b2_pon, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im6          = ax6.contourf(Time_int_poc, Depth_int_poc, IN_DATA, levels,cmap = custom_cmap)
##Set axis info and titles
ax6.set_ylim([200,0]) 
ax6.set_xlim([xaxi[0],xaxi[-1]]) 
ax6.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax6.set_title('(f) Model PN Background', fontsize = Title_font_size, color='m')
ax6.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax6.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar6 = fig.colorbar(im6, ax=ax6, pad = pad_width)
cbar6.ax.locator_params(nbins=5)
cbar6.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar6.ax.tick_params(labelsize = Cbar_label_size)
cbar6.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
ax6.set_xlabel('Year', fontsize=Title_font_size, color='k')

# Save Plot
fig.savefig('plots/HOT_Contour_Model_PON.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()

#%%

### EXTRACT POP MODEL RESULTS & SORT

# CSV filename
filename_1 = 'data/HOT_Bottle_POP_ModelResults.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_pop   = pd.read_csv(filename_1, index_col = 0)

bottle_pop.info()

# Remove NaN fit profiles
# Drop POC profiles not fitted - NaN
#bottle_poc = bottle_poc.dropna(subset=['POC_mod_surface'])

# Sort new df by ID and depth
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_pop = bottle_pop.reset_index(drop=True)

# Save new dataframe with Model output to CSV
bottle_pop.to_csv('data/HOT_Bottle_POP_ModelResults.csv')

### EXTRACT CLEANED DATA & MAKE NEW POC ID LIST ###

### Extract required data from new bottle_6 dataset ###
b3_time     = bottle_pop.loc[:,'time'].to_numpy()
b3_DateTime = bottle_pop.loc[:,'DateTime'].to_numpy()
b3_date     = bottle_pop.loc[:,'Date'].to_numpy()
b3_DecYear = bottle_pop.loc[:,'DecYear'].to_numpy()
b3_depth    = bottle_pop.loc[:,'depth'].to_numpy()
b3_pop      = bottle_pop.loc[:,'POP'].to_numpy()
b3_ID       = bottle_pop.loc[:,'Cruise_ID'].to_numpy()
b3_year     = bottle_pop.loc[:,'yyyy'].to_numpy()
b3_month    = bottle_pop.loc[:,'mm'].to_numpy()

#Convert array object to Datetimeindex type
b3_DateTime = pd.to_datetime(b3_DateTime)

### Cruise_ID list
# Removes Duplicates
ID_list_pop = pd.unique(b3_ID)
print(len(ID_list_pop))
# 265 profiles

# Extract POP Model Results
pop_mod_surface    = bottle_pop.loc[:,'POP_mod_surface'].to_numpy()
pop_mod_subsurface = bottle_pop.loc[:,'POP_mod_subsurface'].to_numpy()
pop_mod_total      = bottle_pop.loc[:,'POP_mod_total'].to_numpy()
pop_mod_bk         = bottle_pop.loc[:,'POP_mod_bk'].to_numpy()
pop_mod_data_diff  = bottle_pop.loc[:,'POP_mod_data_diff'].to_numpy()


# Import POC prof Data
# CSV filename
filename_1 = 'data/HOT_Bottle_POP_ProfData.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_pop_prof   = pd.read_csv(filename_1, index_col = 0)

bottle_pop_prof.info()

# Remove NaN fit profiles
#bottle_poc_prof = bottle_poc_prof.dropna(subset=['C_Chl_ratio_C1'])

# Sort new df by ID and depth
bottle_pop_prof = bottle_pop_prof.sort_values(by=['Cruise_ID'])
# Reset bottle df index replacing old index column
bottle_pop_prof = bottle_pop_prof.reset_index(drop=True)
print(len(bottle_pop_prof))

# Save new dataframe with Model output to CSV
bottle_pop_prof.to_csv('data/HOT_Bottle_POP_ProfData.csv')

# Print start and end dates of bottle data
print("Bottle Dates: "+str(min(bottle_pop_prof['Date']))+" to "+str(max(bottle_pop_prof['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_pop_prof['Date']), max(bottle_pop_prof['Date']))
print("Timespan: "+str(b_date_length))

#%%
######
### POP INTERPOLATION FOR CONTOUR PLOTS ###
#####

# Constants
y = 300  # Depth range
New_depth = np.arange(0, y)  # New depth grid
x2 = len(ID_list_pop)  # Number of time series

# Initialize output arrays
POP = np.empty((x2, y))  # Raw PON
POP_m_surf = np.empty((x2, y))  # Surface PON model
POP_m_subb = np.empty((x2, y))  # Subsurface PON model
POP_m_totl = np.empty((x2, y))  # Total PON model
POP_m_bk = np.empty((x2, y))  # Background PON model

Depth_int_pop = np.empty((x2, y))  # Depth matrix
Time_int_pop = np.empty((x2, y), dtype='datetime64[s]')  # Time matrix

# Function to perform interpolation
def interpolate_pop(data_array, depth_array, new_depth, output_array):
    """
    Interpolates POC data to a uniform depth grid.
    
    Args:
    data_array (ndarray): The POC data to interpolate.
    depth_array (ndarray): Corresponding depth values.
    new_depth (ndarray): The target depth grid.
    output_array (ndarray): The array to store interpolated results.
    """
    count = 0
    for i in ID_list_pop:
        data = data_array[bottle_pop.Cruise_ID == i]
        depth = depth_array[bottle_pop.Cruise_ID == i]
        valid = ~np.isnan(data) & ~np.isnan(depth)
        data, depth = data[valid], depth[valid]
        if len(depth) > 1:
            interp_func = interpolate.interp1d(depth, data, kind='linear', fill_value="extrapolate")
            output_array[count, :] = interp_func(new_depth)
        count += 1

# Perform interpolations
interpolate_pop(b3_pop, b3_depth, New_depth, POP)  # Raw PON
interpolate_pop(pop_mod_surface, b3_depth, New_depth, POP_m_surf)  # Surface model PON
interpolate_pop(pop_mod_subsurface, b3_depth, New_depth, POP_m_subb)  # Subsurface modelPON
interpolate_pop(pop_mod_total, b3_depth, New_depth, POP_m_totl)  # Total model PON
interpolate_pop(pop_mod_bk, b3_depth, New_depth, POP_m_bk)  # Background model PON

# Fill Depth_bot and Time_bot
count = 0
for i in ID_list_pop:
    if len(b3_depth[bottle_pop.Cruise_ID == i]) > 1:
        Depth_int_pop[count, :] = New_depth
        Time_int_pop[count, :] = b3_DateTime[bottle_pop.Cruise_ID == i][0]  # Use first timestamp
    count += 1
    
#%%
### POP CONTOUR PLOTS ###
#Figure parameters that can be changed 
POP_COL = cmocean.cm.matter#mpl.cm.plasma#mpl.cm.jet#cmocean.cm.matter
DIFF_COL          = mpl.cm.bwr
XSIZE             = 14            #Define the xsize of the figure window
YSIZE             = 18           #Define the ysize of the figure window
Title_font_size   = 19            #Define the font size of the titles
Label_font_size_x = 17            #Define the font size of the x-labels
Label_font_size_y = 17            #Define the font size of the y-labels
tick_length       = 6 
Cbar_title_size   = 17            #Define the font size of the Colourbar title
Cbar_label_size   = 17           #Define the font size of the Colourbar labels
pad_width         = 0.02
Percentiles_upper = 99            #Upper percentiles used to constrain the colour scale
Percentiles_lower = 1  

#Define the figure window including 5 subplots orientated vertically
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, sharex=True, figsize=(XSIZE,YSIZE))
fig.subplots_adjust(wspace=0.2,hspace=0.27)

#print(max(bottle_pop_prof['Date']))

# X-axis date range   
#PP Bottle Dates: 1989-10-17 to 2022-09-02
xaxi = [date(1989,10,17),date(2022,9,2)]

# Build modified colormap
custom_cmap = POP_COL.copy()
custom_cmap.set_under('white')

#SUBPLOT 1: POC time series TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP.copy()
PCT_1          = np.nanpercentile(b3_pop, Percentiles_lower)
PCT_2          = np.nanpercentile(b3_pop, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im1            = ax1.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = custom_cmap, alpha =1,extend = 'min')
##Set axis info and titles
ax1.set_ylim([200,0]) 
#ax1.set_xlim([xaxi[0],xaxi[-1]]) 
ax1.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax1.set_title('(a) PP Data', fontsize = Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax1.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar1 = fig.colorbar(im1, ax=ax1, pad = pad_width)
cbar1.ax.locator_params(nbins=5)
cbar1.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar1.ax.tick_params(labelsize = Cbar_label_size)
cbar1.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 2: POC total model time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP_m_totl.copy()
PCT_1          = np.nanpercentile(b3_pop, Percentiles_lower)
PCT_2          = np.nanpercentile(b3_pop, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
#Plot
im2            = ax2.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax2.set_ylim([200,0]) 
ax2.set_xlim([xaxi[0],xaxi[-1]]) 
ax2.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax2.set_title('(b) Model Total PP', fontsize = Title_font_size, color='k')
ax2.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax2.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar2 = fig.colorbar(im2, ax=ax2, pad = pad_width)
cbar2.ax.locator_params(nbins=5)
cbar2.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar2.ax.tick_params(labelsize = Cbar_label_size)
cbar2.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 3: POC total model Difference time series
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP_m_totl - POP
#PCT_1          = np.nanpercentile(pop_mod_data_diff, Percentiles_lower)
#PCT_2          = np.nanpercentile(pop_mod_data_diff, Percentiles_upper)
PCT_1 = -0.5
PCT_2 = 0.5
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im3            = ax3.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = DIFF_COL, extend = 'both')
##Set axis info and titles
ax3.set_ylim([200,0])  
ax3.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax3.set_title('(c) Model - Data', fontsize = Title_font_size, color='k')
ax3.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax3.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar3 = fig.colorbar(im3, ax=ax3, pad = pad_width)
cbar3.ax.locator_params(nbins=5)
cbar3.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar3.ax.tick_params(labelsize = Cbar_label_size)
cbar3.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 4: POC surface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP_m_surf.copy()
PCT_1          = np.nanpercentile(b3_pop, Percentiles_lower)
PCT_2          = np.nanpercentile(b3_pop, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)
im4            = ax4.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax4.set_ylim([200,0]) 
ax4.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax4.set_title('(d) Surface Phytoplankton Phosphorus', fontsize = Title_font_size, color='r')
ax4.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax4.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar4 = fig.colorbar(im4, ax=ax4, pad = pad_width)
cbar4.ax.locator_params(nbins=5)
cbar4.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar4.ax.tick_params(labelsize = Cbar_label_size)
cbar4.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 5: POC subsurface TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP_m_subb.copy()
PCT_1          = np.nanpercentile(b3_pop, Percentiles_lower)
PCT_2          = np.nanpercentile(b3_pop, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im5          = ax5.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = custom_cmap,alpha =1,extend = 'min')
##Set axis info and titles
ax5.set_ylim([200,0]) 
ax5.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax5.set_title('(e) Subsurface Phytoplankton Phosphorus', fontsize = Title_font_size, color='b')
ax5.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax5.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar5 = fig.colorbar(im5, ax=ax5, pad = pad_width)
cbar5.ax.locator_params(nbins=5)
cbar5.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar5.ax.tick_params(labelsize = Cbar_label_size)
cbar5.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))

#SUBPLOT 6: POC background TIME-SERIES
##Constrain data to be between 1 and 99 percentile (avoids outliers in data colour scaling)
IN_DATA        = POP_m_bk.copy()
PCT_1          = np.nanpercentile(b3_pop, Percentiles_lower)
PCT_2          = np.nanpercentile(b3_pop, Percentiles_upper)
valid          = (IN_DATA < PCT_1)
IN_DATA[valid] = PCT_1
valid          = (IN_DATA > PCT_2)
IN_DATA[valid] = PCT_2
##Define colour levels
levels       = np.linspace(PCT_1, PCT_2, 50)

im6          = ax6.contourf(Time_int_pop, Depth_int_pop, IN_DATA, levels,cmap = custom_cmap)
##Set axis info and titles
ax6.set_ylim([200,0]) 
ax6.set_xlim([xaxi[0],xaxi[-1]]) 
ax6.set_ylabel('Depth (m)', fontsize= Cbar_title_size, color='k')
ax6.set_title('(f) Model PP Background', fontsize = Title_font_size, color='m')
ax6.yaxis.set_tick_params(labelsize= Label_font_size_y)##Add colourbar
ax6.xaxis.set_tick_params(labelsize= Label_font_size_y, length = tick_length)##Add colourbar
cbar6 = fig.colorbar(im6, ax=ax6, pad = pad_width)
cbar6.ax.locator_params(nbins=5)
cbar6.set_label("mg m$^{-3}$", size  = Cbar_title_size)
cbar6.ax.tick_params(labelsize = Cbar_label_size)
cbar6.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
ax6.set_xlabel('Year', fontsize=Title_font_size, color='k')

# Save Plot
fig.savefig('plots/HOT_Contour_Model_POP.jpeg', format='jpeg', dpi=300, bbox_inches="tight")
plt.show()


"END"
