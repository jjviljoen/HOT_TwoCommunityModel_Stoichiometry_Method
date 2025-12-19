"""
HOT - Integrate HOT Data and Model Results

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
# Import specific modules from packages
from dateutil import relativedelta
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"  # Set font for all plots
# Supress
import warnings
warnings.filterwarnings("ignore") # Added to remove the warning "UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray." on 2nd to last cell of code
np.seterr(all='ignore');

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    rd = relativedelta.relativedelta( pd.to_datetime( end ), pd.to_datetime( start ) )
    date_len  = '{}y{}m{}d'.format(rd.years,rd.months,rd.days)
    return date_len

#%%

### READ & EXTRACT CTD DATA ###

# CSV filename
filename_1 = 'data/HOT_CTD_Cleaned.csv'

# Load data from csv. "index_col = 0" makes the first column the index.
ctd = pd.read_csv(filename_1, index_col=0)

# Display dataframe info
ctd.info()

### Extract required data from CTD dataframe into numpy arrays ###
ctd_date         = ctd['Date'].to_numpy()
time_year        = ctd['yyyy'].to_numpy()
depth            = ctd['depth'].to_numpy()
temperature      = ctd['temperature'].to_numpy()
salinity         = ctd['salinity'].to_numpy()
density          = ctd['density'].to_numpy()
bvf              = ctd['BVF'].to_numpy()
fluorescence     = ctd['fluorescence'].to_numpy()
ID_ctd           = ctd['cruise_ID'].to_numpy()
ctd_DateTime     = pd.to_datetime(ctd['DateTime'])
ctd_Decimal_year = ctd['Dec_Year'].to_numpy()

### Read CTD prof data ###
# CSV filename
filename_2 = 'data/HOT_CTD_profData.csv'

# Load data from csv. "index_col = 0" makes the first column the index.
ctd_prof = pd.read_csv(filename_2, index_col=0)

# Inspect ctd_prof df
ctd_prof.info()
ctd_prof.head()

print(len(ctd_prof))

# Extract required data from df
ID_list_ctd       = ctd_prof['Cruise_ID'].values
print(len(ID_list_ctd))
ctd_DateTime_prof = pd.to_datetime(ctd_prof['DateTime'].values)
ctd_date_prof     = ctd_prof['Date'].to_numpy()
ctd_DecYear_prof  = ctd_prof['DecYear'].to_numpy()
MLD               = ctd_prof['MLD_boyer'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(ctd_prof['Date']), max(ctd_prof['Date']))
print("Timespan: "+str(ctd_date_length))
print("Min Date: "+str(min(ctd_prof['Date'])))
print("Max Date: "+str(max(ctd_prof['Date'])))


#%%

### READ & EXTRACT BOTTLE PIGMENTS DATA & MODEL RESULTS ###

### Read/Import cleaned Bottle data from CSV including model fit results ###
# CSV filename
filename_1 = 'data/HOT_Bottle_Pigments_ModelResults.csv'
# Load data from CSV. "index_col = 0" makes the first column the index.
bottle_6 = pd.read_csv(filename_1, index_col=0)

# Inspect the dataframe
bottle_6.info()

# Sort dataframe by Cruise_ID and depth
bottle_6 = bottle_6.sort_values(by=['Cruise_ID', 'depth'])
# Reset dataframe index, replacing the old index column
bottle_6 = bottle_6.reset_index(drop=True)

### Extract required data from the dataset ###
b_date         = bottle_6['Date'].to_numpy()
b_depth        = bottle_6['depth'].to_numpy()
b_chla         = bottle_6['Chla'].to_numpy()
b_ID           = bottle_6['Cruise_ID'].to_numpy()
b_year         = bottle_6['yyyy'].to_numpy()
b_month        = bottle_6['mm'].to_numpy()
b_Decimal_year = bottle_6['DecYear'].to_numpy()
b_DateTime     = pd.to_datetime(bottle_6['DateTime'])

### Generate a list of unique Cruise_IDs ###
ID_list_6 = pd.unique(b_ID)
print(len(ID_list_6))

### Extract model results ###
CHL_mod_total      = bottle_6['CHL_mod_total'].to_numpy()      # Extract 'CHL_mod_total' column as numpy array
CHL_mod_surface    = bottle_6['CHL_mod_surface'].to_numpy()    # Extract 'CHL_mod_surface' column as numpy array
CHL_mod_subsurface = bottle_6['CHL_mod_subsurface'].to_numpy() # Extract 'CHL_mod_subsurface' column as numpy array
CHL_mod_data_diff  = bottle_6['CHL_mod_data_diff'].to_numpy()  # Extract 'CHL_mod_data_diff' column as numpy array

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
b_MLD_prof      = bottle_prof['MLD_dens'].to_numpy()             # Extract 'MLD' column as numpy array
b_dec_year_prof = bottle_prof['DecYear'].to_numpy()         # Extract 'DecYear' column as numpy array
b_DCM_depth     = bottle_prof.loc[:,'DCM_depth'].to_numpy()

Kd_prof = bottle_prof['Kd'].to_numpy()
Zp_prof = bottle_prof['Zp'].to_numpy()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_prof['Date']), max(bottle_prof['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(bottle_prof['Date'])))
print("Max Date: "+str(max(bottle_prof['Date'])))

#%%

### READ & EXTRACT BOTTLE POC & PON DATA & MODEL RESULTS ###

# Load and preprocess POC data
filename_poc = 'data/HOT_Bottle_POC_ModelResults.csv'

# Load data from CSV, using the first column as the index
bottle_poc = pd.read_csv(filename_poc, index_col=0)
bottle_poc.info()

# Sort dataframe by Cruise_ID and depth, then reset the index
bottle_poc = bottle_poc.sort_values(by=['Cruise_ID', 'depth']).reset_index(drop=True)

### Extract Required Data ###
b2_DateTime   = pd.to_datetime(bottle_poc['DateTime'])   # Convert DateTime column to pandas datetime
b2_date       = bottle_poc['Date'].to_numpy()           # Extract date as numpy array
b2_depth      = bottle_poc['depth'].to_numpy()          # Extract depth as numpy array
b2_poc        = bottle_poc['POC'].to_numpy()            # Extract POC as numpy array
b2_pon        = bottle_poc['PON'].to_numpy()            # Extract PON as numpy array
b2_ID         = bottle_poc['Cruise_ID'].to_numpy()      # Extract Cruise_ID as numpy array
b2_year       = bottle_poc['yyyy'].to_numpy()           # Extract year as numpy array
b2_month      = bottle_poc['mm'].to_numpy()             # Extract month as numpy array
b2_DecYear    = bottle_poc['DecYear'].to_numpy()        # Extract decimal year as numpy array

### Cruise_ID List ###
ID_list_poc = pd.unique(b2_ID)  # Create unique Cruise_ID list
print(f"Unique Cruise_IDs: {len(ID_list_poc)}")  # Print the count of unique Cruise_IDs

### Extract POC Model Results ###
poc_mod_surface    = bottle_poc['POC_mod_surface'].to_numpy()
poc_mod_subsurface = bottle_poc['POC_mod_subsurface'].to_numpy()
poc_mod_total      = bottle_poc['POC_mod_total'].to_numpy()
poc_mod_bk         = bottle_poc['POC_mod_bk'].to_numpy()
poc_mod_data_diff  = bottle_poc['POC_mod_data_diff'].to_numpy()

### Extract Living POC and Phyto Carbon ###
phytoC        = bottle_poc['Phyto_Carbon'].to_numpy()
phytoC_ratio  = bottle_poc['PhytoC_Ratio'].to_numpy()

### Extract PON Model Results ###
pon_mod_surface    = bottle_poc['PON_mod_surface'].to_numpy()
pon_mod_subsurface = bottle_poc['PON_mod_subsurface'].to_numpy()
pon_mod_total      = bottle_poc['PON_mod_total'].to_numpy()
pon_mod_bk         = bottle_poc['PON_mod_bk'].to_numpy()
pon_mod_data_diff  = bottle_poc['PON_mod_data_diff'].to_numpy()

### Extract Living POC and Phyto Carbon ###
phytoN        = bottle_poc['Phyto_Nitrogen'].to_numpy()
phytoN_ratio  = bottle_poc['PhytoN_Ratio'].to_numpy()

# Load and preprocess POC profile data
filename_poc_prof = 'data/HOT_Bottle_POC_ProfData.csv'

# Load data from CSV, using the first column as the index
bottle_poc_prof = pd.read_csv(filename_poc_prof, index_col=0)
bottle_poc_prof.info()

# Sort dataframe by Cruise_ID, then reset the index
bottle_poc_prof = bottle_poc_prof.sort_values(by=['Cruise_ID']).reset_index(drop=True)
print(f"Number of POC/PON profiles: {len(bottle_poc_prof)}")  # Print the count of profiles

### Extract Profile Data ###
b2_DateTime_prof    = pd.to_datetime(bottle_poc_prof['DateTime'])  # Convert DateTime column to pandas datetime
b2_MLD_prof         = bottle_poc_prof['MLD_used'].to_numpy()       # Extract MLD as numpy array
b2_DecYear_prof     = bottle_poc_prof['DecYear'].to_numpy()        # Extract decimal year as numpy array

### Extract Ratios ###
# C:Chla Ratios
C_Chl_ratio_C1      = bottle_poc_prof['C_Chl_ratio_C1'].to_numpy()
C_Chl_ratio_C2      = bottle_poc_prof['C_Chl_ratio_C2'].to_numpy()

# C:N Molar Ratios
N_Chl_ratio_C1      = bottle_poc_prof['N_Chl_ratio_C1'].to_numpy()
N_Chl_ratio_C2      = bottle_poc_prof['N_Chl_ratio_C2'].to_numpy()
C_N_molar_C1        = bottle_poc_prof['C_N_molar_C1'].to_numpy()
C_N_molar_C2        = bottle_poc_prof['C_N_molar_C2'].to_numpy()
bk_ratio_CN_molar   = bottle_poc_prof['bk_ratio_CN_molar'].to_numpy()

# Cruise_ID list for profile data
ID_list_poc_prof = pd.unique(bottle_poc_prof['Cruise_ID'])
print(f"Unique Cruise_IDs in profile data: {len(ID_list_poc_prof)}")

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(bottle_poc_prof['Date']), max(bottle_poc_prof['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(bottle_poc_prof['Date'])))
print("Max Date: "+str(max(bottle_poc_prof['Date'])))

#%%

### READ & EXTRACT POP DATA & MODEL RESULTS ###

# Import required libraries

# Load and preprocess POP data
filename_pop = 'data/HOT_Bottle_POP_ModelResults.csv'

# Load data from CSV, using the first column as the index
bottle_pop = pd.read_csv(filename_pop, index_col=0)
bottle_pop.info()

# Sort dataframe by Cruise_ID and depth, then reset the index
bottle_pop = bottle_pop.sort_values(by=['Cruise_ID', 'depth']).reset_index(drop=True)

### Extract Required Data ###
b3_time      = bottle_pop['time'].to_numpy()              # Extract time as numpy array
b3_date      = bottle_pop['Date'].to_numpy()              # Extract date as numpy array
b3_DecYear   = bottle_pop['DecYear'].to_numpy()           # Extract decimal year as numpy array
b3_depth     = bottle_pop['depth'].to_numpy()             # Extract depth as numpy array
b3_pop       = bottle_pop['POP'].to_numpy()               # Extract POP as numpy array
b3_ID        = bottle_pop['Cruise_ID'].to_numpy()         # Extract Cruise_ID as numpy array
b3_year      = bottle_pop['yyyy'].to_numpy()              # Extract year as numpy array
b3_month     = bottle_pop['mm'].to_numpy()                # Extract month as numpy array
b3_DateTime  = pd.to_datetime(bottle_pop['DateTime'])     # Convert DateTime column to pandas datetime

### Cruise_ID List ###
ID_list_pop = pd.unique(b3_ID)  # Create unique Cruise_ID list
print(f"Unique Cruise_IDs: {len(ID_list_pop)}")  # Print the count of unique Cruise_IDs

### Extract POP Model Results ###
pop_mod_surface    = bottle_pop['POP_mod_surface'].to_numpy()
pop_mod_subsurface = bottle_pop['POP_mod_subsurface'].to_numpy()
pop_mod_total      = bottle_pop['POP_mod_total'].to_numpy()
pop_mod_bk         = bottle_pop['POP_mod_bk'].to_numpy()
pop_mod_data_diff  = bottle_pop['POP_mod_data_diff'].to_numpy()

### Extract Living POC and Phyto Carbon ###
phytoP        = bottle_pop['Phyto_Phos'].to_numpy()
phytoP_ratio  = bottle_pop['PhytoP_Ratio'].to_numpy()

# Load and preprocess POP profile data
filename_pop_prof = 'data/HOT_Bottle_POP_ProfData.csv'

# Load data from CSV, using the first column as the index
bottle_pop_prof = pd.read_csv(filename_pop_prof, index_col=0)
bottle_pop_prof.info()

# Sort dataframe by Cruise_ID, then reset the index
bottle_pop_prof = bottle_pop_prof.sort_values(by=['Cruise_ID']).reset_index(drop=True)
print(f"Number of profile records: {len(bottle_pop_prof)}")  # Print the count of profiles

### Extract Profile Data ###
b3_DateTime_prof    = pd.to_datetime(bottle_pop_prof['Date'])  # Convert Date column to pandas datetime
b3_DecYear_prof     = bottle_pop_prof['DecYear'].to_numpy()    # Extract decimal year as numpy array

### Extract Ratios ###
# C:P Molar Ratios
P_Chl_ratio_C1      = bottle_pop_prof['P_Chl_ratio_C1'].to_numpy()
P_Chl_ratio_C2      = bottle_pop_prof['P_Chl_ratio_C2'].to_numpy()
C_P_molar_C1        = bottle_pop_prof['C_P_molar_C1'].to_numpy()
C_P_molar_C2        = bottle_pop_prof['C_P_molar_C2'].to_numpy()
bk_ratio_CP_molar   = bottle_pop_prof['bk_ratio_CP_molar'].to_numpy()

# N:P Molar Ratios
N_P_molar_C1        = bottle_pop_prof['N_P_molar_C1'].to_numpy()
N_P_molar_C2        = bottle_pop_prof['N_P_molar_C2'].to_numpy()
bk_ratio_NP_molar   = bottle_pop_prof['bk_ratio_NP_molar'].to_numpy()

#%%
################
### IMPORT NUTRIENT DATA
################

### Import vertical nutrient profiles
# CSV filename
filename_1 = 'data/HOT_Bottle_Nutrients_LLN_Cleaned.csv' # swap for AA based nutrient data 'data/HOT_Bottle_Nutrients_Cleaned.csv'
# Load data from csv. "index_col = 0" make first column the index.
bottle_nut   = pd.read_csv(filename_1, index_col = 0)

# Sort new df by ID and depth
bottle_nut = bottle_nut.sort_values(by=['Cruise_ID','depth'])

# Reset bottle df index replacing old index column
bottle_nut = bottle_nut.reset_index(drop=True)

### EXTRACT CLEANED DATA & MAKE NEW ID LIST ###
bottle_nut.info()
### Extract required data from new bottle_6 dataset ###
nut_time     = bottle_nut.loc[:,'DateTime'].to_numpy()
nut_date     = bottle_nut.loc[:,'Date'].to_numpy()
nut_depth    = bottle_nut.loc[:,'depth'].to_numpy()
#nut_nitrate  = bottle_nut.loc[:,'nit'].to_numpy()
nut_nitrate  = bottle_nut.loc[:,'lln_nmol'].to_numpy()
#nut_phosphate  = bottle_nut.loc[:,'phos'].to_numpy()
#nut_Si       = bottle_nut.loc[:,'sil'].to_numpy()
nut_ID       = bottle_nut.loc[:,'Cruise_ID'].to_numpy()
nut_year     = bottle_nut.loc[:,'yyyy'].to_numpy()
nut_month    = bottle_nut.loc[:,'mm'].to_numpy()
nut_Decimal_year = bottle_nut.loc[:,'DecYear'].to_numpy()

#Convert array object to Datetimeindex type
nut_DateTime = pd.to_datetime(bottle_nut['DateTime'].values)

### Cruise_ID list
ID_list_nut = pd.unique(nut_ID)
print(len(ID_list_nut))

# Load Nutrient profile data - Nutriclines
filename = 'data/HOT_Bottle_Nutrients_LLN_profData.csv' # swap for AA based nutrient data 'data/HOT_Bottle_Nutrients_profData.csv'

# Load data from CSV, using the first column as the index
bottle_nut_prof = pd.read_csv(filename, index_col=0)
bottle_nut_prof.info()

# Sort dataframe by Cruise_ID, then reset the index
bottle_nut_prof = bottle_nut_prof.sort_values(by=['Cruise_ID']).reset_index(drop=True)
print(f"Number of profile records: {len(bottle_nut_prof)}")  # Print the count of profiles

z_nitrate   = bottle_nut_prof['Z_nitrate'].to_numpy()

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

### Extract Zp and Kd from HOT ###
Kd_prof = Zp_Kd_df['Kd_Morel_tuned_sw'].to_numpy()
Zp_prof = Zp_Kd_df['Zeu_Morel_tuned_sw'].to_numpy()

#%%

# DEFINE INTEGRATION FUNCTION

import numpy as np
#from scipy.integrate import trapz
from numpy import trapz

def integrate_sections(depths, conc, mld, zp, dcm_depth=None):
    """
    Integrate a nutrient profile and compute summary statistics:

    Depth intervals (with interpolation at 0 m, MLD, and 1.5*Zp):
      1) Surface → MLD
      2) MLD → 1.5*Zp
      3) Surface → 1.5*Zp

    Additional metrics:
      - median concentration above MLD
      - concentration at DCM depth (if provided)

    Args:
      depths      (1D array): sampled depths (m), unsorted OK
      conc        (1D array): nutrient conc (µmol L⁻¹) at those depths
      mld         (float):     mixed-layer depth (m)
      zp          (float):     euphotic depth (m)
      dcm_depth   (float, optional): depth of DCM (m)

    Returns:
      dict with keys:
        'above_mld'         : ∫₀ᵐˡᵈ conc dz
        'mld_to_1_5_zp'     : ∫ₘˡᵈ¹.⁵Zₚ conc dz
        'above_1_5_zp'      : ∫₀¹.⁵Zₚ conc dz
        'median_above_mld'  : median conc for depths ≤ MLD
        'conc_at_dcm'       : conc at dcm_depth (if given, else None)
    """
    # Convert & sort originals
    depths = np.asarray(depths, float)
    conc   = np.asarray(conc,   float)
    orig_min, orig_max = depths.min(), depths.max()
    sort_idx = np.argsort(depths)
    depths, conc = depths[sort_idx], conc[sort_idx]

    # Boundary depths
    z0    = 0.0
    z_mld = float(mld)
    z_end = 1.5 * float(zp)

    # Warn if euphotic end shallower than MLD
    if z_end <= z_mld:
        message = (
            f"1.5*Zp ({z_end:.2f} m) ≤ MLD ({z_mld:.2f} m); "
            "using surface→1.5Zp for above_mld and skipping MLD→Zp."
        )
        warnings.warn(message, stacklevel=2)
        print(f"Warning: {message}")
        shallow_case = True
    else:
        shallow_case = False

    # 1) Inject surface at 0 m if needed
    if orig_min > 0:
        depths = np.insert(depths, 0, z0)
        conc   = np.insert(conc,   0, conc[0])
        
    # 1b) Inject point at 0.75 × MLD if it's shallower than original data
    z_shallow = 0.75 * z_mld
    if z_shallow < orig_min:
        c_shallow = np.interp(z_shallow, depths, conc)
        depths = np.append(depths, z_shallow)
        conc   = np.append(conc,   c_shallow)

    # 2) Inject MLD point if appropriate
    if not shallow_case and orig_min < z_mld < orig_max and not np.isclose(depths, z_mld).any():
        c_mld = np.interp(z_mld, depths, conc)
        depths = np.append(depths, z_mld)
        conc   = np.append(conc,   c_mld)

    # 3) Inject 1.5*Zp if within original range
    if orig_min < z_end < orig_max and not np.isclose(depths, z_end).any():
        c_end = np.interp(z_end, depths, conc)
        depths = np.append(depths, z_end)
        conc   = np.append(conc,   c_end)

    # Resort after injections
    idx = np.argsort(depths)
    depths, conc = depths[idx], conc[idx]

    # Prepare results
    results = {}

    # 1) Surface → MLD (or → 1.5Zp if shallow_case)
    if shallow_case:
        mask_above = depths <= z_end
    else:
        mask_above = depths <= z_mld
    results['above_mld'] = trapz(conc[mask_above], depths[mask_above]) if mask_above.sum()>=2 else np.nan

    # 2) MLD → 1.5Zp (skip if shallow_case)
    if shallow_case:
        results['below_mld'] = np.nan
    else:
        mask_mid = (depths >= z_mld) & (depths <= z_end)
        results['below_mld'] = trapz(conc[mask_mid], depths[mask_mid]) if mask_mid.sum()>=2 else np.nan

    # 3) Surface → 1.5Zp
    mask_zp = depths <= z_end
    results['above_zp'] = trapz(conc[mask_zp], depths[mask_zp]) if mask_zp.sum()>=2 else np.nan

    # Median concentration above MLD
    mask_med = depths <= z_mld
    results['median_above_mld'] = np.median(conc[mask_med]) if mask_med.sum()>=1 else np.nan

    # Concentration at DCM
    if dcm_depth is not None:
        # interpolate to exactly dcm_depth
        c_dcm = np.interp(float(dcm_depth), depths, conc)
        results['conc_at_dcm'] = c_dcm
    else:
        results['conc_at_dcm'] = None

    return results

#%%

### INTEGRATE OVER DEPTH ###

# Inspect the dataframe
bottle_prof.info()

# Initialize arrays for storing integrated data
len_list = ID_list_6
nan_array = lambda: np.full(len(len_list), np.nan)  # Helper for initializing arrays

# Integrated Chlorophyll data
chla_mod_total_int  = nan_array()
chla_mod_surf_int   = nan_array()
chla_mod_sub_int    = nan_array()
chla_data_int       = nan_array()
chla_data_int_125   = nan_array()
DecYear_prof        = nan_array()

# Direct Chla separations
chla_data_surf_int    = nan_array() # Chl integrated above MLD
chla_data_sub_int     = nan_array() # Chl integrated below MLD to 1.5Zp
chla_data_int2        = nan_array()
chla_data_surf_median = nan_array()
chla_data_dcm_conc    = nan_array()

# Direct Nutrient separations
nitrate_int_surf    = nan_array() # integrated above MLD
nitrate_int_sub     = nan_array() # integrated below MLD to 1.5Zp
nitrate_int_zp      = nan_array() # integrated below MLD to 1.5Zp
nitrate_surf_median = nan_array()
nitrate_dcm_conc    = nan_array()
Znitrate            = nan_array()

phosphate_int_surf    = nan_array() # integrated above MLD
phosphate_int_sub     = nan_array() # integrated below MLD to 1.5Zp
phosphate_int_zp      = nan_array() # integrated below MLD to 1.5Zp
phosphate_surf_median = nan_array()
phosphate_dcm_conc    = nan_array()
Zphosphate            = nan_array()

# Euphotic depth and additional profile metrics
#zp       = nan_array()  # Euphotic depth Zp
bvf_avg   = nan_array()  # Average BVF in euphotic zone * 1.5
b_sst     = nan_array()  # Median temperature in upper 11m 0= SST
temp_ml   = nan_array()  # median temp above MLD
temp_dcm  = nan_array()  # temp at DCM depth

# Integrated POC data
poc_mod_surf_int    = nan_array()
poc_mod_sub_int     = nan_array()
poc_mod_total_int   = nan_array()
poc_mod_bk_int      = nan_array()
poc_mod_bk_1        = nan_array()
poc_data_int        = nan_array()
poc_data_surf_median = nan_array()
poc_data_dcm_conc    = nan_array()
poc_data_int_125    = nan_array()
#phytoC_int_discreet = nan_array()
phytoC_ratio_med    = nan_array()
# Integrated PON data
pon_mod_surf_int    = nan_array()
pon_mod_sub_int     = nan_array()
pon_mod_total_int   = nan_array()
pon_mod_bk_int      = nan_array()
pon_mod_bk_1        = nan_array()
pon_data_int        = nan_array()
pon_data_surf_median = nan_array()
pon_data_dcm_conc    = nan_array()
phytoN_int_discreet = nan_array()
phytoN_ratio_med    = nan_array()
# Integrated POP data
pop_mod_surf_int    = nan_array()
pop_mod_sub_int     = nan_array()
pop_mod_total_int   = nan_array()
pop_mod_bk_int      = nan_array()
pop_mod_bk_1        = nan_array()
pop_data_int        = nan_array()
pop_data_surf_median = nan_array()
pop_data_dcm_conc    = nan_array()
phytoP_int_discreet = nan_array()
phytoP_ratio_med    = nan_array()
# reconcile ratios from POC & POP prof dfs into one df with rest
C_Chl_ratio_C1_df  = nan_array()
C_Chl_ratio_C2_df  = nan_array()
N_Chl_ratio_C1_df  = nan_array()
N_Chl_ratio_C2_df  = nan_array()
P_Chl_ratio_C1_df  = nan_array()
P_Chl_ratio_C2_df  = nan_array()
C_N_molar_C1_df    = nan_array()
C_N_molar_C2_df    = nan_array()
C_N_molar_bk_df    = nan_array()
C_P_molar_C1_df    = nan_array()
C_P_molar_C2_df    = nan_array()
C_P_molar_bk_df    = nan_array()
N_P_molar_C1_df    = nan_array()
N_P_molar_C2_df    = nan_array()
N_P_molar_bk_df    = nan_array()

# Process each profile
for count, i in enumerate(len_list):
    #i=45
    # Extract CHL data
    where_idx = np.where(bottle_6.Cruise_ID == i)
    prof_chla, prof_depth = b_chla[where_idx], b_depth[where_idx]
    prof_chl_mod_surf, prof_chl_mod_sub, prof_chl_mod_total = CHL_mod_surface[where_idx], CHL_mod_subsurface[where_idx], CHL_mod_total[where_idx]

    # Extract CTD data
    where_ctd_idx = np.where(ctd.cruise_ID == i)
    prof_depth_ctd, prof_bvf, prof_temperature = depth[where_ctd_idx], bvf[where_ctd_idx], temperature[where_ctd_idx]

    # Extract POC data
    where_poc_idx = np.where(bottle_poc.Cruise_ID == i)
    prof_poc, prof_depth_poc = b2_poc[where_poc_idx], b2_depth[where_poc_idx]
    prof_poc_mod_bk = poc_mod_bk[where_poc_idx]
    prof_poc_mod_surf, prof_poc_mod_sub, prof_poc_mod_total = (poc_mod_surface[where_poc_idx], poc_mod_subsurface[where_poc_idx], poc_mod_total[where_poc_idx]) 
    prof_phytoC, prof_phytoC_ratio = phytoC[where_poc_idx], phytoC_ratio[where_poc_idx]
    # Extract PON data
    prof_pon = b2_pon[where_poc_idx]
    prof_pon_mod_bk = pon_mod_bk[where_poc_idx]
    prof_pon_mod_surf, prof_pon_mod_sub, prof_pon_mod_total = (pon_mod_surface[where_poc_idx], pon_mod_subsurface[where_poc_idx], pon_mod_total[where_poc_idx])
    prof_phytoN, prof_phytoN_ratio = phytoN[where_poc_idx], phytoN_ratio[where_poc_idx]
    # Extract POP data
    where_pop_idx = np.where(bottle_pop.Cruise_ID == i)
    prof_pop, prof_depth_pop = b3_pop[where_pop_idx], b3_depth[where_pop_idx]
    prof_pop_mod_bk = pop_mod_bk[where_pop_idx]
    prof_pop_mod_surf, prof_pop_mod_sub, prof_pop_mod_total = (pop_mod_surface[where_pop_idx], pop_mod_subsurface[where_pop_idx], pop_mod_total[where_pop_idx]) 
    prof_phytoP, prof_phytoP_ratio = phytoP[where_pop_idx], phytoP_ratio[where_pop_idx]

    # Extract Profile Ratios
    #C:Chl
    where_poc_idx2 = np.where(bottle_poc_prof.Cruise_ID == i)
    C_Chl_ratio_C1_prof, C_Chl_ratio_C2_prof = C_Chl_ratio_C1[where_poc_idx2], C_Chl_ratio_C2[where_poc_idx2]
    #C:N
    C_N_molar_C1_prof, C_N_molar_C2_prof, C_N_molar_bk_prof = C_N_molar_C1[where_poc_idx2], C_N_molar_C2[where_poc_idx2], bk_ratio_CN_molar[where_poc_idx2]
    N_Chl_ratio_C1_prof, N_Chl_ratio_C2_prof = N_Chl_ratio_C1[where_poc_idx2], N_Chl_ratio_C2[where_poc_idx2]
    #C:P
    where_pop_idx2 = np.where(bottle_pop_prof.Cruise_ID == i)
    P_Chl_ratio_C1_prof, P_Chl_ratio_C2_prof = P_Chl_ratio_C1[where_pop_idx2], P_Chl_ratio_C2[where_pop_idx2]
    C_P_molar_C1_prof, C_P_molar_C2_prof, C_P_molar_bk_prof = C_P_molar_C1[where_pop_idx2], C_P_molar_C2[where_pop_idx2], bk_ratio_CP_molar[where_pop_idx2]
    #N:P
    N_P_molar_C1_prof, N_P_molar_C2_prof, N_P_molar_bk_prof = N_P_molar_C1[where_pop_idx2], N_P_molar_C2[where_pop_idx2], bk_ratio_NP_molar[where_pop_idx2]

    # Extract DCM
    where_idx2 = np.where(bottle_prof.Cruise_ID == i)
    dcm_depth = b_DCM_depth[where_idx2]
    # Extract Zp
    where_zp_idx = np.where(Zp_Kd_df.Cruise_ID == i)
    Zp           = Zp_prof[where_zp_idx]
    
    # Extract MLD from ctd prof
    where_ctd_prof_idx = np.where(ctd_prof.Cruise_ID == i)
    prof_MLD           = MLD[where_ctd_prof_idx]
    
    # Integrate Chla Data Split
    chla_data_int_result         = integrate_sections(prof_depth, prof_chla, prof_MLD, Zp, dcm_depth = dcm_depth) #print(res['above_mld']), print(res['below_mld']), print(res['above_zp'])
    chla_data_surf_int[count]    = chla_data_int_result['above_mld']
    chla_data_sub_int[count]     = chla_data_int_result['below_mld']
    chla_data_int[count]         = chla_data_int_result['above_zp']
    chla_data_surf_median[count] = chla_data_int_result['median_above_mld']
    chla_data_dcm_conc[count]    = chla_data_int_result['conc_at_dcm']
    #chla_data_int_125[count]     = integrate_sections_125(prof_depth, prof_chla, prof_MLD, Zp, dcm_depth = dcm_depth)['above_125m']
    
    if i in ID_list_nut:
        # Extract Nutrient data
        where_nut_idx = np.where(bottle_nut.Cruise_ID == i)
        prof_depth_nut, prof_nitrate = nut_depth[where_nut_idx], nut_nitrate[where_nut_idx]
        # Extract Znitrate from nut prof
        where_nut_prof_idx = np.where(bottle_nut_prof.Cruise_ID == i)
        prof_Znitrate      = z_nitrate[where_nut_prof_idx]
        #prof_Zphosphate    = z_phosphate[where_nut_prof_idx]
        Znitrate[count]    = prof_Znitrate
        #Zphosphate[count]  = prof_Zphosphate
        # Integrate Nitrate Data Split
        nitrate_int_result         = integrate_sections(prof_depth_nut, prof_nitrate, prof_MLD, Zp, dcm_depth = dcm_depth) #print(res['above_mld']), print(res['below_mld']), print(res['above_zp'])
        nitrate_int_surf[count]    = nitrate_int_result['above_mld']
        nitrate_int_sub[count]     = nitrate_int_result['below_mld']
        nitrate_int_zp[count]      = nitrate_int_result['above_zp']
        nitrate_surf_median[count] = nitrate_int_result['median_above_mld']
        nitrate_dcm_conc[count]    = nitrate_int_result['conc_at_dcm']
        
# =============================================================================
#         # Integrate Phosphate Data Split
#         phosphate_int_result         = integrate_sections(prof_depth_nut, prof_phosphate, prof_MLD, Zp, dcm_depth = dcm_depth) #print(res['above_mld']), print(res['below_mld']), print(res['above_zp'])
#         phosphate_int_surf[count]    = phosphate_int_result['above_mld']
#         phosphate_int_sub[count]     = phosphate_int_result['below_mld']
#         phosphate_int_zp[count]      = phosphate_int_result['above_zp']
#         phosphate_surf_median[count] = phosphate_int_result['median_above_mld']
#         phosphate_dcm_conc[count]    = phosphate_int_result['conc_at_dcm']
# =============================================================================

    DecYear_prof[count]        = np.median(b_Decimal_year[where_idx])

    # Calculate BVF and temperature
    where_ctd_idx2 = np.where(prof_depth_ctd <= Zp * 1.5)
    bvf_avg[count] = np.nanmean(prof_bvf[where_ctd_idx2])
    where_ctd_idx3 = np.where(prof_depth_ctd <= 11)
    b_sst[count] = np.nanmedian(prof_temperature[where_ctd_idx3])
    
    # Calculate ML and DCM Temp
    temp_result       = integrate_sections(prof_depth_ctd, prof_temperature, prof_MLD, Zp, dcm_depth = dcm_depth) #print(res['above_mld']), print(res['below_mld']), print(res['above_zp'])
    temp_ml[count]    = temp_result['median_above_mld']
    temp_dcm[count]   = temp_result['conc_at_dcm']
    
    # Limit data to 1.5x euphotic depth for POC
    where_poc_idx2 = np.where(prof_depth_poc <= Zp * 1.5)
    prof_phytoC_ratio = prof_phytoC_ratio[where_poc_idx2]
    
    # Limit data to 1.5x euphotic depth for PON
    # No need to limit for pon depth as PON analysed on same sample
    prof_phytoN_ratio = prof_phytoN_ratio[where_poc_idx2]

    # Integrate POC data
    #if i in ID_list_poc:
    # Integrate Chla Data Split
    poc_data_int_result         = integrate_sections(prof_depth_poc, prof_poc, prof_MLD, Zp, dcm_depth = dcm_depth)
    poc_data_int[count]         = poc_data_int_result['above_zp']#spi.trapz(prof_poc, prof_depth_poc)
    poc_data_surf_median[count] = poc_data_int_result['median_above_mld']
    poc_data_dcm_conc[count]    = poc_data_int_result['conc_at_dcm']
    #poc_data_int_125[count]     = integrate_sections_125(prof_depth_poc, prof_poc, prof_MLD, Zp, dcm_depth = dcm_depth)['above_125m']

    poc_mod_bk_1[count]        = prof_poc_mod_bk[0]
    #phytoC_int_discreet[count] = spi.trapz(prof_phytoC, prof_depth_poc)
    phytoC_ratio_med[count]    = np.nanmedian(prof_phytoC_ratio)
    C_Chl_ratio_C1_df[count], C_Chl_ratio_C2_df[count] = C_Chl_ratio_C1_prof, C_Chl_ratio_C2_prof
    
    # Integrate PON data
    pon_data_int_result         = integrate_sections(prof_depth_poc, prof_pon, prof_MLD, Zp, dcm_depth = dcm_depth)
    pon_data_int[count]         = pon_data_int_result['above_zp']#spi.trapz(prof_poc, prof_depth_poc)
    pon_data_surf_median[count] = pon_data_int_result['median_above_mld']
    pon_data_dcm_conc[count]    = pon_data_int_result['conc_at_dcm']
    
    pon_mod_bk_1[count]        = prof_pon_mod_bk[0]
    #phytoN_int_discreet[count] = spi.trapz(prof_phytoN, prof_depth_poc)
    phytoN_ratio_med[count]    = np.nanmedian(prof_phytoN_ratio)
    N_Chl_ratio_C1_df[count], N_Chl_ratio_C2_df[count] = N_Chl_ratio_C1_prof, N_Chl_ratio_C2_prof
    C_N_molar_C1_df[count], C_N_molar_C2_df[count], C_N_molar_bk_df[count] = C_N_molar_C1_prof, C_N_molar_C2_prof, C_N_molar_bk_prof
    
    # Integrate POP Data
    if i in ID_list_pop:
        # Limit data to 1.5x euphotic depth for POP
        where_pop_idx2 = np.where(prof_depth_pop <= Zp * 1.5)
        prof_phytoP_ratio = prof_phytoP_ratio[where_pop_idx2]
        # Integrate
        pop_data_int_result         = integrate_sections(prof_depth_pop, prof_pop, prof_MLD, Zp, dcm_depth = dcm_depth)
        pop_data_int[count]         = pop_data_int_result['above_zp']#spi.trapz(prof_poc, prof_depth_poc)
        pop_data_surf_median[count] = pop_data_int_result['median_above_mld']
        pop_data_dcm_conc[count]    = pop_data_int_result['conc_at_dcm']

        pop_mod_bk_1[count]        = prof_pop_mod_bk[0]
        #phytoP_int_discreet[count] = spi.trapz(prof_phytoP, prof_depth_pop)
        phytoP_ratio_med[count]    = np.nanmedian(prof_phytoP_ratio)
        P_Chl_ratio_C1_df[count], P_Chl_ratio_C2_df[count] = P_Chl_ratio_C1_prof, P_Chl_ratio_C2_prof
        C_P_molar_C1_df[count], C_P_molar_C2_df[count], C_P_molar_bk_df[count]  = C_P_molar_C1_prof, C_P_molar_C2_prof, C_P_molar_bk_prof
        N_P_molar_C1_df[count], N_P_molar_C2_df[count], N_P_molar_bk_df[count] = N_P_molar_C1_prof, N_P_molar_C2_prof, N_P_molar_bk_prof

#%%

### SAVE INTEGRATED DATA TO BOTTLE DF & BOTTLE PROF ###

# Inspect bottle pigment prof df
bottle_prof.info() # column
print(len(ID_list_6))
print(len(bottle_prof))

# Add integrated CTD data - Note these are only for CTD profiles matching bottles, separate scriptto Compute SST for all CTD profiles
bottle_prof['BVF_avg']       = bvf_avg
bottle_prof['SST']           = b_sst
bottle_prof['Temp_ML']       = temp_ml
bottle_prof['Temp_DCM']      = temp_dcm

# Add integrated CHL
bottle_prof['Chla_data_Int']     = chla_data_int
#bottle_prof['Chla_data_Int_125'] = chla_data_int_125

bottle_prof['Chla_data_Surf_Int']  = chla_data_surf_int
bottle_prof['Chla_data_Sub_Int']   = chla_data_sub_int
bottle_prof['Chla_data_ML_median'] = chla_data_surf_median
bottle_prof['Chla_data_DCM_conc']  = chla_data_dcm_conc

# Integrate Nitrate Data Split
bottle_prof['Nitrate_Surf_Int']  = nitrate_int_surf
bottle_prof['Nitrate_Sub_Int']   = nitrate_int_sub
bottle_prof['Nitrate_Int']       = nitrate_int_zp
bottle_prof['Nitrate_ML_median'] = nitrate_surf_median
bottle_prof['Nitrate_DCM_conc']  = nitrate_dcm_conc
bottle_prof['Z_nitrate']         = Znitrate

# Integrate Phosphate Data Split
bottle_prof['Phosphate_Surf_Int']  = phosphate_int_surf
bottle_prof['Phosphate_Sub_Int']   = phosphate_int_sub
bottle_prof['Phosphate_Int']       = phosphate_int_zp
bottle_prof['Phosphate_ML_median'] = phosphate_surf_median
bottle_prof['Phosphate_DCM_conc']  = phosphate_dcm_conc
bottle_prof['Z_phosphate']         = Zphosphate

# Add integrated POC
bottle_prof['POC_data_int']      = poc_data_int/1000 # convert from mg/m2 to g/m2
#bottle_prof['POC_data_int_125']  = poc_data_int_125/1000
bottle_prof['POC_data_ML_median'] = poc_data_surf_median
bottle_prof['POC_data_DCM_conc']  = poc_data_dcm_conc

bottle_prof['POC_mod_bk']        = poc_mod_bk_1

bottle_prof['PhytoC_ratio_med']     = phytoC_ratio_med
#bottle_prof['PhytoC_ratio_int']     = phytoC_ratio_int

bottle_prof['C_Chl_ratio_C1'] = C_Chl_ratio_C1_df
bottle_prof['C_Chl_ratio_C2'] = C_Chl_ratio_C2_df

# Add integrated PON
bottle_prof['PON_data_int']       = pon_data_int/1000
bottle_prof['PON_data_ML_median'] = pon_data_surf_median
bottle_prof['PON_data_DCM_conc']  = pon_data_dcm_conc

bottle_prof['PON_mod_bk']         = pon_mod_bk_1

bottle_prof['PhytoN_ratio_med']     = phytoN_ratio_med
#bottle_prof['PhytoN_ratio_int']     = phytoN_ratio_int

bottle_prof['N_Chl_ratio_C1'] = N_Chl_ratio_C1_df
bottle_prof['N_Chl_ratio_C2'] = N_Chl_ratio_C2_df
bottle_prof['C_N_molar_C1'] = C_N_molar_C1_df
bottle_prof['C_N_molar_C2'] = C_N_molar_C2_df
bottle_prof['C_N_molar_bk'] = C_N_molar_bk_df

# Add integrated POP
bottle_prof['POP_data_int']       = pop_data_int
bottle_prof['POP_data_ML_median'] = pop_data_surf_median
bottle_prof['POP_data_DCM_conc']  = pop_data_dcm_conc

bottle_prof['POP_mod_bk']         = pop_mod_bk_1

bottle_prof['PhytoP_ratio_med']     = phytoP_ratio_med
#bottle_prof['PhytoP_ratio_int']     = phytoP_ratio_int

bottle_prof['P_Chl_ratio_C1'] = P_Chl_ratio_C1_df
bottle_prof['P_Chl_ratio_C2'] = P_Chl_ratio_C2_df
bottle_prof['C_P_molar_C1'] = C_P_molar_C1_df
bottle_prof['C_P_molar_C2'] = C_P_molar_C2_df
bottle_prof['C_P_molar_bk'] = C_P_molar_bk_df
bottle_prof['N_P_molar_C1'] = N_P_molar_C1_df
bottle_prof['N_P_molar_C2'] = N_P_molar_C2_df
bottle_prof['N_P_molar_bk'] = N_P_molar_bk_df

#Calc Particulate C:N:P Ratios and Molar conversion
C_N_ATOMIC_RATIO = 12.01 / 14.01
C_P_ATOMIC_RATIO = 12.01 / 30.97  
N_P_ATOMIC_RATIO = 14.01 / 30.97  
bottle_prof['Data_Part_C_N_molar_ML'] = bottle_prof['POC_data_ML_median']/bottle_prof['PON_data_ML_median']/C_N_ATOMIC_RATIO
bottle_prof['Data_Part_C_N_molar_DCM'] = bottle_prof['POC_data_DCM_conc']/bottle_prof['PON_data_DCM_conc']/C_N_ATOMIC_RATIO
bottle_prof['Data_Part_C_P_molar_ML'] = bottle_prof['POC_data_ML_median']/bottle_prof['POP_data_ML_median']/C_P_ATOMIC_RATIO
bottle_prof['Data_Part_C_P_molar_DCM'] = bottle_prof['POC_data_DCM_conc']/bottle_prof['POP_data_DCM_conc']/C_P_ATOMIC_RATIO
bottle_prof['Data_Part_N_P_molar_ML'] = bottle_prof['PON_data_ML_median']/bottle_prof['POP_data_ML_median']/N_P_ATOMIC_RATIO
bottle_prof['Data_Part_N_P_molar_DCM'] = bottle_prof['PON_data_DCM_conc']/bottle_prof['POP_data_DCM_conc']/N_P_ATOMIC_RATIO

bottle_prof.info()

b_year_prof = bottle_prof['yyyy'].values

# Save new dataframe with Model output to CSV
bottle_prof.to_csv('data/HOT_Bottle_profData_Int.csv')

