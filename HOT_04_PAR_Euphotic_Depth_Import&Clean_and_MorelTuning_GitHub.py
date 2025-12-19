"""
HOT: PAR Euphotic Depth Import & Clean from HOT-DOGS & Compute Morel

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@authors: Johan Viljoen - j.j.viljoen@exeter.ac.uk & Xuerong Sun
"""

#%%

### LOAD PACKAGES ###
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from PyAstronomy import pyasl
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from lmfit import Model

#%%
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

#%%

#Import Three Community Module and Functions
def date_span( start, end ):
    """
    Calculate the difference in years, months, and days between two dates.
    """
    from dateutil import relativedelta
    # Calculate the relative delta between the start and end dates
    rd = relativedelta.relativedelta(pd.to_datetime(end), pd.to_datetime(start))
    # Construct the string representing the duration in years, months, and days
    date_len = '{}y{}m{}d'.format(rd.years, rd.months, rd.days)
    # Return the formatted date length string
    return date_len

## Original Morel Zp KD function
def calculate_Kd_Zp(chla_surf):
    """
    Calculate Kd and Zp based on the given prof_chla_surf.
    Equation for Kd using Morel euphotic depth https://doi.org/10.1016/j.rse.2007.03.012

    Parameters:
    - prof_chla_surf (float): Chlorophyll-a concentration at the surface.

    Returns:
    - tuple: (Kd, Zp)
    """
    Kd = 4.6 / 10.**(1.524 - 0.436*np.log10(chla_surf) - \
                    0.0145*np.log10(chla_surf)**2. + 0.0186*np.log10(chla_surf)**3.)
    Zp = 4.6 / Kd
    return Kd, Zp
    """
    Kd_result, Zp_result = calculate_Kd_Zp(prof_chla_surf_value)
    """

#%%
### HOT-DOGS PAR Euphotic Depths

# File locations for both the PAR euphotic depth and the Mean Kd(PAR) Excel files
filenamePAR = 'data/hot_PAR_EuphoticDepth_16April2025.xlsx' # Euphotic as 1% PAR depth downloaded from HOT-DOGS
filenameKD = 'data/hot_PAR_MeanKd_16June2025.xlsx' # Mean Kd(PAR) 45-125m downloaded from HOT-DOGS same as used on HOT PAR data description for Karl et al. 2021

# Original CTD data from HOT-DOGS site at https://hahana.soest.hawaii.edu/hot/hot-dogs/prrseries.html
# Methods and original data here: https://hahana.soest.hawaii.edu/hot/methods/prr.html
# Data downloaded on 16 April 2025 and copied into an Excel xlsx file
# Data acknowledgement: "Data obtained via the Hawaii Ocean Time-series HOT-DOGS application; University of Hawai'i at Mānoa. National Science Foundation Award # 1756517".

#############################
### READ, CLEAN & FORMAT PAR DATA
#############################

# Load the PAR Excel file, skipping the first 2 rows
par_df = pd.read_excel(filenamePAR, skiprows=2)

# Split the data in the first column into separate columns based on commas
split_columns = par_df.iloc[:, 0].str.split(', ', expand=True)

# Replace empty strings (or cells with only spaces) with NaN
split_columns.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
# Drop columns that are entirely NaN
split_columns = split_columns.dropna(axis=1, how='all')

# Replace the original DataFrame with the cleaned (split) DataFrame
par_df = split_columns.copy()

# Set the first row as column headers then drop that row
par_df.columns = par_df.iloc[0]
par_df = par_df[1:]

# Optionally, extract units (if needed for reference) from the next row
column_names = par_df.columns.tolist()  # original column names
units = par_df.iloc[0].tolist()           # units row (if desired)
print("Columns with Units:")
print(list(zip(column_names, units)))

# Clean column names: remove any leading spaces
cleaned_column_names = [col.lstrip() for col in column_names]
par_df.columns = cleaned_column_names

# Remove the row containing the units
par_df = par_df[1:]

# Rename the relevant column to Zp_PAR_m
par_df.rename(columns={"depth": "Zp_PAR_m"}, inplace=True)

# Remove any leading spaces from all string cells in the DataFrame
par_df = par_df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)

# Convert Zp_PAR_m to numeric; remove negative signs to represent depth as positive values
par_df["Zp_PAR_m"] = pd.to_numeric(par_df["Zp_PAR_m"], errors="coerce").abs()

# Reset the index after cleaning
par_df.reset_index(drop=True, inplace=True)

#############################
### READ, CLEAN & FORMAT KD DATA
#############################

# Load the Kd Excel file, skipping the first 2 rows
kd_df = pd.read_excel(filenameKD, skiprows=2)

# Split the data in the first column into separate columns based on commas
split_columns_kd = kd_df.iloc[:, 0].str.split(', ', expand=True)

# Replace empty strings (or cells with only spaces) with NaN and drop all-NaN columns
split_columns_kd.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
split_columns_kd = split_columns_kd.dropna(axis=1, how='all')

# Update kd_df with the split DataFrame copy
kd_df = split_columns_kd.copy()

# Set the first row as header and drop that row from the DataFrame
kd_df.columns = kd_df.iloc[0]
kd_df = kd_df[1:]

# Clean the column names by removing leading spaces
kd_df.columns = [col.lstrip() for col in kd_df.columns]
# Remove the next row that may contain units
kd_df = kd_df[1:]

# Rename the column with euphotic depth information (assumed to be "depth") to "Kd_PAR"
kd_df.rename(columns={"mean": "Kd_PAR_mean"}, inplace=True)

# Remove leading spaces from string cells throughout the DataFrame
kd_df = kd_df.applymap(lambda x: x.lstrip() if isinstance(x, str) else x)
kd_df.info()

# Convert the Kd_PAR column to numeric (assuming these values don't need to be absolute)
kd_df["Kd_PAR_mean"] = pd.to_numeric(kd_df["Kd_PAR_mean"], errors="coerce")

# Reset the index after cleaning
kd_df.reset_index(drop=True, inplace=True)

#############################
### MERGE THE DATAFRAMES USING CRN
#############################

# Make sure the 'crn' column exists in both DataFrames.
# Merge on the 'crn' column to add the Kd_PAR_mean column to the par_df DataFrame.
final_df = pd.merge(par_df, kd_df[['crn', 'Kd_PAR_mean']], on='crn', how='left', indicator=False)
final_df['crn'] = pd.to_numeric(final_df['crn'], errors='coerce')

# Save the final merged DataFrame to a new CSV file
final_df.to_csv('data/hot_PAR_Zp_Kd_combined.csv', index=False)

# Output the first few rows of the merged DataFrame
print(final_df.head())


#%%

#############################
### CLEAN & FORMAT Further
#############################
# Inspect CTD df
final_df.info()
final_df.head()

par_Date = pd.to_datetime(final_df['date'].values).date

final_df['Date'] = par_Date
final_df.info()

### Convert and separate datetime data into new additional columns ###

# convert to datetime format & extract year into new column called "yyyy"
final_df['yyyy'] = pd.to_datetime(final_df['Date']).dt.year 

# convert to datetime format & extract month into new column called "mm"
final_df['mm'] = pd.to_datetime(final_df['Date']).dt.month

print(np.max(final_df['Date']))
print(np.max(final_df['crn']))

# Use vectorized operations to compute decimal years
par_DateTime = pd.to_datetime(final_df['Date'])
par_DecYear = par_DateTime.apply(pyasl.decimalYear).to_numpy()

# Assign the computed decimal years back to the DataFrame
final_df['DecYear'] = par_DecYear

final_df = final_df[["crn","Date",'DecYear', 'mm','yyyy','Zp_PAR_m','Kd_PAR_mean']]

# Rename Columns
final_df.rename(columns={"crn": "Cruise_ID"},inplace=True)

final_df.head()
final_df.info()

final_df.to_csv('data/hot_PAR_Zp_Kd_combined.csv', index=False)

# Print start and end dates of bottle data
print("PAR Dates: "+str(min(par_Date))+" to "+str(max(par_Date)))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(par_Date), max(par_Date))
print("Timespan: "+str(ctd_date_length))

print("Max Cruise: "+str(max(final_df['Cruise_ID'])))

#%%

#############################
### COMPUTE MOREL ZP & KD FROM SURF CHLA
#############################

### EXTRACT CLEANED PIGMENT BOTTLE DATA & MAKE CHLA ID LIST ###

# CSV filename for the pigment dataset
filename_1 = 'data/HOT_Pigments_Chla_Cleaned.csv'
# Load data from csv. "index_col=0" makes the first column the index.
bottle_6 = pd.read_csv(filename_1, index_col=0)
bottle_6.info()

# Sort by Cruise_ID and depth (make sure the depth column is numeric)
bottle_6 = bottle_6.sort_values(by=['Cruise_ID','depth'])

# Convert DateTime column to datetime format if needed
bottle_6['DateTime'] = pd.to_datetime(bottle_6['DateTime'])

### Cruise ID list for Chla ###
# Extract cruise_ID and remove duplicates
ID_list_6 = pd.unique(bottle_6['Cruise_ID'])

### Morel Zp & Kd loop ###
# Prepare an empty list to store results for each Cruise_ID.
results = []

# Loop over each unique Cruise_ID
for cruise_id in ID_list_6:
    # Subset the bottle data for the current cruise_id
    cruise_data = bottle_6[bottle_6['Cruise_ID'] == cruise_id]
    
    # Initialize sample_date in case no valid surface sample is found.
    sample_date = np.nan
    
    # Check if there is any nonzero Chla in the cruise data.
    if (cruise_data['Chla'] > 0).any():
        # Sort the surface Chla: choose the measurement with the smallest (shallowest) depth
        surf_data = cruise_data[cruise_data['Chla'] > 0].sort_values(by='depth')
        # Get the surface Chla from the shallowest valid measurement
        surf_chla = surf_data.iloc[0]['Chla']
        # Also, get the DateTime from this shallowest sample as the specific date
        sample_date = surf_data.iloc[0]['Date']
        # Calculate Morel Kd and Zp using the provided function
        Kd, Zp = calculate_Kd_Zp(surf_chla)
    else:
        # If no valid nonzero Chla is found, mark values as NaN
        surf_chla = np.nan
        Kd, Zp = np.nan, np.nan

    # Append the results for the current cruise to the list
    results.append({
        'Cruise_ID': cruise_id,
        'Date': sample_date,
        'surf_chla': surf_chla,
        'Kd_chl': Kd,
        'Zp_chl': Zp
    })

# Create a new DataFrame from the results list
morel_df = pd.DataFrame(results)

# Sort the new DataFrame by Cruise_ID and reset the index
morel_df = morel_df.sort_values(by='Cruise_ID').reset_index(drop=True)

# Display the first few rows of the new DataFrame
print(morel_df.head())

# Save the new DataFrame to a CSV file
morel_df.to_csv('data/Morel_Kd_Zp_by_Cruise.csv', index=False)

#%%

#############################
### MERGE HOT PAR & MOREL Zp & Kd
#############################

merged_df = pd.merge(final_df, morel_df, on='Cruise_ID', how='outer', indicator=True)
merged_df = merged_df.sort_values(by='Cruise_ID').reset_index(drop=True)

# save the new DataFrame to a CSV file
merged_df.to_csv('data/HOT_Merged_Measured_Morel_Zp_Kd.csv', index=False)


#%%

#############################
###  IMPORT MERGED HOT DATA
#############################

# CSV filename for dataset
filename_1 = 'data/HOT_Merged_Measured_Morel_Zp_Kd.csv'
# Load data from csv. "index_col=0" makes the first column the index.
merged_df = pd.read_csv(filename_1)
merged_df.info()



#%%

#############################
### PLOT PAR measured Zp vs MOREL Zp & Kd
#############################

# Create output folder if needed
import os
os.makedirs("plots", exist_ok=True)

# Filter rows with data on both sides
both_df = merged_df[merged_df['_merge'] == 'both'].copy()
both_df = both_df.sort_values(by='Cruise_ID').reset_index(drop=True)

#both_df.to_csv('data/HOT_PAR_measured_Zp_surfChl_matched.csv', index=False)

both_df.info()

# --- Line Plot: Zp comparison ---
plt.figure(figsize=(10, 5))
plt.plot(both_df['Cruise_ID'], both_df['Zp_PAR_m'], label='Zp_PAR_m', marker='o')
plt.plot(both_df['Cruise_ID'], both_df['Zp_chl'], label='Zp_chl', marker='x')
plt.xlabel('Cruise_ID')
plt.ylabel('Zp (m)')
plt.title('Line Plot: Zp_PAR_m vs Zp_chl')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/HOT_lineplot_Zp_PAR_vs_Zp_chl.jpeg', dpi=300)
plt.show()

# --- Line Plot: Kd comparison ---
plt.figure(figsize=(10, 5))
plt.plot(both_df['Cruise_ID'], both_df['Kd_PAR_mean'], label='Kd_PAR_mean', marker='o')
plt.plot(both_df['Cruise_ID'], both_df['Kd_chl'], label='Kd_chl', marker='x')
plt.xlabel('Cruise_ID')
plt.ylabel('Kd PAR (1/m)')
plt.title('Line Plot: Kd_PAR_mean vs Kd_chl')
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('plots/HOT_lineplot_Kd_PAR_vs_Kd_chl.jpeg', dpi=300)
plt.show()

# --- Function for scatter plots with Spearman ---
def plot_scatter_with_spearman(x, y, x_label, y_label, title, filename):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=both_df, x=x, y=y)
    
    rho, pval = spearmanr(both_df[x], both_df[y], nan_policy='omit')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}\nSpearman R = {rho:.2f}, p = {pval:.2g}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.jpeg', dpi=300)
    plt.show()

# --- Scatter Plot: Zp ---
plot_scatter_with_spearman(
    x='Zp_PAR_m',
    y='Zp_chl',
    x_label='Zp_PAR_m',
    y_label='Zp_chl',
    title='Scatter: Zp_PAR_m vs Zp_chl',
    filename='HOT_scatter_Zp_PAR_vs_Zp_chl'
)

# --- Scatter Plot: Kd ---
plot_scatter_with_spearman(
    x='Kd_PAR_mean',
    y='Kd_chl',
    x_label='Kd_PAR_mean',
    y_label='Kd_chl',
    title='Scatter: Kd_PAR_mean vs Kd_chl',
    filename='HOT_scatter_Kd_PAR_vs_Kd_chl'
)

#%%

#############################
###  IMPORT MERGED HOT DATA
#############################

# Updated Morel Tuning assisted by Xuerong Sun

data = pd.read_csv('data/HOT_Merged_Measured_Morel_Zp_Kd.csv')

data_model = data.dropna(subset=['Zp_PAR_m','surf_chla']).reset_index(drop=True)
data_model['date'] = pd.to_datetime(data_model['yyyy'].astype(int).astype(str) + '-' + data_model['mm'].astype(int).astype(str) + '-01')

#%% Morel https://doi.org/10.1016/j.rse.2007.03.012
def compute_Zeu(chla):
    X = np.log10(chla)
    log10_Zeu = 1.524 + -0.436 * X + -0.0145 * X**2 + 0.0186 * X**3
    Zeu = 10 ** log10_Zeu
   
    return Zeu

data_model['Zeu_Morel'] = compute_Zeu(data_model['surf_chla'])
merged_df.info()
merged_df['Zeu_Morel'] = compute_Zeu(merged_df['surf_chla'])
merged_df['Kd_Morel'] = 4.6/merged_df['Zeu_Morel']

#%% whole dataset
# Define the model
def log10_zeu_model(chla, a, b, c, d):
    X = np.log10(chla)
    return a + b * X + c * X**2 + d * X**3

# Prepare data
x_data = data_model['surf_chla']
y_data = np.log10(data_model['Zp_PAR_m'])

# Create and fit lmfit model
model = Model(log10_zeu_model)
params = model.make_params(a=1.524, b=-0.436, c=-0.0145, d=0.0186)

result = model.fit(y_data, params, chla=x_data)

# Print fitting report
print(result.fit_report())

# Extract fitted parameters
a_fit = result.params['a'].value
b_fit = result.params['b'].value
c_fit = result.params['c'].value
d_fit = result.params['d'].value

# Compute Zeu using fitted model
def compute_Zeu_tuned(chla):
    X = np.log10(chla)
    log10_Zeu = a_fit + b_fit * X + c_fit * X**2 + d_fit * X**3
    return 10 ** log10_Zeu

# Apply
data_model['Zeu_Morel_tuned'] = compute_Zeu_tuned(data_model['surf_chla'])
merged_df['Zeu_Morel_tuned']  = compute_Zeu_tuned(merged_df['surf_chla'])
merged_df['Kd_Morel_tuned']  = 4.6/merged_df['Zeu_Morel_tuned']

merged_df.info()

#%% summer and winter
# Create lmfit model
model = Model(log10_zeu_model)
initial_params = model.make_params(a=1.524, b=-0.436, c=-0.0145, d=0.0186)

# Masks for seasonal grouping
summer_mask = data_model['mm'].isin([7, 8, 9, 10, 11, 12])
winter_mask = ~summer_mask

# Log Zeu obs
log10_zeu_obs = np.log10(data_model['Zp_PAR_m'])

# Fit summer
result_summer = model.fit(log10_zeu_obs[summer_mask], initial_params, chla=data_model.loc[summer_mask, 'surf_chla'])
params_summer = [result_summer.params[k].value for k in ['a', 'b', 'c', 'd']]

# Fit winter
result_winter = model.fit(log10_zeu_obs[winter_mask], initial_params, chla=data_model.loc[winter_mask, 'surf_chla'])
params_winter = [result_winter.params[k].value for k in ['a', 'b', 'c', 'd']]

# Print results
print(result_summer.fit_report())
print(result_winter.fit_report())

# Compute Zeu row-wise
def compute_Zeu_tuned(chla_series, month_series, params_summer, params_winter):
    zeu_list = []
    for chla, month in zip(chla_series, month_series):
        X = np.log10(chla)
        if 7 <= month <= 12:
            a, b, c, d = params_summer
        else:
            a, b, c, d = params_winter
        log10_Zeu = a + b * X + c * X**2 + d * X**3
        zeu_list.append(10 ** log10_Zeu)
    return pd.Series(zeu_list, index=chla_series.index)

# Apply
data_model['Zeu_Morel_tuned_sw'] = compute_Zeu_tuned(data_model['surf_chla'], data_model['mm'],params_summer, params_winter)
merged_df['Zeu_Morel_tuned_sw']  = compute_Zeu_tuned(merged_df['surf_chla'], merged_df['mm'],params_summer, params_winter)
merged_df['Kd_Morel_tuned_sw']   = 4.6/merged_df['Zeu_Morel_tuned_sw']

merged_df.info()

#%%

#############################
### PLOT NEW Measured vs estimated Zp and Kds
#############################

merged_df.info()

# --- Zp Line Plot: Measured vs Original Morel vs Tuned2 Morel ---
zp_df = merged_df.copy()
zp_df = zp_df.sort_values('Cruise_ID').reset_index(drop=True)

plt.figure(figsize=(16,5))
plt.plot(zp_df['Cruise_ID'], zp_df['Zp_PAR_m'], marker='o', linestyle='-', color='red', label='Measured Zp_PAR_m')
plt.plot(zp_df['Cruise_ID'], zp_df['Zeu_Morel'], marker='s', linestyle='--', color='k', alpha=0.2, label='Morel 2007 Zp')
plt.plot(zp_df['Cruise_ID'], zp_df['Zeu_Morel_tuned'], marker='s', linestyle='--', color='b', alpha=0.5, label='Morel Zp Tuned')
plt.plot(zp_df['Cruise_ID'], zp_df['Zeu_Morel_tuned_sw'], marker='^', linestyle='-.', color='green', label='Morel Zp Tuned SW')
plt.xlabel('Cruise_ID', fontsize=18)
plt.ylabel('Zp (m)', fontsize=18)
plt.title('Zp: Measured vs Morel', fontsize=18)
plt.ylim(80,135)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('plots/HOT_lineplot_Zp_measured_morel_tuned_X.jpeg', dpi=300)
plt.show()


# --- Kd Line Plot: Measured vs Original Morel vs Tuned2 Morel ---
kd_df = merged_df.copy()
kd_df = kd_df.sort_values('Cruise_ID').reset_index(drop=True)

plt.figure(figsize=(16,5))
plt.plot(kd_df['Cruise_ID'], kd_df['Kd_PAR_mean'], marker='o', linestyle='-', color='blue', label='Measured Kd_PAR_mean')
plt.plot(kd_df['Cruise_ID'], kd_df['Kd_Morel'], marker='s', linestyle='--', color='k', label='Morel 2007 Kd')
plt.plot(kd_df['Cruise_ID'], kd_df['Kd_Morel_tuned'], marker='s', linestyle='--', color='green', label='Morel Kd Tuned')
plt.plot(kd_df['Cruise_ID'], kd_df['Kd_Morel_tuned_sw'], marker='^', linestyle='-.', color='red', label='Morel Kd Tuned SW')
plt.xlabel('Cruise_ID', fontsize=18)
plt.ylabel('Kd (1/m)', fontsize=18)
plt.title('Kd: Measured vs Morel', fontsize=18)
plt.xticks(rotation=90, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('plots/HOT_lineplot_Kd_measured_morel_tuned_X.jpeg', dpi=300)
plt.show()

#%%
def errortest_linear(x, y):
    z = pd.concat([x, y], axis = 1).dropna()
    [correlation, p] = stats.spearmanr(z.iloc[:,0],z.iloc[:,1])
    diff = z.iloc[:,1]-z.iloc[:,0]
    bias = sum(diff)/diff.count()
    mae = mean_absolute_error(z.iloc[:,0], z.iloc[:,1])
    #rmse = mean_squared_error(z.iloc[:,0], z.iloc[:,1], squared=False)
    rmse = np.sqrt(mean_squared_error(z.iloc[:,0], z.iloc[:,1]))
    n = len(z)
    slope, intercept, r_value, p_value, std_err = stats.linregress(z.iloc[:,0], z.iloc[:,1])
   
    result = correlation, p, bias, mae, rmse, n, slope, intercept
    return result

Zeu_Morel_error = errortest_linear(data_model['Zp_PAR_m'], data_model['Zeu_Morel'])
Zeu_Morel_tuned_error = errortest_linear(data_model['Zp_PAR_m'], data_model['Zeu_Morel_tuned'])
Zeu_Morel_tuned_sw_error = errortest_linear(data_model['Zp_PAR_m'], data_model['Zeu_Morel_tuned_sw'])

#%% 

### METHOD PAPER Figure S2 - ZP MOREL - Scatter

xx = np.arange(60,140,1)
fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=300)
ax.plot(xx, xx, 'r-')
ax.scatter(data_model['Zp_PAR_m'], data_model['Zeu_Morel'], marker='o', color='b', s=100, alpha=0.5, label='Morel et al. (2007)')
ax.scatter(data_model['Zp_PAR_m'], data_model['Zeu_Morel_tuned'], marker='s', color='orange', s=100, alpha=0.5, label='Morel HOT Tuned')
ax.scatter(data_model['Zp_PAR_m'], data_model['Zeu_Morel_tuned_sw'], marker='h', color='green', s=100, alpha=0.5, label='Morel HOT Tuned Season')
#ax.scatter(data_model['Zp_PAR_m'], data_model['Zeu_PAR_tuned_simple'], marker='h', color='r', s=100, alpha=0.5, label='HOT Tuned Morel 2')

ax.set_xlim([70,135])
ax.set_ylim([70,135])
ax.legend(loc='upper right', fontsize = 12)
ax.set_xlabel('In-situ $Z_{p}$', fontsize = 18)
ax.set_ylabel('Morel $Z_{p}$', fontsize = 18)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)

text = 'R = ' + str('%.3f' %Zeu_Morel_tuned_sw_error[0]) + \
        '\nSlope = ' + str('%.3f' %Zeu_Morel_tuned_sw_error[6]) + \
        '\nRMSD = '+ str('%.3f' %Zeu_Morel_tuned_sw_error[4]) + \
            '\nBias = ' + str('%.3f' %Zeu_Morel_tuned_sw_error[2]) + \
                '\nn = ' + str(int(Zeu_Morel_tuned_sw_error[5]))
ax.annotate(text, xy=(0, 1), xycoords='axes fraction', fontsize=12,
                xytext=(5, -5), textcoords='offset points',
                ha='left', va='top',color='green')

text = 'R = ' + str('%.3f' %Zeu_Morel_tuned_error[0]) + \
        '\nSlope = ' + str('%.3f' %Zeu_Morel_tuned_error[6]) + \
        '\nRMSD = '+ str('%.3f' %Zeu_Morel_tuned_error[4]) + \
            '\nBias = ' + str('%.3f' %Zeu_Morel_tuned_error[2]) + \
            '\nn = ' + str(int(Zeu_Morel_tuned_error[5]))
ax.annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=12,
                xytext=(-5, 5), textcoords='offset points',
                ha='right', va='bottom',color='orange')

# =============================================================================
# text = '$n$ = ' + str(int(Zeu_Morel_tuned_error[5])) + \
#     '\n$r$ = ' + str('%.3f' %Zeu_Morel_tuned_error[0]) + \
#         '\n$\delta$ = ' + str('%.3f' %Zeu_Morel_tuned_error[2]) + \
#         '\n$\epsilon$ = '+ str('%.3f' %Zeu_Morel_tuned_error[3]) + \
#             '\n$\psi$ = ' + str('%.3f' %Zeu_Morel_tuned_error[4])
# ax.annotate(text, xy=(1, 0), xycoords='axes fraction', fontsize=12,
#                 xytext=(-5, 5), textcoords='offset points',
#                 ha='right', va='bottom',color='orange')
# =============================================================================

fig.tight_layout()
fig.savefig('plots/HOT_Scatter_Zp_morel_tuned_X.jpeg', dpi=300)

#%%

#############################
### FINAL Zp and KD dataset
#############################

merged_df.info()

# Create final Date, Zp, and Kd columns, preferring measured values, but filling with tuned Morel where missing
merged_df['Date_final'] = merged_df['Date_x'].combine_first(merged_df['Date_y'])
merged_df['Zp_final'] = merged_df['Zp_PAR_m'].combine_first(merged_df['Zeu_Morel_tuned_sw'])
merged_df['Kd_final'] = merged_df['Kd_PAR_mean'].combine_first(merged_df['Kd_Morel_tuned_sw'])
merged_df['Kd_46Zp'] = 4.6/merged_df['Zp_final']

# Optional: view summary or inspect a few rows
print(merged_df[['Cruise_ID', 'Date_final', 'Zp_final', 'Kd_final']].head())
merged_df[['Cruise_ID', 'Date_final', 'Zp_final', 'Kd_final']].info()

# Make sure 'Date_final' is in datetime format
merged_df['Date'] = pd.to_datetime(merged_df['Date_final']).dt.date
df_DateTime = pd.to_datetime(merged_df['Date_final'])

# Extract year and month
merged_df['yyyy'] = pd.to_datetime(merged_df['Date']).dt.year
merged_df['mm'] = pd.to_datetime(merged_df['Date']).dt.month

# Compute decimal year using PyAstronomy
merged_df['DecYear'] = df_DateTime.apply(pyasl.decimalYear)

# Filter for final columns
final_Zp_Kd_df = merged_df[["Cruise_ID","Date",'DecYear', 'mm','yyyy','Zp_PAR_m','Kd_PAR_mean',
                            'surf_chla', 'Zeu_Morel','Kd_Morel','Zeu_Morel_tuned','Kd_Morel_tuned','Zeu_Morel_tuned_sw', 'Kd_Morel_tuned_sw','Zp_final','Kd_final','Kd_46Zp']]

final_Zp_Kd_df.info()

final_Zp_Kd_df.to_csv('data/HOT_PAR_Zp_Kd_final.csv', index=False)

# Print start and end dates of bottle data
print("PAR Dates: "+str(min(merged_df['Date']))+" to "+str(max(merged_df['Date'])))

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
ctd_date_length = date_span(min(df_DateTime), max(df_DateTime))
print("Timespan: "+str(ctd_date_length))

print("Max Cruise: "+str(max(final_Zp_Kd_df['Cruise_ID'])))

