"""
HOT - Example Fit Two-Community Particulate Model to Chla & POC data

Model developed and modified based on Brewin et al. (2022) Two-Community model and
the extended version used in Viljoen et al. (2024).
Brewin et al. (2022) - https://doi.org/10.1029/2021JC018195
Viljoen et al. (2024) - https://www.nature.com/articles/s41558-024-02136-6

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%
### LOAD PACKAGES ##
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data array
import datetime
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial" # Set font for all plots
from lmfit import report_fit
# Supress
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore');

### IMPORT TWO-COMMUNITY MODEL & CUSTOM FUNCTIONS ###

#Import Three Community Module and Functions
from MODULE_2community_particulate_model_functions_Dec2025 import two_community_model, particulate_model

#%%
### APPLY MODEL TO SINGLE PROFILE ###
ID_1 = 328

# Mixed Layer Depth
prof_MLD = np.array([33.78])
# PAR Euphotic Depth
Zp = np.array([107.03])
# PAR Kd
Kd = np.array([0.0429785])

# Chlorophyll-a profile (µg/L) = mg/m3
prof_chla = np.array([0.103, 0.107, 0.126, 0.142, 0.18 , 0.225, 0.333, 0.429, 0.299,
                      0.158, 0.118, 0.05], dtype=float)

# Depths corresponding to chlorophyll profile (m)
prof_depth = np.array([  5.16729436,  25.63644747,  44.01719502,  60.11141374,
                       74.81366887,  84.94569016,  98.65292181, 114.54421717,
                       124.7735871 , 134.80383317, 148.50773321, 174.12561407], dtype=float)

# Date associated with Chla profile
prof_chla_date = datetime.date(2021, 3, 25)

# POC (mg/m3) and depth data
prof_poc = np.array([24.3763245 , 24.25321175, 24.74566275, 24.00698625, 25.8536775 ,
                     20.190491  , 11.94193675,  5.909412  ,  6.27875025,  7.50987775,
                     5.1707355], dtype=float)

# PON (mg/m3) and depth data
prof_pon = np.array([4.1348664 , 3.93386595, 4.06308052, 4.39329555, 4.8240108 ,
                     3.97693748, 2.2397193 , 1.19164553, 1.148574  , 1.2634314 ,
                     0.89014485], dtype=float)

# Depths corresponding to POC/PON (m)
prof_poc_depth = np.array([ 5.66414271,  24.74220685,  44.71265037,  72.62826526,
                           96.66642329, 123.48252971, 146.81962103, 171.94126827,
                           195.86851705, 245.20346673, 348.10370472], dtype=float)

# POP data

# POP (mg/m3) and depth data
prof_pop = np.array([0.45082308, 0.49209561, 0.51114447, 0.51114447, 0.50162004,
                     0.44129865, 0.14921609, 0.13016723, 0.10159393, 0.07937026], dtype=float)

# Depths corresponding to POP (m)
prof_pop_depth = np.array([ 5.66414271,  24.94092732,  44.71265037,  72.62826526,
                           96.66642329, 123.18459223, 171.94126827, 195.76923961,
                           245.00495947, 348.10370472], dtype=float)

"FILTER DATA for NANs and Depth limit BEFORE USE IN MODEL"

# --- Filter Chlorophyll-a data ---
# Remove NaN values from Chl-a profile and corresponding depths
valid_chla_mask = ~np.isnan(prof_chla)
prof_chla = prof_chla[valid_chla_mask]
prof_chla_depth = prof_depth[valid_chla_mask]

# --- Filter POC data up to 470 m ---
# Remove NaN values and restrict POC profile to depths ≤ 470 m
valid_poc_mask = (~np.isnan(prof_poc)) & (prof_poc_depth <= 470)
prof_poc = prof_poc[valid_poc_mask]
prof_poc_depth = prof_poc_depth[valid_poc_mask]
valid_pon_mask = (~np.isnan(prof_pon)) & (prof_poc_depth <= 470)
prof_pon = prof_pon[valid_pon_mask]

# --- Filter POP data up to 470 m ---
# Remove NaN values and restrict POC profile to depths ≤ 470 m
valid_pop_mask = (~np.isnan(prof_pop)) & (prof_pop_depth <= 470)
prof_pop = prof_pop[valid_pop_mask]
prof_pop_depth = prof_pop_depth[valid_pop_mask]

"FIT CHL & POC TWO-COMMUNITY MODEL"

# Fit model
fit_result = two_community_model(prof_chla, prof_depth, prof_MLD, prof_poc, prof_poc_depth,
                                   data_type = 'bottle', Kd = Kd) # If Kd not given, estimated via Morel surface chl equation
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

"FIT to PON"

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


"FIT to POP"

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
fig.savefig('HOT_ModelFit_Chl_POC_PON_POP_EXAMPLE.jpeg', dpi=300, bbox_inches='tight')
plt.show()


