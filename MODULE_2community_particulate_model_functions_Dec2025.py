"""
Functions for Two-Community Model Fit to HPLC Chla, POC and other particulate data for stoichiometry

Model developed and modified based on Brewin et al. (2022) Two-Community model and
the extended version used in Viljoen et al. (2024).
Brewin et al. (2022) - https://doi.org/10.1029/2021JC018195
Viljoen et al. (2024) - https://www.nature.com/articles/s41558-024-02136-6

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 17 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%
### LOAD PACKAGES ##
#General Python Packages
import pandas as pd # data analysis and manipulation tool
import numpy as np # used to work with data arrays
# Import specific modules from packages
from dateutil import relativedelta
from math import nan
from numpy import trapz
from lmfit import Minimizer, Parameters
# Supress
import warnings
warnings.filterwarnings("ignore") # Added to remove the warning "UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray." on 2nd to last cell of code
# Supresses outputs when trying to "divide by zero" or "divide by NaN" or "treatment for floating-point overflow"
np.seterr(all='ignore');
#%%

### DEFINE ALL FUNCTIONS ###

## Define function to calculate Morel diffuse attenuation coefficient
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

# DEFINE INTEGRATION FUNCTION

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


# Define function to compute time span between dates
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

# Define function for Community 1 normalised Chl-a
# Equation 8 of Brewin et al. 2022 Model
def fcn2min_1pop(params1, X, Y):
    P1 = params1['P1']
    P2 = params1['P2']
    model = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2))) #Reduce equation to two parameters [p1 and p2] (as depth tends to zero)
    return(model-X)

# Define function for Community 1 and 2 normalised Chl-a
# Equation 7 of Brewin et al. 2022 Model
def fcn2min_2pop(params2, X, Y):
    P1 = params2['P1']
    P2 = params2['P2']
    P3 = params2['P3'] # maximum biomass (normalised)
    P4 = params2['P4'] # psuedo parameter controlling depth of maximum biomas (P3)
    P5 = params2['P5'] # width of peak
    MLD_pop = 1 - 1./(1+np.exp(-(P1/P2)*(Y-P2))) #
    DCM_pop = P3*np.exp(-((Y - ((P4+(P5*3.))))/(P5))**2.)  
    model = MLD_pop + DCM_pop
    return(model-X)

# Define function for POC scaling factor for Community 1 (Surface only)
def fcn2min_1pop_poc(params3,X1,A1):
    # Input parameters
    P1 = params3['P1'] # POC C1 ratio
    P3 = params3['P3'] # POC background
    model = P1*A1 + P3
    return(model-X1)

# Define function for POC scaling factors for 2 communities (Surface & Subsurface)
def fcn2min_2pop_poc(params3,X1,A1,A2):
    # Input parameters
    P1 = params3['P1'] # POC C1 ratio
    P2 = params3['P2'] # POC C2 ratio
    P3 = params3['P3'] # POC background
    model = P1*A1 + P2*A2 + P3
    return(model-X1)

# Define function for normalised POC scaling factor for Community 1 (Surface only)
def fcn2min_1pop_poc_norm(params3,X1,A1):
    # Input parameters
    P3 = params3['P3']
    model = (1-P3)*A1 + P3
    return(model-X1)

# Define function for normalised POC scaling factors for 2 communities (Surface & Subsurface)
def fcn2min_2pop_poc_norm(params3,X1,A1,A2):
    # Input parameters
    P2 = params3['P2']
    P3 = params3['P3']
    model = (1-P3)*A1 + P2*A2 + P3
    return(model-X1)

# Define 3rd step function for getting sigmoid parameters
def fcn2min_2pop_3rd_step2(params2, X2, Y2):
    P2 = params2['P2'] # Tau 1
    P3 = params2['P3'] # maximum biomass (normalised)
    P4 = params2['P4'] # psuedo parameter controlling depth of maximum biomas (P3)
    P5 = params2['P5'] # width of peak
    P6 = params2['P6'] # Com 2 weight for POC
    P7 = params2['P7'] # Background
    P1 = 10**(0.08 * P2 + 0.66) # Re
    #P8 = params2['P8'] # Com 1 weight for POC
    # Chl model
    A1 = 1 - 1./(1+np.exp(-(P1/P2)*(Y2-P2))) #
    A2 = P3*np.exp(-((Y2 - ((P4+(P5*3.))))/(P5))**2.) 
    # POC model
    model = (1-P7)*A1 + P6*A2 + P7#P8*A1 + P6*A2 + P7
    return(model-X2)

# Define a sigmoid function for reuse
def sigmoid(p, tau, opt_dim):
    return 1 - 1. / (1 + np.exp(-(p / tau) * (opt_dim - tau)))

# Define a Gaussian function for reuse
def gaussian(bm, opt_dim, tau, sigma):
    return bm * np.exp(-((opt_dim - tau) / sigma) ** 2)

def particulate_model(prof_poc, prof_poc_depth, prof_chla_surf, Kd,
                      P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final,
                      background=1, surf_norm=False,prof_MLD = None):
    FitResults = {}
    #ASD = np.where(prof_poc_depth <= 1 / Kd) # removed requirement for to have at least one POC measurement in 1st optical depth
    # Surface POC normalization control
    if surf_norm: # Trying to add ability of particulate function to do both direct and surface normalised fitting , but will complete and add in next version
        
        if prof_MLD < prof_poc_depth[0]:
            SURF_POC = prof_poc[0]
        else:
            ASD      = np.where(prof_poc_depth <= prof_MLD)
            SURF_POC = np.median(prof_poc[ASD])
        prof_poc_NOM = prof_poc / SURF_POC
    else:
        prof_poc_NOM = prof_poc
        #SURF_POC = 1  # If not normalized, surface value doesn't scale results
        SURF_POC = prof_poc[0]
    
    # POC Optical depth
    OPT_DIM_POC = prof_poc_depth * Kd
    OPT_DIM2    = np.linspace(0,30,2000)
    prof_depth2  = OPT_DIM2/Kd
    
    ##### POC model for 1 pop #####
    if np.isnan(BM2_final):
        MLD_pop_FIT_CHL_POC = (1 - 1. / (1 + np.exp(-(P1_final / TAU1_final) * (OPT_DIM_POC - TAU1_final))))*prof_chla_surf
        MLD_pop_FIT2_POC0 = (1 - 1. / (1 + np.exp(-(P1_final / TAU1_final) * (OPT_DIM2 - TAU1_final))))*prof_chla_surf
        DCM_pop_FIT_CHL_POC = np.full_like(MLD_pop_FIT_CHL_POC, np.nan)
        
        # Fit the 1st population
        params3 = Parameters()
        params3.add('P1', value=0.3, min=0.0) #C1 scaling initial guess
        #params3.add('P3', value=0.2 if background else 0, min=0.01 if background else 0, max=0.95 if background else 0)
        if background:
            params3.add('P3', value=0.2, min=0.0)
        else:
            params3.add('P3', value=0, vary=False)
        X1 = prof_poc#prof_poc_NOM
        A1 = MLD_pop_FIT_CHL_POC # Chl function with POC depths
        out = Minimizer(fcn2min_1pop_poc, params3, fcn_args=(X1, A1))
        particulate_fit_result = out.minimize(method='powell')
        
        # Extract fitted parameters
        background_np  = particulate_fit_result.params['P3'].value if background else 0 #Background
        P_Chl_ratio_C1 = particulate_fit_result.params['P1'].value # C1 scaling  #1 - P3_POC_RATE
        
        # Calculate the model fit
        MLD_pop_FIT_POC = (MLD_pop_FIT_CHL_POC * P_Chl_ratio_C1) #* SURF_POC
        DCM_pop_FIT_POC = ((MLD_pop_FIT_POC * 0))#*SURF_POC
        BACKGROUND_POC  = ((MLD_pop_FIT_CHL_POC * 0) + background_np)#*SURF_POC
        TOTAL_POC_MODEL = MLD_pop_FIT_POC + BACKGROUND_POC
        
        MLD_pop_FIT2_POC = (MLD_pop_FIT2_POC0 * P_Chl_ratio_C1) #* SURF_POC
        DCM_pop_FIT2_POC = ((MLD_pop_FIT2_POC * 0))#*SURF_POC
        BACKGROUND2_POC  = ((MLD_pop_FIT2_POC0 * 0) + background_np)#*SURF_POC
        POC_model_fit2 = MLD_pop_FIT2_POC + BACKGROUND2_POC
        
        # Keep below for future added Surface normalise fit to calculate Chl ratios
        #CHL_POC_C1 = P_Chl_ratio_C1 #/ prof_chla_surf #/ SURF_POC)
        #CHL_POC_C2 = np.nan
        
        print("Particulate 1 Community Fit")
        
        FitResults['fit_type'] = '1pop'
        # store fit result stats
        FitResults['FitReport']        = particulate_fit_result
        FitResults['FitReport_chisq']  = particulate_fit_result.chisqr
        FitResults['FitReport_redchi'] = particulate_fit_result.redchi
        FitResults['FitReport_aic']    = particulate_fit_result.aic
        FitResults['FitReport_bic']    = particulate_fit_result.bic
        FitResults['FitReport_P1_err'] = particulate_fit_result.params['P1'].stderr
        FitResults['FitReport_P2_err'] = np.nan
        FitResults['FitReport_P3_err'] = particulate_fit_result.params['P3'].stderr
    
    ##### POC model for 2 pop #####
    else:
        MLD_pop_FIT_CHL_POC = (1 - 1. / (1 + np.exp(-(P1_final / TAU1_final) * (OPT_DIM_POC - TAU1_final))))*prof_chla_surf
        DCM_pop_FIT_CHL_POC = (BM2_final * np.exp(-((OPT_DIM_POC - TAU2_final) / SIG2_final) ** 2.))*prof_chla_surf
        MLD_pop_FIT2_POC0 = (1 - 1. / (1 + np.exp(-(P1_final / TAU1_final) * (OPT_DIM2 - TAU1_final))))*prof_chla_surf
        DCM_pop_FIT2_POC0 = (BM2_final * np.exp(-((OPT_DIM2 - TAU2_final) / SIG2_final) ** 2.))*prof_chla_surf
        
        params3 = Parameters()
        params3.add('P1', value=3, min=0)
        params3.add('P2', value=3, min=0)
        #params3.add('P3', value=0.2 if background else 0, min=0.01 if background else 0, max=0.95 if background else 0)
        if background:
            params3.add('P3', value=8, min=0)
        else:
            params3.add('P3', value=0, vary=False)
        X1 = prof_poc#prof_poc_NOM
        A1 = MLD_pop_FIT_CHL_POC
        A2 = DCM_pop_FIT_CHL_POC
        out = Minimizer(fcn2min_2pop_poc, params3, fcn_args=(X1, A1, A2))
        particulate_fit_result = out.minimize(method='powell')
        
        P_Chl_ratio_C1 = particulate_fit_result.params['P1'].value# C1 scaling factor
        P_Chl_ratio_C2 = particulate_fit_result.params['P2'].value# C2 scaling factor
        background_np  = particulate_fit_result.params['P3'].value# Background
        
        MLD_pop_FIT_POC = (MLD_pop_FIT_CHL_POC * P_Chl_ratio_C1) #* SURF_POC
        DCM_pop_FIT_POC = (DCM_pop_FIT_CHL_POC * P_Chl_ratio_C2) #* SURF_POC
        BACKGROUND_POC  = ((DCM_pop_FIT_CHL_POC * 0) + background_np)#*SURF_POC
        TOTAL_POC_MODEL = MLD_pop_FIT_POC + DCM_pop_FIT_POC + BACKGROUND_POC
        
        MLD_pop_FIT2_POC = (MLD_pop_FIT2_POC0 * P_Chl_ratio_C1) #* SURF_POC
        DCM_pop_FIT2_POC = (DCM_pop_FIT2_POC0 * P_Chl_ratio_C2) #* SURF_POC
        BACKGROUND2_POC  = ((DCM_pop_FIT2_POC * 0) + background_np)#*SURF_POC
        POC_model_fit2 = MLD_pop_FIT2_POC + DCM_pop_FIT2_POC + BACKGROUND2_POC
        
        # Keep below for future added Surface normalise fit to calculate Chl ratios
        #CHL_POC_C1 = P_Chl_ratio_C1 #/ (prof_chla_surf / SURF_POC)
        #CHL_POC_C2 = P_Chl_ratio_C2 #/ (prof_chla_surf / SURF_POC)
        
        print("Particulate 2 Community Fit")
        
        FitResults['fit_type'] = '2pop'
        # Store Fit Result Stats
        FitResults['FitReport']        = particulate_fit_result
        FitResults['FitReport_chisq']  = particulate_fit_result.chisqr
        FitResults['FitReport_redchi'] = particulate_fit_result.redchi
        FitResults['FitReport_aic']    = particulate_fit_result.aic
        FitResults['FitReport_bic']    = particulate_fit_result.bic
        FitResults['FitReport_P1_err'] = particulate_fit_result.params['P1'].stderr
        FitResults['FitReport_P2_err'] = particulate_fit_result.params['P2'].stderr
        FitResults['FitReport_P3_err'] = particulate_fit_result.params['P3'].stderr
    
    FitResults['prof_depth_HiRes'] = prof_depth2
    FitResults['surface_value']    = SURF_POC
    FitResults['C1_fit']           = MLD_pop_FIT_POC
    FitResults['C2_fit']           = DCM_pop_FIT_POC
    FitResults['background']       = BACKGROUND_POC
    FitResults['Total_fit']        = TOTAL_POC_MODEL
    FitResults['Phyto_fit']        = MLD_pop_FIT_POC + DCM_pop_FIT_POC
    FitResults['C1_fit_HiRes']     = MLD_pop_FIT2_POC
    FitResults['C2_fit_HiRes']     = DCM_pop_FIT2_POC
    FitResults['background_HiRes'] = BACKGROUND2_POC
    FitResults['Total_fit_HiRes']  = POC_model_fit2
    FitResults['Phyto_fit_HiRes']  = MLD_pop_FIT2_POC + DCM_pop_FIT2_POC
    FitResults['P_Chl_ratio_C1']   = P_Chl_ratio_C1
    FitResults['P_Chl_ratio_C2']   = P_Chl_ratio_C2 if 'P_Chl_ratio_C2' in locals() else np.nan
    FitResults['background_np']    = background_np
    #FitResults['P4_POC_RATE']      = P4_POC_RATE if 'P4_POC_RATE' in locals() else np.nan
    
    return FitResults

#%%

def two_community_model(prof_chla, prof_depth, prof_MLD, prof_poc, prof_poc_depth, 
                        Kd = None, data_type = None,c1_R2 = 0.9, use_aic = True):
    """
    Perform chlorophyll fitting using a model with multiple populations.

    Parameters:
    - prof_chla: Chl array
    - prof_depth: Chl depths array
    - prof_MLD: Mixed layer depth
    - prof_poc: POC data
    - prof_poc_depth: POC or BBP depth data
    - Kd: Optional, Kd(PAR) diffuse attenuation coefficient
    - c1_R2: correlation required to fit only surface community. Default = 0.9
    - use_aic: Apply lowest aic result for two community fit. Default = True

    Returns:
    - Dictionary containing fitted chlorophyll and POC parameters and output
    """
    # Data type & length
    if data_type == 'argo':
        data_length = 10                 
    else:
        data_length = 6

    ### FIND SURFACE CHL ### 
    surf_chl_index = np.where(prof_depth == np.min(prof_depth[np.nonzero(prof_chla)]))
    prof_chla_surf = prof_chla[surf_chl_index]
    
    # Check if Kd is given, if not calculate Morel Kd based on surface chla
    if Kd == None:
    # Compute Morel Kd and Zp from surface chl-a 
        Kd, Zp = calculate_Kd_Zp(prof_chla_surf)
    else:
        Zp = np.nan
    
    # Dimensionalise profiles
    CHL_DIM = prof_chla / prof_chla_surf
    OPT_DIM = prof_depth * Kd
    MLD_OD = prof_MLD * Kd

    if len(CHL_DIM) >= data_length: # Chl profile should have at least 6 measurements
        # Fit 1st population - Surface
        params1 = Parameters()
        params1.add('P1', value=9., min=4.6, max=100)
        params1.add('P2', value=MLD_OD[0])
        out = Minimizer(fcn2min_1pop, params1, fcn_args=(CHL_DIM, OPT_DIM))
        fit_result_C1 = out.minimize(method='powell')
        #report_fit(fit_result_C1) #uncomment if want to see results of fit

        P1_FIT = fit_result_C1.params['P1'].value
        P2_FIT = fit_result_C1.params['P2'].value
        C1_P1   = P1_FIT
        C1_TAU1 = P2_FIT
        AIC_FIT1 = fit_result_C1.aic
        MLD_pop  = 1 - 1./(1+np.exp(-(P1_FIT/P2_FIT)*(OPT_DIM-P2_FIT)))
        
        # Correlation of Sigmoid vs Chl data
        r_corr   = np.corrcoef(CHL_DIM, MLD_pop)
        # Calculate r2 - Coefficient of determination - Square of Pearson R
        r2_corr  = r_corr[1,0]**2
        
        # Test if Sigmoid explain 90% portion of variance in Chl data
        if r2_corr >= c1_R2: 
            # assign nan to other parameters if sigmoid explain > 90%
            P3_FIT = nan
            P4_FIT = nan
            P5_FIT = nan

        else:
        ### FIT TWO COMMUNITIES ###
     
            #### FIT TWO COMMUNITY Red Sea ####
                   
            ### Max of DCM 1
            DCM1_MAX   = np.max(CHL_DIM)
            ads = np.where(CHL_DIM == np.max(CHL_DIM))
            DCM1_DEPTH = OPT_DIM[ads]  ###divide by three to account for nature of equation
            DCM1_DEPTH = DCM1_DEPTH[0]/3
            
            # Fit1 two Community
            params2 = Parameters()
            # Estimate Tau1 and P1 from Optical depth of mixed layer from Red Sea relationship
            Tau1_temp = (MLD_OD[0]*0.62)+2.296 # RedSea
            P1_temp   = 10**(0.08 * Tau1_temp + 0.66) # RedSea
            params2.add('P1', value=P1_temp, vary=False) #Fixed
            params2.add('P2', value=Tau1_temp, vary=False) #Fixed
            params2.add('P3', value=DCM1_MAX, min = 0.0, max = 100.0)
            params2.add('P4', value=DCM1_DEPTH, min = 0.0, max = 10)
            params2.add('P5', value=1.0, min = 0.0)
            res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(CHL_DIM, OPT_DIM))
            result_3 = res.minimize(method = 'powell')
            
            P1_FIT = result_3.params['P1'].value
            P2_FIT = result_3.params['P2'].value
            P3_FIT = result_3.params['P3'].value
            P4_FIT = result_3.params['P4'].value
            P5_FIT = result_3.params['P5'].value
            
            # Assign temporary/intermediate parameters for two communities
            P1_TEMP_2   = P1_FIT                                                                                                         
            TAU1_TEMP_2 = P2_FIT
            BM2_TEMP_2  = P3_FIT
            TAU2_TEMP_2 = P4_FIT + P5_FIT * 3.0
            SIG2_TEMP_2 = P5_FIT
            
            ### 3RD STEP: 2nd CHLOROPHYLL FIT USING POC TAU1 ###
            
            ### First Fit Initial POC

            #if np.min(prof_poc_depth) <= 1 / Kd:
            #Get surface POC
            #ASD      = np.where(prof_poc_depth <= 1/Kd)
            if prof_MLD < prof_poc_depth[0]:
                SURF_POC = prof_poc[0]
            else:
                ASD      = np.where(prof_poc_depth <= prof_MLD)
                SURF_POC = np.median(prof_poc[ASD])
            prof_poc_NOM   = prof_poc/SURF_POC
            # POC Optical depth
            OPT_DIM_POC    = prof_poc_depth*Kd

            ##### Fit 3rd Step to Two communities #####
            
            # POC model for 2 communities
            MLD_pop_FIT_CHL_POC  = (1 - 1./(1+np.exp(-(P1_TEMP_2/TAU1_TEMP_2)*(OPT_DIM_POC-TAU1_TEMP_2))))#*prof_chla_surf
            DCM_pop_FIT_CHL_POC  = (BM2_TEMP_2*np.exp(-((OPT_DIM_POC - TAU2_TEMP_2)/SIG2_TEMP_2)**2.))#*prof_chla_surf
            #TOTAL_CHL_POC        = MLD_pop_FIT_CHL_POC + DCM_pop_FIT_CHL_POC
            ###Fit 1st population
            params3  = Parameters()
            #params3.add('P1', value= 0.3, min = 0.01, max = 0.95)
            params3.add('P2', value= 0.3, min = 0.01, max = 0.95)
            params3.add('P3', value= 0.2, min = 0.01, max = 0.95)
            X11 = prof_poc_NOM
            A1 = MLD_pop_FIT_CHL_POC
            A2 = DCM_pop_FIT_CHL_POC            
            out = Minimizer(fcn2min_2pop_poc_norm, params3, fcn_args=(X11, A1, A2))
            result_4   = out.minimize(method = 'powell')
            #report_fit(result_4)
            P2_POC_RATE   = result_4.params['P2'].value
            P3_POC_RATE   = result_4.params['P3'].value
            #P1_POC_RATE   = 1 - P3_POC_RATE
            
            ### Optimise Tau1 using POC scaling factors
            params1  = Parameters()             
            tau1_max3 = TAU2_TEMP_2 - SIG2_TEMP_2*0.6
            params1.add('P2', value=TAU1_TEMP_2, min = MLD_OD[0], max = tau1_max3) # Tau1 still varies
            params1.add('P3', value=P3_FIT, vary=False)
            params1.add('P4', value=P4_FIT, vary=False)
            params1.add('P5', value=P5_FIT, vary=False)
            params1.add('P6', value=P2_POC_RATE, vary=False)
            params1.add('P7', value=P3_POC_RATE, vary=False)
            #params1.add('P8', value=P1_POC_RATE, vary=False)
            Y2 = OPT_DIM_POC
            X2 = prof_poc_NOM
            res      = Minimizer(fcn2min_2pop_3rd_step2,  params1, fcn_args=(X2, Y2))
            fit_result_1 = res.minimize(method = 'powell')
            #report_fit(fit_result_1) ##uncomment if you want to see results 
            P2_FIT_3STEP  = fit_result_1.params['P2'].value # Tau1 still varies
            P1_FIT_3STEP  =  10**(0.08 * P2_FIT_3STEP + 0.66) # Red Sea P1 Tau1 Relationship
            
            ### Fit Chl Again with new Fixed P1 and Tau1 from POC Fit
            params2  = Parameters()
            Tau1_temp = P2_FIT_3STEP # 3rd step
            P1_temp   = P1_FIT_3STEP # 3rd step
            params2.add('P1', value=P1_temp, vary=False) #Fixed
            params2.add('P2', value=Tau1_temp, vary=False) #Fixed
            params2.add('P3', value=P3_FIT, min = 0.0, max = 100.0)
            params2.add('P4', value=P4_FIT, min = 0, max = 10) # Add here min as Tau1?
            params2.add('P5', value=P5_FIT, min = 0.0)
            res      = Minimizer(fcn2min_2pop,  params2, fcn_args=(CHL_DIM, OPT_DIM))
            fit_result_2 = res.minimize(method = 'powell')
            AIC_FIT6 = fit_result_2.aic
            #report_fit(fit_result_2) # uncomment if you want to see results of fit
                
            if use_aic:
                # perform comparison between 1 or 2 commmunity fits
                if AIC_FIT6 < AIC_FIT1: 
                    P1_FIT = fit_result_2.params['P1'].value
                    P2_FIT = fit_result_2.params['P2'].value
                    P3_FIT = fit_result_2.params['P3'].value
                    P4_FIT = fit_result_2.params['P4'].value
                    P5_FIT = fit_result_2.params['P5'].value
                    P4_FIT = P4_FIT + P5_FIT * 3.0
                    print("Chl 2 communities Fitted")
                else:
                    P1_FIT = C1_P1
                    P2_FIT = C1_TAU1
                    P3_FIT = nan
                    P4_FIT = nan
                    P5_FIT = nan
                    print("Chl 2 communities tried but 1 community fits better - NAN")
                    
            else:
                P1_FIT = fit_result_2.params['P1'].value
                P2_FIT = fit_result_2.params['P2'].value
                P3_FIT = fit_result_2.params['P3'].value
                P4_FIT = fit_result_2.params['P4'].value
                P5_FIT = fit_result_2.params['P5'].value
                P4_FIT = P4_FIT + P5_FIT * 3.0
                print("Chl 2 communities Fitted")
                
        # Extract parameters from the chlorophyll fit
        P1_final   = P1_FIT
        TAU1_final = P2_FIT
        BM2_final  = P3_FIT
        TAU2_final = P4_FIT
        SIG2_final = P5_FIT
        
    else:
        print("ERROR - Chl-a measurements count <6")
        #return np.nan, np.nan, np.nan, np.nan, np.nan
        
    ###  CHL MODEL arrays and Uncertainties ###
    OPT_DIM  = prof_depth*Kd
    MLD_pop_FIT   = (1 - 1./(1+np.exp(-(P1_final/TAU1_final)*(OPT_DIM-TAU1_final))))*prof_chla_surf
    if np.isnan(BM2_final):
        ### Only Surface C1 fitted ###
        DCM_pop_FIT   = MLD_pop_FIT *0
        
        # Initialize an empty dictionary to store results
        FitResults = {}
        FitResults['fit_type'] = '1pop'
        # store step-1 stats: All nan if only one community fitted
        FitResults['Chl_FitReport_1'] = np.nan
        FitResults['Chl_FitReport_chisq_1']  = np.nan
        FitResults['Chl_FitReport_redchi_1'] = np.nan
        FitResults['Chl_FitReport_aic_1']    = np.nan
        FitResults['Chl_FitReport_bic_1']    = np.nan
        FitResults['Chl_FitReport_P2_err_1'] = np.nan

        # store step-2 stats
        FitResults['Chl_FitReport_2']        = fit_result_C1
        FitResults['Chl_FitReport_chisq_2']  = fit_result_C1.chisqr
        FitResults['Chl_FitReport_redchi_2'] = fit_result_C1.redchi
        FitResults['Chl_FitReport_aic_2']    = fit_result_C1.aic
        FitResults['Chl_FitReport_bic_2']    = fit_result_C1.bic
        FitResults['Chl_FitReport_P4']       = np.nan
        
        # store step-2 parameter errors
        FitResults['Chl_FitReport_P1_err_2'] = fit_result_C1.params['P1'].stderr
        FitResults['Chl_FitReport_P2_err_2'] = fit_result_C1.params['P2'].stderr
        FitResults['Chl_FitReport_P3_err_2'] = np.nan
        FitResults['Chl_FitReport_P4_err_2'] = np.nan
        FitResults['Chl_FitReport_P5_err_2'] = np.nan
        FitResults['Chl_FitReport_Tau2_err'] = np.nan
    else:
        ### Two communities fitted
        DCM_pop_FIT   = (BM2_final*np.exp(-((OPT_DIM - TAU2_final)/SIG2_final)**2.)) *prof_chla_surf    
        # Initialize an empty dictionary to store results
        FitResults = {}
        FitResults['fit_type'] = '2pop'
        # store step-1 stats
        FitResults['Chl_FitReport_1']        = fit_result_1
        FitResults['Chl_FitReport_chisq_1']  = fit_result_1.chisqr
        FitResults['Chl_FitReport_redchi_1'] = fit_result_1.redchi
        FitResults['Chl_FitReport_aic_1']    = fit_result_1.aic
        FitResults['Chl_FitReport_bic_1']    = fit_result_1.bic
        # store step-1 parameter errors
        for name, par in fit_result_1.params.items():
            FitResults[f'Chl_FitReport_{name}_err_1'] = par.stderr
        # store step-2 stats
        FitResults['Chl_FitReport_2'] = fit_result_2
        FitResults['Chl_FitReport_chisq_2']  = fit_result_2.chisqr
        FitResults['Chl_FitReport_redchi_2'] = fit_result_2.redchi
        FitResults['Chl_FitReport_aic_2']    = fit_result_2.aic
        FitResults['Chl_FitReport_bic_2']    = fit_result_2.bic
        FitResults['Chl_FitReport_P4']       = fit_result_2.params['P4'].value
        
        # store step-2 parameter errors
        for name, par in fit_result_2.params.items():
            FitResults[f'Chl_FitReport_{name}_err_2'] = par.stderr
        FitResults['Chl_FitReport_P2_err_2'] = fit_result_1.params['P2'].stderr #if nan = fit_result_2.params['P2'].stderr
        
        # Tau2 combined parameter error (1‑σ uncertainty) calculation using linear‐propagation‐of‐errors approximation including covariance
        # Tau2 = P4_FIT + P5_FIT * 3.0
        # extract  1‐sigma errors
        err4 = fit_result_2.params['P4'].stderr
        err5 = fit_result_2.params['P5'].stderr    
        # find the indices of P4 and P5 in the covariance matrix
        i4 = fit_result_2.var_names.index('P4')
        i5 = fit_result_2.var_names.index('P5')  
        # extract their covariance
        cov45 = fit_result_2.covar[i4, i5]     
        # compute the propagated error on Tau2 = P4 + 3*P5
        sigma_tau2 = np.sqrt(err4**2 + 9*err5**2 + 6*cov45)
        FitResults['Chl_FitReport_Tau2_err'] = sigma_tau2
        #print("Sigma Prop Error: ",sigma_tau2)

    # Total Model Chl array
    CHL_model_fit = MLD_pop_FIT + DCM_pop_FIT# + DCM2_pop_FIT

    #Higher Res data points for plotting
    OPT_DIM2       = np.linspace(0,30,2000)
    MLD_pop_FIT2   = (1 - 1./(1+np.exp(-(P1_final/TAU1_final)*(OPT_DIM2-TAU1_final))))*prof_chla_surf
    if np.isnan(BM2_final):
        DCM_pop_FIT2   = MLD_pop_FIT2 *0
    else:
        DCM_pop_FIT2   = (BM2_final*np.exp(-((OPT_DIM2 - TAU2_final)/SIG2_final)**2.)) *prof_chla_surf

    CHL_model_fit2 = MLD_pop_FIT2 + DCM_pop_FIT2# + DCM2_pop_FIT2
    # High-Res depth array for plotting
    prof_depth2  = OPT_DIM2/Kd
#%%    
    ### POC Fit after CHL Parameters ###
    poc_fit = particulate_model(prof_poc, prof_poc_depth, prof_chla_surf, Kd, P1_final, TAU1_final, BM2_final, TAU2_final, SIG2_final, surf_norm=False)
    # Assign NaN if 'SURF_POC' does not exist
    SURF_POC = poc_fit.get('surface_value', np.nan)
    
    ### SAVE & RETURN RESULTS ###    
    #FitResults = {}
    # Store each variable in the dictionary
    FitResults['Kd'] = Kd # Kd used used in fitting
    FitResults['Zp'] = Zp # is nan if user defined Kd used, i.e. only given if Morel Kd and Zp calculated
    FitResults['P1_final']       = P1_final
    FitResults['TAU1_final']     = TAU1_final
    FitResults['BM2_final']      = BM2_final
    FitResults['TAU2_final']     = TAU2_final
    FitResults['SIG2_final']     = SIG2_final
    FitResults['Chl_C1_fit']     = MLD_pop_FIT
    FitResults['Chl_C2_fit']     = DCM_pop_FIT
    FitResults['Chl_Total_fit']  = CHL_model_fit
    FitResults['Chl_C1_fit_HiRes']    = MLD_pop_FIT2
    FitResults['Chl_C2_fit_HiRes']    = DCM_pop_FIT2
    FitResults['Chl_Total_fit_HiRes'] = CHL_model_fit2
    FitResults['prof_depth_HiRes']    = prof_depth2
    FitResults['prof_chla_surf']      = prof_chla_surf
    FitResults['DCM1_peak']      = BM2_final*prof_chla_surf
    FitResults['DCM1_depth']     = TAU2_final/Kd
    FitResults['DCM1_width']     = SIG2_final/Kd
    
    # Carbon related variables
    FitResults['prof_poc_surf']     = SURF_POC
    FitResults['POC_background_np'] = poc_fit['background_np'] if not np.isnan(SURF_POC) else np.nan
    FitResults['C_Chl_ratio_C1']    = poc_fit['P_Chl_ratio_C1'] if not np.isnan(SURF_POC) else np.nan
    FitResults['C_Chl_ratio_C2']    = poc_fit['P_Chl_ratio_C2'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_C1_fit']        = poc_fit['C1_fit'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_C2_fit']        = poc_fit['C2_fit'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_background']    = poc_fit['background'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_Total_fit']     = poc_fit['Total_fit'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_Phyto_fit']     = poc_fit['Phyto_fit'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_C1_fit_HiRes']  = poc_fit['C1_fit_HiRes'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_C2_fit_HiRes']  = poc_fit['C2_fit_HiRes'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_background_HiRes']  = poc_fit['background_HiRes'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_Total_fit_HiRes']   = poc_fit['Total_fit_HiRes'] if not np.isnan(SURF_POC) else np.nan
    FitResults['POC_Phyto_fit_HiRes']   = poc_fit['Phyto_fit_HiRes'] if not np.isnan(SURF_POC) else np.nan
    # Save POC fit result Stats
    FitResults['POC_FitReport'] = poc_fit['FitReport'] if not np.isnan(SURF_POC) else np.nan
    # store fit result stats
    FitResults['POC_FitReport_chisq']  = poc_fit['FitReport_chisq']
    FitResults['POC_FitReport_redchi'] = poc_fit['FitReport_redchi']
    FitResults['POC_FitReport_aic']    = poc_fit['FitReport_aic']
    FitResults['POC_FitReport_bic']    = poc_fit['FitReport_bic']
    FitResults['POC_FitReport_P1_err'] = poc_fit['FitReport_P1_err']
    FitResults['POC_FitReport_P2_err'] = poc_fit['FitReport_P2_err']
    FitResults['POC_FitReport_P3_err'] = poc_fit['FitReport_P3_err']
    
    # Return the dictionary
    return FitResults 
