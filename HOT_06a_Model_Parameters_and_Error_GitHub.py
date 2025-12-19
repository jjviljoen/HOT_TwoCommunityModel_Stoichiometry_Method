"""
HOT - Model fitted parameters and errors

Script to extract and plot parameters of tuned two-layer Chl and particulate model results

This script is related to the manuscript by Viljoen et al. (Preprint)
For more details, refer to the project ReadMe: https://github.com/jjviljoen/HOT_TwoCommunityModel_Stoichiometry_Method.

Updated: 19 Dec 2025

@author: Johan Viljoen - j.j.viljoen@exeter.ac.uk
"""

#%%

### LOAD PACKAGES ###
# General Python Packages
import pandas as pd  # data analysis and manipulation tool
import numpy as np   # used to work with data arrays
from dateutil import relativedelta
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation as MAD

# Supress warnings
import warnings
warnings.filterwarnings("ignore")
np.seterr(all='ignore')

plt.rcParams["font.family"] = "Arial"  # Set font for all plots

#%%

### DEFINE ALL FUNCTIONS ###

def date_span( start, end ):
    rd = relativedelta.relativedelta( pd.to_datetime( end ), pd.to_datetime( start ) )
    #date_len = str(rd.years)+"yrs"+str(rd.months)+"m"+str(rd.days+"d")
    date_len  = '{}y{}m{}d'.format(rd.years,rd.months,rd.days)
    #return rd.years, rd.months, rd.days
    return date_len

#%%

### IMPORT MODEL PARAMETERS & ERRORS
# CSV filename
filename_1 = 'data/HOT_ModelFit_StatsErrors.csv'
# Load data from csv. "index_col = 0" make first column the index.
df   = pd.read_csv(filename_1, index_col = 0)
df.info()

# Print timespan of bottle data in '{}y{}m{}d' format using custom function named date_span
b_date_length = date_span(min(df['Date']), max(df['Date']))
print("Timespan: "+str(b_date_length))
print("Min Date: "+str(min(df['Date'])))
print("Max Date: "+str(max(df['Date'])))

#%%

### PLOT TIME SERIES PARAMETERS & ERRORS - CHL

Title_font_size = 10  # Define the font size of the titles
Label_font_size = 9  # Define the font size of the labels

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(7, 7),
      gridspec_kw={'hspace': 0.2})
fig.patch.set_facecolor('White')

# P1
ax1.errorbar(df['DecYear'], df['Chl_Fit_P1'], yerr =df['Chl_Fit_P1_2_error'],  ms=2, linestyle='None',
             marker='o', color='k', alpha=.5)
ax1.set_ylabel('$P_1$', fontsize=Title_font_size, color='k')
ax1.yaxis.set_tick_params(labelsize=Label_font_size)
ax1.xaxis.set_tick_params(labelsize=Label_font_size)

# P2 - Tau 1
ax2.errorbar(df['DecYear'], df['Chl_Fit_P2'], yerr =df['Chl_Fit_P2_1_error'], ms=2,
             linestyle='None',
             marker='o', color='b', alpha=.5)
ax2.set_ylabel('$\\tau_1$', fontsize=Title_font_size, color='b')
ax2.yaxis.set_tick_params(labelsize=Label_font_size)

# P3 - B2m* - DCM peak
ax3.errorbar(df['DecYear'], df['Chl_Fit_P3'], yerr =df['Chl_Fit_P3_2_error'],ms=2, linestyle='None',
             marker='o', color='r', alpha=.5)
ax3.set_ylabel('$B^{*}_{2,m}$', fontsize=Title_font_size, color='r')
ax3.yaxis.set_tick_params(labelsize=Label_font_size)

# P4 - *Tau 2
ax4.errorbar(df['DecYear'], df['Chl_Fit_Tau2'], yerr =df['Chl_FitErr_Tau2'], ms=2,
             linestyle='None',
             marker='o', color='g', alpha=.5)
ax4.set_ylabel('${\\tau}_2$', fontsize=Title_font_size, color='g')
ax4.yaxis.set_tick_params(labelsize=Label_font_size)

# Sigma
ax5.errorbar(df['DecYear'], df['Chl_Fit_P5'], yerr =df['Chl_Fit_P5_2_error'],ms=2,
             linestyle='None',
             marker='o', color='c', alpha=.5)
ax5.set_ylabel('$\sigma$', fontsize=Title_font_size, color='c')
ax5.yaxis.set_tick_params(labelsize=Label_font_size)
ax5.xaxis.set_tick_params(labelsize=Label_font_size)

fig.savefig('plots/HOT_Model_Parameters_Error_ChlFit_paper.jpeg', format='jpeg', dpi=300, bbox_inches="tight")

plt.show()

#%%

### PLOT TIME SERIES PARAMETERS & ERRORS - Particulates

Title_font_size = 9   # font size of the titles
Label_font_size = 8   # font size of the labels
legend_font_size = 7  # font size of the legends

fig, (ax6, ax7, ax8, ax9, ax10, ax11) = plt.subplots(
    6, sharex=True, figsize=(7, 8),
    gridspec_kw={'hspace': 0.2}
)
fig.patch.set_facecolor('white')
sns.set_palette("hls", 12)

# Expand right margin so legends fit
fig.subplots_adjust(right=0.85)

# (a) C:Chl
ax6.errorbar(df['DecYear'], df['POC_Fit_P1'], yerr=df['POC_Fit_P1_error'],
             ms=2, linestyle='None', marker='o', color='r',
             label=r'$C_{1}^{B}$', alpha=.5)
ax6.errorbar(df['DecYear'], df['POC_Fit_P2'], yerr=df['POC_Fit_P2_error'],
             ms=2, linestyle='None', marker='o', color='b',
             label=r'$C_{2}^{B}$', alpha=.5)
ax6.set_ylabel('C:Chl (g g$^{-1}$)', fontsize=Title_font_size, color='k')
ax6.yaxis.set_tick_params(labelsize=Label_font_size)
ax6.set_yscale('log')
#ax6.set_title('(a)', fontsize=Title_font_size, color='k')
ax6.legend(
    bbox_to_anchor=(1.02, 1), loc='upper left',
    fontsize=legend_font_size, borderaxespad=0
)

# (b) PC background
ax7.errorbar(df['DecYear'], df['POC_Fit_P3'], yerr=df['POC_Fit_P3_error'],
             ms=2, linestyle='None', marker='o', color='k', alpha=.2)
ax7.set_ylabel('$C_{bk}$ (mg m$^{-3}$)', fontsize=Title_font_size, color='k')
ax7.yaxis.set_tick_params(labelsize=Label_font_size)

# (c) N:Chl
ax8.errorbar(df['DecYear'], df['PON_Fit_P1'], yerr=df['PON_Fit_P1_error'],
             ms=2, linestyle='None', marker='o', color='r',
             label=r'$N_{1}^{B}$', alpha=.5)
ax8.errorbar(df['DecYear'], df['PON_Fit_P2'], yerr=df['PON_Fit_P2_error'],
             ms=2, linestyle='None', marker='o', color='b',
             label=r'$N_{2}^{B}$', alpha=.5)
ax8.set_ylabel('N:Chl (g g$^{-1}$)', fontsize=Title_font_size, color='k')
ax8.yaxis.set_tick_params(labelsize=Label_font_size)
ax8.set_yscale('log')
#ax8.set_title('(b)', fontsize=Title_font_size, color='k')
ax8.legend(
    bbox_to_anchor=(1.02, 1), loc='upper left',
    fontsize=legend_font_size, borderaxespad=0
)

# (d) PN background
ax9.errorbar(df['DecYear'], df['PON_Fit_P3'], yerr=df['PON_Fit_P3_error'],
             ms=2, linestyle='None', marker='o', color='k', alpha=.2)
ax9.set_ylabel('$N_{bk}$ (mg m$^{-3}$)', fontsize=Title_font_size, color='k')
ax9.yaxis.set_tick_params(labelsize=Label_font_size)

# (e) P:Chl
ax10.errorbar(df['DecYear'], df['POP_Fit_P1'], yerr=df['POP_Fit_P1_error'],
              ms=2, linestyle='None', marker='o', color='r',
              label=r'$P_{1}^{B}$', alpha=.5)
ax10.errorbar(df['DecYear'], df['POP_Fit_P2'], yerr=df['POP_Fit_P2_error'],
              ms=2, linestyle='None', marker='o', color='b',
              label=r'$P_{2}^{B}$', alpha=.5)
ax10.set_ylabel('P:Chl (g g$^{-1}$)', fontsize=Title_font_size, color='k')
ax10.yaxis.set_tick_params(labelsize=Label_font_size)
ax10.set_yscale('log')
#ax10.set_title('(c)', fontsize=Title_font_size, color='k')
ax10.legend(
    bbox_to_anchor=(1.02, 1), loc='upper left',
    fontsize=legend_font_size, borderaxespad=0
)

# (f) PP background
ax11.errorbar(df['DecYear'], df['POP_Fit_P3'], yerr=df['POP_Fit_P3_error'],
              ms=2, linestyle='None', marker='o', color='k', alpha=.2)
ax11.set_ylabel('$P_{bk}$ (mg m$^{-3}$)', fontsize=Title_font_size, color='k')
ax11.yaxis.set_tick_params(labelsize=Label_font_size)
ax11.set_xlabel('Decimal Year', fontsize=Label_font_size, color='k')
ax11.xaxis.set_tick_params(labelsize=Label_font_size)

plt.tight_layout()
fig.savefig(
    'plots/HOT_Model_Parameters_Error_Particulates_paper.jpeg',
    format='jpeg', dpi=300, bbox_inches="tight"
)
plt.show()

#%%

# ERRORS FOR MOLAR RATIOS:

def bootstrap_ratio_error(
    nume,
    sigma_nume,
    deno=None,
    sigma_deno=None,
    M_nume=1.0,
    M_deno=1.0,
    n_boot=5000,
    rng_seed=42
):
    """
    Bootstrap error propagation for (molar) ratios.

    Returns bootstrap uncertainty AND robust spread (MAD) of reported values.
    """
    rng = np.random.default_rng(rng_seed)

    nume = np.asarray(nume)
    sigma_nume = np.asarray(sigma_nume)

    if deno is not None:
        deno = np.asarray(deno)
        sigma_deno = np.asarray(sigma_deno)
        valid_mask = (~np.isnan(nume)) & (~np.isnan(deno))
        x_num = nume[valid_mask]
        s_num = sigma_nume[valid_mask]
        x_den = deno[valid_mask]
        s_den = sigma_deno[valid_mask]

        reported_vals = (x_num / x_den) * (M_deno / M_nume)
    else:
        valid_mask = ~np.isnan(nume)
        x_num = nume[valid_mask]
        s_num = sigma_nume[valid_mask]
        x_den = None
        s_den = None

        reported_vals = x_num

    n = len(reported_vals)
    if n == 0:
        raise ValueError("No valid entries found for numerator (or denominator).")

    # --- Reported statistics ---
    median_reported = np.median(reported_vals)
    mad_reported = MAD(reported_vals, nan_policy='omit')

    # --- Bootstrap ---
    boot_medians = np.zeros(n_boot)

    for i in range(n_boot):
        pert_num = x_num + rng.normal(0, s_num)

        if x_den is not None:
            pert_den = x_den + rng.normal(0, s_den)
            ratio = (pert_num / pert_den) * (M_deno / M_nume)
        else:
            ratio = pert_num

        boot_medians[i] = np.median(ratio)

    p16, p50, p84 = np.percentile(boot_medians, [16, 50, 84])

    err_low = median_reported - p16
    err_high = p84 - median_reported

    if (p16 <= median_reported) and (p84 >= median_reported):
        bias_flag = "interval_straddles_median"
    elif p84 < median_reported:
        bias_flag = "bootstrap_below_median"
    else:
        bias_flag = "bootstrap_above_median"

    return {
        "median": median_reported,
        "mad_reported": mad_reported,
        "err_low": err_low,
        "err_high": err_high,
        "p16": p16,
        "p50": p50,
        "p84": p84,
        "bias_flag": bias_flag,
        "n_used": n,
        "boot_medians": boot_medians
    }

#%%

# Test Function on C:Chl ratios

# C:Chl Surface

res = bootstrap_ratio_error(
    nume=df['POC_Fit_P1'],
    sigma_nume=df['POC_Fit_P1_error'],
    n_boot=5000,
    rng_seed=42
)
print("=== Surface C:Chl ====")
print("Model median (reported):", res['median'])
print("Bootstrap medians p16/p50/p84:", res['p16'], res['p50'], res['p84'])
print("Model uncertainty relative to reported median: +{:.3f} / -{:.3f}".format(res['err_high'], res['err_low']))
print("Bias flag:", res['bias_flag'])

# C:Chl Subsurface

res = bootstrap_ratio_error(
    nume=df['POC_Fit_P2'],
    sigma_nume=df['POC_Fit_P2_error'],
    n_boot=5000,
    rng_seed=42
)
print("=== Subsurface C:Chl ====")
print("Model median (reported):", res['median'])
print("Bootstrap medians p16/p50/p84:", res['p16'], res['p50'], res['p84'])
print("Model uncertainty relative to reported median: +{:.3f} / -{:.3f}".format(res['err_high'], res['err_low']))
print("Bias flag:", res['bias_flag'])

#%%

# Test Function on C:N RATIOS

M_C = 12.01
M_N = 14.01

# C:N Surface

res = bootstrap_ratio_error(
    nume=df['POC_Fit_P1'],
    sigma_nume=df['POC_Fit_P1_error'],
    deno=df['PON_Fit_P1'],
    sigma_deno=df['PON_Fit_P1_error'],
    M_nume=M_C,
    M_deno=M_N,
    n_boot=5000,
    rng_seed=42
)

print("=== Surface C:N ====")
print("Model median (reported):", res['median'])
print("Bootstrap medians p16/p50/p84:", res['p16'], res['p50'], res['p84'])
print("Model uncertainty relative to reported median: +{:.3f} / -{:.3f}".format(res['err_high'], res['err_low']))
print("Bias flag:", res['bias_flag'])

# C:N Subsurface

res = bootstrap_ratio_error(
    nume=df['POC_Fit_P2'],
    sigma_nume=df['POC_Fit_P2_error'],
    deno=df['PON_Fit_P2'],
    sigma_deno=df['PON_Fit_P2_error'],
    M_nume=M_C,
    M_deno=M_N,
    n_boot=5000,
    rng_seed=42
)

print("=== Subsurface C:N ====")
print("Model median (reported):", res['median'])
print("Bootstrap medians p16/p50/p84:", res['p16'], res['p50'], res['p84'])
print("Model uncertainty relative to reported median: +{:.3f} / -{:.3f}".format(res['err_high'], res['err_low']))
print("Bias flag:", res['bias_flag'])

# C:N Non-Algal

res = bootstrap_ratio_error(
    nume=df['POC_Fit_P3'],
    sigma_nume=df['POC_Fit_P3_error'],
    deno=df['PON_Fit_P3'],
    sigma_deno=df['PON_Fit_P3_error'],
    M_nume=M_C,
    M_deno=M_N,
    n_boot=5000,
    rng_seed=42
)

print("=== Non-Algal C:N ====")
print("Model median (reported):", res['median'])
print("Bootstrap medians p16/p50/p84:", res['p16'], res['p50'], res['p84'])
print("Model uncertainty relative to reported median: +{:.3f} / -{:.3f}".format(res['err_high'], res['err_low']))
print("Bias flag:", res['bias_flag'])

#%%

# BUILD TABLE OF MEDIANS AND UNCERTAINTY

def summarize_bootstrap_result(label, res, decimals=2):
    """
    Summarize bootstrap_ratio_error output for table reporting.

    Returns:
    - Median ± MAD (of reported values)
    - Upper uncertainty: p84 - median
    - Lower uncertainty: p16 - median
    - bias_flag
    - n_used
    """

    median = res["median"]
    mad = res.get("mad_reported", np.nan)

    # Signed offsets relative to median (as per column headers)
    upper = res["p84"] - median
    lower = res["p16"] - median

    # Defensive fix: avoid double negatives or sign flips
    # (numerically unnecessary, but guards against formatting artefacts)
    upper = float(upper)
    lower = float(lower)

    return {
        "Ratio": label,
        "Median ± MAD": f"{median:.{decimals}f} ± {mad:.{decimals}f}",
        "Upper Uncertainty (p84−median)": f"{upper:+.{decimals}f}",
        "Lower Uncertainty (p16−median)": f"{lower:+.{decimals}f}",
        "bias_flag": res["bias_flag"],
        "n": res["n_used"]
    }

# Define atomic weights
M_C = 12.01
M_N = 14.01
M_P = 30.97

# Initialise results table
rows = []

############################
# C:Chl (g g-1)
############################

# Surface
res = bootstrap_ratio_error(
    nume=df['POC_Fit_P1'],
    sigma_nume=df['POC_Fit_P1_error']
)
rows.append(summarize_bootstrap_result("C:Chl (g g⁻¹) Surface", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=df['POC_Fit_P2'],
    sigma_nume=df['POC_Fit_P2_error']
)
rows.append(summarize_bootstrap_result("C:Chl (g g⁻¹) Subsurface", res))


############################
# C:N
############################

# Surface
res = bootstrap_ratio_error(
    nume=df['POC_Fit_P1'],
    sigma_nume=df['POC_Fit_P1_error'],
    deno=df['PON_Fit_P1'],
    sigma_deno=df['PON_Fit_P1_error'],
    M_nume=M_C,
    M_deno=M_N
)
rows.append(summarize_bootstrap_result("C:N Surface", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=df['POC_Fit_P2'],
    sigma_nume=df['POC_Fit_P2_error'],
    deno=df['PON_Fit_P2'],
    sigma_deno=df['PON_Fit_P2_error'],
    M_nume=M_C,
    M_deno=M_N
)
rows.append(summarize_bootstrap_result("C:N Subsurface", res))

# Subsurface Non-Algal
res = bootstrap_ratio_error(
    nume=df['POC_Fit_P3'],
    sigma_nume=df['POC_Fit_P3_error'],
    deno=df['PON_Fit_P3'],
    sigma_deno=df['PON_Fit_P3_error'],
    M_nume=M_C,
    M_deno=M_N
)
rows.append(summarize_bootstrap_result("C:N Subsurface Non-Algal", res))


############################
# C:P (Pre)
############################

# filter df for pre and post Nov-2011 PP analytical change
df['Date'] = pd.to_datetime(df['Date'])
before = df[df['Date'] < '2011-11-01']
after  = df[df['Date'] >= '2011-11-01']

# Surface
res = bootstrap_ratio_error(
    nume=before['POC_Fit_P1'],
    sigma_nume=before['POC_Fit_P1_error'],
    deno=before['POP_Fit_P1'],
    sigma_deno=before['POP_Fit_P1_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Surface (Pre)", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=before['POC_Fit_P2'],
    sigma_nume=before['POC_Fit_P2_error'],
    deno=before['POP_Fit_P2'],
    sigma_deno=before['POP_Fit_P2_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Subsurface (Pre)", res))

# Non-Algal
res = bootstrap_ratio_error(
    nume=before['POC_Fit_P3'],
    sigma_nume=before['POC_Fit_P3_error'],
    deno=before['POP_Fit_P3'],
    sigma_deno=before['POP_Fit_P3_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Non-Algal (Pre)", res))


############################
# C:P (Post)
############################

# Surface
res = bootstrap_ratio_error(
    nume=after['POC_Fit_P1'],
    sigma_nume=after['POC_Fit_P1_error'],
    deno=after['POP_Fit_P1'],
    sigma_deno=after['POP_Fit_P1_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Surface (Post)", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=after['POC_Fit_P2'],
    sigma_nume=after['POC_Fit_P2_error'],
    deno=after['POP_Fit_P2'],
    sigma_deno=after['POP_Fit_P2_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Subsurface (Post)", res))

# Non-Algal
res = bootstrap_ratio_error(
    nume=after['POC_Fit_P3'],
    sigma_nume=after['POC_Fit_P3_error'],
    deno=after['POP_Fit_P3'],
    sigma_deno=after['POP_Fit_P3_error'],
    M_nume=M_C,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("C:P Non-Algal (Post)", res))


############################
# N:P (Pre)
############################

# Surface
res = bootstrap_ratio_error(
    nume=before['PON_Fit_P1'],
    sigma_nume=before['PON_Fit_P1_error'],
    deno=before['POP_Fit_P1'],
    sigma_deno=before['POP_Fit_P1_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Surface (Pre)", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=before['PON_Fit_P2'],
    sigma_nume=before['PON_Fit_P2_error'],
    deno=before['POP_Fit_P2'],
    sigma_deno=before['POP_Fit_P2_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Subsurface (Pre)", res))

# Non-Algal
res = bootstrap_ratio_error(
    nume=before['PON_Fit_P3'],
    sigma_nume=before['PON_Fit_P3_error'],
    deno=before['POP_Fit_P3'],
    sigma_deno=before['POP_Fit_P3_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Non-Algal (Pre)", res))


############################
# N:P (Post)
############################

# Surface
res = bootstrap_ratio_error(
    nume=after['PON_Fit_P1'],
    sigma_nume=after['PON_Fit_P1_error'],
    deno=after['POP_Fit_P1'],
    sigma_deno=after['POP_Fit_P1_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Surface (Post)", res))

# Subsurface
res = bootstrap_ratio_error(
    nume=after['PON_Fit_P2'],
    sigma_nume=after['PON_Fit_P2_error'],
    deno=after['POP_Fit_P2'],
    sigma_deno=after['POP_Fit_P2_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Subsurface (Post)", res))

# Non-Algal
res = bootstrap_ratio_error(
    nume=after['PON_Fit_P3'],
    sigma_nume=after['PON_Fit_P3_error'],
    deno=after['POP_Fit_P3'],
    sigma_deno=after['POP_Fit_P3_error'],
    M_nume=M_N,
    M_deno=M_P
)
rows.append(summarize_bootstrap_result("N:P Non-Algal (Post)", res))


#%%

### SAVE TABLE ###

results_table = pd.DataFrame(rows)
results_table


results_table.to_csv("Tables/HOT_Stoichiometry_bootstrap_summary.csv", index=False)
results_table.to_excel("Tables/HOT_Stoichiometry_bootstrap_summary.xlsx", index=False)


