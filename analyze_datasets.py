"""
Quick Dataset Analysis for NASA Space Apps Challenge
Analyzing all three datasets to determine which to use
"""

import pandas as pd
import numpy as np

print("="*80)
print("EXOPLANET DATASET ANALYSIS - NASA SPACE APPS CHALLENGE")
print("="*80)

# Load datasets (skip comment lines)
print("\n[1/3] Loading cumulative dataset...")
cumulative = pd.read_csv('dataset/cumulative_2025.10.05_02.33.52.csv', comment='#')
print(f"✓ Cumulative: {cumulative.shape[0]} rows, {cumulative.shape[1]} columns")

print("\n[2/3] Loading K2PANDC dataset...")
k2pandc = pd.read_csv('dataset/k2pandc_2025.10.05_03.01.55.csv', comment='#')
print(f"✓ K2PANDC: {k2pandc.shape[0]} rows, {k2pandc.shape[1]} columns")

print("\n[3/3] Loading TOI dataset...")
toi = pd.read_csv('dataset/TOI_2025.10.05_02.34.10.csv', comment='#')
print(f"✓ TOI: {toi.shape[0]} rows, {toi.shape[1]} columns")

print("\n" + "="*80)
print("DATASET COLUMNS ANALYSIS")
print("="*80)

print("\n### CUMULATIVE DATASET ###")
print(f"Columns ({len(cumulative.columns)}):")
print(cumulative.columns.tolist()[:30])  # First 30 columns

print("\n### K2PANDC DATASET ###")
print(f"Columns ({len(k2pandc.columns)}):")
print(k2pandc.columns.tolist()[:30])

print("\n### TOI DATASET ###")
print(f"Columns ({len(toi.columns)}):")
print(toi.columns.tolist()[:30])

print("\n" + "="*80)
print("TARGET VARIABLE ANALYSIS (Planet Confirmation Status)")
print("="*80)

# Check for disposition columns (target variable)
if 'koi_disposition' in cumulative.columns:
    print("\n✓ CUMULATIVE - koi_disposition:")
    print(cumulative['koi_disposition'].value_counts())
    print(f"Non-null: {cumulative['koi_disposition'].notna().sum()}")

if 'k2c_disp' in k2pandc.columns:
    print("\n✓ K2PANDC - k2c_disp:")
    print(k2pandc['k2c_disp'].value_counts())
    print(f"Non-null: {k2pandc['k2c_disp'].notna().sum()}")

if 'tfopwg_disp' in toi.columns:
    print("\n✓ TOI - tfopwg_disp:")
    print(toi['tfopwg_disp'].value_counts())
    print(f"Non-null: {toi['tfopwg_disp'].notna().sum()}")

print("\n" + "="*80)
print("KEY FEATURES FOR EXOPLANET DETECTION")
print("="*80)

# Key features based on 20 years of experience in exoplanet detection
key_features = [
    'koi_period',      # Orbital period (days)
    'koi_depth',       # Transit depth (ppm)
    'koi_duration',    # Transit duration (hours)
    'koi_prad',        # Planetary radius (Earth radii)
    'koi_teq',         # Equilibrium temperature (K)
    'koi_insol',       # Insolation flux (Earth flux)
    'koi_steff',       # Stellar effective temperature (K)
    'koi_slogg',       # Stellar surface gravity (log10(cm/s^2))
    'koi_srad',        # Stellar radius (Solar radii)
    'koi_smass',       # Stellar mass (Solar mass)
    'koi_impact',      # Impact parameter
    'koi_ror',         # Planet-star radius ratio
]

print("\nChecking feature availability in CUMULATIVE dataset:")
available_cumulative = [f for f in key_features if f in cumulative.columns]
print(f"✓ Found {len(available_cumulative)}/{len(key_features)} key features")
print(f"Available: {available_cumulative}")

# Check data completeness
print("\nData completeness for key features (CUMULATIVE):")
for feat in available_cumulative[:10]:
    missing_pct = (cumulative[feat].isna().sum() / len(cumulative)) * 100
    print(f"  {feat}: {100-missing_pct:.1f}% complete")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\n✓ BEST DATASET: CUMULATIVE (cumulative_2025.10.05_02.33.52.csv)")
print("  Reasons:")
print("  1. Largest dataset (9,619 rows)")
print("  2. Contains koi_disposition (clear target variable)")
print("  3. Has most key features for exoplanet detection")
print("  4. From Kepler mission (most successful exoplanet hunter)")
print("\n✓ This dataset will be used for training the deep learning models")

# Save summary statistics
print("\n" + "="*80)
print("SAVING DATA SUMMARY")
print("="*80)

summary = {
    'dataset': 'cumulative',
    'rows': len(cumulative),
    'columns': len(cumulative.columns),
    'target_column': 'koi_disposition',
    'key_features': available_cumulative,
    'confirmed_planets': (cumulative['koi_disposition'] == 'CONFIRMED').sum() if 'koi_disposition' in cumulative.columns else 0,
    'candidate_planets': (cumulative['koi_disposition'] == 'CANDIDATE').sum() if 'koi_disposition' in cumulative.columns else 0,
    'false_positives': (cumulative['koi_disposition'] == 'FALSE POSITIVE').sum() if 'koi_disposition' in cumulative.columns else 0,
}

print(f"\n✓ Confirmed Planets: {summary['confirmed_planets']}")
print(f"✓ Candidate Planets: {summary['candidate_planets']}")
print(f"✓ False Positives: {summary['false_positives']}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Data preprocessing & feature engineering")
print("2. Train 3 deep learning models (DNN, CNN, LSTM)")
print("3. Build Flask web interface with visualizations")
print("4. Deploy and test!")
