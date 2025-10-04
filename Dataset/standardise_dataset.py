
import pandas as pd
import numpy as np
import json
import os

# ----------- File Paths (edit these if needed) ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEPLER_PATH = os.path.join(BASE_DIR, "cumulative_2025.10.04_01.30.34.csv")
K2_PATH = os.path.join(BASE_DIR, "k2pandc_2025.10.04_01.30.48.csv")
TESS_PATH = os.path.join(BASE_DIR, "TOI_2025.10.04_01.32.13.csv")
OUT_DIR = os.path.join(BASE_DIR, "standardized")
os.makedirs(OUT_DIR, exist_ok=True)
# -----------------------------------------------------------

def read_exoarchive_csv(path):
    """Read NASA Exoplanet CSV while skipping '#' headers."""
    return pd.read_csv(path, comment="#")

def to_num(serieslike):
    return pd.to_numeric(serieslike, errors="coerce") if serieslike is not None else None

def normalize_labels(raw):
    """Unify labels across missions."""
    up = raw.astype(str).str.strip().str.upper()
    return up.replace({
        "CP": "CONFIRMED",
        "KP": "CONFIRMED",
        "CONFIRMED": "CONFIRMED",
        "PC": "CANDIDATE",
        "APC": "CANDIDATE",
        "CANDIDATE": "CANDIDATE",
        "FP": "FALSE_POSITIVE",
        "FA": "FALSE_POSITIVE",
        "REFUTED": "FALSE_POSITIVE",
        "FALSE POSITIVE": "FALSE_POSITIVE",
        "FALSE_POSITIVE": "FALSE_POSITIVE"
    })

# ---------------------- STANDARDIZERS ----------------------

def standardize_kepler(df):
    return pd.DataFrame({
        "star_id": df["kepid"].astype(str),
        "mission": "Kepler",
        "label": normalize_labels(df["koi_disposition"]),
        "orbital_period": to_num(df.get("koi_period")),
        "transit_duration": to_num(df.get("koi_duration")),
        "transit_depth": to_num(df.get("koi_depth")),
        "planet_radius": to_num(df.get("koi_prad")),
        "impact_parameter": to_num(df.get("koi_impact")),
        "transit_snr": to_num(df.get("koi_snr")),
        "mes": to_num(df.get("koi_mes")),
        "stellar_teff": to_num(df.get("koi_steff")),
        "stellar_logg": to_num(df.get("koi_slogg")),
        "stellar_radius": to_num(df.get("koi_srad")),
        "mag": to_num(df.get("koi_kepmag")),
    }).drop_duplicates(subset=["star_id", "mission"]).reset_index(drop=True)

def standardize_k2(df):
    star_col = None
    for c in ["epic_id", "epic", "epic_number", "tic", "hostname"]:
        if c in df.columns:
            star_col = c
            break
    if star_col is None:
        star_col = df.columns[0]

    labels = normalize_labels(df.get("disposition", pd.Series([""] * len(df))))

    return pd.DataFrame({
        "star_id": df[star_col].astype(str),
        "mission": "K2",
        "label": labels,
        "orbital_period": to_num(df.get("pl_orbper")),
        "transit_duration": to_num(df.get("pl_trandurh")),
        "transit_depth": to_num(df.get("pl_trandep")),
        "planet_radius": to_num(df.get("pl_rade")),
        "impact_parameter": to_num(df.get("pl_imppar")),
        "transit_snr": to_num(df.get("pl_trandsnr") if "pl_trandsnr" in df.columns else df.get("tran_snr")),
        "mes": to_num(df.get("mes")),
        "stellar_teff": to_num(df.get("st_teff")),
        "stellar_logg": to_num(df.get("st_logg")),
        "stellar_radius": to_num(df.get("st_rad")),
        "mag": to_num(df.get("sy_kepmag") if "sy_kepmag" in df.columns else df.get("sy_tmag")),
    }).drop_duplicates(subset=["star_id", "mission"]).reset_index(drop=True)

def standardize_tess(df):
    star_col = "tid" if "tid" in df.columns else ("TIC" if "TIC" in df.columns else df.columns[0])
    labels = normalize_labels(df.get("tfopwg_disp", pd.Series([""] * len(df))))
    return pd.DataFrame({
        "star_id": df[star_col].astype(str),
        "mission": "TESS",
        "label": labels,
        "orbital_period": to_num(df.get("pl_orbper")),
        "transit_duration": to_num(df.get("pl_trandurh") if "pl_trandurh" in df.columns else df.get("dur_hr")),
        "transit_depth": to_num(df.get("pl_trandep") if "pl_trandep" in df.columns else df.get("depth_ppm")),
        "planet_radius": to_num(df.get("pl_rade")),
        "impact_parameter": to_num(df.get("pl_imppar")),
        "transit_snr": to_num(df.get("pl_trandsnr") if "pl_trandsnr" in df.columns else df.get("snr")),
        "mes": to_num(df.get("mes")),
        "stellar_teff": to_num(df.get("st_teff")),
        "stellar_logg": to_num(df.get("st_logg")),
        "stellar_radius": to_num(df.get("st_rad")),
        "mag": to_num(df.get("sy_tmag") if "sy_tmag" in df.columns else df.get("Tmag")),
    }).drop_duplicates(subset=["star_id", "mission"]).reset_index(drop=True)

# ---------------------- MAIN PIPELINE ----------------------

def summarize(df):
    return {
        "rows": int(len(df)),
        "class_counts": df["label"].value_counts(dropna=False).to_dict(),
        "missing_counts": df.isna().sum().to_dict()
    }

def main():
    print("📥 Reading CSV files...")
    kepler = read_exoarchive_csv(KEPLER_PATH)
    k2 = read_exoarchive_csv(K2_PATH)
    tess = read_exoarchive_csv(TESS_PATH)

    print(" Standardizing datasets...")
    std_kepler = standardize_kepler(kepler)
    std_k2 = standardize_k2(k2)
    std_tess = standardize_tess(tess)

    combined = pd.concat([std_kepler, std_k2, std_tess], ignore_index=True)

    print("Saving cleaned CSVs...")
    std_kepler.to_csv(os.path.join(OUT_DIR, "kepler_standardized.csv"), index=False)
    std_k2.to_csv(os.path.join(OUT_DIR, "k2_standardized.csv"), index=False)
    std_tess.to_csv(os.path.join(OUT_DIR, "tess_standardized.csv"), index=False)
    combined.to_csv(os.path.join(OUT_DIR, "all_missions_standardized.csv"), index=False)

    summary = {
        "kepler": summarize(std_kepler),
        "k2": summarize(std_k2),
        "tess": summarize(std_tess),
        "combined": summarize(combined)
    }
    with open(os.path.join(OUT_DIR, "summaries.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n STANDARDIZATION COMPLETE!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
