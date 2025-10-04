# ml_utils.py  (project root)
# Shared utilities: loading, preprocessing, metrics, saving

import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    precision_recall_curve, auc
)

# ---------- Paths ----------
DATA_PATH = "Dataset/standardized/all_missions_standardized.csv"

# ---------- Feature sets ----------
NUM_COLS_BASE = [
    "orbital_period","transit_duration","transit_depth","planet_radius",
    "impact_parameter","transit_snr","mes","stellar_teff",
    "stellar_logg","stellar_radius","mag"
]
CAT_COLS = ["mission"]  # categorical feature

# ---------- Loader ----------
def load_data(path=DATA_PATH, min_num_cols=5, make_report=False):
    """
    Loads standardized dataset, does light feature engineering, sanitizes values,
    and returns (df, X, y, groups, num_cols, cat_cols, classes).
    """
    df = pd.read_csv(path)

    # keep valid labels only
    df = df[df["label"].isin(["CONFIRMED", "CANDIDATE", "FALSE_POSITIVE"])].reset_index(drop=True)

    # ---- Light feature engineering (helpful lifts) ----
    # Avoid div-by-zero by replacing 0 with NaN; later imputed by median
    df["depth_per_hour"]   = df["transit_depth"] / (df["transit_duration"].replace(0, np.nan))
    df["snr_per_dur"]      = df["transit_snr"]   / (df["transit_duration"].replace(0, np.nan))
    df["period_x_duration"] = df["orbital_period"] * df["transit_duration"]

    num_cols = NUM_COLS_BASE + ["depth_per_hour", "snr_per_dur", "period_x_duration"]

    # ---- Sanitize infinities & impossible values ----
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # ---- Drop rows that have too few numeric signals ----
    # Require at least half the numeric features (or min_num_cols) present
    df = df.dropna(subset=num_cols, thresh=max(min_num_cols, len(num_cols)//2)).reset_index(drop=True)

    # Optional: quick missingness report (pre-imputation)
    if make_report:
        miss = df[num_cols].isna().sum().sort_values(ascending=False)
        print("\n[Missingness before imputation]")
        print(miss.to_string())

    X = df[num_cols + CAT_COLS].copy()
    y = df["label"].astype(str)
    groups = df["star_id"].astype(str)
    classes = sorted(y.unique().tolist())
    return df, X, y, groups, num_cols, CAT_COLS, classes

# ---------- Preprocessor ----------
def make_preprocessor(num_cols, cat_cols):
    """
    Numeric: Median imputation + Standard scaling
    Categorical: One-hot (ignore unseen)
    This guarantees no NaN/Inf reaches the classifier.
    """
    num_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # <-- key fix for NaNs
        ("scaler", StandardScaler())
    ])
    cat_tf = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop"
    )
    return pre

# ---------- Metrics ----------
def metrics_with_pr(y_true_text, y_pred_text, proba, classes):
    """
    Returns accuracy, macro F1, macro PR-AUC, PR-AUC for CONFIRMED,
    plus confusion matrix and a full text classification report.
    """
    acc = accuracy_score(y_true_text, y_pred_text)
    f1m = f1_score(y_true_text, y_pred_text, average="macro")
    cm = confusion_matrix(y_true_text, y_pred_text, labels=classes)
    report_text = classification_report(y_true_text, y_pred_text, labels=classes, target_names=classes)

    pr_macro = pr_conf = None
    if proba is not None:
        y_true_arr = np.array(y_true_text)
        pr_list = []
        for i, c in enumerate(classes):
            y_bin = (y_true_arr == c).astype(int)
            pr, rc, _ = precision_recall_curve(y_bin, proba[:, i])
            pr_list.append(auc(rc, pr))
        pr_macro = float(np.mean(pr_list))
        idx_conf = classes.index("CONFIRMED")
        y_bin_conf = (y_true_arr == "CONFIRMED").astype(int)
        pr, rc, _ = precision_recall_curve(y_bin_conf, proba[:, idx_conf])
        pr_conf = float(auc(rc, pr))

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "pr_auc_macro": pr_macro,
        "pr_auc_confirmed": pr_conf,
        "confusion_matrix": cm.tolist(),
        "report": report_text
    }

# ---------- Split helpers ----------
def split_train_test(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

# ---------- Save helpers ----------
def save_outputs(outdir, name, metrics_dict):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"{name}_metrics.json"), "w") as f:
        md = dict(metrics_dict)
        rep = md.pop("report", "")
        json.dump(md, f, indent=2)
    with open(os.path.join(outdir, f"{name}_report.txt"), "w") as f:
        f.write(metrics_dict.get("report", ""))
