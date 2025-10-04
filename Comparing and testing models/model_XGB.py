# model_XGB.py
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from xgboost import XGBClassifier

from ml_utils import (
    load_data, make_preprocessor, split_train_test,
    metrics_with_pr, save_outputs
)

import matplotlib.pyplot as plt
import seaborn as sns

OUTDIR = "outputs/xgb"

def main():
    # Load
    df, X, y_text, groups, num_cols, cat_cols, classes_text = load_data()

    # --- Drop columns that are entirely NaN (prevents imputer warnings)
    all_nan_cols = X.columns[X.isna().all(axis=0)].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
        # also update num_cols/cat_cols
        num_cols = [c for c in num_cols if c not in all_nan_cols]
        cat_cols = [c for c in cat_cols if c not in all_nan_cols]

    # --- Encode labels for XGB: 0..K-1
    le = LabelEncoder()
    y = le.fit_transform(y_text)        # numeric labels
    classes = le.classes_.tolist()      # keep for reports

    # Split
    Xtr, Xte, ytr, yte = split_train_test(X, y)

    # Pipeline:
    #  - our standard preprocessor (impute+scale+one-hot)
    #  - convert to dense to avoid sparse issues on some setups
    #  - XGBClassifier configured for multiclass
    pipe = Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("to_dense", FunctionTransformer(lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x),
                                         accept_sparse=True)),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes),
            eval_metric="mlogloss",
            tree_method="hist",          # fast & stable
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train
    pipe.fit(Xtr, ytr)

    # Predict
    ypred_idx = pipe.predict(Xte)             # ints 0..K-1
    proba = pipe.predict_proba(Xte)           # (n, K)
    yte_text = le.inverse_transform(yte)      # back to strings for metrics
    ypred_text = le.inverse_transform(ypred_idx)

    # Metrics
    metrics = metrics_with_pr(yte_text, ypred_text, proba, classes)
    print(metrics)
    os.makedirs(OUTDIR, exist_ok=True)
    save_outputs(OUTDIR, "xgb", metrics)

    # ---- PLOTS (same style as your LogReg) ----
    # Confusion Matrix
    cm = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - XGBoost")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "xgb_confusion_matrix.png"))
    plt.close()

    # Scores Bar Chart
    scores = {
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["f1_macro"],
        "PR-AUC Macro": metrics["pr_auc_macro"],
        "PR-AUC Confirmed": metrics["pr_auc_confirmed"]
    }
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    plt.title("Performance Metrics - XGBoost")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "xgb_scores.png"))
    plt.close()

if __name__ == "__main__":
    main()
