# model_rf.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from ml_utils import load_data, make_preprocessor, split_train_test, metrics_with_pr, save_outputs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
OUTDIR = "outputs/rf"

def main():
    df, X, y, groups, num_cols, cat_cols, classes = load_data()
    Xtr, Xte, ytr, yte = split_train_test(X, y)

    pipe = Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),  # scaling harmless
        ("clf", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ))
    ])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)
    metrics = metrics_with_pr(yte, ypred, proba, classes)
    print(metrics)
    save_outputs(OUTDIR, "rf", metrics)


    cm = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/rf_confusion_matrix.png")
    plt.close()

    # ---- Plot accuracy / F1 / PR-AUC as bar chart ----
    scores = {
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["f1_macro"],
        "PR-AUC Macro": metrics["pr_auc_macro"],
        "PR-AUC Confirmed": metrics["pr_auc_confirmed"]
    }
    plt.figure(figsize=(6,4))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    plt.title("Performance Metrics - Random Forest")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/rf_scores.png")
    plt.close()




if __name__ == "__main__":
    main()
