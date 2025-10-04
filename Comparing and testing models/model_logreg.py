# model_logreg.py
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ml_utils import load_data, make_preprocessor, split_train_test, metrics_with_pr, save_outputs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


OUTDIR = "outputs/logreg"

def main():
    df, X, y, groups, num_cols, cat_cols, classes = load_data()
    Xtr, Xte, ytr, yte = split_train_test(X, y)

    pipe = Pipeline([
        ("pre", make_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(Xtr, ytr)
    ypred = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)
    metrics = metrics_with_pr(yte, ypred, proba, classes)
    print(metrics)
    save_outputs(OUTDIR, "logreg", metrics)
    
    
    
    cm = pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/logreg_confusion_matrix.png")
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
    plt.title("Performance Metrics - Logistic Regression")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/logreg_scores.png")
    plt.close()


if __name__ == "__main__":
    main()
