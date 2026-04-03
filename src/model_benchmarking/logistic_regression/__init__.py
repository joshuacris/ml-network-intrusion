"""Logistic Regression hyperparameter tuning and evaluation for UNSW-NB15."""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    f1_score, recall_score
)
from scipy.stats import uniform, loguniform

RANDOM_STATE = 42


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(train_path, test_path):
    """Load preprocessed CSVs and split into X/y."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    X_train = train.drop(columns=["label"])
    y_train = train["label"]
    X_test = test.drop(columns=["label"])
    y_test = test["label"]
    return X_train, y_train, X_test, y_test


# ── Course-Pattern C Sweep ────────────────────────────────────────────────────

def sweep_c_values(X_train, y_train, X_val, y_val, c_values=None,
                   penalty="l2", solver="saga", max_iter=2000):
    """
    Manual C sweep following C11 course pattern.
    For each C value: fit LR, record train and val accuracy.
    Returns DataFrame with columns [C, train_acc, val_acc].
    """
    if c_values is None:
        c_values = np.logspace(-3, 3, 20)

    results = []
    for C in c_values:
        clf = LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=max_iter, random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        results.append({"C": C, "train_acc": train_acc, "val_acc": val_acc})

    return pd.DataFrame(results)


def plot_c_sweep(sweep_df):
    """Plot train vs val accuracy over C values (log-scale x-axis)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_df["C"], sweep_df["train_acc"], "o-", label="Train Accuracy")
    ax.plot(sweep_df["C"], sweep_df["val_acc"], "s-", label="Validation Accuracy")
    ax.set_xscale("log")
    ax.set_xlabel("C (Regularization Parameter)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Logistic Regression: C Sweep (Train vs Validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_idx = sweep_df["val_acc"].idxmax()
    best_c = sweep_df.loc[best_idx, "C"]
    best_acc = sweep_df.loc[best_idx, "val_acc"]
    ax.axvline(x=best_c, color="red", linestyle="--", alpha=0.5,
               label=f"Best C={best_c:.4f} (val_acc={best_acc:.4f})")
    ax.legend()

    plt.tight_layout()
    return fig


# ── Randomized Search ─────────────────────────────────────────────────────────

def run_randomized_search(X_train, y_train, n_iter=50, cv=5,
                          scoring="accuracy", n_jobs=-1):
    """
    Broad hyperparameter search using RandomizedSearchCV.
    Uses saga solver (supports l1, l2, elasticnet).
    """
    param_distributions = [
        {
            "C": loguniform(1e-3, 1e3),
            "penalty": ["l1", "l2"],
            "solver": ["saga"],
            "class_weight": [None, "balanced"],
            "max_iter": [2000],
        },
        {
            "C": loguniform(1e-3, 1e3),
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "l1_ratio": uniform(0.1, 0.8),
            "class_weight": [None, "balanced"],
            "max_iter": [2000],
        },
    ]

    lr = LogisticRegression(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        lr, param_distributions,
        n_iter=n_iter, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring, n_jobs=n_jobs, random_state=RANDOM_STATE, verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


# ── Grid Search Refinement ────────────────────────────────────────────────────

def get_refined_grid(random_search, n_points=5):
    """Build a tight grid around the best RandomizedSearchCV params."""
    best = random_search.best_params_
    best_c = best["C"]

    c_range = np.logspace(
        np.log10(best_c) - 0.5,
        np.log10(best_c) + 0.5,
        n_points
    )

    grid = {
        "C": c_range,
        "penalty": [best["penalty"]],
        "solver": ["saga"],
        "class_weight": [None, "balanced"],
        "max_iter": [2000],
    }

    if best["penalty"] == "elasticnet":
        best_l1 = best["l1_ratio"]
        grid["l1_ratio"] = np.clip(
            np.linspace(best_l1 - 0.2, best_l1 + 0.2, n_points), 0.01, 0.99
        ).tolist()

    return grid


def run_grid_search(X_train, y_train, param_grid, cv=5,
                    scoring="accuracy", n_jobs=-1):
    """Run GridSearchCV with the refined parameter grid."""
    lr = LogisticRegression(random_state=RANDOM_STATE)
    search = GridSearchCV(
        lr, param_grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        scoring=scoring, n_jobs=n_jobs, verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Full evaluation: accuracy, confusion matrix, classification report, ROC, PR."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report_text": classification_report(y_test, y_pred,
                                                            target_names=["Normal", "Attack"]),
        "classification_report_dict": classification_report(y_test, y_pred,
                                                            target_names=["Normal", "Attack"],
                                                            output_dict=True),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "precision_curve": precision, "recall_curve": recall,
        "avg_precision": avg_precision,
    }


# ── Threshold Tuning (Originality) ───────────────────────────────────────────

def find_best_threshold(model, X_val, y_val, metric="f1", thresholds=None):
    """
    Tune decision threshold to maximize F1 or recall.
    Returns (best_threshold, scores_df).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)

    y_prob = model.predict_proba(X_val)[:, 1]
    results = []
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        results.append({
            "threshold": t,
            "f1": f1_score(y_val, y_pred_t),
            "recall": recall_score(y_val, y_pred_t),
            "accuracy": accuracy_score(y_val, y_pred_t),
        })

    df = pd.DataFrame(results)
    best_idx = df[metric].idxmax()
    return df.loc[best_idx, "threshold"], df


def plot_threshold_analysis(threshold_df):
    """Plot F1, recall, and accuracy vs decision threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_df["threshold"], threshold_df["f1"], "o-", label="F1", markersize=3)
    ax.plot(threshold_df["threshold"], threshold_df["recall"], "s-", label="Recall", markersize=3)
    ax.plot(threshold_df["threshold"], threshold_df["accuracy"], "^-", label="Accuracy", markersize=3)
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Decision Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(results, title="Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        results["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(results, title="ROC Curve"):
    """Plot ROC curve with AUC annotation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results["fpr"], results["tpr"],
            label=f'ROC Curve (AUC = {results["roc_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(results, title="Precision-Recall Curve"):
    """Plot PR curve with Average Precision annotation."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(results["recall_curve"], results["precision_curve"],
            label=f'PR Curve (AP = {results["avg_precision"]:.4f})')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_search_results(search_cv, n_top=10):
    """Return top-N search results as a DataFrame."""
    results_df = pd.DataFrame(search_cv.cv_results_)
    results_df = results_df.sort_values("rank_test_score").head(n_top)
    display_df = results_df[["params", "mean_test_score", "std_test_score",
                             "mean_train_score"]].reset_index(drop=True)
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"
    display_df.columns = ["Parameters", "Mean CV Score", "Std CV Score", "Mean Train Score"]
    return display_df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path):
    """Save model with joblib. Creates parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def save_results(results, path):
    """Save evaluation results dict with joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    joblib.dump(serializable, path)
    print(f"Results saved to {path}")
