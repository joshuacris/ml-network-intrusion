import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from sklearn.model_selection import RandomizedSearchCV


DATA_TRAIN = '../data/processed/training_xgb_rf.csv'
DATA_TEST  = '../data/processed/test_xgb_rf.csv'
RAW_TEST   = '../data/raw/UNSW_NB15_testing-set.csv'

PARAM_DIST = {
    'n_estimators':      [100, 200, 300, 500],
    'max_depth':         [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf':  [1, 2, 4, 8],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'bootstrap':         [True, False],
}

BEST_PARAMS = {
    'n_estimators':      300,
    'max_depth':         10,
    'min_samples_split': 20,
    'min_samples_leaf':  4,
    'max_features':      0.3,
    'bootstrap':         True,
}


def load_data():
    train = pd.read_csv(DATA_TRAIN)
    test  = pd.read_csv(DATA_TEST)
    X_train = train.drop(columns=['label'])
    y_train = train['label']
    X_test  = test.drop(columns=['label'])
    y_test  = test['label']
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')
    print(f'Class balance (train) — 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}')
    return X_train, y_train, X_test, y_test


def tune_hyperparameters(X_train, y_train):
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=PARAM_DIST,
        n_iter=30,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f'Best ROC-AUC (CV): {search.best_score_:.4f}')
    print(f'Best params:       {search.best_params_}')
    return search


def train_model(X_train, y_train, params=None):
    if params is None:
        params = BEST_PARAMS
    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def evaluate_default(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix (threshold=0.5)')
    plt.tight_layout()
    plt.show()
    return y_prob


def tune_threshold(y_test, y_prob):
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f'Optimal threshold (max F1): {best_threshold:.2f}  — F1: {max(f1_scores):.4f}')
    print(f'Default threshold (0.5):           — F1: {f1_scores[49]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, f1_scores, color='darkorange', label='F1')
    ax.axvline(best_threshold, color='darkorange', linestyle='--',
               label=f'Max F1 threshold = {best_threshold:.2f}')
    ax.axvline(0.5, color='gray', linestyle='--', label='Default = 0.50')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Classification Threshold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (t, title) in zip(axes, [
        (0.5,            'Default (0.50)'),
        (best_threshold, f'Max F1 ({best_threshold:.2f})'),
    ]):
        preds = (y_prob >= t).astype(int)
        sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title(title)
    plt.suptitle('Confusion Matrices — Threshold Comparison', y=1.02)
    plt.tight_layout()
    plt.show()

    print(f'\nChosen threshold: {best_threshold:.2f} (max F1)')
    y_pred_tuned = (y_prob >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_tuned, target_names=['Normal', 'Attack']))
    return best_threshold, y_pred_tuned


def per_attack_breakdown(y_test, y_pred_tuned, y_prob):
    raw_test = pd.read_csv(RAW_TEST)
    raw_test = raw_test.drop_duplicates(
        subset=[c for c in raw_test.columns if c != 'id']
    ).reset_index(drop=True)

    results = raw_test[['attack_cat']].copy()
    results['y_true'] = y_test.values
    results['y_pred'] = y_pred_tuned
    results['y_prob'] = y_prob

    def attack_metrics(group):
        tp = ((group['y_true'] == 1) & (group['y_pred'] == 1)).sum()
        fn = ((group['y_true'] == 1) & (group['y_pred'] == 0)).sum()
        fp = ((group['y_true'] == 0) & (group['y_pred'] == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return pd.Series({'count': len(group), 'precision': round(precision, 4),
                          'recall': round(recall, 4), 'f1': round(f1, 4)})

    attack_results = results[results['attack_cat'] != 'Normal']
    breakdown = attack_results.groupby('attack_cat').apply(
        attack_metrics, include_groups=False
    ).sort_values('recall')
    print(breakdown.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    breakdown[['precision', 'recall', 'f1']].plot(kind='bar', ax=ax, edgecolor='white')
    ax.set_title('Random Forest — Per-Attack-Category Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()
    return breakdown


def plot_feature_importance(model, X_train):
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 5))
    importances.head(20).plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Random Forest — Top 20 Feature Importances')
    ax.set_ylabel('Importance')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return importances


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    # To run hyperparameter tuning:
    # search = tune_hyperparameters(X_train, y_train)
    # model = search.best_estimator_

    # To use best known params directly (skip tuning):
    model = train_model(X_train, y_train)

    y_prob = evaluate_default(model, X_test, y_test)
    best_threshold, y_pred_tuned = tune_threshold(y_test, y_prob)
    per_attack_breakdown(y_test, y_pred_tuned, y_prob)
    plot_feature_importance(model, X_train)
