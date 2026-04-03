import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)


def build_base_model(y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )


def tune_model(base_model, X_train, y_train, n_iter=50, cv=3):
    param_dist = {
        'n_estimators':     [200, 300, 500, 700, 1000],
        'max_depth':        [3, 4, 5, 6, 8],
        'learning_rate':    [0.01, 0.05, 0.1, 0.2],
        'subsample':        [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma':            [0, 0.1, 0.3, 0.5],
        'reg_alpha':        [0, 0.1, 0.5, 1.0],
        'reg_lambda':       [0.5, 1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X_train, y_train)
    print(f'Best ROC-AUC (CV): {search.best_score_:.4f}')
    print(f'Best params:       {search.best_params_}')
    return search.best_estimator_


def evaluate(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    print(f'ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}')
    return y_prob, y_pred


def tune_threshold(y_test, y_prob):
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Optimal threshold (max F1): {best_threshold:.2f}  — F1: {max(f1_scores):.4f}')
    print(f'Default threshold (0.5):           — F1: {f1_scores[49]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds, f1_scores, color='darkorange', label='F1')
    ax.axvline(best_threshold, color='darkorange', linestyle='--', label=f'Max F1 = {best_threshold:.2f}')
    ax.axvline(0.5, color='gray', linestyle='--', label='Default = 0.50')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Classification Threshold')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return best_threshold


def plot_confusion_matrices(y_test, y_prob, best_threshold):
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
    print(classification_report(y_test, (y_prob >= best_threshold).astype(int), target_names=['Normal', 'Attack']))


def per_attack_breakdown(raw_test_path, y_test, y_pred, y_prob):
    raw_test = pd.read_csv(raw_test_path)
    raw_test = raw_test.drop_duplicates(subset=[c for c in raw_test.columns if c != 'id']).reset_index(drop=True)

    results = raw_test[['attack_cat']].copy()
    results['y_true'] = y_test.values
    results['y_pred'] = y_pred
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
    breakdown = attack_results.groupby('attack_cat').apply(attack_metrics, include_groups=False).sort_values('recall')
    print(breakdown.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    breakdown[['precision', 'recall', 'f1']].plot(kind='bar', ax=ax, edgecolor='white')
    ax.set_title('XGBoost — Per-Attack-Category Metrics')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return breakdown


def plot_feature_importance(model, X_train, top_n=20):
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 5))
    importances.head(top_n).plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('XGBoost — Top 20 Feature Importances')
    ax.set_ylabel('Importance')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return importances
