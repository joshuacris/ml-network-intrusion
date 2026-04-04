# Model Comparison Study: Random Forest

## Model Description

Random Forest is an ensemble learning algorithm that builds a large collection of decorrelated decision trees and aggregates their predictions through majority voting (classification) or averaging (regression). Each tree is trained on a bootstrap sample of the training data (bagging), and at each split only a random subset of features is considered as candidates. These two sources of randomness—row sampling and feature subsampling—prevent any single tree from dominating and reduce the variance of the ensemble compared to a single decision tree, without meaningfully increasing bias.

Because trees are grown independently in parallel, Random Forest does not use any form of boosting or sequential correction. This distinguishes it from gradient-boosted methods: RF trades the bias-reduction benefits of boosting for lower sensitivity to noisy or ambiguous samples, since errors are averaged out rather than amplified. The result is a robust, low-variance model that generalizes well across a wide range of tabular datasets with minimal preprocessing.

RF's interpretability is also a practical strength—feature importances derived from impurity reduction provide a straightforward ranking of which inputs drive predictions, and the model requires no assumption about the distribution of input features.

## Model Motivation

We selected Random Forest as a strong tree-based baseline for three main reasons.

1. Random Forest is one of the most well-established and widely-used algorithms for tabular classification tasks. It serves as a natural reference point against which more complex models (XGBoost, MLP) can be compared, particularly on structured network traffic data with many numerical features.

2. We wanted to evaluate whether boosting (XGBoost) meaningfully outperforms bagging (Random Forest) on this specific dataset and problem setting. Network intrusion detection data is relatively clean and moderately sized, which is a regime where the two approaches are known to perform comparably. Including both lets us rigorously test this hypothesis.

3. Random Forest directly extends core lecture concepts—decision trees and ensemble methods—making it a motivated and pedagogically grounded inclusion alongside the more advanced models in the comparison.

## Evaluation Metrics

We evaluate Random Forest using the same metrics applied to all models in this study:

**ROC-AUC (Area Under the ROC Curve):** Our primary metric for model selection and hyperparameter tuning. ROC-AUC measures the model's ability to discriminate between normal and attack traffic, independent of any classification threshold, making it robust to class imbalance. A ROC-AUC closer to 1.0 represents a better classifier.

**Precision, Recall, and F1 Score:** Evaluated at a chosen classification threshold. Precision measures what fraction of predicted attacks are true attacks (minimizing false positives), while recall measures what fraction of actual attacks were detected (minimizing false negatives). F1 score balances precision and recall. In the context of cybersecurity and network intrusion detection, false negatives (missed attacks) are more costly than false positives, since undetected intrusions can have greater consequences.

**Threshold Tuning:** We use a default classification threshold of 0.5 as a baseline, then sweep thresholds from 0.01 to 0.99 and select the value that maximizes F1 score. This yields a better-balanced model than the default.

**Per-Attack-Category Breakdown:** We also evaluate precision, recall, and F1 score separately for each of the 9 attack categories in the UNSW-NB15 dataset by joining predictions against the raw test file (`UNSW_NB15_testing-set.csv`). This allows us to identify which attack types the model handles well versus which are difficult to detect, providing more actionable insight for future research and deployment.

## Hyperparameter Tuning

We use `RandomizedSearchCV` to tune Random Forest's hyperparameters, sampling 30 random configurations from the following parameter grid with 3-fold cross-validation, scored on ROC-AUC:

| Hyperparameter | Search Space |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | None, 10, 20, 30, 40 |
| `min_samples_split` | 2, 5, 10, 20 |
| `min_samples_leaf` | 1, 2, 4, 8 |
| `max_features` | 'sqrt', 'log2', 0.3, 0.5 |
| `bootstrap` | True, False |

The best configuration found was:

| Hyperparameter | Best Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 10 |
| `min_samples_split` | 20 |
| `min_samples_leaf` | 4 |
| `max_features` | 0.3 |
| `bootstrap` | True |

**Best CV ROC-AUC: 0.9484**

## Interpretation of Results

**Overall Performance:**
The tuned Random Forest model achieves a test set ROC-AUC of **0.9729**, confirming strong discriminative ability between normal and attack traffic. The test ROC-AUC exceeds the cross-validation ROC-AUC of 0.9484, indicating the model generalizes well to unseen data without overfitting.

**Base Evaluation (threshold = 0.5):**
At the default threshold, the model achieves high recall for the Attack class (0.99) at the cost of lower recall for Normal traffic (0.71). Normal traffic precision is 0.99, while Attack precision is 0.68, producing an overall accuracy of 0.82. This trade-off reflects the model's tendency to predict attacks at the default threshold.

**After Threshold Tuning:**
Sweeping the threshold and selecting the value that maximizes F1 score yields a substantially better-balanced model. This yields only **282 false positives** across the 55,945-row test set—a very low false alarm rate that minimizes alert fatigue in a Security Operations Center (SOC) context.

**Per-Attack-Category Breakdown:**
At the tuned threshold, all attack categories achieve perfect precision (1.0), meaning no normal traffic is misclassified as any specific attack type. Recall varies substantially by category:

| Attack Type | Recall | F1 |
|---|---|---|
| Fuzzers | 0.6255 | 0.7696 |
| Analysis | 0.8812 | 0.9368 |
| Shellcode | 0.9656 | 0.9825 |
| Exploits | 0.9823 | 0.9910 |
| DoS | 0.9866 | 0.9933 |
| Reconnaissance | 0.9970 | 0.9985 |
| Backdoor | 0.9971 | 0.9986 |
| Generic | 0.9973 | 0.9986 |
| Worms | 1.0000 | 1.0000 |

**Fuzzers** remain the hardest attack type to detect (recall = 0.6255), consistent with how fuzzing generates randomized or mutated inputs that can closely resemble normal traffic. **Worms** and **Generic** attacks are detected with near-perfect recall, likely because they produce distinctive and consistent network signatures.

**Comparison with XGBoost:**
Random Forest achieves a higher test ROC-AUC (0.9729 vs XGBoost's 0.9695), despite a lower cross-validation ROC-AUC (0.9484 vs 0.9601). This reversal is notable: XGBoost generalizes better during tuning but RF edges it out on the held-out test set.

RF outperforming XGBoost on held-out data despite losing on CV can be explained by several factors specific to this dataset. First, the training set is nearly balanced (51,890 normal vs 55,850 attack), which limits XGBoost's boosting advantage—when classes are balanced, there is less gradient signal to exploit from hard samples. Second, RF's bagging averages over noise from ambiguous samples (such as Fuzzers), while XGBoost's sequential correction can amplify noise on those samples. Finally, XGBoost's advantage over RF is most pronounced on large-scale datasets with heavy feature engineering; on clean, moderately-sized tabular data, RF often matches or exceeds it.

Overall, both tree-based models perform at a high level and significantly outperform naive baselines, reinforcing the strength of ensemble methods on tabular network traffic data.

## Source

Breiman, Leo. "Random forests." Machine learning 45 (2001): 5-32.
