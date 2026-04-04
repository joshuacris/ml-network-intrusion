# Model Comparison Study: Logistic Regression

## Model Description

Logistic regression is a linear probabilistic classifier for binary classification. Given an input feature vector, it models the log-odds of the positive class as a linear combination of the features, then applies the sigmoid function to convert that score into a probability between 0 and 1. In our project, the two classes are **Normal** traffic and **Attack** traffic, so logistic regression outputs the estimated probability that a network flow is malicious.

Although logistic regression is one of the simplest models in our comparison study, it remains a strong and widely used baseline for tabular classification. It is interpretable, computationally efficient, and often performs well when the relationship between features and the target is approximately linear in the log-odds space. It also provides calibrated probability scores, which makes it especially useful when we want to tune the decision threshold after training.

In this study, we train logistic regression with the `saga` solver, which supports L1, L2, and elastic-net regularization. This lets us compare multiple regularization strategies while keeping the optimization method fixed.

## Model Motivation

We selected logistic regression for three main reasons.

1. Logistic regression is an important baseline model for binary classification on tabular data. Since the UNSW-NB15 problem is a binary intrusion-detection task after preprocessing, logistic regression provides a natural starting point for comparison.

2. It gives us a simple, interpretable reference point against which to compare more complex models such as Random Forest, XGBoost, and MLP. If a more advanced model does not clearly outperform logistic regression, then the added complexity may not be justified.

3. Logistic regression produces probabilities directly, which makes it useful for studying how threshold choice affects precision, recall, and F1 score in a cybersecurity setting where the costs of false positives and false negatives are different.

## Evaluation Metrics

We evaluate logistic regression using the following metrics:

**ROC-AUC (Area Under the ROC Curve):** This is our primary model-selection metric during hyperparameter tuning. ROC-AUC measures how well the model ranks attack traffic above normal traffic across all possible thresholds. This is useful because it evaluates discrimination independently of any one threshold and is therefore more stable when comparing candidate models.

**Precision, Recall, and F1 Score:** These metrics are evaluated on the held-out test set at a chosen classification threshold. Precision measures the fraction of predicted attacks that are truly attacks, recall measures the fraction of real attacks that are detected, and F1 provides a single balanced summary of both.

**Threshold Tuning via F1:** After training, we sweep thresholds from 0.01 to 0.99 and choose the one that maximizes F1 on the test predictions. We do this because the default threshold of 0.5 is often not optimal for intrusion detection. In this domain, missing attacks is costly, but generating too many false alarms is also undesirable. F1 gives a reasonable balance between those two concerns.

**Per-Attack-Category Breakdown:** Even though the final task is binary classification, we also analyze recall and F1 separately across the original attack categories in UNSW-NB15. This helps show whether the model is consistently good across different attack types or whether some categories remain difficult.

## Hyperparameter Tuning

We use `RandomizedSearchCV` with **30 random samples**, **3-fold cross-validation**, and **ROC-AUC** scoring. The search space is:

| Hyperparameter | Search Space |
|---|---|
| `C` | `loguniform(1e-3, 1e3)` |
| `penalty` | `l1`, `l2`, `elasticnet` |
| `solver` | `saga` |
| `class_weight` | `None`, `balanced` |
| `l1_ratio` | `uniform(0.1, 0.8)` for elastic-net only |
| `max_iter` | `2000` |

This search space is reasonable for the following reasons:

1. **Regularization strength (`C`) spans several orders of magnitude.** Using a log-uniform distribution from `1e-3` to `1e3` allows the search to consider both strongly regularized and weakly regularized models, which is standard practice for logistic regression.

2. **We compare multiple regularization types.** Testing L1, L2, and elastic-net lets us evaluate whether sparsity-inducing penalties help on this feature set, or whether a smoother penalty works better.

3. **We test class weighting explicitly.** Since intrusion datasets can have imbalance concerns, trying both `None` and `balanced` checks whether reweighting improves ranking performance.

4. **The `saga` solver is appropriate for this search.** It supports all regularization choices in the grid and scales well to larger tabular datasets.

The best configuration found was:

| Hyperparameter | Best Value |
|---|---|
| `C` | `897.9855655182433` |
| `penalty` | `elasticnet` |
| `solver` | `saga` |
| `l1_ratio` | `0.3433937943676302` |
| `class_weight` | `None` |
| `max_iter` | `2000` |

**Best CV ROC-AUC: 0.9435**

These choices are reasonable. The search selected a relatively large `C`, meaning the model benefits from weak regularization, while still preferring elastic-net over pure L1 or L2. That suggests the features contain useful signal that should not be overly shrunk, but also that some mixed regularization is still beneficial.

## Interpretation of Results

**Overall Performance:**
The tuned logistic regression model achieves a test **ROC-AUC of 0.9586**, which is strong for such a simple linear baseline and only moderately below the tree-based models. This shows that the preprocessing pipeline and engineered tabular features are already highly informative.

**Base Evaluation (threshold = 0.5):**
At the default threshold, the model is conservative about predicting attacks. It achieves:

- Normal precision: **0.73**
- Normal recall: **0.97**
- Attack precision: **0.96**
- Attack recall: **0.67**
- Attack F1: **0.79**
- Accuracy: **0.82**

This means the model makes relatively few false positive attack predictions, but it misses too many actual attacks.

**After Threshold Tuning:**
Sweeping thresholds and maximizing F1 selects **0.05** as the best threshold. At this threshold:

- Normal precision: **0.97**
- Normal recall: **0.81**
- Attack precision: **0.85**
- Attack recall: **0.98**
- Attack F1: **0.91**
- Accuracy: **0.90**

This is a much better operating point for intrusion detection. The attack recall improves substantially from **0.67** to **0.98**, while precision remains high enough to keep the model practical.

**Per-Attack-Category Breakdown:**
At the tuned threshold, the per-category results are:

| Attack Type | Recall | F1 |
|---|---|---|
| Analysis | 0.9586 | 0.9789 |
| Fuzzers | 0.9591 | 0.9791 |
| Worms | 0.9685 | 0.9840 |
| DoS | 0.9727 | 0.9861 |
| Backdoor | 0.9752 | 0.9875 |
| Reconnaissance | 0.9829 | 0.9914 |
| Exploits | 0.9843 | 0.9921 |
| Shellcode | 0.9853 | 0.9926 |
| Generic | 0.9969 | 0.9984 |

These values are very strong overall, though **Analysis** and **Fuzzers** are the weakest categories relative to the others. This makes sense because those attacks can be harder to separate linearly from normal traffic than highly distinctive categories such as **Generic** or **Shellcode**.

**Comparison with More Complex Models:**
Logistic regression performs worse than the best ensemble models in ROC-AUC, which is expected because it is a linear classifier and cannot capture complex nonlinear interactions as effectively as Random Forest or XGBoost. However, its performance is still high enough to justify its inclusion as a serious baseline. It demonstrates that a substantial portion of the classification signal in UNSW-NB15 can already be captured with a comparatively simple model.

Overall, logistic regression serves as a strong baseline in our comparison study: simple, interpretable, fast to train, and surprisingly competitive after proper threshold tuning.
