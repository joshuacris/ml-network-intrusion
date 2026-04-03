# Model Comparison Study: XGBoost

## Model Description

XGBoost stands for Extreme Gradient Boosting and is an ML framework that provides a gradient-boosted decision tree (GBDT) implementation that is scalable and distributed. Similar to Random Forest and the topics we learned in class, a GBDT is an ensemble decision tree learning algorithm. It works by combining multiple decision trees to build the final model, but structures and mixes trees differently compared to the RF algorithm. RF builds decision trees independently in parallel via bagging, while XGBoost uses gradient boosting to build a new tree that learns from the previous tree's "mistakes." Boosting is a technique that combines smaller and weaker models together to create a stronger, collective model. Gradient boosting improves regular boosting by building models, in XGBoost's case decision trees, sequentially, where each new model learns and corrects the errors of the previous ensemble. These errors are represented by setting an objective function and minimizing it using the gradient descent algorithm to create the targeted outcome for the next model. More precisely, each tree fits the negative gradient of the loss. This boosting process thus reduces bias while greatly improving our prediction power.

XGBoost extends the idea of gradient boosting through its use of both the first and second order gradient statistics (gradient and Hessian) on the loss function to find the optimal score, leading to fast convergence and a better tree structure. The objective is also regularized using both L1 (lasso) and L2 (ridge) regularization in the objective function itself to increase its resistance to overfitting. It also takes advantage of shrinkage, which scales new weights by a factor after a step of boosting and thus ensures no one tree has too much influence, and row/column subsampling, which introduces randomness similar to the random forest algorithm. XGBoost also uses sparsity-aware split finding to handle missing values and/or sparse features, which learns a default direction for missing values at each split. Lastly, a weighted quantile sketch is used to determine split efficiency instead of the greedy algorithm other tree-based methods use, which is why XGBoost is scalable to large datasets.

Together these properties make XGBoost faster, more accurate, and more resistant to overfitting than standard gradient boosting and typically other tree-based models such as Random Forest.

## Model Motivation

We selected XGBoost as the advanced/novel model in our project for three main reasons.

1. XGBoost is widely known as the SOTA ML when it comes to classification problems on tabular data. It has been widely used to create benchmarks for
various datasets and is an algorithm that is computationally fast and scales well to large data. We wanted to gauge XGBoost's performance on this set of tabular data with a large number of features (35+) and compare it to the models we learned in class. We are further motivated by the tabularity of our UNSW-NB15 dataset and wanted to see how XGBoost handles these features. 

2. We wanted to compare tree-based models vs neural networks (MLP) on tabular data (with a logistic regression model as a baseline). Neural networks and deep learning are rising extremely fast in popularity and many people are steering toward using these as solutions to their use-cases. One motivation for this project is to show that neural networks/deep learning approaches are not always the best approaches, especially when it  comes to tabular data, and can be outperformed by tree-based models. XGBoost is a popular tree-based model and we wanted to use multiple (adding on random forest) tree-based models to prove this point. 

3. XGBoost represents the current SOTA in tree-based models and extends many concepts covered in lecture (decision trees, ensembles, random forest, gradient descent), making it a natural and well-motivated choice for this comparison.

## Evaluation Metrics

We evaluate XGBoost using the following metrics:

**ROC-AUC (Area Under the ROC Curve):** This is our primary metric for model selection and hyperparameter tuning. ROC-AUC measures the model's ability to discriminate between normal and attack traffic (0 and 1 in this binary classification problem), independent of any classification thresholds, making it robust to class imbalance. A ROC-AUC closer to 1.0 represents a better classifier.

**Precision, Recall, and F1 Score:** These metrics are evaluated at a chosen classification threshold. Precision measures what fraction of predicted attacks are true attacks (minimizing false positives), while recall measures what fraction of actual attacks were detected (minimizing false negatives). F1 score balances precision and recall. In the context of cybersecurity and network intrusion detection, false negatives (missed attacks) are more costly than false positives (flagging normal traffic as malicious), since undetected intrusions can have greater consequences.

**Threshold Tuning:** We use a default classification threshold of 0.5. However, since the optimal model depends on the trade-off between false positives and false negatives, we perform threshold tuning by sweeping thresholds from 0.01 to 0.99 and selecting the value that maximizes F1 score. This yields a better-balanced model than the default using 0.5.

**Per-Attack-Category Breakdown:** We also evaluate and compare precision, recall, and F1 score separately for each of the 9 attack categories in the UNSW-NB15 dataset. This allows us to identify which attack types the model handles well versus which are difficult to detect, providing more actionable insight for future research and deployment.

## Hyperparameter Tuning

We use `RandomizedSearchCV` to tune XGBoost's hyperparameters, sampling 50 random configurations from the following parameter grid with 3-fold cross-validation, scored on ROC-AUC:

| Hyperparameter | Search Space |
|---|---|
| `n_estimators` | 200, 300, 500, 700, 1000 |
| `max_depth` | 3, 4, 5, 6, 8 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `subsample` | 0.6, 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.6, 0.7, 0.8, 1.0 |
| `min_child_weight` | 1, 3, 5 |
| `gamma` | 0, 0.1, 0.3, 0.5 |
| `reg_alpha` | 0, 0.1, 0.5, 1.0 |
| `reg_lambda` | 0.5, 1.0, 2.0, 5.0 |

We also set `scale_pos_weight` to the ratio of negative to positive training samples (approximately 0.929) to account for the slight class imbalance in the training set. This scales up the gradient contribution of the minority class during training.

Compared to an initial search (n_iter=30, no L1/L2 regularization tuning), this expanded search addresses two identified weaknesses: the initial search found `n_estimators=100` as optimal despite the validation loss still decreasing at that point, and `reg_alpha`/`reg_lambda` (XGBoost's explicit L1 and L2 penalties) were not being tuned at all. Increasing `n_iter` to 50 also provides better coverage of the larger 9-parameter space.

The best configuration found was:

| Hyperparameter | Best Value |
|---|---|
| `subsample` | 0.7 |
| `n_estimators` | 1000 |
| `min_child_weight` | 1 |
| `max_depth` | 4 |
| `learning_rate` | 0.01 |
| `gamma` | 0.1 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 1.0 |
| `reg_lambda` | 2.0 |

**Best CV ROC-AUC: 0.9601**

## Interpretation of Results

**Overall Performance:**
The tuned XGBoost model achieves a test set ROC-AUC of **0.9695**, confirming strong ability to distinguish normal and attack traffic. This closely matches the cross-validation ROC-AUC of 0.9601, meaning the model generalizes well and is not overfit.

**Base Evaluation (threshold = 0.5):**
At the default threshold, the model achieves high recall for the Attack class (0.98) at the cost of lower recall for Normal traffic (0.74). Normal traffic precision is 0.98, while Attack precision is 0.70, producing an overall accuracy of 0.83.

**After Threshold Tuning:**
Sweeping the threshold and selecting the value that maximizes F1 score yields a better-balanced model, improving the trade-off between attack detection rate and false alarm rate.

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

**Fuzzers** remain the hardest attack type to detect (recall = 0.6255), consistent with how fuzzing generates randomized or mutated inputs that can resemble normal traffic. **Worms** and **Generic** attacks are detected with near-perfect recall, likely because they produce distinctive and consistent network signatures.

**Comparison with Random Forest:**
XGBoost achieves a higher cross-validation ROC-AUC (0.9601 vs RF's 0.9484), suggesting better generalization during tuning. Random Forest achieves a marginally higher test ROC-AUC (0.9729 vs XGBoost's 0.9695), though the gap narrowed significantly after the extended search (from 0.0088 to 0.0034). Fuzzers recall is now equivalent between the two models (0.6255 vs 0.6255).

RF outperforming XGBoost on held-out data despite losing on CV is surprising given XGBoost's reputation, but can be explained by several factors specific to this dataset and setup. First, XGBoost's sequential boosting mechanism focuses later trees on samples that earlier trees misclassified. When those hard samples are ambiguous (such as Fuzzers), boosting can increase noise rather than signal. RF's bagging averages over this noise instead. Second, the training set is nearly balanced (51890 normal vs 55850 attack), so `scale_pos_weight` provides minimal benefit and may slightly distort gradient updates. Finally, XGBoost's advantage over RF is most pronounced on large-scale datasets with heavy feature engineering; on clean, moderately-sized tabular data it is common for the two models to perform comparably, and RF sometimes wins.

Overall, both tree-based models perform at a high level and both significantly outperform a naive baseline, demonstrating the strength of ensemble methods on tabular network traffic data.

## Source

Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016.