# ml-network-intrusion

CSCC11 Final Project — Binary network intrusion detection comparison study using the UNSW-NB15 dataset.

Dataset: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data

## Report

The final PDF report is at [`final_report_project/main.pdf`](final_report_project/main.pdf).

LaTeX source is in [`final_report_project/`](final_report_project/):

```
final_report_project/
├── main.tex                          # Root document
├── Title/
│   ├── coverpage.tex
│   └── Abstract.tex
├── Ch1-Introduction/
├── Ch2-Data/
├── Ch3-Model-Comparison/             # Model descriptions, tuning, main results
├── Ch4-Originality-Future-Work/
├── Ch6-Appendix/
└── Ref/References.bib
```

## Notebooks

Main work is in [`notebooks/`](notebooks/). Each notebook covers one model end-to-end (data loading → hyperparameter tuning → threshold tuning → per-attack breakdown):

| Notebook | Description |
|---|---|
| [`preprocessing.ipynb`](notebooks/preprocessing.ipynb) | Duplicate removal, feature engineering, produces processed CSVs |
| [`logistic_regression.ipynb`](notebooks/logistic_regression.ipynb) | Logistic Regression baseline |
| [`mlp.ipynb`](notebooks/mlp.ipynb) | Multi-Layer Perceptron |
| [`random_forest_classifier.ipynb`](notebooks/random_forest_classifier.ipynb) | Random Forest |
| [`xgboost.ipynb`](notebooks/xgboost.ipynb) | XGBoost (selected final model) |

PDF exports of all notebooks are in [`report/notebook_pdfs/`](report/notebook_pdfs/).

## Modularized Source

Key logic is also modularized into [`src/`](src/) for standalone use:

```
src/
├── preprocessing/
│   └── preprocessing.py              # Full preprocessing pipeline
└── model_benchmarking/
    ├── logistic_regression/
    ├── mlp/
    │   └── mlp.py
    ├── random_forest/
    │   └── random_forest.py          # load_data, train_model, evaluate, per_attack_breakdown
    └── xgboost/
        └── xgboost.py
```

## Data

```
data/
├── raw/                              # Original UNSW-NB15 CSV files (gitignored)
└── processed/                        # Output of preprocessing.ipynb
    ├── training_lr_mlp.csv           # Scaled — used by LR and MLP
    ├── test_lr_mlp.csv
    ├── training_xgb_rf.csv           # Unscaled — used by Random Forest and XGBoost
    └── test_xgb_rf.csv
```

Preprocessed files are committed. Raw CSVs and `artifacts.pkl` are gitignored.

## Plots

Figures embedded in the report are saved to [`report/plots/`](report/plots/).
