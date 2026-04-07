# ml-network-intrusion

CSCC11 Final Project — Binary network intrusion detection comparison study using the UNSW-NB15 dataset.

Dataset: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/data

## Setup

**Requirements:** Python 3.12+

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Notebooks

Start JupyterLab from the repo root:

```bash
jupyter lab
```

Run notebooks in this order:

1. `notebooks/preprocessing.ipynb` — must run first; produces the processed CSVs in `data/processed/`
2. Any model notebook in any order (`logistic_regression`, `mlp`, `random_forest_classifier`, `xgboost`)

Each model notebook skips hyperparameter tuning by default (best params are hardcoded). To re-run tuning, comment out the direct instantiation cell and uncomment the `RandomizedSearchCV` cell.

## Running the Modularized Scripts

Each script under `src/` can be run standalone from the repo root:

```bash
# Example: run Random Forest pipeline
python src/model_benchmarking/random_forest/random_forest.py

# Example: run Logistic Regression pipeline
python src/model_benchmarking/logistic_regression/logistic_regression.py

# Example: run MLP pipeline
python src/model_benchmarking/mlp/mlp.py

# Example: run XGBoost pipeline
python src/model_benchmarking/xgboost/xgboost.py
```

Scripts expect the processed CSVs to already exist in `data/processed/`. Run `preprocessing.ipynb` first if they are missing.

## Building the Report

The PDF is already compiled at `final_report_project/main.pdf`. To recompile from LaTeX source (requires [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/)):

```bash
cd final_report_project
latexmk -pdf main.tex
```

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
