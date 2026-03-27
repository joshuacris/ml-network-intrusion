import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLS_TO_DROP = ['id', 'attack_cat', 'stcpb', 'dtcpb']

PROTO_TOP   = ['tcp', 'udp', 'unas', 'arp', 'ospf', 'sctp']
SERVICE_TOP = ['-', 'http', 'dns', 'smtp', 'ftp-data', 'ftp', 'ssh', 'pop3']
STATE_TOP   = ['FIN', 'INT', 'CON', 'REQ']

SKEWED_COLS = [
    'sload', 'dload', 'sbytes', 'dbytes', 'sjit', 'djit',
    'rate', 'sinpkt', 'dinpkt', 'spkts', 'dpkts',
    'sloss', 'dloss', 'response_body_len'
]

def drop_irrelevant(df):
    return df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

def categorical_processing(df):
    df['proto_bucket']   = df['proto'].where(df['proto'].isin(PROTO_TOP), other='other')
    df['service_bucket'] = df['service'].where(df['service'].isin(SERVICE_TOP), other='other')
    df['state_bucket']   = df['state'].where(df['state'].isin(STATE_TOP), other='other')
    return df

def one_hot_encoding(df, train_ohe_cols=None):
    """
    One-hot encode categorical columns
    train_ohe_cols: dict of {prefix: [columns]} from training
    Returns (df_with_ohe, ohe_cols_dict)
    """
    ohe_cols = {}
    for col, prefix in [('proto_bucket', 'proto'), ('service_bucket', 'service'), ('state_bucket', 'state')]:
        dummies = pd.get_dummies(df[col], prefix=prefix).astype(int)
        if train_ohe_cols is not None:
            dummies = dummies.reindex(columns=train_ohe_cols[prefix], fill_value=0)
        ohe_cols[prefix] = dummies.columns.tolist()
        df = pd.concat([df, dummies], axis=1)
    return df, ohe_cols

def log_transform(df, num_cols):
    skewed = [c for c in SKEWED_COLS if c in num_cols]
    for col in skewed:
        df[f'{col}_log'] = np.log1p(df[col])
    log_cols = [f'{c}_log' for c in skewed]
    return df, log_cols

def scale(df, log_cols, non_skewed_cols, scaler=None):
    cols_to_scale = log_cols + non_skewed_cols
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[cols_to_scale])
    else:
        scaled_values = scaler.transform(df[cols_to_scale])
    scaled_cols = [f'{c}_scaled' for c in cols_to_scale]
    df = pd.concat([df, pd.DataFrame(scaled_values, columns=scaled_cols, index=df.index)], axis=1)
    return df, scaled_cols, scaler

def build_outputs(df, ohe_cols, scaled_cols, num_cols):
    all_ohe = [c for cols in ohe_cols.values() for c in cols]
    df_full = df.copy()
    df_model = df[all_ohe + scaled_cols + ['label']].copy()
    df_model_tree = df[all_ohe + num_cols + ['label']].copy()
    return df_full, df_model, df_model_tree

def preprocess_train(df):
    """
    Fit and apply all preprocessing to the training set.
    Returns: df_full (every column, original + preprocessed),
    df_model: df for LR/MLP training (OHE + scaled),
    df_model_tree: df for RF/XGB (OHE + raw numerical),
    artifacts: dicts for test set preprocessing
    """
    df = df.copy()

    assert df.isnull().sum().sum() == 0, f"Null values found in training data"

    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'id']).reset_index(drop=True)
    df = drop_irrelevant(df)
    df = categorical_processing(df)

    num_cols = df.select_dtypes(include='number').columns.difference(['label']).tolist()
    non_skewed_cols = [c for c in num_cols if c not in SKEWED_COLS]

    df, ohe_cols = one_hot_encoding(df)

    df, log_cols = log_transform(df, num_cols)
    df, scaled_cols, scaler = scale(df, log_cols, non_skewed_cols)

    df_full, df_model, df_model_tree = build_outputs(df, ohe_cols, scaled_cols, num_cols)

    artifacts = {
        'scaler': scaler,
        'ohe_cols': ohe_cols,
        'num_cols': num_cols,
        'non_skewed_cols': non_skewed_cols,
        'log_cols': log_cols,
        'scaled_cols': scaled_cols,
    }

    return df_full, df_model, df_model_tree, artifacts

def preprocess_test(df, artifacts):
    df = df.copy()

    df = df.drop_duplicates(subset=[c for c in df.columns if c != 'id']).reset_index(drop=True)
    df = drop_irrelevant(df)
    df = categorical_processing(df)
    df, _ = one_hot_encoding(df, train_ohe_cols=artifacts['ohe_cols'])

    df, _ = log_transform(df, artifacts['num_cols'])
    df, _, _ = scale(df, artifacts['log_cols'], artifacts['non_skewed_cols'], scaler=artifacts['scaler'])

    df_full, df_model, df_model_tree = build_outputs(df, artifacts['ohe_cols'], 
    artifacts['scaled_cols'], artifacts['num_cols'])

    return df_full, df_model, df_model_tree

def save_artifacts(artifacts, path):
    joblib.dump(artifacts, path)
    print(f"Artifacts saved to {path}")

def load_artifacts(path):
    return joblib.load(path)