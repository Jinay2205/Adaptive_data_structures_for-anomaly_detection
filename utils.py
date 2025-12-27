import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

def load_nab_file(filepath):
    df = pd.read_csv(filepath)
    # Ensure datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def get_oracle_labels(df, contamination=0.03):
    """
    Trains a Batch Isolation Forest on the ENTIRE dataset to generate
    'Silver Standard' ground truth labels for evaluation.
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    # Train on value column
    X = df['value'].values.reshape(-1, 1)
    preds = model.fit_predict(X)
    # IsolationForest: -1 is anomaly, 1 is normal
    return (preds == -1).astype(int)

def quantize(value, scale=10):
    """Discretizes continuous values into integer buckets for Sketches."""
    return int(value * scale)

def evaluate_predictions(y_true, y_pred):
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }