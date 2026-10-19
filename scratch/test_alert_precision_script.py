import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
from src.train_model import stratified_patient_split, compute_alert_burden, evaluate_alert_baselines

def main():
    feat_path = Path("models/feature_columns.json")
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    df = pd.read_csv("data/labeled_dataset.csv")
    _, val_df, test_df, _ = stratified_patient_split(df, seed=42)

    # Let's test with the middle variant model
    # We can load or train it
    pass

if __name__ == "__main__":
    main()
