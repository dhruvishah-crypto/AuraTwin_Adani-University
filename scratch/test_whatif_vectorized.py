import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import joblib

from src.train_model import stratified_patient_split
from src.digital_twin import DigitalTwin

def main():
    feat_cols_path = Path("models/feature_columns.json")
    with open(feat_cols_path, "r") as f:
        feature_cols = json.load(f)
        
    model = joblib.load("models/model.joblib")
    twin = DigitalTwin(data_dir="data")
    
    labeled_path = Path("data/labeled_dataset.csv")
    df = pd.read_csv(labeled_path)
    _, _, test_df, _ = stratified_patient_split(df, seed=42)
    
    # Pick 5 test rows and compare what_if with manual modification
    for i in range(5):
        row = test_df.iloc[i]
        pid = row["patient_id"]
        feat_dict = {c: float(row[c]) for c in feature_cols if c in row}
        twin.active_patient_id = pid
        twin.current_state[pid] = {
            "patient_id": pid,
            "timestamp": str(row.get("timestamp")),
            "features": feat_dict,
            "personal_baseline": {
                "resting_resp_rate": float(feat_dict.get("baseline_resp_rate", 16.0)),
                "resting_spo2": float(feat_dict.get("baseline_spo2", 96.0)),
            },
            "latest_vitals": {},
            "data_quality": {"stale_channels": []},
            "source": "observed",
        }
        twin.state_history[pid] = []
        twin.observations[pid] = pd.DataFrame()
        
        # Test PM2.5 override
        res_twin = twin.what_if(pid, {"pm25_level": 150.0})
        twin_risk = res_twin["scenario_risk"]["probability"]
        
        # Vectorized modification
        mod_dict = dict(feat_dict)
        for c in ["pm25_rolling_mean_6h", "pm25_rolling_max_6h", "pm25_rolling_mean_24h", "pm25_rolling_max_24h", "pm25_rolling_mean_72h", "pm25_rolling_max_72h"]:
            mod_dict[c] = 150.0
        X_row = pd.DataFrame([mod_dict])[feature_cols]
        vec_risk = float(model.predict_proba(X_row)[0, 1])
        
        assert abs(twin_risk - vec_risk) < 1e-6, f"Mismatch: twin={twin_risk}, vec={vec_risk}"
        
    print("Vectorized parity verified successfully!")

if __name__ == "__main__":
    main()
