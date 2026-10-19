import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from src.train_model import stratified_patient_split
import joblib

def main():
    labeled_path = Path("data/labeled_dataset.csv")
    episodes_path = Path("data/episodes.csv")
    feat_cols_path = Path("models/feature_columns.json")
    model_path = Path("models/model.joblib")
    metrics_path = Path("models/metrics.json")
    
    with open(feat_cols_path, "r") as f:
        feature_cols = json.load(f)
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        decision_thresh = metrics_data["risk_thresholds"]["decision_threshold"]
        
    model = joblib.load(model_path)
    df = pd.read_csv(labeled_path)
    _, _, test_df, _ = stratified_patient_split(df, seed=42)
    test_patients = set(test_df["patient_id"].unique())
    print(f"Number of test patients: {len(test_patients)}")
    
    episodes_df = pd.read_csv(episodes_path)
    episodes_df["onset_timestamp"] = pd.to_datetime(episodes_df["onset_timestamp"])
    episodes_df["end_timestamp"] = pd.to_datetime(episodes_df["end_timestamp"])
    
    test_episodes = episodes_df[episodes_df["patient_id"].isin(test_patients)].copy()
    print(f"Total episodes in test set: {len(test_episodes)}")
    print(f"Non-silent in test: {(~test_episodes['is_silent']).sum()}, Silent in test: {test_episodes['is_silent'].sum()}")
    
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    
    # We want episodes where onset_hour >= 48 and max_hour >= onset_hour + 12
    valid_non_silent = []
    valid_silent = []
    
    for idx, ep in test_episodes.iterrows():
        pid = ep["patient_id"]
        onset_ts = ep["onset_timestamp"]
        pat_df = test_df[test_df["patient_id"] == pid].sort_values("timestamp")
        
        # Check window: [onset - 48h, onset + 12h]
        start_ts = onset_ts - pd.Timedelta(hours=48)
        end_ts = onset_ts + pd.Timedelta(hours=12)
        
        window_df = pat_df[(pat_df["timestamp"] >= start_ts) & (pat_df["timestamp"] <= end_ts)]
        if len(window_df) >= 60: # 48 + 12 + 1 = 61 hours
            # Evaluate predictions along the window
            X_win = window_df[feature_cols]
            probs = model.predict_proba(X_win)[:, 1]
            alerts = probs >= decision_thresh
            
            # Check pre-onset alert in [onset - 24h, onset - 1h]
            pre_onset_mask = (window_df["timestamp"] >= onset_ts - pd.Timedelta(hours=24)) & (window_df["timestamp"] < onset_ts)
            pre_alerts = alerts[pre_onset_mask]
            has_pre_alert = bool(pre_alerts.any())
            
            # Post-onset alert in [onset, onset + 6h]
            post_mask = (window_df["timestamp"] >= onset_ts) & (window_df["timestamp"] <= onset_ts + pd.Timedelta(hours=6))
            post_alerts = alerts[post_mask]
            has_post_alert = bool(post_alerts.any())
            
            info = {
                "patient_id": pid,
                "episode_id": ep["episode_id"],
                "onset_timestamp": str(onset_ts),
                "is_silent": ep["is_silent"],
                "has_pre_alert": has_pre_alert,
                "has_post_alert": has_post_alert,
                "first_pre_alert_hours_before": None,
                "max_prob_pre": float(np.max(probs[pre_onset_mask])) if pre_onset_mask.sum() > 0 else 0.0,
            }
            if has_pre_alert:
                alert_times = window_df.loc[pre_onset_mask & alerts, "timestamp"]
                first_alert = alert_times.min()
                info["first_pre_alert_hours_before"] = (onset_ts - first_alert).total_seconds() / 3600.0
                
            if ep["is_silent"]:
                valid_silent.append(info)
            else:
                valid_non_silent.append(info)
                
    print(f"Eligible valid non-silent episodes: {len(valid_non_silent)}")
    print(f"Eligible valid silent episodes: {len(valid_silent)}")
    
    print("\nSample Non-Silent candidates:")
    for c in valid_non_silent[:5]:
        print(c)
        
    print("\nSample Silent candidates (anticipated):")
    for c in [x for x in valid_silent if x['has_pre_alert']][:3]:
        print(c)
        
    print("\nSample Silent candidates (missed/late):")
    for c in [x for x in valid_silent if not x['has_pre_alert']][:3]:
        print(c)

if __name__ == "__main__":
    main()
