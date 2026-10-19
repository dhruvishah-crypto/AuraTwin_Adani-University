import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd

from src.build_features import FEATURE_COLUMNS
from src.explain_prediction import SIGNAL_FAMILIES

def get_baseline_feature_vector(feature_dict, personal_baseline=None):
    pb = personal_baseline or {}
    base_rr = float(pb.get("baseline_resp_rate", feature_dict.get("baseline_resp_rate", 16.0)))
    base_spo2 = float(pb.get("baseline_spo2", feature_dict.get("baseline_spo2", 96.0)))
    usual_pm25 = float(pb.get("usual_pm25", 35.0))
    usual_adh = float(pb.get("usual_controller_adherence", 1.0))
    usual_symptom = float(pb.get("usual_symptom_score", 1.0))
    usual_hr = float(pb.get("usual_heart_rate", 72.0))
    usual_hrv = float(pb.get("usual_hrv", 38.0))

    defaults = {
        # Pollution
        "pm25_rolling_mean_6h": usual_pm25,
        "pm25_rolling_max_6h": usual_pm25,
        "pm25_rolling_mean_24h": usual_pm25,
        "pm25_rolling_max_24h": usual_pm25,
        "pm25_rolling_mean_72h": usual_pm25,
        "pm25_rolling_max_72h": usual_pm25,
        "pm10_rolling_mean_24h": usual_pm25 * 1.85,
        "pm10_trend_24h": 0.0,
        # Breathing / oxygenation
        "respiratory_rate_raw": base_rr,
        "rr_deviation_from_baseline": 0.0,
        "rr_rolling_mean_6h": base_rr,
        "rr_rolling_mean_24h": base_rr,
        "rr_rolling_std_24h": 0.8,
        "rr_recent_vs_usual": 0.0,
        "spo2_raw": base_spo2,
        "spo2_mean_6h": base_spo2,
        "spo2_min_6h": base_spo2,
        "spo2_mean_24h": base_spo2,
        "spo2_min_24h": base_spo2,
        "spo2_drop_from_baseline": 0.0,
        "spo2_recent_vs_usual": 0.0,
        # Adherence
        "controller_adherence_7d": usual_adh,
        # Rescue & symptoms
        "rescue_puffs_sum_6h": 0.0,
        "rescue_puffs_sum_24h": 0.0,
        "rescue_puffs_trend_24h": 0.0,
        "symptom_score_raw": usual_symptom,
        "symptom_score_mean_6h": usual_symptom,
        "symptom_score_mean_24h": usual_symptom,
        "symptom_score_trend_24h": 0.0,
        # History / Autonomic
        "heart_rate_raw": usual_hr,
        "hr_rolling_mean_6h": usual_hr,
        "hr_rolling_mean_24h": usual_hr,
        "hr_trend_24h": 0.0,
        "hr_recent_vs_usual": 0.0,
        "hrv_rmssd_raw": usual_hrv,
        "hrv_rolling_mean_6h": usual_hrv,
        "hrv_rolling_mean_24h": usual_hrv,
        "sleep_hours": 7.5,
        "sleep_deficit": 0.0,
        "steps_rolling_sum_24h": 6500.0,
    }
    
    base_vec = dict(feature_dict)
    for k, v in defaults.items():
        if k in base_vec:
            base_vec[k] = v
    return base_vec

def compute_occlusion_contributions(model, feature_dict, personal_baseline=None):
    X_curr = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
    base_prob = float(model.predict_proba(X_curr)[0, 1])
    
    # Safe logit
    eps = 1e-5
    p_clip = np.clip(base_prob, eps, 1.0 - eps)
    base_logit = np.log(p_clip / (1.0 - p_clip))
    
    baseline_vec = get_baseline_feature_vector(feature_dict, personal_baseline)
    
    family_contributions = {}
    
    for fam_key, fam_meta in SIGNAL_FAMILIES.items():
        fam_cols = fam_meta["features"]
        X_occ = feature_dict.copy()
        for col in fam_cols:
            if col in baseline_vec:
                X_occ[col] = baseline_vec[col]
                
        X_occ_df = pd.DataFrame([X_occ])[FEATURE_COLUMNS]
        occ_prob = float(model.predict_proba(X_occ_df)[0, 1])
        occ_p_clip = np.clip(occ_prob, eps, 1.0 - eps)
        occ_logit = np.log(occ_p_clip / (1.0 - occ_p_clip))
        
        prob_delta = base_prob - occ_prob
        logit_delta = base_logit - occ_logit
        
        direction = "increases_risk" if prob_delta > 0.005 else ("decreases_risk" if prob_delta < -0.005 else "neutral")
        
        family_contributions[fam_key] = {
            "family": fam_key,
            "label": fam_meta["label"],
            "risk_delta": round(prob_delta, 5),
            "summed_logit_contribution": round(logit_delta, 4),
            "direction": direction,
            "base_prob": round(base_prob, 4),
            "occluded_prob": round(occ_prob, 4),
        }
        
    return base_prob, family_contributions

def main():
    model = joblib.load("models/model.joblib")
    labeled_path = Path("data/labeled_dataset.csv")
    df = pd.read_csv(labeled_path)
    
    # Pick an episode onset hour where patient is having an exacerbation
    sample_row = df[df["event_next_24h"] == 1].iloc[10].to_dict()
    
    base_prob, fam_contrib = compute_occlusion_contributions(model, sample_row)
    print(f"Base predicted probability: {base_prob:.4f}")
    print("\nOcclusion Family Contributions:")
    for fam, c in fam_contrib.items():
        print(f"  {c['label']:<26}: Delta={c['risk_delta']:+.4f} (Logit={c['summed_logit_contribution']:+.4f}) -> {c['direction']}")

if __name__ == "__main__":
    main()
