import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from src.train_model import stratified_patient_split, CalibratedModel
from src.digital_twin import DigitalTwin

def main():
    data_path = Path("data/labeled_dataset.csv")
    feat_path = Path("models/feature_columns.json")
    
    with open(feat_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
        
    df = pd.read_csv(data_path)
    train_df, val_df, test_df, _ = stratified_patient_split(df, seed=42)
    
    X_train = train_df[feature_cols]
    y_train = train_df["event_next_24h"].to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["event_next_24h"].to_numpy()
    X_test = test_df[feature_cols]
    y_test = test_df["event_next_24h"].to_numpy()

    # Relaxed constraints
    relaxed_constraints = {c: 0 for c in feature_cols}
    for c in feature_cols:
        if c.startswith("pm25_"):
            relaxed_constraints[c] = 1
        elif c == "controller_adherence_7d":
            relaxed_constraints[c] = -1
        elif c.startswith("rescue_puffs_"):
            relaxed_constraints[c] = 1
        elif c.startswith("symptom_score_"):
            relaxed_constraints[c] = 1
        elif c in ["rr_deviation_from_baseline", "spo2_drop_from_baseline"]:
            relaxed_constraints[c] = 1

    relaxed_cst = [relaxed_constraints[c] for c in feature_cols]
    
    clf = HistGradientBoostingClassifier(
        monotonic_cst=relaxed_cst,
        learning_rate=0.02,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    pipe.fit(X_train, y_train)
    print("Fitted candidate model.", flush=True)

    # Precompute and cache feature_importances_ on HGB to avoid recomputing across 600 trees on every what_if call
    clf_hgb = pipe.named_steps["clf"]
    if hasattr(clf_hgb, "_predictors"):
        n_feat = len(feature_cols)
        importances = np.zeros(n_feat)
        for predictor_list in clf_hgb._predictors:
            for tree in predictor_list:
                nodes = tree.nodes
                gain_mask = nodes["gain"] > 0
                np.add.at(importances, nodes["feature_idx"][gain_mask], nodes["gain"][gain_mask])
        tot = np.sum(importances)
        if tot > 0:
            importances /= tot
        clf_hgb.feature_importances_ = importances
    print("Precomputed feature importances.", flush=True)
    
    val_raw_probs = pipe.predict_proba(X_val)[:, 1]
    val_pr = average_precision_score(y_val, val_raw_probs)
    print(f"Candidate Relaxed HGB Validation PR-AUC: {val_pr:.4f} (iters: {clf.n_iter_})", flush=True)
    
    # Isotonic calibration on validation only
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_raw_probs, y_val)
    calibrated_model = CalibratedModel(pipe, calibrator, method="isotonic")
    
    # Now set up DigitalTwin with this calibrated model
    twin = DigitalTwin(data_dir="data")
    twin.predictor.model = calibrated_model
    twin.predictor.feature_columns = feature_cols
    
    # Sample 200 test patient-hours (satisfies "at least 200 randomly sampled test patient-hours")
    np.random.seed(42)
    sample_indices = np.random.choice(test_df.index, size=200, replace=False)
    sampled_test_rows = test_df.loc[sample_indices]
    
    sweeps = {
        "pm25": {
            "levels": [10.0, 30.0, 60.0, 100.0, 150.0],
            "key": "pm25_level",
            "direction": "non-decreasing", # r[i+1] >= r[i]
        },
        "adherence": {
            "levels": [10.0, 30.0, 60.0, 80.0, 100.0],
            "key": "controller_adherence_pct",
            "direction": "non-increasing", # r[i+1] <= r[i]
        },
        "rr_shift": {
            "levels": [-3.0, -1.0, 0.0, 2.0, 5.0],
            "key": "respiratory_rate_shift",
            "direction": "non-decreasing", # r[i+1] >= r[i]
        },
        "rescue_puffs": {
            "levels": [0.0, 1.0, 2.0, 4.0, 8.0],
            "key": "rescue_puffs",
            "direction": "non-decreasing", # r[i+1] >= r[i]
        },
        "spo2": {
            "levels": [98.0, 96.0, 94.0, 91.0, 88.0],
            "key": "spo2",
            "direction": "non-decreasing", # dropping spo2 -> non-decreasing risk
        },
    }
    
    results = {k: {"monotone_count": 0, "risk_deltas": []} for k in sweeps}
    
    count = 0
    for idx, row in sampled_test_rows.iterrows():
        count += 1
        pid = row["patient_id"]
        feat_dict = {c: float(row[c]) for c in feature_cols if c in row}
        
        # Mock twin state for this hour
        twin.active_patient_id = pid
        twin.current_state[pid] = {
            "patient_id": pid,
            "timestamp": str(row.get("timestamp", "2026-08-01")),
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
        
        for sweep_name, s_info in sweeps.items():
            risks = []
            for val in s_info["levels"]:
                res = twin.what_if(pid, {s_info["key"]: val})
                risks.append(res["scenario_risk"]["probability"])
            
            # Check monotonicity
            is_mono = True
            for i in range(len(risks) - 1):
                if s_info["direction"] == "non-decreasing":
                    if risks[i+1] < risks[i] - 1e-6: # small epsilon for numerical stability
                        is_mono = False
                        break
                else: # non-increasing
                    if risks[i+1] > risks[i] + 1e-6:
                        is_mono = False
                        break
            
            if is_mono:
                results[sweep_name]["monotone_count"] += 1
            
            # Risk delta: end minus start
            delta = risks[-1] - risks[0]
            results[sweep_name]["risk_deltas"].append(delta)

        if count % 25 == 0:
            print(f"Processed {count}/{len(sampled_test_rows)} patient-hours...", flush=True)

    total_sampled = len(sampled_test_rows)
    print(f"\n--- What-If Sweep Monotonicity Results on {total_sampled} Test Patient-Hours (Relaxed Model) ---")
    for sweep_name, r in results.items():
        frac = r["monotone_count"] / total_sampled
        med_delta = float(np.median(r["risk_deltas"]))
        mean_delta = float(np.mean(r["risk_deltas"]))
        print(f"Sweep [{sweep_name:<12}]: Monotone Fraction = {frac*100:.2f}% ({r['monotone_count']}/{total_sampled}), Median Risk Change = {med_delta:+.4f} (Mean = {mean_delta:+.4f})")

if __name__ == "__main__":
    main()
