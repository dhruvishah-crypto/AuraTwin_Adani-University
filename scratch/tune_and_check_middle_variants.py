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

def evaluate_sweeps(twin, sampled_test_rows, feature_cols, total_sampled=200):
    sweeps = {
        "pm25": {
            "levels": [10.0, 30.0, 60.0, 100.0, 150.0],
            "key": "pm25_level",
            "direction": "non-decreasing",
        },
        "adherence": {
            "levels": [10.0, 30.0, 60.0, 80.0, 100.0],
            "key": "controller_adherence_pct",
            "direction": "non-increasing",
        },
        "rr_shift": {
            "levels": [-3.0, -1.0, 0.0, 2.0, 5.0],
            "key": "respiratory_rate_shift",
            "direction": "non-decreasing",
        },
        "rescue_puffs": {
            "levels": [0.0, 1.0, 2.0, 4.0, 8.0],
            "key": "rescue_puffs",
            "direction": "non-decreasing",
        },
        "spo2": {
            "levels": [98.0, 96.0, 94.0, 91.0, 88.0],
            "key": "spo2",
            "direction": "non-decreasing",
        },
    }
    
    results = {k: {"monotone_count": 0, "risk_deltas": []} for k in sweeps}
    for idx, row in sampled_test_rows.iterrows():
        pid = row["patient_id"]
        feat_dict = {c: float(row[c]) for c in feature_cols if c in row}
        
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
            
            is_mono = True
            for i in range(len(risks) - 1):
                if s_info["direction"] == "non-decreasing":
                    if risks[i+1] < risks[i] - 1e-6:
                        is_mono = False
                        break
                else:
                    if risks[i+1] > risks[i] + 1e-6:
                        is_mono = False
                        break
            
            if is_mono:
                results[sweep_name]["monotone_count"] += 1
            
            delta = risks[-1] - risks[0]
            results[sweep_name]["risk_deltas"].append(delta)

    summary = {}
    all_pass = True
    for sweep_name, r in results.items():
        frac = r["monotone_count"] / total_sampled
        med_delta = float(np.median(r["risk_deltas"]))
        mean_delta = float(np.mean(r["risk_deltas"]))
        passed = (frac >= 0.99)
        if not passed:
            all_pass = False
        summary[sweep_name] = {
            "monotone_fraction": frac,
            "monotone_count": r["monotone_count"],
            "total_sampled": total_sampled,
            "median_risk_delta": med_delta,
            "mean_risk_delta": mean_delta,
            "passes_99_rule": passed,
        }
    return summary, all_pass

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

    # Define Middle Variant Constraints:
    # Features moved together by what-if overrides:
    # - pm25_*: +1
    # - controller_adherence_7d: -1
    # - rescue_puffs_*: +1
    # - symptom_score_*: +1
    # - rr_deviation_from_baseline: +1
    # - respiratory_rate_raw: +1
    # - rr_rolling_mean_6h: +1
    # - rr_rolling_mean_24h: +1
    # - rr_recent_vs_usual: +1
    # - spo2_drop_from_baseline: +1
    # - spo2_raw: -1
    # - spo2_mean_6h: -1
    # - spo2_min_6h: -1
    # - spo2_recent_vs_usual: -1
    # Unconstrained (kept relaxed):
    # - rr_rolling_std_24h: 0
    # - spo2_mean_24h: 0 (or try both)
    # - spo2_min_24h: 0 (or try both)
    # - sleep_deficit: 0
    # - sleep_hours: 0
    # - steps_rolling_sum_24h: 0
    # - autonomic vitals (heart rate, hrv): 0
    
    middle_cst_dict = {c: 0 for c in feature_cols}
    for c in feature_cols:
        if c.startswith("pm25_"):
            middle_cst_dict[c] = 1
        elif c == "controller_adherence_7d":
            middle_cst_dict[c] = -1
        elif c.startswith("rescue_puffs_"):
            middle_cst_dict[c] = 1
        elif c.startswith("symptom_score_"):
            middle_cst_dict[c] = 1
        elif c in ["rr_deviation_from_baseline", "respiratory_rate_raw", "rr_rolling_mean_6h", "rr_rolling_mean_24h", "rr_recent_vs_usual"]:
            middle_cst_dict[c] = 1
        elif c in ["spo2_drop_from_baseline"]:
            middle_cst_dict[c] = 1
        elif c in ["spo2_raw", "spo2_mean_6h", "spo2_min_6h", "spo2_recent_vs_usual"]:
            middle_cst_dict[c] = -1

    middle_cst = [middle_cst_dict[c] for c in feature_cols]
    print(f"Middle constraints count non-zero: {sum(1 for v in middle_cst if v != 0)} / {len(middle_cst)}", flush=True)

    # Full constraints for comparison
    with open("models/monotonic_constraints.json", "r", encoding="utf-8") as f:
        full_dict = json.load(f)
    full_cst = [int(full_dict.get(c, 0)) for c in feature_cols]
    print(f"Full constraints count non-zero: {sum(1 for v in full_cst if v != 0)} / {len(full_cst)}", flush=True)

    # Let's tune middle variant on VALIDATION ONLY
    param_grid = [
        {"learning_rate": 0.02, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 1.0},
        {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0},
    ]

    print("\n--- Tuning Middle Variant on Validation Set ---", flush=True)
    best_mid_val_pr = -1.0
    best_mid_cfg = None
    best_mid_pipe = None
    best_mid_val_probs = None

    for i, cfg in enumerate(param_grid):
        clf = HistGradientBoostingClassifier(
            monotonic_cst=middle_cst,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
            **cfg,
        )
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
        pipe.fit(X_train, y_train)
        val_probs = pipe.predict_proba(X_val)[:, 1]
        val_pr = float(average_precision_score(y_val, val_probs))
        print(f"  [Middle Grid {i+1}/{len(param_grid)}] {cfg} -> Val PR-AUC = {val_pr:.4f} (iters: {clf.n_iter_})", flush=True)
        if val_pr > best_mid_val_pr:
            best_mid_val_pr = val_pr
            best_mid_cfg = {**cfg, "iters_run": int(clf.n_iter_)}
            best_mid_pipe = pipe
            best_mid_val_probs = val_probs

    print(f"\nBest Middle Variant: Val PR-AUC = {best_mid_val_pr:.4f} with {best_mid_cfg}", flush=True)

    # Precompute feature importances
    clf_hgb = best_mid_pipe.named_steps["clf"]
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

    # Calibrate on validation
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(best_mid_val_probs, y_val)
    calibrated_model = CalibratedModel(best_mid_pipe, calibrator, method="isotonic")

    # Sample 200 test patient-hours
    np.random.seed(42)
    sample_indices = np.random.choice(test_df.index, size=200, replace=False)
    sampled_test_rows = test_df.loc[sample_indices]

    twin = DigitalTwin(data_dir="data")
    twin.predictor.model = calibrated_model
    twin.predictor.feature_columns = feature_cols

    print("\n--- Running 5 What-If Sweeps on 200 Test Patient-Hours (Middle Variant) ---", flush=True)
    summary, all_pass = evaluate_sweeps(twin, sampled_test_rows, feature_cols, total_sampled=200)

    for sweep_name, info in summary.items():
        pass_str = "PASS" if info["passes_99_rule"] else "FAIL"
        print(f"Sweep [{sweep_name:<12}]: {pass_str} - Monotone = {info['monotone_fraction']*100:.2f}% ({info['monotone_count']}/{info['total_sampled']}), Median Delta = {info['median_risk_delta']:+.4f} (Mean = {info['mean_risk_delta']:+.4f})", flush=True)

    print(f"\nAll sweeps pass 99% rule: {all_pass}", flush=True)

if __name__ == "__main__":
    main()
