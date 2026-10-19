import re
from pathlib import Path

train_model_path = Path("src/train_model.py")
content = train_model_path.read_text(encoding="utf-8")

# 1. Add run_what_if_validity_check function
what_if_func = '''

def run_what_if_validity_check(
    calibrated_model: CalibratedModel,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    total_sampled: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate what-if counterfactual monotonicity across five coherent sweeps on test patient-hours.
    
    Sweeps:
    1. pm25: Ambient PM2.5 surge [10 -> 150 ug/m3] (non-decreasing risk)
    2. adherence: Controller adherence [10% -> 100%] (non-increasing risk)
    3. rr_shift: Respiratory rate shift [-3 -> +5 bpm] (non-decreasing risk)
    4. rescue_puffs: Acute SABA rescue puffs [0 -> 8 puffs] (non-decreasing risk)
    5. spo2: Arterial oxygen desaturation [98% -> 88%] (non-decreasing risk as SpO2 drops)
    """
    from src.digital_twin import DigitalTwin

    twin = DigitalTwin(data_dir="data")
    twin.predictor.model = calibrated_model
    twin.predictor.feature_columns = feature_cols

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(test_df.index, size=total_sampled, replace=False)
    sampled_test_rows = test_df.loc[sample_indices]

    sweeps = {
        "pm25": {
            "levels": [10.0, 30.0, 60.0, 100.0, 150.0],
            "key": "pm25_level",
            "direction": "non-decreasing",
            "description": "Ambient PM2.5 surge [10 -> 150 ug/m3]",
        },
        "adherence": {
            "levels": [10.0, 30.0, 60.0, 80.0, 100.0],
            "key": "controller_adherence_pct",
            "direction": "non-increasing",
            "description": "Controller medication adherence [10% -> 100%]",
        },
        "rr_shift": {
            "levels": [-3.0, -1.0, 0.0, 2.0, 5.0],
            "key": "respiratory_rate_shift",
            "direction": "non-decreasing",
            "description": "Tachypneic respiratory rate shift [-3 -> +5 bpm]",
        },
        "rescue_puffs": {
            "levels": [0.0, 1.0, 2.0, 4.0, 8.0],
            "key": "rescue_puffs",
            "direction": "non-decreasing",
            "description": "Escalating SABA rescue inhaler puffs [0 -> 8 puffs]",
        },
        "spo2": {
            "levels": [98.0, 96.0, 94.0, 91.0, 88.0],
            "key": "spo2",
            "direction": "non-decreasing",
            "description": "Acute arterial oxygen desaturation [98% -> 88%]",
        },
    }

    results: Dict[str, Any] = {k: {"monotone_count": 0, "risk_deltas": []} for k in sweeps}

    for _, row in sampled_test_rows.iterrows():
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
                    if risks[i + 1] < risks[i] - 1e-6:
                        is_mono = False
                        break
                else:
                    if risks[i + 1] > risks[i] + 1e-6:
                        is_mono = False
                        break

            if is_mono:
                results[sweep_name]["monotone_count"] += 1

            delta = risks[-1] - risks[0]
            results[sweep_name]["risk_deltas"].append(delta)

    sweep_summaries = {}
    all_passed = True
    for sweep_name, r in results.items():
        frac = r["monotone_count"] / total_sampled
        med_delta = float(np.median(r["risk_deltas"]))
        mean_delta = float(np.mean(r["risk_deltas"]))
        passed = frac >= 0.99
        if not passed:
            all_passed = False
        sweep_summaries[sweep_name] = {
            "sweep_description": sweeps[sweep_name]["description"],
            "sweep_levels": sweeps[sweep_name]["levels"],
            "expected_direction": sweeps[sweep_name]["direction"],
            "monotone_patient_hours": int(r["monotone_count"]),
            "total_sampled_patient_hours": int(total_sampled),
            "monotone_fraction": round(float(frac), 4),
            "median_risk_change": round(float(med_delta), 4),
            "mean_risk_change": round(float(mean_delta), 4),
            "passed_99_rule": bool(passed),
        }

    return {
        "total_sampled_patient_hours": int(total_sampled),
        "sampling_method": "Random patient-hour sampling without replacement from holdout test set",
        "monotonicity_threshold": 0.99,
        "all_sweeps_passed_99_rule": bool(all_passed),
        "sweeps": sweep_summaries,
    }
'''

# Insert run_what_if_validity_check before compute_patient_bootstrap_ci if not present
if "def run_what_if_validity_check" not in content:
    content = content.replace("def compute_patient_bootstrap_ci(", what_if_func + "\n\ndef compute_patient_bootstrap_ci(")
    print("Added run_what_if_validity_check.")

train_model_path.write_text(content, encoding="utf-8")
print("Saved updated train_model.py")
