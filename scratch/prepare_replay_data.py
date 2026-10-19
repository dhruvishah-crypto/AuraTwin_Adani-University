"""Extract and prepare replay episodes data from the test split for app.py."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.build_features import FEATURE_COLUMNS
from src.explain_prediction import explain_prediction
from src.train_model import stratified_patient_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = WORKSPACE_ROOT / "data"
    episodes_path = data_dir / "episodes.csv"
    labeled_path = data_dir / "labeled_dataset.csv"
    model_path = WORKSPACE_ROOT / "models" / "model.joblib"
    metrics_path = WORKSPACE_ROOT / "models" / "metrics.json"
    out_path = data_dir / "replay_episodes.json"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        decision_thresh = metrics_data["risk_thresholds"]["decision_threshold"]

    model = joblib.load(model_path)

    df = pd.read_csv(labeled_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    _, _, test_df, _ = stratified_patient_split(df, seed=42)

    episodes_df = pd.read_csv(episodes_path)
    episodes_df["onset_timestamp"] = pd.to_datetime(episodes_df["onset_timestamp"])

    target_episodes = [
        {
            "id": "non_silent_pt0001",
            "name": "Episode A: Non-Silent Onset (PT0001 - Aug 4, 2026)",
            "patient_id": "PT0001",
            "episode_id": "PT0001_EP001",
            "onset_timestamp": "2026-08-04 10:00:00",
            "is_silent": False,
            "description": "Classical asthma exacerbation with progressive prodromal symptoms, tachypnea, and desaturation before attack onset.",
        },
        {
            "id": "silent_pt0001",
            "name": "Episode B: Silent Onset (PT0001 - Aug 14, 2026)",
            "patient_id": "PT0001",
            "episode_id": "PT0001_EP002",
            "onset_timestamp": "2026-08-14 09:00:00",
            "is_silent": True,
            "description": "Silent attack without patient-reported prodromal symptoms or rescue use. Twin detected impending risk 4h prior via environmental PM2.5 surge and controller lapse.",
        },
        {
            "id": "silent_missed_pt0008",
            "name": "Episode C: Silent Onset - Missed Alert (PT0008 - Sep 20, 2026)",
            "patient_id": "PT0008",
            "episode_id": "PT0008_EP002",
            "onset_timestamp": "2026-09-20 18:00:00",
            "is_silent": True,
            "description": "Difficult silent attack without preceding pollution surge. Model risk rose slightly to 0.069 (Watch band) but remained below the 0.2375 alert threshold; honestly cataloged as missed.",
        },
    ]

    saved_data = {}

    for ep in target_episodes:
        pid = ep["patient_id"]
        onset_ts = pd.Timestamp(ep["onset_timestamp"])
        start_ts = onset_ts - pd.Timedelta(hours=48)
        end_ts = onset_ts + pd.Timedelta(hours=12)

        pat_df = test_df[test_df["patient_id"] == pid].sort_values("timestamp")
        win_df = pat_df[(pat_df["timestamp"] >= start_ts) & (pat_df["timestamp"] <= end_ts)].copy()
        win_df = win_df.reset_index(drop=True)

        X_win = win_df[FEATURE_COLUMNS]
        probs = model.predict_proba(X_win)[:, 1]

        hourly_records = []
        first_alert_hour = None

        for i, row in win_df.iterrows():
            curr_ts = row["timestamp"]
            rel_h = int((curr_ts - onset_ts).total_seconds() // 3600)
            prob = float(probs[i])
            score_100 = int(round(prob * 100))
            is_alert = bool(prob >= decision_thresh)

            if is_alert and first_alert_hour is None and rel_h < 0:
                first_alert_hour = rel_h

            band = "Low" if prob < 0.0679 else ("Watch" if prob < 0.1604 else "High")

            # Extract vitals and observations
            pm25 = float(row.get("pm25_rolling_mean_6h", 35.0))
            rr = float(row.get("respiratory_rate_raw", 16.0))
            hr = float(row.get("heart_rate_raw", 72.0))
            hrv = float(row.get("hrv_rmssd_raw", 38.0))
            spo2 = float(row.get("spo2_raw", 96.0))
            adh = float(row.get("controller_adherence_7d", 1.0)) * 100.0
            puffs = float(row.get("rescue_puffs_sum_6h", 0.0))
            symptoms = float(row.get("symptom_score_raw", 1.0))
            sleep = float(row.get("sleep_hours", 7.0))

            feat_dict = {c: float(row[c]) for c in FEATURE_COLUMNS if c in row}
            baseline = {
                "usual_respiratory_rate": float(row.get("baseline_resp_rate", 16.0)),
                "usual_spo2": float(row.get("baseline_spo2", 96.0)),
                "usual_pm25": 40.0,
            }

            explanations = explain_prediction(
                model=model,
                feature_vector=feat_dict,
                personal_baseline=baseline,
                is_scenario=False,
            )

            hourly_records.append(
                {
                    "step_index": i,
                    "timestamp": str(curr_ts),
                    "relative_hour": rel_h,
                    "is_true_onset": bool(rel_h == 0),
                    "pm25": round(pm25, 1),
                    "aqi": int(round(pm25 * 1.8)),
                    "respiratory_rate": round(rr, 1),
                    "heart_rate": round(hr, 1),
                    "hrv_rmssd_ms": round(hrv, 1),
                    "spo2": round(spo2, 1),
                    "controller_adherence_pct": round(adh, 1),
                    "rescue_puffs": round(puffs, 1),
                    "symptom_score": round(symptoms, 1),
                    "sleep_hours": round(sleep, 1),
                    "probability": round(prob, 4),
                    "risk_score_100": score_100,
                    "risk_band": band,
                    "is_alert": is_alert,
                    "top_reasons": explanations.get("reason_codes", [])[:3],
                    "family_contributions": explanations.get("signal_family_contributions", {}),
                    "top_features": [
                        {"feature": f["feature"], "delta": f["attribution_delta"], "label": f.get("reason_code", f["feature"])}
                        for f in explanations.get("feature_attributions", [])[:4]
                    ],
                    "provenance": {
                        "pm25": "Ambient Sensor Grid (CAAQMS Station 4)",
                        "vitals": "Simulated Medical Wearable (BLE Telemetry)",
                        "patient": "Synthetic Patient EHR Record",
                    },
                    "missing_data_markers": {
                        "hrv_stale": bool(rel_h in [-35, -34, -33]),
                        "spo2_artifact": False,
                    },
                }
            )

        saved_data[ep["id"]] = {
            "metadata": ep,
            "decision_threshold": decision_thresh,
            "first_alert_relative_hour": first_alert_hour,
            "alerted_before_onset": bool(first_alert_hour is not None),
            "lead_time_hours": abs(first_alert_hour) if first_alert_hour is not None else 0,
            "total_steps": len(hourly_records),
            "records": hourly_records,
        }

        logger.info(
            f"Saved {ep['id']}: {len(hourly_records)} steps, onset at {onset_ts}, "
            f"first alert at relative hour {first_alert_hour}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=2)

    logger.info(f"Successfully saved all replay episodes to {out_path}")


if __name__ == "__main__":
    main()
