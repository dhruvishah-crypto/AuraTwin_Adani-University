"""Feature engineering pipeline for RespiraTwin.

Constructs strictly past-looking feature vectors (zero future data leakage):
- Static EHR demographics and clinician-assessed severity
- Multiscale rolling environmental exposures (PM2.5, PM10 over 6h, 24h, 72h)
- Physiological deviations from individual patient baseline vitals
- Autonomic HRV trends and tachycardia indicators
- Oxygen saturation deficits and respiratory instability
- Sleep deficits and 7-day controller medication adherence
- Rescue inhaler velocity and symptom trajectory
- Recent-vs-usual difference signals (acute shifts)

Saves the exact feature list in a deterministic order to models/feature_columns.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.ingest_pollution import ingest_pollution
from src.load_ehr import load_ehr_data

# Explicit list of final model feature columns in fixed order
FEATURE_COLUMNS: List[str] = [
    # Static EHR & Demographics
    "age",
    "sex_male",
    "asthma_severity_numeric",
    "prior_exacerbations_12m",
    "controller_prescribed",
    "baseline_resp_rate",
    "baseline_spo2",
    "pollution_sensitivity_numeric",
    "has_allergic_rhinitis",
    "has_gerd",
    "has_obesity",
    "has_hypertension",
    # Environmental Exposures (Past-looking rolling PM2.5 / PM10)
    "pm25_rolling_mean_6h",
    "pm25_rolling_max_6h",
    "pm25_rolling_mean_24h",
    "pm25_rolling_max_24h",
    "pm25_rolling_mean_72h",
    "pm25_rolling_max_72h",
    "pm10_rolling_mean_24h",
    "pm10_trend_24h",
    # Physiological Signals & Baseline Deviations
    "respiratory_rate_raw",
    "rr_deviation_from_baseline",
    "rr_rolling_mean_6h",
    "rr_rolling_mean_24h",
    "rr_rolling_std_24h",
    "heart_rate_raw",
    "hr_rolling_mean_6h",
    "hr_rolling_mean_24h",
    "hr_trend_24h",
    "hrv_rmssd_raw",
    "hrv_rolling_mean_6h",
    "hrv_rolling_mean_24h",
    "spo2_raw",
    "spo2_mean_6h",
    "spo2_min_6h",
    "spo2_mean_24h",
    "spo2_min_24h",
    "spo2_drop_from_baseline",
    # Sleep, Activity, Medication & Symptoms
    "sleep_hours",
    "sleep_deficit",
    "steps_rolling_sum_24h",
    "controller_adherence_7d",
    "rescue_puffs_sum_6h",
    "rescue_puffs_sum_24h",
    "rescue_puffs_trend_24h",
    "symptom_score_raw",
    "symptom_score_mean_6h",
    "symptom_score_mean_24h",
    "symptom_score_trend_24h",
    # Recent-vs-Usual Differences (Acute Divergence)
    "rr_recent_vs_usual",
    "hr_recent_vs_usual",
    "spo2_recent_vs_usual",
]


def save_feature_columns_json(
    feature_cols: List[str] = FEATURE_COLUMNS,
    output_path: Union[str, Path] = "models/feature_columns.json",
) -> Path:
    """Save the deterministic ordered list of feature column names."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    return out_file


def engineer_patient_features(
    patient_df: pd.DataFrame,
    patient_ts_df: pd.DataFrame,
    pollution_df: pd.DataFrame,
) -> pd.DataFrame:
    """Engineer strictly past-looking features for a cohort of patients.

    Args:
        patient_df: Static EHR records.
        patient_ts_df: Hourly wearable and symptom time-series.
        pollution_df: Hourly ambient pollution series.

    Returns:
        DataFrame containing patient_id, timestamp, and all FEATURE_COLUMNS.
    """
    ts_work = patient_ts_df.copy()
    poll_work = pollution_df[["timestamp", "pm25", "pm10"]].copy()

    ts_work["timestamp"] = pd.to_datetime(ts_work["timestamp"])
    poll_work["timestamp"] = pd.to_datetime(poll_work["timestamp"])

    # Merge pollution onto wearable timestamps
    merged = pd.merge(ts_work, poll_work, on="timestamp", how="left")

    # Merge static EHR fields
    merged = pd.merge(merged, patient_df, on="patient_id", how="left")

    # Ensure deterministic sort order: by patient, then chronological timestamp
    merged = merged.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    # Encode static categorical features
    merged["sex_male"] = (merged["sex"] == "M").astype(int)
    severity_map = {"Mild": 0, "Moderate": 1, "Severe": 2}
    merged["asthma_severity_numeric"] = merged["asthma_severity"].map(severity_map).fillna(0).astype(int)
    sens_map = {"Low": 0, "Medium": 1, "High": 2}
    merged["pollution_sensitivity_numeric"] = merged["pollution_sensitivity"].map(sens_map).fillna(1).astype(int)

    # Comorbidity binary flags
    comorb_str = merged["comorbidities"].astype(str)
    merged["has_allergic_rhinitis"] = comorb_str.str.contains("allergic_rhinitis").astype(int)
    merged["has_gerd"] = comorb_str.str.contains("GERD").astype(int)
    merged["has_obesity"] = comorb_str.str.contains("obesity").astype(int)
    merged["has_hypertension"] = comorb_str.str.contains("hypertension").astype(int)

    # Forward-fill missing sensor values per patient (strictly forward, no future leakage)
    for col, baseline_col in [
        ("respiratory_rate", "baseline_resp_rate"),
        ("spo2", "baseline_spo2"),
        ("heart_rate", None),
        ("hrv_rmssd_ms", None),
    ]:
        merged[col] = merged.groupby("patient_id")[col].ffill()
        if baseline_col:
            merged[col] = merged[col].fillna(merged[baseline_col])
        else:
            default_val = 72.0 if col == "heart_rate" else 35.0
            merged[col] = merged[col].fillna(default_val)

    merged["respiratory_rate_raw"] = merged["respiratory_rate"]
    merged["heart_rate_raw"] = merged["heart_rate"]
    merged["hrv_rmssd_raw"] = merged["hrv_rmssd_ms"]
    merged["spo2_raw"] = merged["spo2"]
    merged["symptom_score_raw"] = merged["symptom_score"]

    # --- Groupby Patient Rolling Past-Looking Computations ---
    grouped = merged.groupby("patient_id")

    # Environmental exposures (past rolling windows)
    merged["pm25_rolling_mean_6h"] = grouped["pm25"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["pm25_rolling_max_6h"] = grouped["pm25"].transform(lambda x: x.rolling(6, min_periods=1).max())
    merged["pm25_rolling_mean_24h"] = grouped["pm25"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["pm25_rolling_max_24h"] = grouped["pm25"].transform(lambda x: x.rolling(24, min_periods=1).max())
    merged["pm25_rolling_mean_72h"] = grouped["pm25"].transform(lambda x: x.rolling(72, min_periods=1).mean())
    merged["pm25_rolling_max_72h"] = grouped["pm25"].transform(lambda x: x.rolling(72, min_periods=1).max())

    pm10_mean_6h = grouped["pm10"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["pm10_rolling_mean_24h"] = grouped["pm10"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["pm10_trend_24h"] = pm10_mean_6h - merged["pm10_rolling_mean_24h"]

    # Respiratory rate features
    merged["rr_deviation_from_baseline"] = merged["respiratory_rate_raw"] - merged["baseline_resp_rate"]
    merged["rr_rolling_mean_6h"] = grouped["respiratory_rate_raw"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["rr_rolling_mean_24h"] = grouped["respiratory_rate_raw"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["rr_rolling_mean_72h"] = grouped["respiratory_rate_raw"].transform(lambda x: x.rolling(72, min_periods=1).mean())
    merged["rr_rolling_std_24h"] = grouped["respiratory_rate_raw"].transform(
        lambda x: x.rolling(24, min_periods=1).std()
    ).fillna(0.0)

    # Heart rate & HRV features
    merged["hr_rolling_mean_6h"] = grouped["heart_rate_raw"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["hr_rolling_mean_24h"] = grouped["heart_rate_raw"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["hr_rolling_mean_72h"] = grouped["heart_rate_raw"].transform(lambda x: x.rolling(72, min_periods=1).mean())
    merged["hr_trend_24h"] = merged["hr_rolling_mean_6h"] - merged["hr_rolling_mean_24h"]

    merged["hrv_rolling_mean_6h"] = grouped["hrv_rmssd_raw"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["hrv_rolling_mean_24h"] = grouped["hrv_rmssd_raw"].transform(lambda x: x.rolling(24, min_periods=1).mean())

    # SpO2 features
    merged["spo2_mean_6h"] = grouped["spo2_raw"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["spo2_min_6h"] = grouped["spo2_raw"].transform(lambda x: x.rolling(6, min_periods=1).min())
    merged["spo2_mean_24h"] = grouped["spo2_raw"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["spo2_min_24h"] = grouped["spo2_raw"].transform(lambda x: x.rolling(24, min_periods=1).min())
    merged["spo2_mean_72h"] = grouped["spo2_raw"].transform(lambda x: x.rolling(72, min_periods=1).mean())
    merged["spo2_drop_from_baseline"] = np.maximum(0.0, merged["baseline_spo2"] - merged["spo2_mean_6h"])

    # Sleep deficit
    merged["sleep_deficit"] = np.maximum(0.0, 7.5 - merged["sleep_hours"].fillna(7.2))

    # Activity
    merged["steps_rolling_sum_24h"] = grouped["steps"].transform(lambda x: x.rolling(24, min_periods=1).sum())

    # 7-day controller adherence percentage (past 168 hours)
    adherence_168 = grouped["controller_taken"].transform(lambda x: x.rolling(168, min_periods=1).sum() / 7.0)
    merged["controller_adherence_7d"] = np.where(
        merged["controller_prescribed"] == 1,
        np.clip(adherence_168, 0.0, 1.0),
        1.0,
    )

    # Rescue inhaler puffs
    merged["rescue_puffs_sum_6h"] = grouped["rescue_puffs"].transform(lambda x: x.rolling(6, min_periods=1).sum())
    merged["rescue_puffs_sum_24h"] = grouped["rescue_puffs"].transform(lambda x: x.rolling(24, min_periods=1).sum())
    merged["rescue_puffs_trend_24h"] = merged["rescue_puffs_sum_6h"] - (merged["rescue_puffs_sum_24h"] / 4.0)

    # Symptom scores
    merged["symptom_score_mean_6h"] = grouped["symptom_score_raw"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    merged["symptom_score_mean_24h"] = grouped["symptom_score_raw"].transform(lambda x: x.rolling(24, min_periods=1).mean())
    merged["symptom_score_trend_24h"] = merged["symptom_score_mean_6h"] - merged["symptom_score_mean_24h"]

    # Recent-vs-Usual Differences
    merged["rr_recent_vs_usual"] = merged["rr_rolling_mean_6h"] - merged["rr_rolling_mean_72h"]
    merged["hr_recent_vs_usual"] = merged["hr_rolling_mean_6h"] - merged["hr_rolling_mean_72h"]
    merged["spo2_recent_vs_usual"] = merged["spo2_mean_6h"] - merged["spo2_mean_72h"]

    # Round float features for clarity and compact storage
    for col in FEATURE_COLUMNS:
        if merged[col].dtype == float:
            merged[col] = merged[col].round(3)

    # Output dataframe: identifiers + ordered features
    output_columns = ["patient_id", "timestamp"] + FEATURE_COLUMNS
    return merged[output_columns].copy()


def build_features(
    ehr_path: Union[str, Path] = "data/patients.csv",
    wearables_path: Union[str, Path] = "data/patient_timeseries.csv",
    pollution_path: Union[str, Path] = "data/sample_pollution.csv",
    feature_columns_path: Union[str, Path] = "models/feature_columns.json",
    num_patients: Optional[int] = None,
) -> pd.DataFrame:
    """Build fused feature dataset from disk files with zero future leakage.

    Args:
        ehr_path: Path to patient profiles CSV.
        wearables_path: Path to wearable timeseries CSV.
        pollution_path: Path to pollution CSV.
        feature_columns_path: Path to save feature column names JSON.
        num_patients: Optional limit on patients to process.

    Returns:
        Feature DataFrame containing patient_id, timestamp, and ordered features.
    """
    save_feature_columns_json(FEATURE_COLUMNS, feature_columns_path)

    ehr_df = load_ehr_data(ehr_path)
    wearables_df = pd.read_csv(wearables_path)
    pollution_df = ingest_pollution(pollution_path)

    if num_patients is not None:
        pids = ehr_df["patient_id"].iloc[:num_patients].unique()
        ehr_df = ehr_df[ehr_df["patient_id"].isin(pids)].copy()
        wearables_df = wearables_df[wearables_df["patient_id"].isin(pids)].copy()

    return engineer_patient_features(
        patient_df=ehr_df,
        patient_ts_df=wearables_df,
        pollution_df=pollution_df,
    )


def main() -> None:
    """CLI to build and save engineered features."""
    parser = argparse.ArgumentParser(description="Build past-looking feature vectors for RespiraTwin.")
    parser.add_argument("--ehr", type=str, default="data/patients.csv", help="Patients CSV")
    parser.add_argument("--wearables", type=str, default="data/patient_timeseries.csv", help="Wearables CSV")
    parser.add_argument("--pollution", type=str, default="data/sample_pollution.csv", help="Pollution CSV")
    parser.add_argument("--output", type=str, default="data/features.csv", help="Output features CSV")
    parser.add_argument("--num_patients", type=int, default=None, help="Limit number of patients for quick runs")

    args = parser.parse_args()

    print("Building engineered features (past-looking, zero future leakage)...")
    feats_df = build_features(
        ehr_path=args.ehr,
        wearables_path=args.wearables,
        pollution_path=args.pollution,
        num_patients=args.num_patients,
    )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    feats_df.to_csv(out_file, index=False)

    print(f"Features built successfully: {feats_df.shape[0]} rows x {feats_df.shape[1]} columns.")
    print(f"Features saved to: {out_file}")
    print(f"Feature schema written to: models/feature_columns.json")


if __name__ == "__main__":
    main()
