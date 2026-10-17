"""Physiological wearable and symptom time series simulator for RespiraTwin.

Simulates coherent, multi-day hourly wearable signals for asthma patients:
- Circadian rhythms across vitals (HR, SpO2, RR, HRV).
- Lagged inflammatory responses to elevated ambient PM2.5 in pollution-sensitive patients.
- Daily controller medication adherence dynamics.
- Symptom diary scoring (0-10) and responsive rescue inhaler puffs.
- Realistic sensor noise and wearable dropout (missingness / charging off-wrist).
- Provenance tagging ('simulated') and reproducibility sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.ingest_pollution import ingest_pollution
from src.load_ehr import load_ehr_data

WEARABLE_COLUMNS = [
    "patient_id",
    "timestamp",
    "respiratory_rate",
    "heart_rate",
    "hrv_rmssd_ms",
    "spo2",
    "sleep_hours",
    "steps",
    "controller_taken",
    "rescue_puffs",
    "symptom_score",
    "provenance",
]


def compute_lagged_pollution_stress(
    pm25_series: np.ndarray,
    sensitivity: str,
    lag_hours: int = 12,
) -> np.ndarray:
    """Calculate delayed inflammatory stress from ambient PM2.5 based on patient phenotype.

    Args:
        pm25_series: Array of ambient hourly PM2.5 concentrations.
        sensitivity: 'Low', 'Medium', or 'High'.
        lag_hours: Rolling window lag for airway hyperresponsiveness onset.

    Returns:
        Array of normalized stress increments [0.0, 3.5].
    """
    n = len(pm25_series)
    # Rolling 12-hour average PM2.5 to simulate cumulative exposure lag
    kernel = np.ones(lag_hours) / lag_hours
    padded_pm = np.pad(pm25_series, (lag_hours - 1, 0), mode="edge")
    rolling_pm = np.convolve(padded_pm, kernel, mode="valid")

    # Excess over clean air threshold (40 ug/m3)
    excess_pm = np.maximum(0.0, rolling_pm - 40.0)

    # Sensitivity scaling multiplier
    multiplier = {"Low": 0.008, "Medium": 0.022, "High": 0.048}.get(sensitivity, 0.02)
    return np.minimum(3.5, excess_pm * multiplier)


def simulate_single_patient_timeseries(
    patient: pd.Series,
    pollution_df: pd.DataFrame,
    seed: int = 42,
    sensor_dropout_rate: float = 0.035,
) -> pd.DataFrame:
    """Generate hourly wearable time series for a single patient record.

    Args:
        patient: Row from patients.csv.
        pollution_df: Hourly pollution dataframe with 'timestamp' and 'pm25'.
        seed: Random seed for reproducibility.
        sensor_dropout_rate: Probability of missing wearable sensor readings.

    Returns:
        DataFrame containing simulated hourly observations.
    """
    rng = np.random.default_rng(seed)
    n_hours = len(pollution_df)
    timestamps = pd.to_datetime(pollution_df["timestamp"]).tolist()

    pid = str(patient["patient_id"])
    severity = str(patient["asthma_severity"])
    pollution_sens = str(patient["pollution_sensitivity"])
    baseline_rr = float(patient["baseline_resp_rate"])
    baseline_spo2 = float(patient["baseline_spo2"])
    has_controller_rx = int(patient["controller_prescribed"]) == 1

    # Base heart rate (derived from respiratory baseline & age)
    base_hr = 70.0 + (baseline_rr - 15.0) * 1.5 + (0.15 * max(0, float(patient["age"]) - 40))
    # Base HRV RMSSD (ms) - inversely correlated with age & severity
    base_hrv = max(22.0, 55.0 - (0.4 * float(patient["age"])) - (6.0 if severity == "Severe" else 0.0))

    # Calculate pollution stress curve
    pm25_vals = pollution_df["pm25"].to_numpy()
    pollution_stress = compute_lagged_pollution_stress(pm25_vals, pollution_sens, lag_hours=12)

    # Daily adherence simulation:
    # Patients take daily controller (morning ~08:00 or evening ~20:00)
    # Adherent patients take it ~88% of days; occasional non-adherent streaks of 2-4 days
    n_days = (n_hours + 23) // 24
    daily_controller_taken = np.zeros(n_days, dtype=int)
    daily_sleep = np.zeros(n_days, dtype=float)

    if has_controller_rx:
        day_idx = 0
        while day_idx < n_days:
            # Random adherence streak vs lapse
            streak_adherent = rng.random() < 0.85
            streak_len = int(rng.integers(1, 6))
            end_day = min(n_days, day_idx + streak_len)
            daily_controller_taken[day_idx:end_day] = 1 if streak_adherent else 0
            day_idx = end_day

    # Daily sleep hours (calculated once per day)
    for d in range(n_days):
        daily_sleep[d] = round(float(np.clip(rng.normal(7.2, 0.9), 4.5, 9.8)), 1)

    # Hourly signal simulation arrays
    hr_arr = np.zeros(n_hours, dtype=float)
    rr_arr = np.zeros(n_hours, dtype=float)
    hrv_arr = np.zeros(n_hours, dtype=float)
    spo2_arr = np.zeros(n_hours, dtype=float)
    steps_arr = np.zeros(n_hours, dtype=int)
    controller_arr = np.zeros(n_hours, dtype=int)
    rescue_arr = np.zeros(n_hours, dtype=int)
    symptom_arr = np.zeros(n_hours, dtype=int)
    sleep_col_arr = np.zeros(n_hours, dtype=float)

    # Base severity symptom baseline
    sev_sym_base = 0 if severity == "Mild" else (1 if severity == "Moderate" else 2)

    for i in range(n_hours):
        dt = timestamps[i]
        hour = dt.hour
        day = i // 24
        day_sleep_val = daily_sleep[min(day, n_days - 1)]
        sleep_col_arr[i] = day_sleep_val

        # Circadian rhythm phase [-1, +1]
        circadian_hr = np.sin(2 * np.pi * (hour - 9) / 24)
        circadian_rr = np.sin(2 * np.pi * (hour - 10) / 24)
        is_sleeping = 0 <= hour <= 6

        # Step count: 0 during sleep, peak during morning & evening active hours
        if is_sleeping:
            steps = int(rng.poisson(lam=5))
        elif 8 <= hour <= 10 or 17 <= hour <= 19:
            steps = int(np.clip(rng.normal(650, 180), 50, 2000))
        else:
            steps = int(np.clip(rng.normal(250, 100), 20, 1200))
        steps_arr[i] = steps

        # Controller medication intake timing (logged at 08:00 if taken that day)
        if hour == 8 and has_controller_rx and daily_controller_taken[min(day, n_days - 1)] == 1:
            controller_arr[i] = 1

        # Non-adherence risk penalty: missed medication adds vulnerability
        adherence_penalty = 0.0
        if has_controller_rx and daily_controller_taken[min(day, n_days - 1)] == 0:
            adherence_penalty = 1.2

        # Symptom score generation (0 - 10 scale)
        # Driven by baseline severity + pollution stress + adherence penalty + stochastic flare
        stress = pollution_stress[i]
        raw_symptom = sev_sym_base + stress + adherence_penalty + rng.normal(0, 0.45)
        # Occasional episodic exacerbation spike if high stress
        if stress > 1.8 and rng.random() < 0.15:
            raw_symptom += rng.uniform(2.0, 4.0)

        symptom_score = int(np.clip(np.round(raw_symptom), 0, 10))
        symptom_arr[i] = symptom_score

        # Rescue puffs follow symptom level
        if symptom_score <= 2:
            puffs = int(rng.random() < 0.04)
        elif symptom_score <= 4:
            puffs = int(rng.choice([1, 2], p=[0.75, 0.25]))
        elif symptom_score <= 7:
            puffs = int(rng.choice([2, 3, 4], p=[0.5, 0.35, 0.15]))
        else:
            puffs = int(rng.choice([3, 4, 5, 6], p=[0.3, 0.4, 0.2, 0.1]))
        rescue_arr[i] = puffs

        # Physiological response to symptoms / distress:
        distress_factor = max(0.0, (symptom_score - 2) * 0.7)

        # Heart rate (bpm)
        activity_hr = min(25.0, steps * 0.015)
        hr_val = base_hr + (circadian_hr * 7.0) + activity_hr + (distress_factor * 3.5) + rng.normal(0, 2.5)
        hr_arr[i] = round(float(np.clip(hr_val, 45.0, 160.0)), 1)

        # Respiratory rate (breaths/min)
        rr_val = baseline_rr + (circadian_rr * 1.5) + (distress_factor * 1.8) + (steps * 0.002) + rng.normal(0, 0.8)
        rr_arr[i] = round(float(np.clip(rr_val, 10.0, 42.0)), 1)

        # SpO2 (%)
        night_spo2_dip = 0.8 if is_sleeping else 0.0
        spo2_drop = distress_factor * 0.9
        spo2_val = baseline_spo2 - night_spo2_dip - spo2_drop + rng.normal(0, 0.4)
        spo2_arr[i] = round(float(np.clip(spo2_val, 75.0, 100.0)), 1)

        # HRV RMSSD (ms) - drops under distress / elevation of HR
        hrv_val = base_hrv - (distress_factor * 3.0) - (hr_val - base_hr) * 0.25 + rng.normal(0, 3.0)
        hrv_arr[i] = round(float(np.clip(hrv_val, 8.0, 110.0)), 1)

    # Apply sensor dropouts / charging periods
    # Simulate missingness in vitals
    dropout_mask = rng.random(n_hours) < sensor_dropout_rate
    # Also occasional 3-4 hour contiguous charging block every ~4 days
    for charge_start in range(12, n_hours, 96):
        if rng.random() < 0.6:
            block_len = rng.integers(2, 5)
            dropout_mask[charge_start : min(n_hours, charge_start + block_len)] = True

    hr_arr[dropout_mask] = np.nan
    rr_arr[dropout_mask] = np.nan
    hrv_arr[dropout_mask] = np.nan
    spo2_arr[dropout_mask] = np.nan

    return pd.DataFrame(
        {
            "patient_id": pid,
            "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
            "respiratory_rate": rr_arr,
            "heart_rate": hr_arr,
            "hrv_rmssd_ms": hrv_arr,
            "spo2": spo2_arr,
            "sleep_hours": sleep_col_arr,
            "steps": steps_arr,
            "controller_taken": controller_arr,
            "rescue_puffs": rescue_arr,
            "symptom_score": symptom_arr,
            "provenance": "simulated",
        }
    )


def simulate_wearables(
    patients_df: pd.DataFrame,
    pollution_df: Optional[pd.DataFrame] = None,
    days: int = 60,
    seed: int = 42,
    num_patients: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate multi-day hourly wearable signals for all patients.

    Args:
        patients_df: EHR records for patients.
        pollution_df: Ambient pollution time-series. If None, ingested automatically.
        days: Duration in days.
        seed: Random seed.
        num_patients: Optional limit on number of patients to simulate.

    Returns:
        Concatenated DataFrame with all patients' hourly records.
    """
    if pollution_df is None:
        pollution_df = ingest_pollution(days=days, seed=seed)

    # Slice pollution to match days if needed
    target_hours = days * 24
    if len(pollution_df) > target_hours:
        pollution_df = pollution_df.iloc[:target_hours].copy()

    records_to_simulate = patients_df
    if num_patients is not None and num_patients < len(patients_df):
        records_to_simulate = patients_df.iloc[:num_patients].copy()

    patient_dfs = []
    for idx, (_, patient) in enumerate(records_to_simulate.iterrows()):
        p_seed = seed + idx * 37
        pdf = simulate_single_patient_timeseries(
            patient=patient,
            pollution_df=pollution_df,
            seed=p_seed,
        )
        patient_dfs.append(pdf)

    return pd.concat(patient_dfs, ignore_index=True)


def save_wearable_timeseries(
    timeseries_df: pd.DataFrame,
    output_path: Union[str, Path] = "data/patient_timeseries.csv",
    seed: int = 42,
) -> Dict[str, str]:
    """Persist simulated timeseries CSV and companion metadata JSON.

    Args:
        timeseries_df: Concatenated patient timeseries DataFrame.
        output_path: Path to target CSV.
        seed: Random seed used.

    Returns:
        Dict of file paths.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timeseries_df.to_csv(out_file, index=False)

    # Also update sample_wearables.csv with a small attributed subset (e.g. 500 rows)
    sample_path = out_file.parent / "sample_wearables.csv"
    timeseries_df.head(1000).to_csv(sample_path, index=False)

    # Save companion metadata JSON
    meta_file = out_file.with_suffix(".json")
    metadata = {
        "dataset": "patient_timeseries.csv",
        "provenance": "simulated",
        "total_rows": len(timeseries_df),
        "unique_patients": int(timeseries_df["patient_id"].nunique()),
        "columns": list(timeseries_df.columns),
        "seed": seed,
        "date_range": [timeseries_df["timestamp"].iloc[0], timeseries_df["timestamp"].iloc[-1]],
        "disclaimer": "All signals are synthetically simulated for research digital twin evaluation. Not real patient data.",
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "patient_timeseries_csv": str(out_file),
        "sample_wearables_csv": str(sample_path),
        "metadata_json": str(meta_file),
    }


def main() -> None:
    """CLI to execute wearable simulation."""
    parser = argparse.ArgumentParser(description="Simulate physiological wearable time series for RespiraTwin.")
    parser.add_argument("--patients_file", type=str, default="data/patients.csv", help="Patients CSV file")
    parser.add_argument("--pollution_file", type=str, default="data/sample_pollution.csv", help="Pollution CSV file")
    parser.add_argument("--days", type=int, default=60, help="Days of hourly data to generate (default: 60)")
    parser.add_argument("--num_patients", type=int, default=None, help="Number of patients to simulate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="data/patient_timeseries.csv", help="Output timeseries CSV")

    args = parser.parse_args()

    patients_df = load_ehr_data(args.patients_file)
    pollution_df = ingest_pollution(file_path=args.pollution_file, days=args.days, seed=args.seed)

    print(f"Simulating {args.days} days of hourly wearable signals...")
    ts_df = simulate_wearables(
        patients_df=patients_df,
        pollution_df=pollution_df,
        days=args.days,
        seed=args.seed,
        num_patients=args.num_patients,
    )
    saved = save_wearable_timeseries(ts_df, output_path=args.output, seed=args.seed)

    print(f"Simulation completed: {len(ts_df)} rows for {ts_df['patient_id'].nunique()} patients.")
    for key, val in saved.items():
        print(f"  - {key}: {val}")


if __name__ == "__main__":
    main()
