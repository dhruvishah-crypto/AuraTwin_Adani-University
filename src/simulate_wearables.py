"""Physiological wearable and symptom time series simulator for RespiraTwin.

Simulates coherent, multi-day hourly wearable signals for asthma patients:
- Exogenous exacerbation onset generation driven strictly by exogenous hazards
  (recent PM2.5 exposure, controller adherence, clinical severity, prior exacerbations,
   phenotypic pollution sensitivity, hidden latent vulnerability, noise, and stochastic shocks).
- Post-episode refractory period (72 hours) during which no new exacerbation can initiate.
- Realistic episode duration (12-48h) and prodromal window (6-24h) before onset, with
  rising respiratory rate, falling SpO2, elevated rescue puffs, and worsening symptom diary scores.
- ~30% of episodes modeled as 'silent' (weak or no prodrome) to prevent trivial detection.
- Post-episode recovery phase returning signals smoothly to baseline.
- Circadian rhythms across vitals (HR, SpO2, RR, HRV).
- Sensor noise, off-wrist charging, and wearable dropout (missingness).
- Provenance tagging ('simulated') and reproducibility sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.ingest_pollution import ingest_pollution
from src.load_ehr import load_ehr_data

WEARABLE_COLUMNS: List[str] = [
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

EPISODE_COLUMNS: List[str] = [
    "patient_id",
    "episode_id",
    "onset_timestamp",
    "end_timestamp",
    "duration_hours",
    "prodrome_hours",
    "is_silent",
    "amplitude",
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
    kernel = np.ones(lag_hours) / lag_hours
    padded_pm = np.pad(pm25_series, (lag_hours - 1, 0), mode="edge")
    rolling_pm = np.convolve(padded_pm, kernel, mode="valid")

    # Excess over clean air threshold (40 ug/m3)
    excess_pm = np.maximum(0.0, rolling_pm - 40.0)

    # Sensitivity scaling multiplier
    multiplier = {"Low": 0.008, "Medium": 0.022, "High": 0.048}.get(sensitivity, 0.02)
    return np.minimum(3.5, excess_pm * multiplier)


def simulate_patient_episodes(
    patient: pd.Series,
    pollution_df: pd.DataFrame,
    daily_controller_taken: np.ndarray,
    latent_vuln: float = 0.5,
    seed: int = 42,
    refractory_hours: int = 72,
    silent_fraction: float = 0.30,
) -> List[Dict[str, Any]]:
    """Generate discrete exacerbation episodes for a patient from exogenous hazard.

    Hazard is driven strictly by exogenous variables:
    - Recent PM2.5 exposure (24h rolling mean excess)
    - Controller medication adherence (daily intake/lapse)
    - Clinical asthma severity (Mild / Moderate / Severe)
    - Prior 12m exacerbations
    - Phenotypic pollution sensitivity (Low / Medium / High)
    - Latent ground-truth vulnerability (simulator only)
    - Gaussian noise and random shocks

    Args:
        patient: Patient EHR series.
        pollution_df: Ambient pollution dataframe.
        daily_controller_taken: Array of daily binary adherence values.
        latent_vuln: Ground-truth unobserved latent vulnerability [0.0, 1.0].
        seed: Random seed for reproducibility.
        refractory_hours: Hours following episode end where hazard is strictly 0 (default: 72h).
        silent_fraction: Fraction of episodes with weak/no prodrome (default: 0.30).

    Returns:
        List of episode dictionaries with onset, duration, prodrome, and amplitude.
    """
    rng = np.random.default_rng(seed)
    n_hours = len(pollution_df)
    timestamps = pd.to_datetime(pollution_df["timestamp"]).tolist()

    pid = str(patient.get("patient_id", "PT0000"))
    severity = str(patient.get("asthma_severity", "Moderate"))
    sev_num = {"Mild": 1, "Moderate": 2, "Severe": 3}.get(severity, 2)
    poll_sens = str(patient.get("pollution_sensitivity", "Medium"))
    poll_sens_num = {"Low": 1, "Medium": 2, "High": 3}.get(poll_sens, 2)
    prior_exac = float(patient.get("prior_exacerbations_12m", 1.0))
    has_ctrl = int(patient.get("controller_prescribed", 1)) == 1

    # 1. Exogenous recent PM2.5 exposure (24h rolling mean)
    pm25_vals = pollution_df["pm25"].to_numpy(dtype=float)
    kernel = np.ones(24) / 24.0
    padded_pm = np.pad(pm25_vals, (23, 0), mode="edge")
    rolling_pm25 = np.convolve(padded_pm, kernel, mode="valid")
    pm25_excess = np.maximum(0.0, rolling_pm25 - 35.0)

    # 2. Exogenous adherence lapse per hour
    n_days = len(daily_controller_taken)
    adherence_lapse = np.zeros(n_hours, dtype=float)
    for h in range(n_hours):
        d = h // 24
        if has_ctrl and d < n_days and daily_controller_taken[d] == 0:
            adherence_lapse[h] = 0.55

    # 3. Exogenous Hourly Hazard Logit
    base_logit = -5.85 + 0.65 * (latent_vuln - 0.5) + 0.35 * (sev_num - 2) + 0.12 * (min(prior_exac, 4.0) - 1.5)
    pm_stress = pm25_excess * (0.012 * (1.0 + 0.35 * poll_sens_num))
    noise = rng.normal(0.0, 0.35, size=n_hours)
    shocks = (rng.random(size=n_hours) < 0.012) * rng.uniform(1.0, 2.0, size=n_hours)

    logits = base_logit + pm_stress + adherence_lapse + noise + shocks
    hazards = 1.0 / (1.0 + np.exp(-logits))

    # 4. Sequential Onset Generation with 72h Refractory Period
    episodes: List[Dict[str, Any]] = []
    t = 24  # Leave room for initial pre-prodromal baseline
    epi_count = 0

    while t < n_hours - 24:
        if rng.random() < hazards[t]:
            epi_count += 1
            duration = int(rng.integers(12, 49))   # 12 to 48 hours
            prodrome = int(rng.integers(6, 25))    # 6 to 24 hours
            is_silent = bool(rng.random() < silent_fraction)
            amplitude = round(float(rng.uniform(0.75, 1.35)), 3)

            onset_hour = t
            end_hour = min(n_hours, t + duration)

            episodes.append(
                {
                    "patient_id": pid,
                    "episode_id": f"{pid}_EP{epi_count:03d}",
                    "onset_hour": onset_hour,
                    "onset_timestamp": timestamps[onset_hour].strftime("%Y-%m-%d %H:%M:%S"),
                    "end_hour": end_hour,
                    "end_timestamp": timestamps[min(n_hours - 1, end_hour)].strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_hours": duration,
                    "prodrome_hours": min(prodrome, onset_hour),
                    "is_silent": is_silent,
                    "amplitude": amplitude,
                }
            )
            # Refractory period: 72 hours after episode ends
            t = end_hour + refractory_hours
        else:
            t += 1

    return episodes


def simulate_single_patient_timeseries(
    patient: pd.Series,
    pollution_df: pd.DataFrame,
    seed: int = 42,
    sensor_dropout_rate: float = 0.035,
    latent_vulnerability: Optional[float] = None,
    episodes: Optional[List[Dict[str, Any]]] = None,
    return_episodes: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
    """Generate hourly wearable time series for a single patient record conditioned on episodes.

    Args:
        patient: Row from patients.csv.
        pollution_df: Hourly pollution dataframe with 'timestamp' and 'pm25'.
        seed: Random seed for reproducibility.
        sensor_dropout_rate: Probability of missing wearable sensor readings.
        latent_vulnerability: Unobserved ground truth vulnerability (0 to 1).
        episodes: Optional pre-generated episode schedule.
        return_episodes: Whether to return (timeseries_df, episodes_list).

    Returns:
        DataFrame containing simulated hourly observations, or tuple with episode list.
    """
    rng = np.random.default_rng(seed)
    n_hours = len(pollution_df)
    timestamps = pd.to_datetime(pollution_df["timestamp"]).tolist()

    pid = str(patient.get("patient_id", "PT0000"))
    severity = str(patient.get("asthma_severity", "Moderate"))
    pollution_sens = str(patient.get("pollution_sensitivity", "Medium"))
    baseline_rr = float(patient.get("baseline_resp_rate", 16.0))
    baseline_spo2 = float(patient.get("baseline_spo2", 96.0))
    has_controller_rx = int(patient.get("controller_prescribed", 1)) == 1

    # Base heart rate (derived from respiratory baseline & age)
    age = float(patient.get("age", 40.0))
    base_hr = 70.0 + (baseline_rr - 15.0) * 1.5 + (0.15 * max(0.0, age - 40.0))
    # Base HRV RMSSD (ms) - inversely correlated with age & severity
    base_hrv = max(22.0, 55.0 - (0.4 * age) - (6.0 if severity == "Severe" else 0.0))

    # Calculate pollution stress curve
    pm25_vals = pollution_df["pm25"].to_numpy(dtype=float)
    pollution_stress = compute_lagged_pollution_stress(pm25_vals, pollution_sens, lag_hours=12)

    # Daily adherence simulation:
    # Patients take daily controller (morning ~08:00 or evening ~20:00)
    # Adherent patients take it ~88% of days; occasional non-adherent streaks of 2-4 days
    n_days = (n_hours + 23) // 24
    daily_controller_taken = np.zeros(n_days, dtype=int)
    if has_controller_rx:
        day_idx = 0
        while day_idx < n_days:
            streak_adherent = rng.random() < 0.85
            streak_len = int(rng.integers(1, 6))
            end_day = min(n_days, day_idx + streak_len)
            daily_controller_taken[day_idx:end_day] = 1 if streak_adherent else 0
            day_idx = end_day
    else:
        daily_controller_taken.fill(0)

    # Daily sleep hours: night d runs from (d-1) 23:00 to (d) 07:00
    daily_sleep = np.zeros(n_days + 1, dtype=float)
    for d in range(n_days + 1):
        daily_sleep[d] = round(float(np.clip(rng.normal(7.2, 0.9), 4.5, 9.8)), 1)

    # Resolve latent ground-truth vulnerability
    if latent_vulnerability is None:
        latent_vulnerability = float(patient.get("latent_vulnerability", 0.5))

    # Generate or adopt exacerbation episode schedule
    if episodes is None:
        patient_episodes = simulate_patient_episodes(
            patient=patient,
            pollution_df=pollution_df,
            daily_controller_taken=daily_controller_taken,
            latent_vuln=latent_vulnerability,
            seed=seed,
            refractory_hours=72,
            silent_fraction=0.30,
        )
    else:
        patient_episodes = episodes

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

        # Causal Sleep Assignment
        if hour < 7:
            sleep_col_arr[i] = daily_sleep[max(0, day)]
        else:
            sleep_col_arr[i] = daily_sleep[min(len(daily_sleep) - 1, day + 1)]

        # Determine Episode Dynamics at Hour i
        # 1. Acute Episode
        # 2. Prodrome (6-24h before onset)
        # 3. Post-Episode Recovery (~18h decay)
        is_in_acute = False
        acute_amplitude = 0.0
        prodrome_factor = 0.0
        recovery_factor = 0.0

        for ep in patient_episodes:
            onset = ep["onset_hour"]
            end = ep["end_hour"]
            prod_len = ep["prodrome_hours"]
            amp = ep["amplitude"]
            silent = ep["is_silent"]

            # Acute window
            if onset <= i < end:
                is_in_acute = True
                acute_amplitude = amp
                break
            # Prodrome window
            elif (onset - prod_len) <= i < onset:
                progress = (i - (onset - prod_len)) / max(1.0, float(prod_len))
                ramp = (progress ** 1.3) * amp
                # ~30% silent episodes exhibit minimal/weak prodromal deviation
                p_mult = 0.08 if silent else 1.0
                prodrome_factor = max(prodrome_factor, ramp * p_mult)
            # Recovery window
            elif end <= i < (end + 18):
                rec_progress = (i - end) / 18.0
                rec_decay = (1.0 - rec_progress) * amp * 0.7
                recovery_factor = max(recovery_factor, rec_decay)

        # Circadian rhythm phase [-1, +1]
        circadian_hr = np.sin(2 * np.pi * (hour - 9) / 24)
        circadian_rr = np.sin(2 * np.pi * (hour - 10) / 24)
        circadian_spo2 = 0.25 * np.sin(2 * np.pi * (hour - 15) / 24)
        is_sleeping = 0 <= hour <= 6

        # Step count: reduced significantly during acute episodes, slightly during prodrome
        if is_sleeping:
            steps = int(rng.poisson(lam=5))
        elif 8 <= hour <= 10 or 17 <= hour <= 19:
            steps = int(np.clip(rng.normal(650, 180), 50, 2000))
        else:
            steps = int(np.clip(rng.normal(250, 100), 20, 1200))

        if is_in_acute:
            steps = int(steps * 0.35)
        elif prodrome_factor > 0.4:
            steps = int(steps * 0.75)
        steps_arr[i] = steps

        # Controller medication intake timing (logged at 08:00 if taken that day)
        if hour == 8 and has_controller_rx and daily_controller_taken[min(day, n_days - 1)] == 1:
            controller_arr[i] = 1

        # Non-adherence risk penalty: missed medication adds vulnerability
        adherence_penalty = 0.0
        if has_controller_rx and daily_controller_taken[min(day, n_days - 1)] == 0:
            adherence_penalty = 1.2

        # Symptom score generation (0 - 10 scale)
        # Combines baseline severity + pollution stress + adherence lapse + prodrome / acute / recovery
        stress = pollution_stress[i]
        acute_boost = 5.5 * acute_amplitude if is_in_acute else 0.0
        prodrome_boost = 3.2 * prodrome_factor
        recovery_boost = 2.2 * recovery_factor
        raw_symptom = sev_sym_base + stress + adherence_penalty + acute_boost + prodrome_boost + recovery_boost + rng.normal(0, 0.35)
        symptom_score = int(np.clip(np.round(raw_symptom), 0, 10))
        symptom_arr[i] = symptom_score

        # Rescue puffs follow symptoms & prodrome
        if is_in_acute:
            puffs = int(rng.choice([3, 4, 5, 6], p=[0.20, 0.40, 0.30, 0.10]))
        elif prodrome_factor > 0.35:
            puffs = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
        elif symptom_score <= 2:
            puffs = int(rng.random() < 0.03)
        elif symptom_score <= 4:
            puffs = int(rng.choice([1, 2], p=[0.75, 0.25]))
        else:
            puffs = int(rng.choice([2, 3, 4], p=[0.50, 0.35, 0.15]))
        rescue_arr[i] = puffs

        # Physiological response to symptoms / distress / prodromal drift
        distress_factor = max(0.0, (symptom_score - 2) * 0.7)
        if prodrome_factor > 0:
            distress_factor += 1.2 * prodrome_factor

        # Heart rate (bpm): drifts upward during prodrome & acute flare
        activity_hr = min(25.0, steps * 0.015)
        hr_val = base_hr + (circadian_hr * 7.0) + activity_hr + (distress_factor * 3.5) + rng.normal(0, 2.2)
        hr_arr[i] = round(float(np.clip(hr_val, 45.0, 160.0)), 1)

        # Respiratory rate (breaths/min): drifts upward 6-24h before onset, elevates in acute
        rr_elevation = (5.5 * acute_amplitude) if is_in_acute else (3.5 * prodrome_factor + 2.5 * recovery_factor)
        rr_val = baseline_rr + (circadian_rr * 1.5) + rr_elevation + (steps * 0.002) + rng.normal(0, 0.6)
        rr_arr[i] = round(float(np.clip(rr_val, 10.0, 42.0)), 1)

        # SpO2 (%): drops during nocturnal sleep, prodrome, and acute flare
        night_spo2_dip = 0.8 if is_sleeping else 0.0
        spo2_drop = (4.0 * acute_amplitude) if is_in_acute else (2.5 * prodrome_factor + 1.8 * recovery_factor)
        spo2_val = baseline_spo2 + circadian_spo2 - night_spo2_dip - spo2_drop + rng.normal(0, 0.35)
        spo2_arr[i] = round(float(np.clip(spo2_val, 86.0, 100.0)), 1)

        # HRV RMSSD (ms): vagal withdrawal, drops early during prodrome and acute flare
        hrv_val = base_hrv - (distress_factor * 3.2) - (hr_val - base_hr) * 0.25 + rng.normal(0, 2.8)
        hrv_arr[i] = round(float(np.clip(hrv_val, 8.0, 110.0)), 1)

    # Apply sensor dropouts / charging periods
    dropout_mask = rng.random(n_hours) < sensor_dropout_rate
    for charge_start in range(12, n_hours, 96):
        if rng.random() < 0.6:
            block_len = rng.integers(2, 5)
            dropout_mask[charge_start : min(n_hours, charge_start + block_len)] = True

    hr_arr[dropout_mask] = np.nan
    rr_arr[dropout_mask] = np.nan
    hrv_arr[dropout_mask] = np.nan
    spo2_arr[dropout_mask] = np.nan

    ts_df = pd.DataFrame(
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

    if return_episodes:
        return ts_df, patient_episodes
    return ts_df


def simulate_wearables(
    patients_df: pd.DataFrame,
    pollution_df: Optional[pd.DataFrame] = None,
    latent_truth_df: Optional[pd.DataFrame] = None,
    days: int = 60,
    seed: int = 42,
    num_patients: Optional[int] = None,
    return_episodes: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Simulate multi-day hourly wearable signals for all patients conditioned on episode schedule.

    Args:
        patients_df: EHR records for patients.
        pollution_df: Ambient pollution time-series. If None, ingested automatically.
        latent_truth_df: Ground truth latent vulnerability table.
        days: Duration in days.
        seed: Random seed.
        num_patients: Optional limit on number of patients to simulate.
        return_episodes: If True, returns tuple (timeseries_df, episodes_df).

    Returns:
        Concatenated DataFrame with all patients' hourly records (or tuple with episodes DataFrame).
    """
    if pollution_df is None:
        pollution_df = ingest_pollution(days=days, seed=seed)

    target_hours = days * 24
    if len(pollution_df) > target_hours:
        pollution_df = pollution_df.iloc[:target_hours].copy()

    records_to_simulate = patients_df
    if num_patients is not None and num_patients < len(patients_df):
        records_to_simulate = patients_df.iloc[:num_patients].copy()

    # Attempt to load latent truth if not provided
    vuln_map: Dict[str, float] = {}
    if latent_truth_df is not None:
        for _, row in latent_truth_df.iterrows():
            vuln_map[str(row["patient_id"])] = float(row["latent_vulnerability"])
    else:
        candidates = [Path("data/latent_truth.csv"), WORKSPACE_ROOT / "data" / "latent_truth.csv"]
        for c in candidates:
            if c.is_file():
                ldf = pd.read_csv(c)
                for _, row in ldf.iterrows():
                    vuln_map[str(row["patient_id"])] = float(row["latent_vulnerability"])
                break

    patient_dfs: List[pd.DataFrame] = []
    all_episodes: List[Dict[str, Any]] = []

    for idx, (_, patient) in enumerate(records_to_simulate.iterrows()):
        p_seed = seed + idx * 37
        pid = str(patient["patient_id"])
        latent_val = vuln_map.get(pid, float(patient.get("latent_vulnerability", 0.5)))

        pdf, ep_list = simulate_single_patient_timeseries(
            patient=patient,
            pollution_df=pollution_df,
            seed=p_seed,
            latent_vulnerability=latent_val,
            return_episodes=True,
        )
        patient_dfs.append(pdf)
        all_episodes.extend(ep_list)

    concatenated_ts = pd.concat(patient_dfs, ignore_index=True)
    episodes_df = pd.DataFrame(all_episodes) if all_episodes else pd.DataFrame(columns=EPISODE_COLUMNS)

    if return_episodes:
        return concatenated_ts, episodes_df
    return concatenated_ts


def save_wearable_timeseries(
    timeseries_df: pd.DataFrame,
    episodes_df: Optional[pd.DataFrame] = None,
    output_path: Union[str, Path] = "data/patient_timeseries.csv",
    episodes_output_path: Union[str, Path] = "data/episodes.csv",
    seed: int = 42,
) -> Dict[str, str]:
    """Persist simulated timeseries CSV, companion metadata JSON, and evaluation episodes CSV.

    Args:
        timeseries_df: Concatenated patient timeseries DataFrame.
        episodes_df: Optional DataFrame of exacerbation episodes.
        output_path: Path to target timeseries CSV.
        episodes_output_path: Path to target episodes CSV.
        seed: Random seed used.

    Returns:
        Dict of file paths.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    timeseries_df.to_csv(out_file, index=False)

    # Update sample_wearables.csv with a small attributed subset (e.g. 1000 rows)
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

    results = {
        "patient_timeseries_csv": str(out_file),
        "sample_wearables_csv": str(sample_path),
        "metadata_json": str(meta_file),
    }

    # Save episodes CSV for evaluation
    if episodes_df is not None:
        ep_file = Path(episodes_output_path)
        ep_file.parent.mkdir(parents=True, exist_ok=True)
        episodes_df.to_csv(ep_file, index=False)

        sample_ep_path = ep_file.parent / "sample_episodes.csv"
        episodes_df.head(100).to_csv(sample_ep_path, index=False)

        ep_meta_file = ep_file.with_suffix(".json")
        ep_metadata = {
            "dataset": "episodes.csv",
            "provenance": "simulated_evaluation_only",
            "total_episodes": len(episodes_df),
            "unique_patients": int(episodes_df["patient_id"].nunique()) if not episodes_df.empty else 0,
            "columns": list(episodes_df.columns),
            "seed": seed,
            "disclaimer": "Episode onsets saved strictly for model evaluation and test metrics. Never used as features.",
        }
        with open(ep_meta_file, "w", encoding="utf-8") as f:
            json.dump(ep_metadata, f, indent=2)

        results["episodes_csv"] = str(ep_file)
        results["sample_episodes_csv"] = str(sample_ep_path)

    return results


def main() -> None:
    """CLI to execute wearable simulation."""
    parser = argparse.ArgumentParser(description="Simulate physiological wearable time series for RespiraTwin.")
    parser.add_argument("--patients_file", type=str, default="data/patients.csv", help="Patients CSV file")
    parser.add_argument("--pollution_file", type=str, default="data/sample_pollution.csv", help="Pollution CSV file")
    parser.add_argument("--days", type=int, default=60, help="Days of hourly data to generate (default: 60)")
    parser.add_argument("--num_patients", type=int, default=None, help="Number of patients to simulate (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default="data/patient_timeseries.csv", help="Output timeseries CSV")
    parser.add_argument("--episodes_output", type=str, default="data/episodes.csv", help="Output episodes CSV")

    args = parser.parse_args()

    patients_df = load_ehr_data(args.patients_file)
    pollution_df = ingest_pollution(file_path=args.pollution_file, days=args.days, seed=args.seed)

    print(f"Simulating {args.days} days of hourly wearable signals conditioned on episode schedule...")
    ts_df, ep_df = simulate_wearables(
        patients_df=patients_df,
        pollution_df=pollution_df,
        days=args.days,
        seed=args.seed,
        num_patients=args.num_patients,
        return_episodes=True,
    )
    saved = save_wearable_timeseries(
        timeseries_df=ts_df,
        episodes_df=ep_df,
        output_path=args.output,
        episodes_output_path=args.episodes_output,
        seed=args.seed,
    )

    print(f"Simulation completed: {len(ts_df)} rows for {ts_df['patient_id'].nunique()} patients.")
    print(f"Total episodes generated: {len(ep_df)} across {ep_df['patient_id'].nunique()} patients.")
    for key, val in saved.items():
        print(f"  - {key}: {val}")


if __name__ == "__main__":
    main()
