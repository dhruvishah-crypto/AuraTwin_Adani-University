"""Episode-based exacerbation event simulation for RespiraTwin.

Defines target label:
    event_next_24h(t) = 1 if an exacerbation episode onset occurs in (t, t + 24h], else 0.

Episode onsets are generated from strictly exogenous drivers (PM2.5, adherence,
severity, prior history, pollution sensitivity, latent vulnerability, noise, and shocks),
and saved to data/episodes.csv strictly for clinical evaluation.

CRITICAL NOTICE:
All coefficients and formulations represent simulator assumptions for evaluating
the digital twin algorithm, NOT clinical ground truth facts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.build_features import build_features


SIMULATION_CONFIG: Dict[str, Any] = {
    "target_label": "event_next_24h",
    "definition": "event_next_24h(t) = 1 if an exacerbation episode onset occurs in (t, t + 24h], else 0",
    "lookahead_hours": 24,
    "refractory_hours": 72,
    "silent_episode_fraction": 0.30,
    "duration_hours_range": [12, 48],
    "prodrome_hours_range": [6, 24],
    "target_positive_fraction_range": [0.05, 0.15],
    "target_inter_episode_days_range": [10.0, 25.0],
    "exogenous_drivers": [
        "pm25_rolling_mean_24h",
        "controller_adherence_lapse",
        "asthma_severity_numeric",
        "prior_exacerbations_12m",
        "pollution_sensitivity_numeric",
        "latent_vulnerability",
        "gaussian_noise",
        "stochastic_shocks",
    ],
    "disclaimer": (
        "All parameters and equations represent simulation assumptions for evaluating "
        "the RespiraTwin digital twin, not validated clinical facts."
    ),
}

# Backward compatibility alias
SIMULATION_COEFFICIENTS: Dict[str, Any] = SIMULATION_CONFIG


def compute_exacerbation_risk_logits(
    features_df: pd.DataFrame,
    latent_truth_df: Optional[pd.DataFrame] = None,
    coefficients: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward compatibility wrapper returning (logits, probs, events)."""
    labeled = simulate_events(features_df, latent_truth_df=latent_truth_df, seed=seed)
    events = labeled["event_next_24h"].to_numpy()
    probs = labeled["simulated_risk_prob"].to_numpy()
    clipped_probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped_probs / (1.0 - clipped_probs))
    return logits, probs, events


def label_events_from_onsets(
    features_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    lookahead_hours: int = 24,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized calculation of event_next_24h and simulated_risk_prob from episode onsets.

    Condition:
        event_next_24h(t) = 1 if exists onset t_onset such that t < t_onset <= t + 24h.
        Equivalently: t_onset - 24h <= t < t_onset.

    Args:
        features_df: Features table with patient_id and timestamp.
        episodes_df: Episodes table with patient_id and onset_timestamp.
        lookahead_hours: Prediction horizon (default: 24h).
        seed: Random seed for continuous probability noise.

    Returns:
        Tuple of (binary_events_array, continuous_probs_array).
    """
    rng = np.random.default_rng(seed)
    n = len(features_df)

    feat_timestamps = pd.to_datetime(features_df["timestamp"]).to_numpy()
    feat_pids = features_df["patient_id"].to_numpy()

    events = np.zeros(n, dtype=int)
    min_lead_times = np.full(n, 999.0, dtype=float)

    # Group episodes by patient
    ep_df = episodes_df.copy()
    ep_df["onset_dt"] = pd.to_datetime(ep_df["onset_timestamp"])

    pids_in_features = set(features_df["patient_id"].unique())

    # Build index map for speed
    for pid in pids_in_features:
        pat_episodes = ep_df[ep_df["patient_id"] == pid]
        if pat_episodes.empty:
            continue

        pat_mask = (feat_pids == pid)
        pat_ts = feat_timestamps[pat_mask]
        pat_events = np.zeros(len(pat_ts), dtype=int)
        pat_leads = np.full(len(pat_ts), 999.0, dtype=float)

        lookahead_ns = np.timedelta64(lookahead_hours, "h")

        for _, ep in pat_episodes.iterrows():
            onset_ns = np.datetime64(ep["onset_dt"])
            # Window condition: onset_ns - 24h <= t < onset_ns
            in_window = (pat_ts < onset_ns) & (pat_ts >= (onset_ns - lookahead_ns))
            pat_events[in_window] = 1

            # Track lead time in hours
            diff_hours = (onset_ns - pat_ts[in_window]) / np.timedelta64(1, "h")
            pat_leads[in_window] = np.minimum(pat_leads[in_window], diff_hours)

        events[pat_mask] = pat_events
        min_lead_times[pat_mask] = pat_leads

    # Compute continuous calibrated risk probability
    # Higher as lead time approaches 0; lower when distant
    probs = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, 0.03, size=n)

    for i in range(n):
        if events[i] == 1:
            lead = min_lead_times[i]
            # Lead ranges from 0 to 24h: prob ranges ~0.65 to 0.95
            p = 0.92 - (0.011 * lead) + noise[i]
            probs[i] = np.clip(p, 0.52, 0.98)
        else:
            base_p = 0.045
            if "pm25_rolling_mean_24h" in features_df.columns:
                pm_val = float(features_df["pm25_rolling_mean_24h"].iloc[i])
                base_p += max(0.0, pm_val - 35.0) * 0.0008
            if "asthma_severity_numeric" in features_df.columns:
                sev_val = float(features_df["asthma_severity_numeric"].iloc[i])
                base_p += (sev_val - 1.0) * 0.015
            probs[i] = np.clip(base_p + noise[i], 0.01, 0.38)

    return events, np.round(probs, 4)


def simulate_events(
    features_df: pd.DataFrame,
    latent_truth_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
    episodes_df: Optional[pd.DataFrame] = None,
    coefficients: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate simulated exacerbation events and attach event_next_24h label.

    Condition:
        event_next_24h(t) = 1 if an onset occurs in (t, t + 24h] for this patient.

    Args:
        features_df: Feature DataFrame.
        latent_truth_df: Optional ground truth latent vulnerability DataFrame.
        seed: Random seed.
        episodes_df: Optional DataFrame containing true episode onsets.
        coefficients: Optional parameter dictionary.

    Returns:
        DataFrame with features plus 'event_next_24h' and 'simulated_risk_prob'.
    """
    resolved_episodes = episodes_df

    # If episodes not passed and seed is default (42), attempt to load data/episodes.csv if it covers the feature patients
    if resolved_episodes is None and seed == 42:
        candidates = [Path("data/episodes.csv"), WORKSPACE_ROOT / "data" / "episodes.csv"]
        for c in candidates:
            if c.is_file():
                loaded_ep = pd.read_csv(c)
                feat_pids = set(features_df["patient_id"].unique())
                ep_pids = set(loaded_ep["patient_id"].unique())
                if feat_pids.issubset(ep_pids):
                    resolved_episodes = loaded_ep
                    break

    # If still None (e.g., in unit tests on synthetic fixture), generate onsets on the fly
    if resolved_episodes is None:
        from src.simulate_wearables import simulate_patient_episodes

        pids = features_df["patient_id"].unique()
        timestamps = pd.to_datetime(features_df["timestamp"]).drop_duplicates().sort_values()
        dummy_poll = pd.DataFrame(
            {
                "timestamp": timestamps,
                "pm25": np.full(len(timestamps), 45.0),
                "pm10": np.full(len(timestamps), 80.0),
            }
        )

        vuln_dict: Dict[str, float] = {}
        if latent_truth_df is not None:
            for _, r in latent_truth_df.iterrows():
                vuln_dict[str(r["patient_id"])] = float(r["latent_vulnerability"])

        generated_episodes: List[Dict[str, Any]] = []
        for idx, pid in enumerate(pids):
            p_seed = seed + idx * 37
            pat_feat = features_df[features_df["patient_id"] == pid].iloc[0]
            pat_series = pd.Series(
                {
                    "patient_id": pid,
                    "asthma_severity": "Moderate",
                    "prior_exacerbations_12m": float(pat_feat.get("prior_exacerbations_12m", 1.0)),
                    "pollution_sensitivity": "Medium",
                    "controller_prescribed": int(pat_feat.get("controller_prescribed", 1)),
                }
            )
            n_days = (len(dummy_poll) + 23) // 24
            daily_ctrl = np.ones(n_days, dtype=int)
            vuln_val = vuln_dict.get(pid, 0.5)

            p_eps = simulate_patient_episodes(
                patient=pat_series,
                pollution_df=dummy_poll,
                daily_controller_taken=daily_ctrl,
                latent_vuln=vuln_val,
                seed=p_seed,
                refractory_hours=72,
                silent_fraction=0.30,
            )
            generated_episodes.extend(p_eps)

        resolved_episodes = (
            pd.DataFrame(generated_episodes)
            if generated_episodes
            else pd.DataFrame(
                columns=[
                    "patient_id",
                    "episode_id",
                    "onset_timestamp",
                    "end_timestamp",
                    "duration_hours",
                    "prodrome_hours",
                    "is_silent",
                    "amplitude",
                ]
            )
        )

    # Label events from episode onsets
    events, probs = label_events_from_onsets(
        features_df=features_df,
        episodes_df=resolved_episodes,
        lookahead_hours=24,
        seed=seed,
    )

    result_df = features_df.copy()
    result_df["event_next_24h"] = events
    result_df["simulated_risk_prob"] = probs

    return result_df


def save_event_simulation_config(
    output_path: Union[str, Path] = "data/events_config.json",
    seed: int = 42,
    event_rate: Optional[float] = None,
    total_samples: Optional[int] = None,
    total_episodes: Optional[int] = None,
) -> Path:
    """Persist simulation metadata and explicit assumptions to config JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    config = {
        **SIMULATION_CONFIG,
        "seed": seed,
        "event_rate": round(float(event_rate), 4) if event_rate is not None else None,
        "total_samples": int(total_samples) if total_samples is not None else None,
        "total_episodes": int(total_episodes) if total_episodes is not None else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return out_file


def main() -> None:
    """CLI to generate labeled dataset with event_next_24h from episode onsets."""
    parser = argparse.ArgumentParser(description="Generate episode-based event labels for RespiraTwin.")
    parser.add_argument("--features", type=str, default="data/features.csv", help="Features CSV path")
    parser.add_argument("--episodes", type=str, default="data/episodes.csv", help="Episodes CSV path")
    parser.add_argument("--latent", type=str, default="data/latent_truth.csv", help="Latent truth CSV path")
    parser.add_argument("--output", type=str, default="data/labeled_dataset.csv", help="Output labeled CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    print("Loading features and episode onsets...")
    features_df = pd.read_csv(args.features)
    episodes_df = pd.read_csv(args.episodes) if Path(args.episodes).is_file() else None
    latent_truth_df = pd.read_csv(args.latent) if Path(args.latent).is_file() else None

    print("Labeling target event_next_24h from onset schedule...")
    labeled_df = simulate_events(
        features_df=features_df,
        latent_truth_df=latent_truth_df,
        seed=args.seed,
        episodes_df=episodes_df,
    )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_file, index=False)

    # Also save sample labeled dataset
    sample_path = out_file.parent / "sample_labeled_dataset.csv"
    labeled_df.head(1000).to_csv(sample_path, index=False)

    event_rate = float(labeled_df["event_next_24h"].mean())
    n_episodes = len(episodes_df) if episodes_df is not None else 0

    save_event_simulation_config(
        output_path=out_file.parent / "events_config.json",
        seed=args.seed,
        event_rate=event_rate,
        total_samples=len(labeled_df),
        total_episodes=n_episodes,
    )

    print(f"Labeled dataset created: {labeled_df.shape[0]} rows x {labeled_df.shape[1]} columns.")
    print(f"Target event rate (event_next_24h): {event_rate * 100:.2f}% (Target: 5% - 15%)")
    print(f"Saved to: {out_file} and {sample_path}")


if __name__ == "__main__":
    main()
