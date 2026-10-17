"""Ground-truth exacerbation event simulation for RespiraTwin.

Generates the target label 'event_next_24h' (exacerbation risk in next 24 hours)
from a transparent risk formulation combining:
- Recent PM2.5 exposure and pollution sensitivity
- Clinical history (prior exacerbations, asthma severity)
- Controller medication adherence lapses
- Rising rescue inhaler use and acute symptom surges
- Physiological deviations (respiratory rate rise, SpO2 drop, tachycardia)
- The isolated ground-truth 'latent_vulnerability' (for simulator only)
- Unobserved stochastic shocks and Gaussian noise (non-deterministic label)

CRITICAL NOTICE:
All coefficients represent modeling assumptions for evaluating the digital twin,
NOT validated clinical facts. Results measure learning of a simulated process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd

from src.build_features import build_features


# Explicit transparent simulation coefficients (assumptions for simulated learning)
SIMULATION_COEFFICIENTS: Dict[str, Any] = {
    "intercept": -7.40,
    "beta_latent_vulnerability": 1.60,
    "beta_prior_exacerbations": 0.28,
    "beta_asthma_severity": 0.45,
    "beta_pm25_exposure": 0.018,
    "beta_missed_controller": 0.85,
    "beta_rescue_puffs": 0.32,
    "beta_symptom_score": 0.25,
    "beta_rr_deviation": 0.15,
    "beta_spo2_drop": 0.35,
    "beta_hr_elevation": 0.04,
    "noise_std": 0.45,
    "shock_probability": 0.035,
    "shock_magnitude_min": 1.20,
    "shock_magnitude_max": 2.50,
}


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def compute_exacerbation_risk_logits(
    features_df: pd.DataFrame,
    latent_truth_df: pd.DataFrame,
    coefficients: Dict[str, Any] = SIMULATION_COEFFICIENTS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute risk logits, continuous probabilities, and binary event labels.

    Args:
        features_df: Engineered feature matrix containing patient_id and feature columns.
        latent_truth_df: Hidden ground truth containing patient_id and latent_vulnerability.
        coefficients: Dictionary of simulation weights and noise parameters.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (risk_logits, risk_probabilities, binary_events).
    """
    rng = np.random.default_rng(seed)
    n = len(features_df)

    # Merge latent vulnerability strictly for the simulator
    merged = pd.merge(
        features_df[["patient_id"]],
        latent_truth_df[["patient_id", "latent_vulnerability"]],
        on="patient_id",
        how="left",
    )
    latent_vuln = merged["latent_vulnerability"].fillna(0.5).to_numpy()

    # Extract feature vectors
    prior_exac = np.minimum(features_df["prior_exacerbations_12m"].to_numpy(), 5.0)
    severity_num = features_df["asthma_severity_numeric"].to_numpy()
    pm25_roll = np.maximum(0.0, features_df["pm25_rolling_mean_24h"].to_numpy() - 35.0)
    poll_sens = features_df["pollution_sensitivity_numeric"].to_numpy()
    adh_lapse = (1.0 - features_df["controller_adherence_7d"].to_numpy()) * features_df["controller_prescribed"].to_numpy()
    rescue_puffs = np.minimum(features_df["rescue_puffs_sum_24h"].to_numpy(), 6.0)
    symptoms = np.maximum(0.0, features_df["symptom_score_mean_6h"].to_numpy() - 1.0)
    rr_dev = np.maximum(0.0, features_df["rr_deviation_from_baseline"].to_numpy())
    spo2_drop = features_df["spo2_drop_from_baseline"].to_numpy()
    hr_elev = np.maximum(0.0, features_df["hr_recent_vs_usual"].to_numpy())

    # Calculate deterministic component of risk logit
    b = coefficients
    logits = (
        b["intercept"]
        + b["beta_latent_vulnerability"] * (latent_vuln - 0.5)
        + b["beta_prior_exacerbations"] * prior_exac
        + b["beta_asthma_severity"] * severity_num
        + b["beta_pm25_exposure"] * pm25_roll * (1.0 + 0.5 * poll_sens)
        + b["beta_missed_controller"] * adh_lapse
        + b["beta_rescue_puffs"] * rescue_puffs
        + b["beta_symptom_score"] * symptoms
        + b["beta_rr_deviation"] * rr_dev
        + b["beta_spo2_drop"] * spo2_drop
        + b["beta_hr_elevation"] * hr_elev
    )

    # Add Gaussian noise
    noise = rng.normal(0.0, b["noise_std"], size=n)
    logits += noise

    # Add unobserved stochastic shock (viral exposure, unmeasured allergen, etc.)
    shock_occurred = rng.random(size=n) < b["shock_probability"]
    shock_values = rng.uniform(b["shock_magnitude_min"], b["shock_magnitude_max"], size=n)
    logits += np.where(shock_occurred, shock_values, 0.0)

    # Convert to probability via logistic function
    probs = sigmoid(logits)

    # Sample binary event outcome (non-deterministic)
    events = (rng.random(size=n) < probs).astype(int)

    return logits, probs, events


def simulate_events(
    features_df: pd.DataFrame,
    latent_truth_df: pd.DataFrame,
    seed: int = 42,
    coefficients: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate simulated exacerbation events and attach event_next_24h label.

    Args:
        features_df: Feature DataFrame.
        latent_truth_df: Ground truth latent vulnerability DataFrame.
        seed: Random seed.
        coefficients: Optional parameter overrides.

    Returns:
        DataFrame with features plus 'event_next_24h' and 'simulated_risk_prob'.
    """
    coeffs = coefficients or SIMULATION_COEFFICIENTS
    logits, probs, events = compute_exacerbation_risk_logits(
        features_df=features_df,
        latent_truth_df=latent_truth_df,
        coefficients=coeffs,
        seed=seed,
    )

    result_df = features_df.copy()
    result_df["event_next_24h"] = events
    result_df["simulated_risk_prob"] = np.round(probs, 4)

    return result_df


def save_event_simulation_config(
    output_path: Union[str, Path] = "data/events_config.json",
    seed: int = 42,
    coefficients: Dict[str, Any] = SIMULATION_COEFFICIENTS,
    event_rate: Optional[float] = None,
    total_samples: Optional[int] = None,
) -> Path:
    """Persist simulation metadata and explicit assumptions to config JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "simulator": "src/simulate_events.py",
        "target_label": "event_next_24h",
        "seed": seed,
        "event_rate": round(float(event_rate), 4) if event_rate is not None else None,
        "total_samples": total_samples,
        "coefficients_assumptions": coefficients,
        "provenance": "simulated",
        "clinical_disclaimer": (
            "Coefficients represent modeling assumptions for evaluating digital twin learning "
            "mechanisms on a simulated process. They are NOT clinical predictions or medical guidance."
        ),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return out_file


def main() -> None:
    """CLI to generate labeled dataset with event_next_24h."""
    parser = argparse.ArgumentParser(description="Simulate 24h exacerbation events for RespiraTwin.")
    parser.add_argument("--features", type=str, default="data/features.csv", help="Features CSV path")
    parser.add_argument("--latent_truth", type=str, default="data/latent_truth.csv", help="Hidden latent truth CSV")
    parser.add_argument("--seed", type=int, default=42, help="Seed for event generation (default: 42)")
    parser.add_argument("--output", type=str, default="data/labeled_dataset.csv", help="Output labeled CSV path")
    parser.add_argument("--config_out", type=str, default="data/events_config.json", help="Config JSON output path")

    args = parser.parse_args()

    feats_path = Path(args.features)
    if not feats_path.is_file():
        print(f"Features file not found at {args.features}. Building features now...")
        features_df = build_features(num_patients=None)
        features_df.to_csv(feats_path, index=False)
    else:
        features_df = pd.read_csv(feats_path)

    latent_truth_df = pd.read_csv(args.latent_truth)

    print(f"Simulating exacerbation events for {len(features_df)} records (seed={args.seed})...")
    labeled_df = simulate_events(
        features_df=features_df,
        latent_truth_df=latent_truth_df,
        seed=args.seed,
    )

    # Save labeled dataset
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(out_file, index=False)

    event_rate = float(labeled_df["event_next_24h"].mean())
    save_event_simulation_config(
        output_path=args.config_out,
        seed=args.seed,
        event_rate=event_rate,
        total_samples=len(labeled_df),
    )

    print(f"Labeled dataset created successfully: {len(labeled_df)} rows.")
    print(f"Target event rate (event_next_24h): {event_rate * 100:.2f}% (Target: 5% - 15%)")
    print(f"Saved labeled data to: {out_file}")
    print(f"Saved simulation config to: {args.config_out}")


if __name__ == "__main__":
    main()
