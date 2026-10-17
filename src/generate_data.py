"""Synthetic patient EHR profile generator for RespiraTwin.

Generates realistic synthetic EHR baseline profiles with clinically plausible
correlations (e.g. asthma severity correlating with prior exacerbations,
vitals, controller medications, and pollution sensitivity).
Saves hidden latent vulnerability separately to ensure zero leakage.

DISCLAIMER:
All generated patient profiles are synthetic and intended strictly for
evaluating simulated digital twin workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEVERITY_LEVELS: List[str] = ["Mild", "Moderate", "Severe"]
SENSITIVITY_LEVELS: List[str] = ["Low", "Medium", "High"]
SEX_CATEGORIES: List[str] = ["M", "F"]

COMORBIDITY_POOL: List[str] = [
    "allergic_rhinitis",
    "GERD",
    "obesity",
    "hypertension",
    "eczema",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_patient_profiles(
    num_patients: int = 300,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic patient EHR profiles with controlled correlations.

    Args:
        num_patients: Number of patient records to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
          - patients_df: Observable EHR profiles (for model features)
          - latent_df: Ground truth latent vulnerability (for simulator only)
    """
    rng = np.random.default_rng(seed)

    # 1. Patient Identifiers
    patient_ids = [f"PT{i + 1:04d}" for i in range(num_patients)]

    # 2. Demographics
    # Age range 6 to 82, mean ~38
    ages = np.clip(np.round(rng.normal(loc=38.0, scale=18.0, size=num_patients)), 6, 82).astype(int)
    sexes = rng.choice(SEX_CATEGORIES, size=num_patients, p=[0.48, 0.52])

    # 3. Asthma Severity (Mild: 45%, Moderate: 35%, Severe: 20%)
    severity_probs = [0.45, 0.35, 0.20]
    severities = rng.choice(SEVERITY_LEVELS, size=num_patients, p=severity_probs)

    # Severity index mapping for correlation calculations (0: Mild, 1: Moderate, 2: Severe)
    severity_idx = np.array([SEVERITY_LEVELS.index(s) for s in severities])

    # 4. Prior Exacerbations in last 12 months (strongly correlated with severity)
    # Mild: Poisson(0.5), Moderate: Poisson(1.8), Severe: Poisson(3.6)
    prior_exacerbations: List[int] = []
    for s_idx in severity_idx:
        if s_idx == 0:
            pe = int(rng.poisson(lam=0.5))
        elif s_idx == 1:
            pe = int(rng.poisson(lam=1.8))
        else:
            pe = int(rng.poisson(lam=3.6))
        prior_exacerbations.append(min(pe, 15))
    prior_exacerbations_arr = np.array(prior_exacerbations, dtype=int)

    # 5. Controller Prescribed (Daily ICS/LABA)
    # Probability increases with severity and prior exacerbations
    controller_prescribed: List[int] = []
    for s_idx, pe in zip(severity_idx, prior_exacerbations_arr):
        base_p = 0.30 if s_idx == 0 else (0.75 if s_idx == 1 else 0.95)
        # Higher prior exacerbations further nudges controller prescription
        p = min(0.99, base_p + 0.05 * min(pe, 3))
        controller_prescribed.append(int(rng.random() < p))
    controller_arr = np.array(controller_prescribed, dtype=int)

    # 6. Baseline Respiratory Rate (breaths/min)
    # Normal resting: 12-20, higher severity tends toward higher baseline
    resp_rates: List[float] = []
    for s_idx in severity_idx:
        loc = 15.0 if s_idx == 0 else (17.5 if s_idx == 1 else 20.5)
        rr = rng.normal(loc=loc, scale=1.8)
        resp_rates.append(round(float(np.clip(rr, 10.0, 32.0)), 1))
    resp_rates_arr = np.array(resp_rates, dtype=float)

    # 7. Baseline SpO2 (%)
    # Normal: 95-99%, severe or older patients experience slightly lower baselines
    spo2_vals: List[float] = []
    for s_idx, age in zip(severity_idx, ages):
        loc = 98.2 if s_idx == 0 else (96.5 if s_idx == 1 else 94.5)
        if age > 65:
            loc -= 0.5
        val = rng.normal(loc=loc, scale=1.0)
        spo2_vals.append(round(float(np.clip(val, 88.0, 100.0)), 1))
    spo2_arr = np.array(spo2_vals, dtype=float)

    # 8. Comorbidities
    comorbidities_list: List[str] = []
    for s_idx, age in zip(severity_idx, ages):
        selected: List[str] = []
        # Allergic rhinitis common across asthma phenotypes
        if rng.random() < (0.55 if s_idx == 0 else 0.45):
            selected.append("allergic_rhinitis")
        # GERD more prevalent in moderate/severe
        if rng.random() < (0.15 + 0.15 * s_idx):
            selected.append("GERD")
        # Obesity & Hypertension correlated with age
        if age > 40 and rng.random() < 0.25:
            selected.append("obesity")
        if age > 55 and rng.random() < 0.30:
            selected.append("hypertension")
        if rng.random() < 0.12:
            selected.append("eczema")

        comorbidities_list.append(",".join(selected) if selected else "none")

    # 9. Pollution Sensitivity (Low / Medium / High)
    # Correlated with severity and allergic rhinitis
    pollution_sensitivities: List[str] = []
    for s_idx, cm in zip(severity_idx, comorbidities_list):
        has_rhinitis = "allergic_rhinitis" in cm
        if s_idx == 0:
            probs = [0.45, 0.40, 0.15] if has_rhinitis else [0.60, 0.30, 0.10]
        elif s_idx == 1:
            probs = [0.20, 0.55, 0.25] if has_rhinitis else [0.30, 0.50, 0.20]
        else:
            probs = [0.08, 0.32, 0.60] if has_rhinitis else [0.15, 0.45, 0.40]

        # Normalize probs in case of floating precision
        p_arr = np.array(probs) / sum(probs)
        pollution_sensitivities.append(str(rng.choice(SENSITIVITY_LEVELS, p=p_arr)))

    # Assemble Observable EHR DataFrame (NO latent variables)
    patients_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "age": ages,
            "sex": sexes,
            "asthma_severity": severities,
            "prior_exacerbations_12m": prior_exacerbations_arr,
            "controller_prescribed": controller_arr,
            "baseline_resp_rate": resp_rates_arr,
            "baseline_spo2": spo2_arr,
            "comorbidities": comorbidities_list,
            "pollution_sensitivity": pollution_sensitivities,
        }
    )

    # 10. Hidden Latent Vulnerability [0.0, 1.0]
    # Ground truth biological factor for the event simulator only.
    # NEVER exposed to model training or feature pipelines.
    # Combines severity, prior exacerbations, vitals deficit, and intrinsic noise.
    sens_numeric = np.array([SENSITIVITY_LEVELS.index(ps) for ps in pollution_sensitivities])
    raw_score = (
        0.8 * severity_idx
        + 0.25 * prior_exacerbations_arr
        + 0.15 * (resp_rates_arr - 15.0)
        + 0.35 * (98.0 - spo2_arr)
        + 0.30 * sens_numeric
        - 0.40 * controller_arr
        + rng.normal(loc=0.0, scale=0.6, size=num_patients)
    )
    # Scale via sigmoid to strictly (0, 1) and round
    latent_vulnerability = np.round(sigmoid((raw_score - 1.0) / 1.5), 4)

    latent_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "latent_vulnerability": latent_vulnerability,
        }
    )

    return patients_df, latent_df


def save_synthetic_data(
    patients_df: pd.DataFrame,
    latent_df: pd.DataFrame,
    output_dir: str | Path = "data",
    seed: int = 42,
) -> Dict[str, str]:
    """Persist generated data and configuration JSON sidecar.

    Args:
        patients_df: Observable synthetic EHR dataframe.
        latent_df: Hidden latent vulnerability dataframe.
        output_dir: Output directory to write files.
        seed: Random seed used in generation.

    Returns:
        Dictionary of created file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patients_csv = out_path / "patients.csv"
    sample_patients_csv = out_path / "sample_patients.csv"
    latent_csv = out_path / "latent_truth.csv"
    config_json = out_path / "patients_config.json"

    # Save observable patients
    patients_df.to_csv(patients_csv, index=False)
    patients_df.to_csv(sample_patients_csv, index=False)

    # Save hidden latent truth (MUST be isolated)
    latent_df.to_csv(latent_csv, index=False)

    # Save reproducibility metadata JSON sidecar
    config_data = {
        "generator": "generate_data.py",
        "num_patients": len(patients_df),
        "seed": seed,
        "provenance": {
            "patients_csv": "simulated",
            "latent_truth_csv": "simulated (ground truth for event simulator only, not model feature)",
        },
        "schema": {
            "observable_columns": list(patients_df.columns),
            "latent_columns": list(latent_df.columns),
        },
    }
    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return {
        "patients_csv": str(patients_csv),
        "sample_patients_csv": str(sample_patients_csv),
        "latent_csv": str(latent_csv),
        "config_json": str(config_json),
    }


def main() -> None:
    """CLI entrypoint for data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic EHR patient data for RespiraTwin.")
    parser.add_argument("--num_patients", type=int, default=300, help="Number of synthetic patients (default: 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory path (default: data)")

    args = parser.parse_args()

    patients_df, latent_df = generate_patient_profiles(
        num_patients=args.num_patients,
        seed=args.seed,
    )
    saved_files = save_synthetic_data(
        patients_df=patients_df,
        latent_df=latent_df,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print(f"Generated {len(patients_df)} synthetic patients successfully.")
    for key, path in saved_files.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
