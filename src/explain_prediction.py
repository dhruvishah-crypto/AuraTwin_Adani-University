"""Model interpretability, signal-family grouping, and transparent reason codes for RespiraTwin.

Provides transparent, human-readable explanations of risk predictions:
- Signal-Family Grouping: Primary grouping by clinical signal family
  ('pollution', 'breathing/oxygenation', 'adherence', 'rescue/symptoms', 'history')
  with summed logit contribution per family. Per-feature attributions are secondary.
- Transparent Reason Codes: Human-readable clinical bullet points derived from
  deviations from the patient's personal baseline.
- Model Response to Hypothetical Input: Clear tagging for counterfactual what-if scenarios.
- Optional SHAP integration if installed.
- Mandatory Clinical Disclaimer: States that these are associative model-based signals,
  not causal explanations.

Independent of Streamlit.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MANDATORY_EXPLANATION_DISCLAIMER: str = (
    "CLINICAL EXPLANATION DISCLAIMER: Attributions and reason codes represent associative model-based "
    "signals derived from feature deviations, NOT causal clinical explanations or medical determinations. "
    "All outputs reflect learning on a simulated process and must not be used for diagnosis or treatment."
)

SCENARIO_LABEL: str = "model response to hypothetical input"

# Explicit categorization of the 52 features into 5 clinical signal families
SIGNAL_FAMILIES: Dict[str, Dict[str, Any]] = {
    "pollution": {
        "label": "Pollution Exposure",
        "features": [
            "pm25_rolling_mean_6h",
            "pm25_rolling_max_6h",
            "pm25_rolling_mean_24h",
            "pm25_rolling_max_24h",
            "pm25_rolling_mean_72h",
            "pm25_rolling_max_72h",
            "pm10_rolling_mean_24h",
            "pm10_trend_24h",
        ],
    },
    "breathing/oxygenation": {
        "label": "Breathing & Oxygenation",
        "features": [
            "respiratory_rate_raw",
            "rr_deviation_from_baseline",
            "rr_rolling_mean_6h",
            "rr_rolling_mean_24h",
            "rr_rolling_std_24h",
            "rr_recent_vs_usual",
            "spo2_raw",
            "spo2_mean_6h",
            "spo2_min_6h",
            "spo2_mean_24h",
            "spo2_min_24h",
            "spo2_drop_from_baseline",
            "spo2_recent_vs_usual",
        ],
    },
    "adherence": {
        "label": "Medication Adherence",
        "features": [
            "controller_adherence_7d",
        ],
    },
    "rescue/symptoms": {
        "label": "Rescue Inhaler & Symptoms",
        "features": [
            "rescue_puffs_sum_6h",
            "rescue_puffs_sum_24h",
            "rescue_puffs_trend_24h",
            "symptom_score_raw",
            "symptom_score_mean_6h",
            "symptom_score_mean_24h",
            "symptom_score_trend_24h",
        ],
    },
    "history": {
        "label": "Clinical Profile & History",
        "features": [
            # Static EHR & demographics
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
            # Autonomic vitals, sleep, activity
            "heart_rate_raw",
            "hr_rolling_mean_6h",
            "hr_rolling_mean_24h",
            "hr_trend_24h",
            "hr_recent_vs_usual",
            "hrv_rmssd_raw",
            "hrv_rolling_mean_6h",
            "hrv_rolling_mean_24h",
            "sleep_hours",
            "sleep_deficit",
            "steps_rolling_sum_24h",
        ],
    },
}

# Check for optional SHAP
try:
    import shap  # type: ignore

    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False


def extract_base_model_and_scaler(model: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """Extract classifier and scaler from CalibratedModel or Pipeline if present."""
    base_pipe = getattr(model, "base_pipeline", model)
    scaler = None
    clf = base_pipe

    if hasattr(base_pipe, "named_steps"):
        scaler = base_pipe.named_steps.get("scaler")
        clf = base_pipe.named_steps.get("clf", base_pipe)

    return clf, scaler


def get_personal_baseline_features(
    feature_dict: Dict[str, Any],
    personal_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Construct baseline feature vector using personal baseline norms."""
    pb = personal_baseline or {}
    base_rr = float(pb.get("baseline_resp_rate", feature_dict.get("baseline_resp_rate", 16.0)))
    base_spo2 = float(pb.get("baseline_spo2", feature_dict.get("baseline_spo2", 96.0)))
    usual_pm25 = float(pb.get("usual_pm25", 35.0))
    usual_adh = float(pb.get("usual_controller_adherence", 1.0))
    usual_symptom = float(pb.get("usual_symptom_score", 1.0))
    usual_hr = float(pb.get("usual_heart_rate", 72.0))
    usual_hrv = float(pb.get("usual_hrv", 38.0))

    defaults = {
        # Pollution (Environmental)
        "pm25_rolling_mean_6h": usual_pm25,
        "pm25_rolling_max_6h": usual_pm25,
        "pm25_rolling_mean_24h": usual_pm25,
        "pm25_rolling_max_24h": usual_pm25,
        "pm25_rolling_mean_72h": usual_pm25,
        "pm25_rolling_max_72h": usual_pm25,
        "pm10_rolling_mean_24h": usual_pm25 * 1.85,
        "pm10_trend_24h": 0.0,
        # Breathing & Oxygenation
        "respiratory_rate_raw": base_rr,
        "rr_deviation_from_baseline": 0.0,
        "rr_rolling_mean_6h": base_rr,
        "rr_rolling_mean_24h": base_rr,
        "rr_rolling_std_24h": 0.8,
        "rr_recent_vs_usual": 0.0,
        "spo2_raw": base_spo2,
        "spo2_mean_6h": base_spo2,
        "spo2_min_6h": base_spo2,
        "spo2_mean_24h": base_spo2,
        "spo2_min_24h": base_spo2,
        "spo2_drop_from_baseline": 0.0,
        "spo2_recent_vs_usual": 0.0,
        # Medication Adherence
        "controller_adherence_7d": usual_adh,
        # Rescue & Symptoms
        "rescue_puffs_sum_6h": 0.0,
        "rescue_puffs_sum_24h": 0.0,
        "rescue_puffs_trend_24h": 0.0,
        "symptom_score_raw": usual_symptom,
        "symptom_score_mean_6h": usual_symptom,
        "symptom_score_mean_24h": usual_symptom,
        "symptom_score_trend_24h": 0.0,
        # History & Autonomic Baseline
        "heart_rate_raw": usual_hr,
        "hr_rolling_mean_6h": usual_hr,
        "hr_rolling_mean_24h": usual_hr,
        "hr_trend_24h": 0.0,
        "hr_recent_vs_usual": 0.0,
        "hrv_rmssd_raw": usual_hrv,
        "hrv_rolling_mean_6h": usual_hrv,
        "hrv_rolling_mean_24h": usual_hrv,
        "sleep_hours": 7.5,
        "sleep_deficit": 0.0,
        "steps_rolling_sum_24h": 6500.0,
    }

    base_vec = {k: float(v) for k, v in feature_dict.items() if isinstance(v, (int, float, np.number))}
    for k, v in defaults.items():
        if k in base_vec:
            base_vec[k] = float(v)
    return base_vec


def compute_occlusion_attributions(
    model: Any,
    feature_dict: Dict[str, Any],
    feature_names: List[str],
    personal_baseline: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """Compute local signal-family and feature attributions via personal-baseline occlusion on the operational model.

    Occlusion replaces each family or feature with the patient's personal baseline values,
    measuring the resulting change in calibrated risk probability and logit.
    Vectorized into a single batch prediction call across all 5 families and 52 features.

    Args:
        model: Trained calibrated model or pipeline.
        feature_dict: Current point-in-time feature dictionary.
        feature_names: Ordered list of feature column names.
        personal_baseline: Patient personal rolling baseline dict.

    Returns:
        Tuple of (signal_families_dict, ranked_families_list, per_feature_attributions_dict).
    """
    baseline_vec = get_personal_baseline_features(feature_dict, personal_baseline)
    curr_row = [float(feature_dict.get(c, 0.0)) for c in feature_names]

    rows = [curr_row]  # Index 0 is unoccluded baseline state

    # Rows 1..N_fam: family occlusions
    fam_keys = list(SIGNAL_FAMILIES.keys())
    for fam_key in fam_keys:
        fam_cols = set(SIGNAL_FAMILIES[fam_key]["features"])
        occ_row = [
            float(baseline_vec.get(c, feature_dict.get(c, 0.0))) if c in fam_cols else float(feature_dict.get(c, 0.0))
            for c in feature_names
        ]
        rows.append(occ_row)

    # Rows (1 + N_fam)..: per-feature occlusions
    for col in feature_names:
        f_occ_row = [
            float(baseline_vec.get(c, feature_dict.get(c, 0.0))) if c == col else float(feature_dict.get(c, 0.0))
            for c in feature_names
        ]
        rows.append(f_occ_row)

    # Single batch prediction across all 58 rows for maximum speed
    X_batch = pd.DataFrame(rows, columns=feature_names)
    all_probs = model.predict_proba(X_batch)[:, 1]

    eps = 1e-5
    all_clips = np.clip(all_probs, eps, 1.0 - eps)
    all_logits = np.log(all_clips / (1.0 - all_clips))

    base_prob = float(all_probs[0])
    base_logit = float(all_logits[0])

    family_results: Dict[str, Dict[str, Any]] = {}
    ranked_list: List[Dict[str, Any]] = []

    for i, fam_key in enumerate(fam_keys):
        occ_prob = float(all_probs[1 + i])
        occ_logit = float(all_logits[1 + i])
        prob_delta = base_prob - occ_prob
        logit_delta = base_logit - occ_logit
        direction = "increases_risk" if prob_delta > 0.005 else ("decreases_risk" if prob_delta < -0.005 else "neutral")

        entry = {
            "family": fam_key,
            "label": SIGNAL_FAMILIES[fam_key]["label"],
            "summed_logit_contribution": round(logit_delta, 4),
            "risk_delta": round(prob_delta, 5),
            "direction": direction,
            "base_prob": round(base_prob, 4),
            "occluded_prob": round(occ_prob, 4),
            "feature_count": len([c for c in SIGNAL_FAMILIES[fam_key]["features"] if c in feature_names]),
            "per_feature_detail": {},
        }
        family_results[fam_key] = entry
        ranked_list.append(entry)

    ranked_list.sort(key=lambda x: x["summed_logit_contribution"], reverse=True)

    offset = 1 + len(fam_keys)
    per_feature_attributions: Dict[str, float] = {}
    for j, col in enumerate(feature_names):
        f_occ_logit = float(all_logits[offset + j])
        per_feature_attributions[col] = round(base_logit - f_occ_logit, 4)

    for fam_key, fam_meta in SIGNAL_FAMILIES.items():
        fam_cols = [c for c in fam_meta["features"] if c in feature_names]
        feat_detail = {c: per_feature_attributions.get(c, 0.0) for c in fam_cols}
        family_results[fam_key]["per_feature_detail"] = feat_detail
        if feat_detail:
            family_results[fam_key]["primary_driver"] = max(feat_detail.items(), key=lambda kv: abs(kv[1]))[0]

    return family_results, ranked_list, per_feature_attributions


def generate_transparent_reason_codes(
    features: Dict[str, Any],
    personal_baseline: Optional[Dict[str, Any]] = None,
    family_ranking: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Generate transparent clinical reason codes derived from deviations from personal baseline.

    Args:
        features: Dictionary of engineered features at current observation.
        personal_baseline: Personal rolling baseline for the patient.
        family_ranking: Ranked list of signal family contributions.

    Returns:
        List of formatted human-readable reason code strings.
    """
    pb = personal_baseline or {}
    reasons: List[str] = []

    # 1. PM2.5 environmental surge vs personal baseline
    pm25_24h = float(features.get("pm25_rolling_mean_24h", 0.0))
    usual_pm25 = float(pb.get("usual_pm25", 35.0))
    if usual_pm25 > 0 and pm25_24h > 0:
        ratio = pm25_24h / usual_pm25
        if ratio >= 1.4:
            reasons.append(
                f"Elevated PM2.5 exposure: 24h mean is {ratio:.1f}x this patient's personal baseline "
                f"({pm25_24h:.1f} vs usual {usual_pm25:.1f} µg/m³)"
            )
        elif pm25_24h >= 45.0:
            reasons.append(
                f"High ambient PM2.5: 24h mean of {pm25_24h:.1f} µg/m³ exceeds clean air threshold (35 µg/m³)"
            )

    # 2. Controller medication adherence lapses
    is_controller_prescribed = int(features.get("controller_prescribed", 1)) == 1
    if is_controller_prescribed and "controller_adherence_7d" in features:
        adh = float(features["controller_adherence_7d"])
        adh_pct = adh * 100.0
        if adh_pct < 60.0:
            reasons.append(
                f"Severe controller adherence lapse: {adh_pct:.0f}% over past 7 days (target: ≥ 80%)"
            )
        elif adh_pct < 80.0:
            reasons.append(
                f"Sub-optimal controller adherence: {adh_pct:.0f}% over past 7 days (target: ≥ 80%)"
            )

    # 3. Respiratory rate elevation
    rr_raw = float(features.get("respiratory_rate_raw", 0.0))
    base_rr = float(features.get("baseline_resp_rate", pb.get("baseline_resp_rate", 16.0)))
    rr_dev = float(features.get("rr_deviation_from_baseline", rr_raw - base_rr))
    rr_recent_vs_usual = float(features.get("rr_recent_vs_usual", 0.0))

    if rr_dev >= 2.0:
        reasons.append(
            f"Tachypnea / respiratory rate elevated by +{rr_dev:.1f} bpm above baseline "
            f"({rr_raw:.1f} vs baseline {base_rr:.1f} bpm)"
        )
    elif rr_recent_vs_usual >= 1.5:
        reasons.append(
            f"Acute respiratory rate shift: 6h mean is +{rr_recent_vs_usual:.1f} bpm above 72h usual"
        )

    # 4. Oxygen saturation (SpO2) dip
    spo2_raw = float(features.get("spo2_raw", 98.0))
    base_spo2 = float(features.get("baseline_spo2", pb.get("baseline_spo2", 96.0)))
    spo2_drop = float(features.get("spo2_drop_from_baseline", max(0.0, base_spo2 - spo2_raw)))
    spo2_min_6h = float(features.get("spo2_min_6h", spo2_raw))

    if spo2_drop >= 1.5:
        reasons.append(
            f"Oxygen saturation (SpO2) dropped {spo2_drop:.1f}% below patient baseline "
            f"({spo2_raw:.1f}% vs baseline {base_spo2:.1f}%)"
        )
    elif spo2_min_6h < 92.0:
        reasons.append(
            f"Acute nocturnal/recent desaturation: 6h minimum SpO2 reached {spo2_min_6h:.1f}%"
        )

    # 5. Rescue inhaler usage surge
    rescue_24h = float(features.get("rescue_puffs_sum_24h", 0.0))
    usual_rescue = float(pb.get("usual_rescue_puffs", 0.2))
    if rescue_24h >= 2.0:
        reasons.append(
            f"Rescue inhaler surge: {int(rescue_24h)} puffs in last 24h (patient usual: {usual_rescue:.1f} puffs/day)"
        )
    elif float(features.get("rescue_puffs_trend_24h", 0.0)) > 0.5:
        reasons.append("Accelerating rescue inhaler frequency over past 6 hours")

    # 6. Patient-reported symptoms
    symptom_mean_6h = float(features.get("symptom_score_mean_6h", features.get("symptom_score_raw", 0.0)))
    usual_symptom = float(pb.get("usual_symptom_score", 1.0))
    if symptom_mean_6h >= 3.0:
        reasons.append(
            f"Acute symptom diary elevation: 6h mean is {symptom_mean_6h:.1f} / 10 (usual: {usual_symptom:.1f} / 10)"
        )

    # 7. Autonomic / Heart Rate
    hr_shift = float(features.get("hr_recent_vs_usual", 0.0))
    if hr_shift >= 6.0:
        reasons.append(f"Elevated resting heart rate (+{hr_shift:.1f} bpm above 72h usual)")

    hrv_raw = float(features.get("hrv_rmssd_raw", 35.0))
    if hrv_raw > 0 and hrv_raw < 22.0:
        reasons.append(f"Autonomic vagal withdrawal: HRV RMSSD reduced to {hrv_raw:.1f} ms")

    # 8. Add dominant signal family summary code
    if family_ranking:
        top_fam = family_ranking[0]
        if top_fam["summed_logit_contribution"] > 0.20:
            reasons.append(
                f"Dominant signal family: {top_fam['label']} (summed logit contribution: "
                f"{top_fam['summed_logit_contribution']:+.3f}, primary feature: {top_fam['primary_driver']})"
            )

    if not reasons:
        reasons.append("All key physiological, environmental, and adherence channels remain within normal personal bounds.")

    return reasons


def explain_prediction(
    model: Any,
    feature_vector: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    personal_baseline: Optional[Dict[str, Any]] = None,
    is_scenario: bool = False,
) -> Dict[str, Any]:
    """Compute transparent grouped explanations, reason codes, and model attributions.

    Signal families (pollution, breathing/oxygenation, adherence, rescue/symptoms, history)
    form the primary explanation tier with summed logit contributions; per-feature attributions
    are kept as secondary detail.

    Args:
        model: Trained pipeline or CalibratedModel.
        feature_vector: Dictionary, Series, or single-row DataFrame of features.
        personal_baseline: Personal rolling baseline for the patient.
        is_scenario: True if evaluating a hypothetical counterfactual scenario.

    Returns:
        Dictionary containing grouped signal families, ranked families, reason codes,
        secondary per-feature attributions, and clinical disclaimer.
    """
    if isinstance(feature_vector, pd.DataFrame):
        row_dict = feature_vector.iloc[0].to_dict()
    elif isinstance(feature_vector, pd.Series):
        row_dict = feature_vector.to_dict()
    elif isinstance(feature_vector, dict):
        row_dict = feature_vector.copy()
    else:
        raise ValueError(f"Unsupported feature_vector type: {type(feature_vector)}")

    # Extract ordered feature names
    feature_columns = getattr(model, "feature_cols", None)
    if feature_columns is None:
        base_pipe = getattr(model, "base_pipeline", model)
        feature_columns = getattr(base_pipe, "feature_names_in_", None)

    if feature_columns is None:
        feature_columns = [k for k, v in row_dict.items() if isinstance(v, (int, float, np.number))]
    else:
        feature_columns = list(feature_columns)

    # 1 & 2. Compute primary signal families and per-feature attributions directly from the operational model via baseline occlusion
    signal_families, family_ranking, per_feature_attributions = compute_occlusion_attributions(
        model=model,
        feature_dict=row_dict,
        feature_names=feature_columns,
        personal_baseline=personal_baseline,
    )

    # 3. Generate transparent clinical reason codes
    reason_codes = generate_transparent_reason_codes(
        features=row_dict,
        personal_baseline=personal_baseline,
        family_ranking=family_ranking,
    )

    output_label = SCENARIO_LABEL if is_scenario else "observed"

    # Form feature attributions list for UI convenience
    feature_attributions_list = [
        {"feature": feat, "attribution_delta": delta, "reason_code": f"{feat}: {delta:+.4f}"}
        for feat, delta in sorted(per_feature_attributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    ]

    return {
        "output_label": output_label,
        "signal_families": signal_families,
        "signal_family_contributions": signal_families,
        "family_ranking": family_ranking,
        "reason_codes": reason_codes,
        "per_feature_attributions": per_feature_attributions,
        "feature_attributions": feature_attributions_list,
        "method": "operational_model_baseline_occlusion",
        "disclaimer": MANDATORY_EXPLANATION_DISCLAIMER,
    }
