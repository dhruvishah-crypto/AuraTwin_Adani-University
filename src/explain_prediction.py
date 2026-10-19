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


def compute_feature_attributions(
    model: Any,
    feature_names: List[str],
    feature_values: np.ndarray,
) -> Dict[str, float]:
    """Compute local feature attributions using standardized model coefficients or importances."""
    clf, scaler = extract_base_model_and_scaler(model)
    x = np.asarray(feature_values, dtype=float).reshape(1, -1)

    # 1. If SHAP is installed, attempt SHAP explanation
    if HAS_SHAP and clf is not None:
        try:
            x_scaled = x
            if scaler is not None and hasattr(scaler, "transform"):
                x_scaled = scaler.transform(x)
            explainer = shap.Explainer(clf, x_scaled)
            shap_values = explainer(x_scaled)
            if hasattr(shap_values, "values"):
                vals = shap_values.values[0]
                if vals.ndim > 1:
                    vals = vals[:, 1]
                return {name: round(float(v), 4) for name, v in zip(feature_names, vals)}
        except Exception as e:
            logger.debug("SHAP explanation fallback triggered: %s", e)

    # 2. Standardized linear model contributions: w_i * (x_i - mean_i) / std_i
    if clf is not None and hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        if scaler is not None and hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            means = scaler.mean_
            scales = scaler.scale_
            scales_safe = np.where(scales == 0, 1.0, scales)
            z = (x[0] - means) / scales_safe
            contributions = coefs * z
        else:
            contributions = coefs * x[0]

        return {name: round(float(c), 4) for name, c in zip(feature_names, contributions)}

    # 3. Tree-based model fallback (feature_importances_)
    if clf is not None and hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}

    # 4. HistGradientBoostingClassifier fallback (accumulate split gains across boosting trees)
    if clf is not None and hasattr(clf, "_predictors"):
        n_feat = len(feature_names)
        importances = np.zeros(n_feat)
        for predictor_list in clf._predictors:
            for tree in predictor_list:
                nodes = tree.nodes
                gain_mask = nodes["gain"] > 0
                np.add.at(importances, nodes["feature_idx"][gain_mask], nodes["gain"][gain_mask])
        tot = np.sum(importances)
        if tot > 0:
            importances /= tot
        return {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}

    # Fallback to zero attributions
    return {name: 0.0 for name in feature_names}


def group_attributions_by_family(
    attributions: Dict[str, float],
    feature_values: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Group feature attributions by signal family and calculate summed logit contributions.

    Args:
        attributions: Mapping of feature_name to attribution score (logit contribution).
        feature_values: Mapping of feature_name to raw feature values.

    Returns:
        Tuple of (signal_families_dict, ranked_families_list).
    """
    family_results: Dict[str, Dict[str, Any]] = {}
    ranked_list: List[Dict[str, Any]] = []

    for fam_key, fam_meta in SIGNAL_FAMILIES.items():
        fam_label = fam_meta["label"]
        fam_cols = fam_meta["features"]

        per_feature: Dict[str, float] = {}
        summed_logit = 0.0

        for col in fam_cols:
            attr = float(attributions.get(col, 0.0))
            per_feature[col] = attr
            summed_logit += attr

        # Identify primary driver feature within the family
        if per_feature:
            primary_feat = max(per_feature.items(), key=lambda kv: abs(kv[1]))[0]
        else:
            primary_feat = fam_cols[0] if fam_cols else "unknown"

        direction = "increases_risk" if summed_logit > 0.005 else ("decreases_risk" if summed_logit < -0.005 else "neutral")

        entry = {
            "family": fam_key,
            "label": fam_label,
            "summed_logit_contribution": round(summed_logit, 4),
            "direction": direction,
            "primary_driver": primary_feat,
            "feature_count": len(fam_cols),
            "per_feature_detail": per_feature,
        }
        family_results[fam_key] = entry
        ranked_list.append(entry)

    # Rank families primarily by summed logit contribution (highest risk driver first)
    ranked_list.sort(key=lambda x: x["summed_logit_contribution"], reverse=True)

    return family_results, ranked_list


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

    feature_values = np.array([float(row_dict.get(c, 0.0)) for c in feature_columns])

    # 1. Compute secondary per-feature attributions
    per_feature_attributions = compute_feature_attributions(model, feature_columns, feature_values)

    # 2. Compute primary signal families with summed logit contributions
    signal_families, family_ranking = group_attributions_by_family(per_feature_attributions, row_dict)

    # 3. Generate transparent clinical reason codes
    reason_codes = generate_transparent_reason_codes(
        features=row_dict,
        personal_baseline=personal_baseline,
        family_ranking=family_ranking,
    )

    output_label = SCENARIO_LABEL if is_scenario else "observed"

    return {
        "output_label": output_label,
        "signal_families": signal_families,
        "family_ranking": family_ranking,
        "reason_codes": reason_codes,
        "per_feature_attributions": per_feature_attributions,
        "method": "shap" if HAS_SHAP else "model_coefficients",
        "disclaimer": MANDATORY_EXPLANATION_DISCLAIMER,
    }
