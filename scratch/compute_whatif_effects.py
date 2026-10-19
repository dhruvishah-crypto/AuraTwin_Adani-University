"""Compute and save what-if effect sizes across test set risk bands and pre-onset hours.

Outputs saved to models/whatif_effects.json.
"""

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

from src.train_model import stratified_patient_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    data_path = WORKSPACE_ROOT / "data" / "labeled_dataset.csv"
    feature_cols_path = WORKSPACE_ROOT / "models" / "feature_columns.json"
    metrics_path = WORKSPACE_ROOT / "models" / "metrics.json"
    model_path = WORKSPACE_ROOT / "models" / "model.joblib"
    out_path = WORKSPACE_ROOT / "models" / "whatif_effects.json"

    logger.info("Loading model and metadata...")
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols: list[str] = json.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        thresholds = metrics_data.get("risk_thresholds", {})
        low_thresh = float(thresholds.get("low_upper_threshold", 0.0679))
        high_thresh = float(thresholds.get("high_lower_threshold", 0.1604))

    model = joblib.load(model_path)

    logger.info(f"Loading {data_path} and splitting into test set...")
    df = pd.read_csv(data_path)
    _, _, test_df, _ = stratified_patient_split(df, seed=42)
    test_df = test_df.reset_index(drop=True)
    num_test = len(test_df)
    logger.info(f"Test split rows: {num_test}")

    # Compute baseline predictions on test set
    X_test = test_df[feature_cols].copy()
    base_probs = model.predict_proba(X_test)[:, 1]
    test_df["base_prob"] = base_probs

    # Identify cohorts/subsets
    masks = {
        "low_band": test_df["base_prob"] < low_thresh,
        "watch_band": (test_df["base_prob"] >= low_thresh) & (test_df["base_prob"] < high_thresh),
        "high_band": test_df["base_prob"] >= high_thresh,
        "pre_onset_1_to_24h": test_df["event_next_24h"] == 1,
    }

    cohort_counts = {k: int(m.sum()) for k, m in masks.items()}
    logger.info(f"Cohort counts: {cohort_counts}")

    # Define the 5 sliders and their transformation across range
    # 1. PM2.5: 10 -> 150 ug/m3
    def apply_pm25(X: pd.DataFrame, val: float) -> pd.DataFrame:
        X_mod = X.copy()
        for col in [
            "pm25_rolling_mean_6h",
            "pm25_rolling_max_6h",
            "pm25_rolling_mean_24h",
            "pm25_rolling_max_24h",
            "pm25_rolling_mean_72h",
            "pm25_rolling_max_72h",
        ]:
            if col in X_mod.columns:
                X_mod[col] = val
        return X_mod

    # 2. Adherence: 10% -> 100% (0.1 -> 1.0)
    def apply_adherence(X: pd.DataFrame, val: float) -> pd.DataFrame:
        X_mod = X.copy()
        if "controller_adherence_7d" in X_mod.columns:
            X_mod["controller_adherence_7d"] = val
        return X_mod

    # 3. RR Shift: -3 -> +5 bpm
    def apply_rr_shift(X: pd.DataFrame, shift: float) -> pd.DataFrame:
        X_mod = X.copy()
        base_rr = X_mod["baseline_resp_rate"] if "baseline_resp_rate" in X_mod.columns else 16.0
        if "respiratory_rate_raw" in X_mod.columns:
            X_mod["respiratory_rate_raw"] = X["respiratory_rate_raw"] + shift
        if "rr_rolling_mean_6h" in X_mod.columns:
            X_mod["rr_rolling_mean_6h"] = X["rr_rolling_mean_6h"] + shift
        if "rr_rolling_mean_24h" in X_mod.columns:
            X_mod["rr_rolling_mean_24h"] = X["rr_rolling_mean_24h"] + (shift * 0.25)
        if "rr_deviation_from_baseline" in X_mod.columns:
            X_mod["rr_deviation_from_baseline"] = X_mod["respiratory_rate_raw"] - base_rr
        if "rr_recent_vs_usual" in X_mod.columns:
            X_mod["rr_recent_vs_usual"] = X["rr_recent_vs_usual"] + shift
        return X_mod

    # 4. Rescue Puffs: 0 -> 8 puffs
    def apply_rescue_puffs(X: pd.DataFrame, puffs: float) -> pd.DataFrame:
        X_mod = X.copy()
        if "rescue_puffs_sum_24h" in X_mod.columns:
            X_mod["rescue_puffs_sum_24h"] = puffs
        if "rescue_puffs_sum_6h" in X_mod.columns:
            X_mod["rescue_puffs_sum_6h"] = min(puffs, puffs / 2.0)
        return X_mod

    # 5. SpO2: 98% -> 88% (acute desaturation)
    def apply_spo2(X: pd.DataFrame, spo2_val: float) -> pd.DataFrame:
        X_mod = X.copy()
        base_spo2 = X_mod["baseline_spo2"] if "baseline_spo2" in X_mod.columns else 96.0
        if "spo2_raw" in X_mod.columns:
            X_mod["spo2_raw"] = spo2_val
        if "spo2_mean_6h" in X_mod.columns:
            X_mod["spo2_mean_6h"] = spo2_val
        if "spo2_min_6h" in X_mod.columns:
            X_mod["spo2_min_6h"] = np.minimum(spo2_val, X["spo2_min_6h"])
        if "spo2_drop_from_baseline" in X_mod.columns:
            X_mod["spo2_drop_from_baseline"] = np.maximum(0.0, base_spo2 - spo2_val)
        if "spo2_recent_vs_usual" in X_mod.columns:
            X_mod["spo2_recent_vs_usual"] = spo2_val - base_spo2
        return X_mod

    sliders_spec = {
        "pm25": {
            "name": "PM2.5 Exposure",
            "unit": "µg/m³",
            "min_val": 10.0,
            "max_val": 150.0,
            "fn": apply_pm25,
            "direction": "increase",
        },
        "adherence": {
            "name": "Controller Adherence",
            "unit": "%",
            "min_val": 0.10,
            "max_val": 1.00,
            "fn": apply_adherence,
            "direction": "increase",  # from 10% to 100% adherence reduces risk
        },
        "respiratory_rate_shift": {
            "name": "Respiratory Rate Shift",
            "unit": "bpm",
            "min_val": -3.0,
            "max_val": 5.0,
            "fn": apply_rr_shift,
            "direction": "increase",
        },
        "rescue_puffs": {
            "name": "SABA Rescue Puffs (24h)",
            "unit": "puffs",
            "min_val": 0.0,
            "max_val": 8.0,
            "fn": apply_rescue_puffs,
            "direction": "increase",
        },
        "spo2": {
            "name": "Oxygen Saturation (SpO2)",
            "unit": "%",
            "min_val": 98.0,
            "max_val": 88.0,  # desaturation from 98 down to 88
            "fn": apply_spo2,
            "direction": "desaturation",
        },
    }

    results: dict[str, dict[str, Any]] = {
        "meta": {
            "test_rows": num_test,
            "risk_thresholds": {
                "low_upper_threshold": low_thresh,
                "high_lower_threshold": high_thresh,
            },
            "cohort_hours": cohort_counts,
        },
        "by_cohort": {},
        "summary_table": {},
    }

    for cohort_key, mask in masks.items():
        X_sub = X_test[mask].copy()
        cohort_n = len(X_sub)
        results["by_cohort"][cohort_key] = {
            "num_hours": cohort_n,
            "fraction_of_test": round(cohort_n / num_test, 4),
            "effects": {},
        }

        logger.info(f"\nComputing effects for cohort: {cohort_key} (N={cohort_n})...")
        for s_key, s_spec in sliders_spec.items():
            fn = s_spec["fn"]
            X_low = fn(X_sub, s_spec["min_val"])
            X_high = fn(X_sub, s_spec["max_val"])

            p_low = model.predict_proba(X_low)[:, 1]
            p_high = model.predict_proba(X_high)[:, 1]
            deltas = p_high - p_low

            mean_delta = float(np.mean(deltas))
            median_delta = float(np.median(deltas))
            abs_mean = float(np.mean(np.abs(deltas)))
            abs_median = float(np.median(np.abs(deltas)))
            p25 = float(np.percentile(deltas, 25))
            p75 = float(np.percentile(deltas, 75))

            is_small = bool(abs_median < 0.005)

            effect_record = {
                "slider_key": s_key,
                "name": s_spec["name"],
                "unit": s_spec["unit"],
                "range": [s_spec["min_val"], s_spec["max_val"]],
                "mean_risk_change": round(mean_delta, 5),
                "median_risk_change": round(median_delta, 5),
                "abs_mean_risk_change": round(abs_mean, 5),
                "abs_median_risk_change": round(abs_median, 5),
                "p25_risk_change": round(p25, 5),
                "p75_risk_change": round(p75, 5),
                "typical_effect": round(median_delta, 5),
                "typical_effect_abs": round(abs_median, 5),
                "typical_effect_pct": round(abs_median * 100.0, 2),
                "is_small_effect": is_small,
            }
            results["by_cohort"][cohort_key]["effects"][s_key] = effect_record
            logger.info(
                f"  [{s_key:<22}]: Median Delta={median_delta:+.4f} (Mean={mean_delta:+.4f}, AbsMedian={abs_median:.4f})"
                + (" [SMALL EFFECT < 0.005]" if is_small else "")
            )

    # Compile summary table for easy lookup in app.py
    for s_key in sliders_spec:
        high_abs_median = results["by_cohort"]["high_band"]["effects"][s_key]["abs_median_risk_change"]
        high_is_small = results["by_cohort"]["high_band"]["effects"][s_key]["is_small_effect"]

        results["summary_table"][s_key] = {
            "name": sliders_spec[s_key]["name"],
            "unit": sliders_spec[s_key]["unit"],
            "low_band_typical_effect": results["by_cohort"]["low_band"]["effects"][s_key]["median_risk_change"],
            "watch_band_typical_effect": results["by_cohort"]["watch_band"]["effects"][s_key]["median_risk_change"],
            "high_band_typical_effect": results["by_cohort"]["high_band"]["effects"][s_key]["median_risk_change"],
            "pre_onset_typical_effect": results["by_cohort"]["pre_onset_1_to_24h"]["effects"][s_key]["median_risk_change"],
            "high_band_is_small": high_is_small,
            "display_label_high_band": (
                "small effect in this model (<0.005)"
                if high_is_small
                else f"typical effect: {results['by_cohort']['high_band']['effects'][s_key]['typical_effect_pct']}%"
            ),
        }

    # Save to models/whatif_effects.json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved effect sizes to {out_path}")
    print("\n================== WHAT-IF EFFECT SIZES SUMMARY TABLE ==================")
    summary_df = pd.DataFrame(
        [
            {
                "Slider": s_spec["name"],
                "Range": f"{s_spec['min_val']} -> {s_spec['max_val']} {s_spec['unit']}",
                "Low Band": f"{results['by_cohort']['low_band']['effects'][k]['median_risk_change']:+.4f}",
                "Watch Band": f"{results['by_cohort']['watch_band']['effects'][k]['median_risk_change']:+.4f}",
                "High Band": f"{results['by_cohort']['high_band']['effects'][k]['median_risk_change']:+.4f}",
                "Pre-Onset 1-24h": f"{results['by_cohort']['pre_onset_1_to_24h']['effects'][k]['median_risk_change']:+.4f}",
                "High Band Note": "small effect in this model" if results["summary_table"][k]["high_band_is_small"] else "active effect",
            }
            for k, s_spec in sliders_spec.items()
        ]
    )
    print(summary_df.to_string(index=False))
    print("========================================================================\n")


if __name__ == "__main__":
    main()
