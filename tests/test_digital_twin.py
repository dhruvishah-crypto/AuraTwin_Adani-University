"""Unit tests for Stage 5b Digital Twin hardening, feature parity, coherent what-if, monotonicity, and stale data."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import engineer_patient_features
from src.digital_twin import DigitalTwin
from src.explain_prediction import SCENARIO_LABEL, explain_prediction


class TestDigitalTwinHardening:
    """Test suite verifying feature parity, coherent overrides, monotonicity, and data quality."""

    @pytest.fixture
    def twin(self) -> DigitalTwin:
        """Instantiate a DigitalTwin instance for PT0001."""
        return DigitalTwin(patient_id="PT0001")

    def test_feature_vector_equals_build_features_output_for_5_hours(self, twin: DigitalTwin) -> None:
        """Verify twin's feature vector exactly matches build_features.py output for 5 random patient-hours."""
        patients = pd.read_csv("data/sample_patients.csv")
        wearables = pd.read_csv("data/sample_wearables.csv")
        pollution = pd.read_csv("data/sample_pollution.csv")

        p1 = patients.iloc[[0]]
        pid = str(p1["patient_id"].iloc[0])
        w1 = wearables[wearables["patient_id"] == pid].copy()

        # Generate golden feature matrix via build_features.py
        full_feats = engineer_patient_features(p1, w1, pollution)

        with open("models/feature_columns.json", "r", encoding="utf-8") as f:
            expected_cols: list[str] = json.load(f)

        # 5 distinct representative observation hours across the time series
        test_hours = [48, 120, 240, 480, 750]

        for idx in test_hours:
            expected_row = full_feats.iloc[idx][expected_cols].to_dict()

            # Pass historical slice up to that exact hour to twin
            sub_w = w1.iloc[: idx + 1].copy()
            twin.observations[pid] = sub_w
            twin_res = twin.update_twin(pid)
            twin_features = twin_res["current_state"]["features"]

            # 1. Assert exact column order match
            assert list(twin_features.keys()) == expected_cols, f"Column order mismatch at hour {idx}"

            # 2. Assert exact numerical equality across all 52 features
            for col in expected_cols:
                exp_v = float(expected_row[col])
                twin_v = float(twin_features[col])
                assert abs(exp_v - twin_v) < 1e-4, (
                    f"Feature mismatch for '{col}' at hour index {idx}: expected {exp_v}, got {twin_v}"
                )

    def test_personal_baseline_excludes_recent_72h(self, twin: DigitalTwin) -> None:
        """Verify personal baseline is computed over 14-30d window excluding most recent 72 hours."""
        twin.update_twin("PT0001")
        baseline = twin.current_state["PT0001"]["personal_baseline"]

        assert baseline["excluded_recent_hours"] == 72
        assert baseline["baseline_window_hours"] > 0
        assert "usual_respiratory_rate" in baseline
        assert "usual_pm25" in baseline
        assert "usual_controller_adherence" in baseline

    def test_what_if_coherent_overrides(self, twin: DigitalTwin) -> None:
        """Verify what_if overrides update all dependent features consistently."""
        twin.update_twin("PT0001")
        base_features = twin.current_state["PT0001"]["features"]
        base_rr = float(base_features.get("baseline_resp_rate", 16.0))
        orig_rr = float(base_features["respiratory_rate_raw"])
        orig_rr_dev = float(base_features["rr_deviation_from_baseline"])
        orig_rr_trend = float(base_features["rr_recent_vs_usual"])

        # 1. Coherent PM2.5 override: updates all rolling means/maxes
        pm_val = 115.0
        pm_res = twin.what_if("PT0001", {"pm25_level": pm_val})
        pm_feats = pm_res["scenario_state"]["features"]

        assert pm_feats["pm25_rolling_mean_6h"] == pm_val
        assert pm_feats["pm25_rolling_max_6h"] == pm_val
        assert pm_feats["pm25_rolling_mean_24h"] == pm_val
        assert pm_feats["pm25_rolling_max_24h"] == pm_val
        assert pm_feats["pm25_rolling_mean_72h"] == pm_val
        assert pm_feats["pm25_rolling_max_72h"] == pm_val

        # 2. Coherent adherence override: updates 7d adherence
        adh_res = twin.what_if("PT0001", {"controller_adherence_pct": 85.0})
        assert abs(adh_res["scenario_state"]["features"]["controller_adherence_7d"] - 0.85) < 1e-4

        # 3. Coherent RR shift: shifts all RR-derived features consistently
        shift = 3.5
        rr_res = twin.what_if("PT0001", {"respiratory_rate_shift": shift})
        rr_feats = rr_res["scenario_state"]["features"]

        assert abs(rr_feats["respiratory_rate_raw"] - (orig_rr + shift)) < 1e-4
        assert abs(rr_feats["rr_rolling_mean_6h"] - (float(base_features["rr_rolling_mean_6h"]) + shift)) < 1e-4
        assert abs(rr_feats["rr_rolling_mean_24h"] - (float(base_features["rr_rolling_mean_24h"]) + (shift * 0.25))) < 1e-4
        assert abs(rr_feats["rr_deviation_from_baseline"] - (orig_rr_dev + shift)) < 1e-4
        assert abs(rr_feats["rr_recent_vs_usual"] - (orig_rr_trend + shift)) < 1e-4

    def test_what_if_monotonicity(self, twin: DigitalTwin) -> None:
        """Verify monotonicity: higher PM2.5 -> risk not lower; higher adherence -> risk not higher; larger RR shift -> risk not lower; sweeps for rescue puffs and SpO2; and check >= 0.02 responsiveness across PM2.5 10 to 150."""
        twin.update_twin("PT0001")

        # 1. Monotonicity for PM2.5: risk not lower as PM2.5 increases
        pm_levels = [10.0, 30.0, 60.0, 100.0, 150.0]
        pm_risks = [twin.what_if("PT0001", {"pm25_level": val})["scenario_risk"]["probability"] for val in pm_levels]
        for i in range(len(pm_risks) - 1):
            assert pm_risks[i + 1] >= pm_risks[i], f"PM2.5 monotonicity violated: {pm_risks}"

        # Plausible range responsiveness check: risk increases by at least 0.02 across PM2.5 10 to 150
        pm_delta = pm_risks[-1] - pm_risks[0]
        assert pm_delta >= 0.02, (
            f"PM2.5 what-if responsiveness too low: delta={pm_delta:.4f} < 0.02 across {pm_levels} (risks={pm_risks})"
        )

        # 2. Monotonicity for Adherence: risk not higher as adherence increases
        adh_levels = [10.0, 30.0, 60.0, 80.0, 100.0]
        adh_risks = [
            twin.what_if("PT0001", {"controller_adherence_pct": val})["scenario_risk"]["probability"]
            for val in adh_levels
        ]
        for i in range(len(adh_risks) - 1):
            assert adh_risks[i + 1] <= adh_risks[i], f"Adherence monotonicity violated: {adh_risks}"

        # 3. Monotonicity for Respiratory Rate shift: risk not lower as RR shifts upward
        rr_shifts = [-3.0, -1.0, 0.0, 2.0, 5.0]
        rr_risks = [
            twin.what_if("PT0001", {"respiratory_rate_shift": val})["scenario_risk"]["probability"]
            for val in rr_shifts
        ]
        for i in range(len(rr_risks) - 1):
            assert rr_risks[i + 1] >= rr_risks[i], f"RR shift monotonicity violated: {rr_risks}"

        # 4. Monotonicity for Rescue Inhaler Puffs: risk not lower as rescue puffs increase
        puffs_levels = [0.0, 1.0, 2.0, 4.0, 8.0]
        puffs_risks = [
            twin.what_if("PT0001", {"rescue_puffs": val})["scenario_risk"]["probability"]
            for val in puffs_levels
        ]
        for i in range(len(puffs_risks) - 1):
            assert puffs_risks[i + 1] >= puffs_risks[i], f"Rescue puffs monotonicity violated: {puffs_risks}"

        # 5. Monotonicity for SpO2: risk not lower as SpO2 drops (decreasing SpO2 -> non-decreasing risk)
        spo2_levels = [98.0, 96.0, 94.0, 91.0, 88.0]
        spo2_risks = [
            twin.what_if("PT0001", {"spo2": val})["scenario_risk"]["probability"]
            for val in spo2_levels
        ]
        for i in range(len(spo2_risks) - 1):
            assert spo2_risks[i + 1] >= spo2_risks[i], f"SpO2 drop monotonicity violated: {spo2_risks}"

    def test_stale_data_quality_flags(self, twin: DigitalTwin) -> None:
        """Verify channels forward-filled for > 6 hours are flagged in current_state['data_quality']."""
        twin.update_twin("PT0001")

        # Ingest 8 consecutive hours where SpO2 and HRV are missing/unobserved
        last_t = twin.observations["PT0001"]["timestamp"].max()
        stale_records = []
        for h in range(1, 9):
            stale_records.append(
                {
                    "patient_id": "PT0001",
                    "timestamp": last_t + pd.Timedelta(hours=h),
                    "respiratory_rate": 18.0,
                    "heart_rate": 74.0,
                    "spo2": np.nan,  # Missing / forward-filled
                    "hrv_rmssd_ms": None,  # Missing / forward-filled
                    "sleep_hours": 7.0,
                    "steps": 100,
                    "controller_taken": 1,
                    "rescue_puffs": 0,
                    "symptom_score": 1,
                }
            )

        res = twin.update_twin("PT0001", stale_records)
        dq = res["current_state"]["data_quality"]

        assert dq["has_stale_data"] is True
        assert "spo2" in dq["stale_channels"]
        assert dq["stale_channels"]["spo2"]["hours_stale"] == 8
        assert "hrv_rmssd_ms" in dq["stale_channels"]
        assert len(dq["flags"]) >= 2

        # Ingest fresh observation restoring SpO2
        fresh_obs = {
            "patient_id": "PT0001",
            "timestamp": last_t + pd.Timedelta(hours=9),
            "respiratory_rate": 18.0,
            "heart_rate": 74.0,
            "spo2": 97.0,  # Fresh value restored
            "hrv_rmssd_ms": 36.0,  # Fresh value restored
        }
        res_fresh = twin.update_twin("PT0001", fresh_obs)
        dq_fresh = res_fresh["current_state"]["data_quality"]

        # Staleness for restored channels is cleared
        assert "spo2" not in dq_fresh["stale_channels"]
        assert "hrv_rmssd_ms" not in dq_fresh["stale_channels"]
        assert dq_fresh["has_stale_data"] is False

    def test_grouped_explanation_by_signal_family(self, twin: DigitalTwin) -> None:
        """Verify explanations group by signal families with summed logit contributions as primary tier."""
        twin.update_twin("PT0001")
        features = twin.current_state["PT0001"]["features"]
        baseline = twin.current_state["PT0001"]["personal_baseline"]

        exp = explain_prediction(
            model=twin.predictor.model,
            feature_vector=features,
            personal_baseline=baseline,
        )

        assert "signal_families" in exp
        assert "family_ranking" in exp
        assert "reason_codes" in exp
        assert "per_feature_attributions" in exp

        families = exp["signal_families"]
        expected_families = {"pollution", "breathing/oxygenation", "adherence", "rescue/symptoms", "history"}
        assert set(families.keys()) == expected_families

        # Check family structure
        for fam_name in expected_families:
            fam_data = families[fam_name]
            assert "label" in fam_data
            assert "summed_logit_contribution" in fam_data
            assert "direction" in fam_data
            assert "primary_driver" in fam_data
            assert "per_feature_detail" in fam_data

        # Check ranking is sorted descending by summed contribution
        ranking = exp["family_ranking"]
        assert len(ranking) == 5
        for i in range(len(ranking) - 1):
            assert ranking[i]["summed_logit_contribution"] >= ranking[i + 1]["summed_logit_contribution"]

    def test_scenario_outputs_labeled_model_response_to_hypothetical_input(self, twin: DigitalTwin) -> None:
        """Verify scenario outputs are tagged 'model response to hypothetical input' in code and strings."""
        twin.update_twin("PT0001")
        res = twin.what_if("PT0001", {"pm25_level": 25.0})

        assert res["source"] == SCENARIO_LABEL
        assert res["scenario_label"] == SCENARIO_LABEL
        assert res["scenario_risk"]["source"] == SCENARIO_LABEL
        assert res["scenario_state"]["source"] == SCENARIO_LABEL
        assert "hypothetical input" in res["disclaimer"].lower()
