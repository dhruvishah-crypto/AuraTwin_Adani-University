"""Unit tests for Stage 6b explainability integrity, operational model attribution, and reason code sign agreement."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.build_features import FEATURE_COLUMNS
from src.explain_prediction import explain_prediction


class TestExplainabilityIntegrity:
    """Verify that signal-family contributions and reason codes are derived from the operational model and agree in sign."""

    @pytest.fixture
    def operational_model(self):
        model_path = Path("models/model.joblib")
        assert model_path.is_file(), "Operational model not found."
        return joblib.load(model_path)

    @pytest.fixture
    def sample_features(self):
        data_path = Path("data/sample_labeled_dataset.csv")
        assert data_path.is_file(), "sample_labeled_dataset.csv not found."
        df = pd.read_csv(data_path)
        # Use first nominal (non-exacerbation) row from PT0001
        p1 = df[df["patient_id"] == "PT0001"].sort_values("timestamp")
        row = p1.iloc[10].to_dict()
        return {col: float(row.get(col, 0.0)) for col in FEATURE_COLUMNS}

    @pytest.fixture
    def personal_baseline(self):
        return {
            "usual_respiratory_rate": 16.0,
            "usual_spo2": 96.0,
            "usual_pm25": 35.0,
            "usual_controller_adherence": 1.0,
            "usual_rescue_puffs": 0.2,
            "usual_symptom_score": 1.0,
            "usual_heart_rate": 72.0,
            "usual_hrv": 38.0,
            "baseline_resp_rate": 16.0,
            "baseline_spo2": 96.0,
        }

    def test_explanations_computed_from_operational_model(self, operational_model, sample_features, personal_baseline):
        """Verify explanations are derived from operational HGB via personal-baseline occlusion."""
        exp = explain_prediction(
            model=operational_model,
            feature_vector=sample_features,
            personal_baseline=personal_baseline,
        )

        assert exp["method"] == "operational_model_baseline_occlusion"
        assert "signal_families" in exp
        assert "family_ranking" in exp
        assert "reason_codes" in exp

        # Ensure all 5 families exist
        families = exp["signal_families"]
        expected = {"pollution", "breathing/oxygenation", "adherence", "rescue/symptoms", "history"}
        assert set(families.keys()) == expected

    def test_reason_codes_agree_in_sign_with_pollution_surge(self, operational_model, sample_features, personal_baseline):
        """Verify reason code agrees in positive sign with pollution family contribution during PM2.5 surge."""
        surge_feats = dict(sample_features)
        # Spike PM2.5 to 120 ug/m3
        for col in ["pm25_rolling_mean_6h", "pm25_rolling_max_6h", "pm25_rolling_mean_24h", "pm25_rolling_max_24h", "pm25_rolling_mean_72h", "pm25_rolling_max_72h"]:
            surge_feats[col] = 120.0
        surge_feats["pm10_rolling_mean_24h"] = 220.0

        exp = explain_prediction(
            model=operational_model,
            feature_vector=surge_feats,
            personal_baseline=personal_baseline,
        )

        fam_pollution = exp["signal_families"]["pollution"]
        # Pollution should have positive contribution (increases risk)
        assert fam_pollution["summed_logit_contribution"] > 0, "Pollution logit contribution should be positive"
        assert fam_pollution["risk_delta"] > 0, "Pollution risk delta should be positive"
        assert fam_pollution["direction"] == "increases_risk"

        # Check reason codes include PM2.5 warning
        reasons_text = " ".join(exp["reason_codes"])
        assert "PM2.5" in reasons_text, f"Reason codes must mention PM2.5: {exp['reason_codes']}"

    def test_reason_codes_agree_in_sign_with_tachypnea_and_desaturation(self, operational_model, sample_features, personal_baseline):
        """Verify reason code agrees in positive sign with breathing/oxygenation family during tachypnea + desaturation."""
        attack_feats = dict(sample_features)
        attack_feats["respiratory_rate_raw"] = 24.0
        attack_feats["rr_rolling_mean_6h"] = 23.0
        attack_feats["rr_rolling_mean_24h"] = 20.0
        attack_feats["rr_deviation_from_baseline"] = 8.0
        attack_feats["spo2_raw"] = 89.0
        attack_feats["spo2_mean_6h"] = 90.0
        attack_feats["spo2_min_6h"] = 88.0
        attack_feats["spo2_drop_from_baseline"] = 7.0

        exp = explain_prediction(
            model=operational_model,
            feature_vector=attack_feats,
            personal_baseline=personal_baseline,
        )

        fam_resp = exp["signal_families"]["breathing/oxygenation"]
        assert fam_resp["summed_logit_contribution"] > 0, "Breathing contribution should be positive during attack"
        assert fam_resp["risk_delta"] > 0, "Breathing risk delta should be positive"
        assert fam_resp["direction"] == "increases_risk"

        reasons_text = " ".join(exp["reason_codes"])
        assert any(k in reasons_text for k in ["Tachypnea", "respiratory rate", "Oxygen saturation", "SpO2"]), (
            f"Reason codes must capture respiratory deterioration: {exp['reason_codes']}"
        )

    def test_reason_codes_agree_in_sign_with_rescue_surge(self, operational_model, personal_baseline):
        """Verify reason code agrees in positive sign with rescue/symptoms family during rescue inhaler surge."""
        data_path = Path("data/sample_labeled_dataset.csv")
        df = pd.read_csv(data_path)
        exac_rows = df[df["event_next_24h"] == 1]
        rescue_feats = exac_rows.iloc[0].to_dict()

        exp = explain_prediction(
            model=operational_model,
            feature_vector=rescue_feats,
            personal_baseline=personal_baseline,
        )

        fam_rescue = exp["signal_families"]["rescue/symptoms"]
        assert fam_rescue["summed_logit_contribution"] > 0, "Rescue/symptoms contribution should be positive"
        assert fam_rescue["risk_delta"] > 0, "Rescue risk delta should be positive"
        assert fam_rescue["direction"] == "increases_risk"

        reasons_text = " ".join(exp["reason_codes"])
        assert "Rescue inhaler" in reasons_text or "symptom" in reasons_text.lower() or "Dominant signal family" in reasons_text, (
            f"Reason codes must capture rescue/symptom surge: {exp['reason_codes']}"
        )
