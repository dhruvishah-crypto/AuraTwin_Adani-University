"""Unit tests for risk prediction pipeline, model loading, and thresholding."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.predict_risk import RiskPredictor, predict_risk
from src.train_model import stratified_patient_split


class TestRiskPredictions:
    """Tests for trained model loading, feature invariance, and prediction output ranges."""

    @pytest.fixture
    def sample_record(self) -> pd.Series:
        """Load a sample patient feature record."""
        sample_path = Path("data/sample_labeled_dataset.csv")
        if sample_path.is_file():
            df = pd.read_csv(sample_path)
            return df.iloc[0]
        # Fallback to feature_columns with dummy values
        with open("models/feature_columns.json", "r", encoding="utf-8") as f:
            cols = json.load(f)
        dummy_data = {c: 1.0 for c in cols}
        dummy_data["patient_id"] = "PT0001"
        return pd.Series(dummy_data)

    def test_model_loads_cleanly_from_disk(self) -> None:
        """Verify model.joblib and feature_columns.json exist and load properly."""
        predictor = RiskPredictor()
        assert predictor.model is not None
        assert len(predictor.feature_columns) > 0

    def test_prediction_output_ranges_and_keys(self, sample_record: pd.Series) -> None:
        """Verify probability is in [0, 1], risk_score in [0, 100], and valid demo band."""
        result = predict_risk(sample_record)

        assert "probability" in result
        assert "risk_score_100" in result
        assert "risk_band" in result
        assert "disclaimer" in result

        prob = result["probability"]
        score = result["risk_score_100"]
        band = result["risk_band"]

        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1] bounds"
        assert 0 <= score <= 100, f"Score {score} out of [0, 100] bounds"
        assert band in {"Low", "Watch", "High"}, f"Unexpected band {band}"
        assert "demo" in result["band_description"].lower()

    def test_prediction_works_with_dict_and_series(self, sample_record: pd.Series) -> None:
        """Verify prediction handles both dictionary and pandas Series inputs identically."""
        rec_dict = sample_record.to_dict()

        res_series = predict_risk(sample_record)
        res_dict = predict_risk(rec_dict)

        assert res_series["probability"] == res_dict["probability"]
        assert res_series["risk_score_100"] == res_dict["risk_score_100"]
        assert res_series["risk_band"] == res_dict["risk_band"]

    def test_missing_features_raises_error(self, sample_record: pd.Series) -> None:
        """Verify missing feature columns raises a clear error."""
        bad_dict = sample_record.to_dict()
        # Drop key feature
        del bad_dict["pm25_rolling_mean_24h"]

        with pytest.raises(ValueError, match="missing required columns"):
            predict_risk(bad_dict)

    def test_patient_split_leakage_assertion(self) -> None:
        """Verify stratified_patient_split guarantees zero train/test patient overlap."""
        dummy_df = pd.DataFrame(
            {
                "patient_id": [f"PT{i:03d}" for i in range(20)] * 10,
                "event_next_24h": np.random.choice([0, 1], size=200, p=[0.9, 0.1]),
            }
        )
        train_df, val_df, test_df, summary = stratified_patient_split(dummy_df, seed=42)

        train_pids = set(train_df["patient_id"])
        test_pids = set(test_df["patient_id"])
        val_pids = set(val_df["patient_id"])

        # No overlap between any set
        assert len(train_pids.intersection(test_pids)) == 0
        assert len(train_pids.intersection(val_pids)) == 0
        assert len(val_pids.intersection(test_pids)) == 0

    def test_demo_bands_event_rates_strictly_monotonic(self) -> None:
        """Verify observed test event rates per demo band strictly satisfy Low < Watch < High."""
        metrics_file = Path("models/metrics.json")
        assert metrics_file.is_file(), "models/metrics.json must exist"
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        risk_thresh = metrics_data["risk_thresholds"]
        observed = risk_thresh["observed_test_event_rates"]

        low_rate = observed["Low"]
        watch_rate = observed["Watch"]
        high_rate = observed["High"]

        # Strict monotonicity assertion required by Stage 4b
        assert low_rate < watch_rate < high_rate, (
            f"Observed test rates must satisfy Low < Watch < High: "
            f"Low={low_rate:.4f}, Watch={watch_rate:.4f}, High={high_rate:.4f}"
        )
        assert risk_thresh.get("monotonicity_verified", False) is True

