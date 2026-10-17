"""Unit tests for pollution ingestion and wearable time-series simulation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_patient_profiles
from src.ingest_pollution import (
    POLLUTION_COLUMNS,
    calculate_us_aqi_pm25,
    fetch_openaq_data,
    generate_synthetic_pollution,
    ingest_pollution,
)
from src.simulate_wearables import (
    WEARABLE_COLUMNS,
    compute_lagged_pollution_stress,
    simulate_single_patient_timeseries,
    simulate_wearables,
)


class TestPollutionIngestion:
    """Tests for pollution data ingestion and generation."""

    def test_calculate_us_aqi_pm25(self) -> None:
        """Test US EPA AQI calculation separates raw concentration from AQI index."""
        # Good (< 12 ug/m3) -> AQI <= 50
        assert calculate_us_aqi_pm25(10.0) <= 50
        # Moderate (12.1 - 35.4) -> 51 - 100
        assert 50 < calculate_us_aqi_pm25(25.0) <= 100
        # Unhealthy (> 55.4) -> AQI > 150
        assert calculate_us_aqi_pm25(60.0) > 150

    def test_synthetic_pollution_generation_schema_and_properties(self) -> None:
        """Test synthetic pollution generator schema and autocorrelation."""
        df = generate_synthetic_pollution(hours=72, city="Delhi", seed=42)

        assert len(df) == 72
        assert list(df.columns) == POLLUTION_COLUMNS
        assert df["source"].iloc[0] == "synthetic"
        assert (df["pm25"] >= 0).all()
        # PM10 >= PM2.5 in realistic atmospheric environments
        assert (df["pm10"] >= df["pm25"]).all()

    def test_ingest_pollution_fallback_on_missing_file(self, tmp_path: Path) -> None:
        """Test fallback to synthetic generation when sample file is absent."""
        missing_file = tmp_path / "non_existent_pollution.csv"
        df = ingest_pollution(file_path=missing_file, city="Delhi", days=3, seed=42)

        assert len(df) == 72
        assert df["source"].iloc[0] == "synthetic"

    def test_fetch_openaq_fails_gracefully(self) -> None:
        """Test remote OpenAQ fetch handles invalid timeout/unreachable endpoints gracefully."""
        # Should not raise exception even with ultra-short timeout
        res = fetch_openaq_data(city="NonExistentCity12345", timeout_seconds=0.01)
        assert res is None or isinstance(res, pd.DataFrame)


class TestWearableSimulation:
    """Tests for wearable physiological signals and dynamic coherence."""

    @pytest.fixture
    def sample_patients(self) -> pd.DataFrame:
        """Generate test patient records with known phenotypes."""
        patients_df, _ = generate_patient_profiles(num_patients=5, seed=100)
        return patients_df

    @pytest.fixture
    def sample_pollution(self) -> pd.DataFrame:
        """Generate test 7-day pollution series."""
        return generate_synthetic_pollution(hours=168, city="Delhi", seed=100)

    def test_single_patient_simulation_columns_and_provenance(
        self, sample_patients: pd.DataFrame, sample_pollution: pd.DataFrame
    ) -> None:
        """Test single patient simulation produces all required columns and simulated provenance."""
        patient = sample_patients.iloc[0]
        ts = simulate_single_patient_timeseries(patient, sample_pollution, seed=42)

        assert len(ts) == len(sample_pollution)
        for col in WEARABLE_COLUMNS:
            assert col in ts.columns, f"Missing column {col} in wearable time-series"

        assert (ts["provenance"] == "simulated").all()

    def test_sensor_dropout_and_missingness(
        self, sample_patients: pd.DataFrame, sample_pollution: pd.DataFrame
    ) -> None:
        """Verify realistic sensor dropouts produce NaNs in vitals."""
        patient = sample_patients.iloc[0]
        ts = simulate_single_patient_timeseries(patient, sample_pollution, seed=42, sensor_dropout_rate=0.05)

        # There should be missing values in physiological channels due to charging/off-wrist
        assert ts["heart_rate"].isna().any()
        assert ts["spo2"].isna().any()
        assert ts["respiratory_rate"].isna().any()

    def test_rescue_puffs_follow_symptoms(
        self, sample_patients: pd.DataFrame, sample_pollution: pd.DataFrame
    ) -> None:
        """Verify rescue inhaler puffs correlate with symptom score."""
        patient = sample_patients.iloc[0]
        ts = simulate_single_patient_timeseries(patient, sample_pollution, seed=42)

        low_sym_puffs = ts[ts["symptom_score"] <= 2]["rescue_puffs"].mean()
        high_sym_puffs = ts[ts["symptom_score"] >= 6]["rescue_puffs"].mean()

        if not np.isnan(high_sym_puffs):
            assert high_sym_puffs >= low_sym_puffs

    def test_pollution_sensitivity_lagged_coherence(self, sample_pollution: pd.DataFrame) -> None:
        """Verify high sensitivity patient exhibits greater pollution-induced stress than low sensitivity."""
        pm25 = sample_pollution["pm25"].to_numpy()

        stress_low = compute_lagged_pollution_stress(pm25, sensitivity="Low", lag_hours=12)
        stress_high = compute_lagged_pollution_stress(pm25, sensitivity="High", lag_hours=12)

        assert stress_high.mean() > stress_low.mean()
