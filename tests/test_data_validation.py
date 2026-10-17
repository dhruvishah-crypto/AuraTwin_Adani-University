"""Unit tests for synthetic patient EHR data generation and validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.generate_data import generate_patient_profiles, save_synthetic_data
from src.load_ehr import (
    EXPECTED_COLUMNS,
    FORBIDDEN_COLUMNS,
    EHRDataLeakageError,
    EHRValidationError,
    load_ehr_data,
    validate_ranges_and_types,
    validate_schema,
)


class TestDataGeneration:
    """Tests for synthetic patient EHR generator."""

    def test_default_patient_generation_shape_and_columns(self) -> None:
        """Test default generation of 300 patients with exact expected columns."""
        patients_df, latent_df = generate_patient_profiles(num_patients=300, seed=42)

        assert len(patients_df) == 300
        assert len(latent_df) == 300

        # Check all required columns are present
        for col in EXPECTED_COLUMNS:
            assert col in patients_df.columns, f"Missing column {col} in generated patients_df"

        # Check latent columns
        assert list(latent_df.columns) == ["patient_id", "latent_vulnerability"]

    def test_latent_truth_isolation_zero_leakage(self) -> None:
        """Verify latent_vulnerability is strictly isolated and never in patients_df."""
        patients_df, latent_df = generate_patient_profiles(num_patients=50, seed=123)

        assert "latent_vulnerability" not in patients_df.columns
        assert not any(col in FORBIDDEN_COLUMNS for col in patients_df.columns)

        # Ensure latent vulnerability values are bounded within (0, 1)
        assert (latent_df["latent_vulnerability"] >= 0.0).all()
        assert (latent_df["latent_vulnerability"] <= 1.0).all()

    def test_clinical_correlations(self) -> None:
        """Verify controlled clinical correlations (e.g., severity vs exacerbations & medications)."""
        patients_df, _ = generate_patient_profiles(num_patients=600, seed=999)

        mild_exac = patients_df[patients_df["asthma_severity"] == "Mild"]["prior_exacerbations_12m"].mean()
        mod_exac = patients_df[patients_df["asthma_severity"] == "Moderate"]["prior_exacerbations_12m"].mean()
        sev_exac = patients_df[patients_df["asthma_severity"] == "Severe"]["prior_exacerbations_12m"].mean()

        # Severe > Moderate > Mild prior exacerbations
        assert sev_exac > mod_exac > mild_exac

        # Controller prescription probability should be higher in Severe than Mild
        mild_ctrl = patients_df[patients_df["asthma_severity"] == "Mild"]["controller_prescribed"].mean()
        sev_ctrl = patients_df[patients_df["asthma_severity"] == "Severe"]["controller_prescribed"].mean()
        assert sev_ctrl > mild_ctrl

        # Resting SpO2 should trend lower for Severe than Mild
        mild_spo2 = patients_df[patients_df["asthma_severity"] == "Mild"]["baseline_spo2"].mean()
        sev_spo2 = patients_df[patients_df["asthma_severity"] == "Severe"]["baseline_spo2"].mean()
        assert mild_spo2 > sev_spo2

    def test_reproducibility_with_seed(self) -> None:
        """Test identical outputs are produced given the same seed."""
        df1, lat1 = generate_patient_profiles(num_patients=100, seed=77)
        df2, lat2 = generate_patient_profiles(num_patients=100, seed=77)
        pd.testing.assert_frame_equal(df1, df2)
        pd.testing.assert_frame_equal(lat1, lat2)

        # Different seeds produce different data
        df3, _ = generate_patient_profiles(num_patients=100, seed=88)
        assert not df1.equals(df3)

    def test_persistence_and_config_sidecar(self, tmp_path: Path) -> None:
        """Test file saving and metadata config JSON creation."""
        patients_df, latent_df = generate_patient_profiles(num_patients=50, seed=42)
        saved = save_synthetic_data(patients_df, latent_df, output_dir=tmp_path, seed=42)

        assert Path(saved["patients_csv"]).is_file()
        assert Path(saved["sample_patients_csv"]).is_file()
        assert Path(saved["latent_csv"]).is_file()
        assert Path(saved["config_json"]).is_file()

        # Validate config contents
        with open(saved["config_json"], "r", encoding="utf-8") as f:
            cfg = json.load(f)
            assert cfg["seed"] == 42
            assert cfg["num_patients"] == 50
            assert "simulated" in cfg["provenance"]["patients_csv"]


class TestEHRValidation:
    """Tests for EHR loading, schema verification, and range validation."""

    def test_successful_load(self, tmp_path: Path) -> None:
        """Test loading a properly formatted synthetic EHR CSV."""
        patients_df, latent_df = generate_patient_profiles(num_patients=100, seed=42)
        file_path = tmp_path / "patients.csv"
        patients_df.to_csv(file_path, index=False)

        loaded_df = load_ehr_data(file_path)
        assert len(loaded_df) == 100
        assert list(loaded_df.columns) == EXPECTED_COLUMNS
        assert loaded_df["age"].dtype == np.int32 or loaded_df["age"].dtype == np.int64

    def test_missing_file_raises_error(self) -> None:
        """Test FileNotFoundError on non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_ehr_data("non_existent_directory/missing_patients.csv")

    def test_data_leakage_detection(self) -> None:
        """Test that attempting to load latent_vulnerability raises EHRDataLeakageError."""
        bad_df = pd.DataFrame(
            {
                "patient_id": ["PT0001"],
                "age": [30],
                "sex": ["M"],
                "asthma_severity": ["Mild"],
                "prior_exacerbations_12m": [0],
                "controller_prescribed": [0],
                "baseline_resp_rate": [16.0],
                "baseline_spo2": [98.0],
                "comorbidities": ["none"],
                "pollution_sensitivity": ["Low"],
                "latent_vulnerability": [0.42],  # FORBIDDEN!
            }
        )
        with pytest.raises(EHRDataLeakageError, match="Data leakage detected"):
            validate_schema(bad_df)

    def test_missing_column_raises_validation_error(self) -> None:
        """Test missing required column triggers EHRValidationError."""
        bad_df = pd.DataFrame(
            {
                "patient_id": ["PT0001"],
                "age": [30],
                # sex is missing
                "asthma_severity": ["Mild"],
            }
        )
        with pytest.raises(EHRValidationError, match="Missing required EHR column"):
            validate_schema(bad_df)

    def test_duplicate_patient_id_raises_error(self) -> None:
        """Test duplicate patient_id triggers error."""
        patients_df, _ = generate_patient_profiles(num_patients=10, seed=42)
        # Duplicate first row id
        patients_df.loc[1, "patient_id"] = patients_df.loc[0, "patient_id"]

        with pytest.raises(EHRValidationError, match="Duplicate patient_id"):
            validate_ranges_and_types(patients_df)

    def test_invalid_categorical_raises_error(self) -> None:
        """Test invalid categorical values trigger error."""
        patients_df, _ = generate_patient_profiles(num_patients=10, seed=42)
        patients_df.loc[0, "asthma_severity"] = "Catastrophic"

        with pytest.raises(EHRValidationError, match="Invalid values in 'asthma_severity'"):
            validate_ranges_and_types(patients_df)

    def test_out_of_range_spo2_raises_error(self) -> None:
        """Test out-of-range SpO2 triggers error."""
        patients_df, _ = generate_patient_profiles(num_patients=10, seed=42)
        patients_df.loc[0, "baseline_spo2"] = 105.0  # Invalid SpO2

        with pytest.raises(EHRValidationError, match="outside valid clinical range"):
            validate_ranges_and_types(patients_df)
