"""Unit tests for feature engineering pipeline and event simulation.

Includes strict temporal anti-leakage test verifying that features at time t
are invariant to any alterations in future data (t' > t).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.build_features import (
    FEATURE_COLUMNS,
    build_features,
    engineer_patient_features,
    save_feature_columns_json,
)
from src.generate_data import generate_patient_profiles
from src.ingest_pollution import generate_synthetic_pollution
from src.simulate_events import (
    SIMULATION_COEFFICIENTS,
    compute_exacerbation_risk_logits,
    simulate_events,
)
from src.simulate_wearables import simulate_single_patient_timeseries


class TestFeatureEngineering:
    """Tests for past-looking feature engineering and leakage prevention."""

    @pytest.fixture
    def cohort_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate test cohort of 2 patients over 7 days (168 hours)."""
        patients_df, _ = generate_patient_profiles(num_patients=2, seed=123)
        poll_df = generate_synthetic_pollution(hours=168, city="Delhi", seed=123)

        ts_list = []
        for _, pat in patients_df.iterrows():
            pts = simulate_single_patient_timeseries(pat, poll_df, seed=123)
            ts_list.append(pts)
        ts_df = pd.concat(ts_list, ignore_index=True)

        return patients_df, ts_df, poll_df

    def test_feature_columns_schema_and_json(self, tmp_path: Path) -> None:
        """Verify models/feature_columns.json matches FEATURE_COLUMNS exactly."""
        json_path = tmp_path / "feature_columns.json"
        save_feature_columns_json(FEATURE_COLUMNS, json_path)

        assert json_path.is_file()
        with open(json_path, "r", encoding="utf-8") as f:
            loaded_cols = json.load(f)

        assert loaded_cols == FEATURE_COLUMNS
        assert len(loaded_cols) == len(FEATURE_COLUMNS)

    def test_engineered_features_shape_and_no_nans(
        self, cohort_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test feature engineering produces all columns with complete imputation (no NaNs)."""
        patients_df, ts_df, poll_df = cohort_data
        feats = engineer_patient_features(patients_df, ts_df, poll_df)

        assert len(feats) == len(ts_df)
        assert "patient_id" in feats.columns
        assert "timestamp" in feats.columns

        # Verify all FEATURE_COLUMNS are present
        for col in FEATURE_COLUMNS:
            assert col in feats.columns, f"Missing feature column: {col}"
            assert not feats[col].isna().any(), f"NaN values detected in feature: {col}"

    def test_strict_temporal_leakage_proof(
        self, cohort_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    ) -> None:
        """CRITICAL LEAKAGE TEST:
        Prove that feature values at time t do NOT change if future rows (t' > t) are altered.
        """
        patients_df, ts_df, poll_df = cohort_data

        # 1. Compute features on original data
        orig_feats = engineer_patient_features(patients_df, ts_df, poll_df)

        # Focus on patient PT0001 at t_cutoff = 40 (row 40)
        t_cutoff_idx = 40
        pt1_orig = orig_feats[orig_feats["patient_id"] == "PT0001"].iloc[t_cutoff_idx]
        orig_feature_vector = pt1_orig[FEATURE_COLUMNS].to_dict()

        # 2. Drastically corrupt all future rows (t > 40) for PT0001 and pollution
        corrupted_ts = ts_df.copy()
        mask_future = (corrupted_ts["patient_id"] == "PT0001") & (corrupted_ts.index > t_cutoff_idx)
        corrupted_ts.loc[mask_future, "respiratory_rate"] = 48.0  # extreme tachypnea
        corrupted_ts.loc[mask_future, "heart_rate"] = 175.0       # extreme tachycardia
        corrupted_ts.loc[mask_future, "spo2"] = 72.0             # critical hypoxia
        corrupted_ts.loc[mask_future, "rescue_puffs"] = 8        # heavy inhaler use
        corrupted_ts.loc[mask_future, "symptom_score"] = 10      # maximum symptoms

        corrupted_poll = poll_df.copy()
        corrupted_poll.loc[corrupted_poll.index > t_cutoff_idx, "pm25"] = 490.0  # hazardous spike
        corrupted_poll.loc[corrupted_poll.index > t_cutoff_idx, "pm10"] = 750.0

        # 3. Recompute features on corrupted future data
        corrupted_feats = engineer_patient_features(patients_df, corrupted_ts, corrupted_poll)
        pt1_corrupted = corrupted_feats[corrupted_feats["patient_id"] == "PT0001"].iloc[t_cutoff_idx]
        corrupted_feature_vector = pt1_corrupted[FEATURE_COLUMNS].to_dict()

        # 4. Prove that every feature at time t is strictly identical before and after future alteration
        for col in FEATURE_COLUMNS:
            orig_val = orig_feature_vector[col]
            corrupt_val = corrupted_feature_vector[col]
            assert np.isclose(orig_val, corrupt_val, rtol=1e-5, atol=1e-5), (
                f"LEAKAGE DETECTED in feature '{col}' at t={t_cutoff_idx}: "
                f"Original={orig_val}, After future alteration={corrupt_val}"
            )


class TestEventSimulation:
    """Tests for simulated exacerbation risk labels and stochastic properties."""

    @pytest.fixture
    def test_features_and_latent(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate features and latent truth for event testing."""
        patients_df, latent_df = generate_patient_profiles(num_patients=10, seed=42)
        poll_df = generate_synthetic_pollution(hours=240, city="Delhi", seed=42)

        ts_list = []
        for _, pat in patients_df.iterrows():
            pts = simulate_single_patient_timeseries(pat, poll_df, seed=42)
            ts_list.append(pts)
        ts_df = pd.concat(ts_list, ignore_index=True)

        feats = engineer_patient_features(patients_df, ts_df, poll_df)
        return feats, latent_df

    def test_event_rate_in_realistic_demo_range(
        self, test_features_and_latent: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Target event rate must land in the realistic 5% to 15% range."""
        feats, latent_df = test_features_and_latent
        labeled_df = simulate_events(feats, latent_df, seed=42)

        assert "event_next_24h" in labeled_df.columns
        assert "simulated_risk_prob" in labeled_df.columns

        event_rate = float(labeled_df["event_next_24h"].mean())
        assert 0.05 <= event_rate <= 0.15, (
            f"Event rate {event_rate * 100:.2f}% is outside the target 5% - 15% range"
        )

    def test_label_is_non_deterministic(
        self, test_features_and_latent: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Verify label is not a deterministic function (different random seeds yield stochastic differences)."""
        feats, latent_df = test_features_and_latent

        labeled_run1 = simulate_events(feats, latent_df, seed=101)
        labeled_run2 = simulate_events(feats, latent_df, seed=202)

        # Probabilities & events should differ due to Gaussian noise and stochastic shocks
        diff_count = (labeled_run1["event_next_24h"] != labeled_run2["event_next_24h"]).sum()
        assert diff_count > 0, "Labels are completely deterministic! Shock/noise stochasticity failed."

    def test_latent_truth_isolation_in_features(
        self, test_features_and_latent: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Verify latent_vulnerability is NOT present in feature columns."""
        feats, _ = test_features_and_latent
        assert "latent_vulnerability" not in feats.columns
        assert "latent_vulnerability" not in FEATURE_COLUMNS

    def test_no_feature_uses_episode_info(self) -> None:
        """Verify model feature columns contain zero episode ground-truth or future onset leakage."""
        forbidden_substrings = [
            "episode",
            "onset",
            "prodrome",
            "silent",
            "refractory",
            "hazard",
            "duration",
            "event_next_24h",
            "simulated_risk_prob",
        ]
        with open("models/feature_columns.json", "r", encoding="utf-8") as f:
            cols = json.load(f)

        for c in cols:
            for forbidden in forbidden_substrings:
                assert forbidden not in c.lower(), f"Feature '{c}' leaks episode simulator metadata '{forbidden}'!"

    def test_prodrome_signals_rise_before_onset_on_average(self) -> None:
        """Verify prodromal signals (RR, symptoms, rescue puffs) rise and SpO2 drops before onset on average."""
        patients_df, _ = generate_patient_profiles(num_patients=10, seed=42)
        poll_df = generate_synthetic_pollution(hours=720, city="Delhi", seed=42)

        prodrome_rrs, baseline_rrs = [], []
        prodrome_syms, baseline_syms = [], []
        prodrome_puffs, baseline_puffs = [], []
        prodrome_spo2s, baseline_spo2s = [], []

        for _, pat in patients_df.iterrows():
            ts, eps = simulate_single_patient_timeseries(pat, poll_df, seed=42, return_episodes=True)
            for ep in eps:
                if ep["is_silent"]:
                    continue  # Only evaluate non-silent episodes for prodromal drift
                onset = ep["onset_hour"]
                prod_len = ep["prodrome_hours"]
                if onset - prod_len < 24:
                    continue

                prod_window = ts.iloc[onset - prod_len : onset]
                base_window = ts.iloc[max(0, onset - prod_len - 48) : max(1, onset - prod_len - 12)]

                prodrome_rrs.append(prod_window["respiratory_rate"].mean())
                baseline_rrs.append(base_window["respiratory_rate"].mean())

                prodrome_syms.append(prod_window["symptom_score"].mean())
                baseline_syms.append(base_window["symptom_score"].mean())

                prodrome_puffs.append(prod_window["rescue_puffs"].mean())
                baseline_puffs.append(base_window["rescue_puffs"].mean())

                prodrome_spo2s.append(prod_window["spo2"].mean())
                baseline_spo2s.append(base_window["spo2"].mean())

        assert len(prodrome_rrs) > 0, "No non-silent episodes were found for prodromal testing!"
        assert np.nanmean(prodrome_rrs) > np.nanmean(baseline_rrs), "RR did not rise during prodrome on average!"
        assert np.nanmean(prodrome_syms) > np.nanmean(baseline_syms), "Symptom score did not rise during prodrome on average!"
        assert np.nanmean(prodrome_puffs) > np.nanmean(baseline_puffs), "Rescue puffs did not rise during prodrome on average!"
        assert np.nanmean(prodrome_spo2s) < np.nanmean(baseline_spo2s), "SpO2 did not drop during prodrome on average!"
