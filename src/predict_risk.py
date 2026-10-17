"""Risk prediction pipeline for RespiraTwin.

Given an observed feature vector, computes:
- Calibrated exacerbation-risk probability (next 24 hours)
- 0 to 100 integer Risk Score
- Demonstration Risk Band (Low / Watch / High) based on validation-tuned thresholds

CRITICAL NOTICE:
Risk bands are demonstration categories for research evaluation on a simulated process,
NOT validated medical cutoffs. Never use for clinical decision-making.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.train_model import CalibratedModel

MANDATORY_DISCLAIMER: str = (
    "CLINICAL ACCURACY DISCLAIMER: Results measure learning of a simulated process, "
    "not real-world prediction. Risk bands are demo categories, not clinical cutoffs."
)


class RiskPredictor:
    """Production predictor wrapper for RespiraTwin exacerbation risk estimation."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/model.joblib",
        feature_columns_path: Union[str, Path] = "models/feature_columns.json",
        metrics_path: Union[str, Path] = "models/metrics.json",
    ):
        self.model_path = Path(model_path)
        self.feature_columns_path = Path(feature_columns_path)
        self.metrics_path = Path(metrics_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}. Train the model first.")

        self.model = joblib.load(self.model_path)

        with open(self.feature_columns_path, "r", encoding="utf-8") as f:
            self.feature_columns: List[str] = json.load(f)

        # Load validation thresholds
        if self.metrics_path.is_file():
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                thresholds = metrics_data.get("risk_thresholds", {})
                self.threshold_low = thresholds.get("low_upper_threshold", 0.15)
                self.threshold_high = thresholds.get("high_lower_threshold", 0.55)
        else:
            self.threshold_low = 0.15
            self.threshold_high = 0.55

    def format_input(self, features: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Format input into DataFrame with exact feature order."""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features.to_dict()])
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError(f"Unsupported features input type: {type(features)}")

        # Check for missing expected columns
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input features missing required columns ({len(missing)} missing): {missing[:5]}")

        # Reorder to match exact training column schema
        return df[self.feature_columns]

    def predict_risk(
        self,
        features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Predict 24-hour exacerbation risk for a single patient record or observation.

        Args:
            features: Dictionary or DataFrame with feature columns.

        Returns:
            Dictionary with probability, risk_score (0-100), risk_band, and disclaimers.
        """
        X = self.format_input(features)
        prob = float(self.model.predict_proba(X)[0, 1])
        score = int(round(np.clip(prob * 100, 0, 100)))

        # Assign demo risk band
        if prob < self.threshold_low:
            band = "Low"
            desc = "Routine observation (Demo category, not medical cutoff)"
        elif prob < self.threshold_high:
            band = "Watch"
            desc = "Elevated trend / close monitoring (Demo category, not medical cutoff)"
        else:
            band = "High"
            desc = "High-risk alert (Demo category, not medical cutoff)"

        return {
            "probability": round(prob, 4),
            "risk_score_100": score,
            "risk_band": band,
            "band_description": desc,
            "thresholds_used": {
                "low_threshold": self.threshold_low,
                "high_threshold": self.threshold_high,
            },
            "disclaimer": MANDATORY_DISCLAIMER,
        }


def predict_risk(
    features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_path: Union[str, Path] = "models/model.joblib",
) -> Dict[str, Any]:
    """Convenience functional interface for risk prediction."""
    predictor = RiskPredictor(model_path=model_path)
    return predictor.predict_risk(features)


def main() -> None:
    """CLI to test prediction on a sample patient record."""
    parser = argparse.ArgumentParser(description="Predict 24h exacerbation risk.")
    parser.add_argument("--sample_data", type=str, default="data/sample_labeled_dataset.csv", help="Sample CSV")
    parser.add_argument("--row_index", type=int, default=0, help="Row index to evaluate (default: 0)")

    args = parser.parse_args()

    df = pd.read_csv(args.sample_data)
    row = df.iloc[args.row_index]

    result = predict_risk(row)
    print("\n--- RespiraTwin Risk Prediction Output ---")
    print(f"Patient ID:      {row.get('patient_id', 'Unknown')}")
    print(f"Timestamp:       {row.get('timestamp', 'Unknown')}")
    print(f"Probability:     {result['probability']:.4f}")
    print(f"Risk Score:      {result['risk_score_100']} / 100")
    print(f"Risk Band:       {result['risk_band']} ({result['band_description']})")
    print(f"Disclaimer:      {result['disclaimer']}")


if __name__ == "__main__":
    main()
