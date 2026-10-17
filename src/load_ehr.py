"""EHR data loader and validator for RespiraTwin.

Provides robust validation of synthetic EHR profiles, ensuring proper
schema structure, allowed categorical levels, physiological ranges,
and absence of latent truth columns (preventing data leakage).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Union

import pandas as pd

# Expected columns for patient EHR profiles
EXPECTED_COLUMNS: List[str] = [
    "patient_id",
    "age",
    "sex",
    "asthma_severity",
    "prior_exacerbations_12m",
    "controller_prescribed",
    "baseline_resp_rate",
    "baseline_spo2",
    "comorbidities",
    "pollution_sensitivity",
]

# Forbidden columns that must NEVER appear in patient features (leakage prevention)
FORBIDDEN_COLUMNS: Set[str] = {
    "latent_vulnerability",
    "simulated_vulnerability",
    "ground_truth_risk",
}

# Allowed categorical values
ALLOWED_SEX: Set[str] = {"M", "F"}
ALLOWED_SEVERITY: Set[str] = {"Mild", "Moderate", "Severe"}
ALLOWED_SENSITIVITY: Set[str] = {"Low", "Medium", "High"}

# Physiological / Demographic validation ranges
RANGE_CHECKS: Dict[str, Tuple[float, float]] = {
    "age": (1.0, 120.0),
    "prior_exacerbations_12m": (0.0, 50.0),
    "baseline_resp_rate": (8.0, 50.0),
    "baseline_spo2": (70.0, 100.0),
}


class EHRValidationError(ValueError):
    """Exception raised when EHR data fails schema or range validation."""
    pass


class EHRDataLeakageError(ValueError):
    """Exception raised when forbidden latent ground-truth variables are detected."""
    pass


def validate_schema(df: pd.DataFrame) -> None:
    """Validate presence of required columns and absence of forbidden columns.

    Args:
        df: DataFrame to validate.

    Raises:
        EHRValidationError: If required columns are missing.
        EHRDataLeakageError: If forbidden latent ground-truth columns are present.
    """
    # Check for data leakage first
    detected_forbidden = [col for col in df.columns if col in FORBIDDEN_COLUMNS]
    if detected_forbidden:
        raise EHRDataLeakageError(
            f"Data leakage detected! Forbidden latent column(s) found in EHR dataset: {detected_forbidden}. "
            "Latent vulnerability must remain isolated in data/latent_truth.csv and never exposed to model features."
        )

    # Check for missing required columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise EHRValidationError(f"Missing required EHR column(s): {missing_cols}")


def validate_ranges_and_types(df: pd.DataFrame) -> None:
    """Validate value ranges, types, and uniqueness.

    Args:
        df: DataFrame to validate.

    Raises:
        EHRValidationError: If null values, duplicates, invalid categories, or out-of-range values exist.
    """
    if df.empty:
        raise EHRValidationError("EHR dataset is empty (0 rows).")

    # Check for missing / null values
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise EHRValidationError(f"Null/NaN values detected in columns: {cols_with_nulls.to_dict()}")

    # Check unique patient_ids
    if df["patient_id"].duplicated().any():
        duplicate_ids = df[df["patient_id"].duplicated()]["patient_id"].unique().tolist()
        raise EHRValidationError(f"Duplicate patient_id values found: {duplicate_ids[:5]}")

    # Check categoricals
    invalid_sex = set(df["sex"].dropna()) - ALLOWED_SEX
    if invalid_sex:
        raise EHRValidationError(f"Invalid values in 'sex': {invalid_sex}. Allowed: {ALLOWED_SEX}")

    invalid_severity = set(df["asthma_severity"].dropna()) - ALLOWED_SEVERITY
    if invalid_severity:
        raise EHRValidationError(
            f"Invalid values in 'asthma_severity': {invalid_severity}. Allowed: {ALLOWED_SEVERITY}"
        )

    invalid_sens = set(df["pollution_sensitivity"].dropna()) - ALLOWED_SENSITIVITY
    if invalid_sens:
        raise EHRValidationError(
            f"Invalid values in 'pollution_sensitivity': {invalid_sens}. Allowed: {ALLOWED_SENSITIVITY}"
        )

    # Controller prescribed check (must be binary 0 or 1)
    if not set(df["controller_prescribed"].unique()).issubset({0, 1, True, False}):
        raise EHRValidationError(
            f"Invalid values in 'controller_prescribed': {set(df['controller_prescribed'].unique())}. Must be 0 or 1."
        )

    # Range checks
    for col, (min_val, max_val) in RANGE_CHECKS.items():
        # Ensure column is numeric
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                numeric_series = pd.to_numeric(df[col])
            except Exception as e:
                raise EHRValidationError(f"Column '{col}' must be numeric: {e}") from e
        else:
            numeric_series = df[col]

        out_of_bounds = df[(numeric_series < min_val) | (numeric_series > max_val)]
        if not out_of_bounds.empty:
            sample_bad = out_of_bounds[["patient_id", col]].head(3).to_dict(orient="records")
            raise EHRValidationError(
                f"Column '{col}' contains values outside valid clinical range [{min_val}, {max_val}]. "
                f"Offending samples: {sample_bad}"
            )


def load_ehr_data(file_path: Union[str, Path] = "data/patients.csv") -> pd.DataFrame:
    """Load and validate synthetic patient EHR records from CSV.

    Args:
        file_path: Relative or absolute path to the patients CSV file.

    Returns:
        Validated pandas DataFrame with standardized types.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        EHRValidationError: If schema, types, or ranges are invalid.
        EHRDataLeakageError: If forbidden latent truth columns are included.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"EHR data file not found at: {file_path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise EHRValidationError(f"Failed to read CSV at {file_path}: {e}") from e

    # Perform validation
    validate_schema(df)
    validate_ranges_and_types(df)

    # Standardize data types
    df["patient_id"] = df["patient_id"].astype(str)
    df["age"] = df["age"].astype(int)
    df["prior_exacerbations_12m"] = df["prior_exacerbations_12m"].astype(int)
    df["controller_prescribed"] = df["controller_prescribed"].astype(int)
    df["baseline_resp_rate"] = df["baseline_resp_rate"].astype(float)
    df["baseline_spo2"] = df["baseline_spo2"].astype(float)
    df["comorbidities"] = df["comorbidities"].astype(str)

    return df
