"""Pollution data ingestion stub for RespiraTwin."""
from typing import Optional
import pandas as pd


def ingest_pollution(source_path: Optional[str] = None) -> pd.DataFrame:
    """Ingest air quality observations from sample CSV or fallback.
    
    Returns:
        pd.DataFrame containing timestamp, pm25, pm10, and optionally aqi.
    """
    raise NotImplementedError("Scheduled for implementation in later stages.")
