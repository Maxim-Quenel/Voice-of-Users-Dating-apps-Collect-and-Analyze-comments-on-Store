from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

import pandas as pd

from clustering_service import read_csv_safely

MAX_SAMPLE_ROWS = int(os.getenv("VIZ_MAX_ROWS", "5000"))


def _infer_column_types(sample: pd.DataFrame) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for col in sample.columns:
        series = sample[col]
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "date"
            continue
        name_lower = col.lower()
        if "date" in name_lower or "time" in name_lower:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.6:
                types[col] = "date"
                continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() >= 0.8:
            types[col] = "numeric"
        else:
            types[col] = "text"
    return types


def build_chart_payload_from_csv(csv_path: Path, sample_percent: float = 20.0) -> Dict[str, Any]:
    logs = []
    df = read_csv_safely(csv_path, logs)
    if df.empty:
        raise ValueError("Le fichier CSV est vide.")

    if not (0 < sample_percent <= 100):
        raise ValueError("Le pourcentage doit etre entre 1 et 100.")
    requested_size = max(1, int(round(len(df) * (sample_percent / 100.0))))
    sample_size = min(len(df), requested_size, MAX_SAMPLE_ROWS)
    sample_df = df.sample(n=sample_size, random_state=42, replace=False).copy()

    for col in ("text", "reply_content"):
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].astype(str).str.slice(0, 160)

    sample_df = sample_df.where(pd.notna(sample_df), None)
    column_types = _infer_column_types(sample_df)
    effective_percent = round((sample_size / len(df)) * 100, 2) if len(df) else 0.0
    return {
        "columns": sample_df.columns.tolist(),
        "chart_sample": sample_df.to_dict(orient="records"),
        "column_types": column_types,
        "chart_sample_size": sample_size,
        "requested_percent": sample_percent,
        "effective_percent": effective_percent,
        "total_rows": len(df),
        "logs": logs,
    }
