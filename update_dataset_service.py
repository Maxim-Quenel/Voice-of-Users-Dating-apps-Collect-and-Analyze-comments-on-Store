from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import update_dataset as updater


_BASE_DIR = Path(__file__).resolve().parent
_RAW_OUTPUT = Path(updater.OUTPUT_FILE)
_OUTPUT_PATH = _RAW_OUTPUT if _RAW_OUTPUT.is_absolute() else (_BASE_DIR / _RAW_OUTPUT)
_LAST_OUTPUT_PATH: Path | None = None
_LAST_SOURCE_NAME: str | None = None
_LAST_GOOGLE_COUNTRY: str | None = None
_LAST_GOOGLE_LANG: str | None = None
_LAST_TARGET_YEAR: int | None = None


def _safe_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        try:
            return int(pd.read_csv(path).shape[0])
        except Exception:
            return 0


def _latest_date(path: Path) -> datetime:
    if not path.exists():
        return updater.DEFAULT_START_DATE
    try:
        df = pd.read_csv(path, usecols=["date"])
    except Exception:
        return updater.DEFAULT_START_DATE
    if "date" not in df.columns:
        return updater.DEFAULT_START_DATE
    dt_series = pd.to_datetime(df["date"], utc=True, errors="coerce")
    max_date = dt_series.max()
    if pd.isna(max_date):
        return updater.DEFAULT_START_DATE
    return max_date.to_pydatetime()


def _current_output_path() -> Path:
    return _LAST_OUTPUT_PATH or _OUTPUT_PATH


def get_update_status() -> Dict[str, Any]:
    output_path = _current_output_path()
    source_name = _LAST_SOURCE_NAME
    if source_name is None and output_path.exists():
        source_name = output_path.name
    google_country = _LAST_GOOGLE_COUNTRY or updater.GOOGLE_DEFAULT_COUNTRY
    google_lang = _LAST_GOOGLE_LANG or updater.GOOGLE_DEFAULT_LANG
    target_year = _LAST_TARGET_YEAR or updater.TARGET_YEAR
    return {
        "output_file": str(output_path),
        "output_exists": output_path.exists(),
        "existing_rows": _safe_row_count(output_path),
        "latest_date": _latest_date(output_path),
        "source_name": source_name,
        "source_kind": "upload" if _LAST_OUTPUT_PATH is not None else "default",
        "google_apps": updater.GOOGLE_APPS,
        "google_country": google_country,
        "google_lang": google_lang,
        "target_year": target_year,
        "apple_country": google_country,
        "apple_lang": google_lang,
        "apple_year": target_year,
        "apple_apps": updater.APPLE_APPS,
    }


def run_update(
    source_path: Path | None = None,
    source_name: str | None = None,
    google_country: str | None = None,
    google_lang: str | None = None,
    target_year: int | None = None,
) -> Dict[str, Any]:
    global _LAST_OUTPUT_PATH, _LAST_SOURCE_NAME, _LAST_GOOGLE_COUNTRY, _LAST_GOOGLE_LANG, _LAST_TARGET_YEAR
    logs: List[str] = []
    output_path = source_path or _OUTPUT_PATH
    _LAST_OUTPUT_PATH = output_path
    if source_name:
        _LAST_SOURCE_NAME = source_name
    updater.OUTPUT_FILE = str(output_path)
    summary = updater.update_database(
        logs=logs,
        google_country=google_country,
        google_lang=google_lang,
        target_year=target_year,
    )
    _LAST_GOOGLE_COUNTRY = summary.get("google_country")
    _LAST_GOOGLE_LANG = summary.get("google_lang")
    _LAST_TARGET_YEAR = summary.get("target_year")
    summary["logs"] = logs
    summary["output_file"] = str(output_path)
    summary["output_exists"] = output_path.exists()
    summary["source_name"] = _LAST_SOURCE_NAME or output_path.name
    return summary


def get_output_path() -> Path:
    return _current_output_path()
