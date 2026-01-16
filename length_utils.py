from __future__ import annotations

from typing import List, Optional

import pandas as pd


def _count_words(text: object) -> int:
    if not isinstance(text, str):
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return len([part for part in stripped.split() if part])


def add_length_column(
    df: pd.DataFrame,
    text_column: str = "text",
    length_column: str = "longueur",
    threshold: int = 4,
    logs: Optional[List[str]] = None,
) -> pd.DataFrame:
    if text_column not in df.columns:
        raise ValueError(f"Missing required column '{text_column}' to compute longueur.")

    working = df.copy()
    word_counts = working[text_column].fillna("").apply(_count_words)
    working[length_column] = (word_counts >= threshold).astype(int)

    if logs is not None:
        short_count = int((working[length_column] == 0).sum())
        long_count = int((working[length_column] == 1).sum())
        logs.append(
            f"Colonne '{length_column}' ajoutee (0:<{threshold} mots, 1:>={threshold}). "
            f"Courts={short_count}, longs={long_count}."
        )

    return working


def filter_by_length(
    df: pd.DataFrame,
    text_column: str = "text",
    length_column: str = "longueur",
    threshold: int = 4,
    keep_value: int = 1,
    logs: Optional[List[str]] = None,
) -> pd.DataFrame:
    working = add_length_column(
        df,
        text_column=text_column,
        length_column=length_column,
        threshold=threshold,
        logs=logs,
    )
    filtered = working[working[length_column] == keep_value].copy()

    if logs is not None:
        logs.append(
            f"Filtrage sur '{length_column}={keep_value}': "
            f"{len(filtered)} lignes retenues sur {len(working)}."
        )

    return filtered
