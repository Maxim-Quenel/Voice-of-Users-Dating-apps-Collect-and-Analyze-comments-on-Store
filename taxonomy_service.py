from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os

import pandas as pd
import torch
from transformers import pipeline

from clustering_service import read_csv_safely
from length_utils import add_length_column


TAXONOMY_SCHEMA = {
    "Technique / Bugs": [
        "Crash / Stabilité de l'app",
        "Connexion / Authentification",
        "Notifications / Push",
        "Chargement profil / média",
        "Géolocalisation / GPS",
        "Bug upload / synchronisation",
    ],
    "UX / Ergonomie": [
        "Navigation / Interface confuse",
        "Performance / Lenteur",
        "Onboarding / Tutoriel",
        "Publicités / Distractions",
        "Design / Mode sombre",
    ],
    "Suggestions / Fonctionnalités": [
        "Filtres / Recherche",
        "Chat / Messagerie",
        "Profil / Options personnalisées",
        "Sécurité / Vérification",
        "Amélioration matching / tri",
        "Mode premium / avantages",
    ],
    "Communauté / Sécurité": [
        "Bots / Faux profils / Spam",
        "Harcèlement / Comportement inapproprié",
        "Bannissement / Suspension",
        "Vérification / Modération",
        "Arnaque / Fraude",
        "Contenu sensible",
    ],
    "Paiement / Abonnement": [
        "Prix / Coût abonnement",
        "Valeur / Rapport qualité-prix",
        "Paywall / Limitation fonctionnalités",
        "Paiement / Facturation",
        "Remboursement / Support",
        "Désabonnement / Suppression compte",
    ],
    "Matching / Algorithme": [
        "Qualité des matchs",
        "Critères / Préférences",
        "Limite de likes / swipes",
        "Visibilité / Shadowban",
        "Distance / Localisation",
    ],
    "Retours Positifs": [
        "Match trouvé / Success story",
        "UX agréable / Interface",
        "Qualité des matchs",
        "Concept / Mission",
        "Application utile / efficace",
    ],
    "Autres / Indéterminé": [
        "Hors sujet",
        "Avis générique",
        "Langue / International",
    ],
}

DEFAULT_LABEL = "Autres / Indéterminé - Avis générique"
HYPOTHESIS_TEMPLATE = os.getenv("TAXONOMY_HYPOTHESIS", "This review is about {}.")
HYPOTHESIS_TEMPLATE_FR = os.getenv("TAXONOMY_HYPOTHESIS_FR", "Cet avis porte sur {}.")
MODEL_NAME = os.getenv("TAXONOMY_MODEL_NAME")
BATCH_SIZE = int(os.getenv("TAXONOMY_BATCH_SIZE", "32"))

_CLASSIFIER = None
_DEVICE = 0 if torch.cuda.is_available() else -1

CANDIDATE_LABELS = [
    f"{category} - {sub_category}"
    for category, sub_categories in TAXONOMY_SCHEMA.items()
    for sub_category in sub_categories
]


def _get_classifier():
    global _CLASSIFIER
    if _CLASSIFIER is None:
        print(f"[taxonomy] Loading model {MODEL_NAME} (device={_DEVICE})...", flush=True)
        _CLASSIFIER = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=_DEVICE,
            batch_size=BATCH_SIZE,
        )
    return _CLASSIFIER


def _log(logs: List[str], message: str) -> None:
    logs.append(message)
    print(f"[taxonomy] {message}", flush=True)


def _classify_reviews_zero_shot(
    df: pd.DataFrame,
    text_column: str,
    logs: List[str],
    hypothesis_template: str,
) -> pd.Series:
    classifier = _get_classifier()
    valid_mask = df[text_column].apply(lambda x: isinstance(x, str) and bool(x.strip()))
    valid_indices = df.index[valid_mask].tolist()
    reviews = df.loc[valid_indices, text_column].tolist()

    if not reviews:
        _log(logs, "No valid reviews found, skipping classification.")
        return pd.Series(DEFAULT_LABEL, index=df.index)

    _log(logs, f"Classifying {len(reviews)} reviews in batches of {BATCH_SIZE}.")
    results: List[str] = []
    for start in range(0, len(reviews), BATCH_SIZE):
        batch = reviews[start:start + BATCH_SIZE]
        preds = classifier(
            batch,
            CANDIDATE_LABELS,
            multi_label=False,
            hypothesis_template=hypothesis_template,
        )
        for pred in preds:
            results.append(pred["labels"][0])
        if start == 0 or (start // BATCH_SIZE) % 10 == 0:
            _log(logs, f"Processed {min(start + BATCH_SIZE, len(reviews))}/{len(reviews)} reviews.")

    results_series = pd.Series(DEFAULT_LABEL, index=df.index)
    results_series.loc[valid_indices] = results
    return results_series


def _detect_locale_from_data(df: pd.DataFrame) -> str:
    if "country" not in df.columns:
        return "en"
    countries = df["country"].dropna().astype(str).str.lower()
    if countries.empty:
        return "en"
    top_country = countries.value_counts().idxmax()
    return "fr" if top_country == "fr" else "en"


def _select_hypothesis_template(df: pd.DataFrame) -> str:
    return HYPOTHESIS_TEMPLATE_FR if _detect_locale_from_data(df) == "fr" else HYPOTHESIS_TEMPLATE


def _build_breakdown_matrix(pivot_df: pd.DataFrame) -> Dict[str, Any]:
    pivot_df = pivot_df.copy()
    pivot_df = pivot_df.fillna(0.0)
    return {
        "main_categories": pivot_df.index.tolist(),
        "taxonomy_categories": pivot_df.columns.tolist(),
        "matrix": pivot_df.round(2).values.tolist(),
    }


def _build_rating_matrix(df: pd.DataFrame, main_category_col: str) -> Dict[str, Any] | None:
    if "rating" not in df.columns or main_category_col not in df.columns:
        return None
    rating_counts = df.groupby([main_category_col, "rating"]).size().unstack(fill_value=0)
    if rating_counts.empty:
        return None
    rating_order = sorted(rating_counts.columns, reverse=True)
    rating_counts = rating_counts[rating_order]
    return {
        "main_categories": rating_counts.index.tolist(),
        "ratings": [str(rating) for rating in rating_order],
        "matrix": rating_counts.values.tolist(),
    }


def _build_weekly_trends(
    df: pd.DataFrame,
    logs: List[str],
    main_category_col: str,
) -> Dict[str, Any] | None:
    if "date" not in df.columns or main_category_col not in df.columns:
        return None
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"])
    if working.empty:
        _log(logs, "Date column exists but no valid datetime rows found.")
        return None

    top_issue_categories = (
        working[~working[main_category_col].isin({"Positive Feedback", "Retours Positifs"})][main_category_col]
        .value_counts()
        .head(4)
        .index.tolist()
    )
    if not top_issue_categories:
        return None

    time_series_df = working[working[main_category_col].isin(top_issue_categories)]
    weekly_trends = (
        time_series_df.groupby([pd.Grouper(key="date", freq="W"), main_category_col])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if weekly_trends.empty:
        return None

    return {
        "weeks": [ts.strftime("%Y-%m-%d") for ts in weekly_trends.index],
        "categories": weekly_trends.columns.tolist(),
        "matrix": weekly_trends.values.tolist(),
    }


def _build_taxonomy_outputs(df: pd.DataFrame, logs: List[str], text_column: str) -> Dict[str, Any]:
    category_counts = df["taxonomy_category"].value_counts()
    main_category_col = "taxonomy_main_category" if "taxonomy_main_category" in df.columns else "main_category"
    main_counts = df[main_category_col].value_counts()
    main_percents = (main_counts / main_counts.sum() * 100).round(2)
    _log(logs, f"{len(category_counts)} granular categories found.")

    pivot_df = pd.crosstab(df[main_category_col], df["taxonomy_category"])
    normalized_pivot = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    normalized_pivot = normalized_pivot.reindex(main_counts.index)

    rating_matrix = _build_rating_matrix(df, main_category_col)
    weekly_trends = _build_weekly_trends(df, logs, main_category_col)

    preview_columns: List[str] = []
    for col in (text_column, "longueur", "taxonomy_category", "taxonomy_main_category", "rating", "date"):
        if col in df.columns and col not in preview_columns:
            preview_columns.append(col)
    if not preview_columns:
        preview_columns = df.columns[:4].tolist()
    preview = df[preview_columns].head(100).to_dict(orient="records")

    return {
        "category_counts": [
            {"category": category, "count": int(count)}
            for category, count in category_counts.items()
        ],
        "main_counts": [
            {"category": category, "count": int(main_counts[category]), "percent": float(main_percents[category])}
            for category in main_counts.index
        ],
        "taxonomy_breakdown": _build_breakdown_matrix(normalized_pivot),
        "rating_breakdown": rating_matrix,
        "weekly_trends": weekly_trends,
        "preview": preview,
        "preview_columns": preview_columns,
        "num_reviews": len(df),
        "num_categories": len(category_counts),
    }


def _append_short_rows(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    null_columns: List[str],
) -> pd.DataFrame:
    if short_df.empty:
        return long_df
    aligned = short_df.copy()
    for col in null_columns:
        aligned[col] = None
    for col in long_df.columns:
        if col not in aligned.columns:
            aligned[col] = None
    aligned = aligned[long_df.columns]
    return pd.concat([long_df, aligned], ignore_index=True)


def run_taxonomy_classification(csv_path: Path, text_column: str = "text") -> Dict[str, Any]:
    logs: List[str] = []

    _log(logs, f"Reading CSV: {csv_path}")
    df = read_csv_safely(csv_path, logs)
    if text_column not in df.columns:
        raise ValueError(f"Missing required column '{text_column}'.")

    full_df = add_length_column(df, text_column=text_column, logs=logs)
    short_df = full_df[full_df["longueur"] == 0].copy()
    df = full_df[full_df["longueur"] == 1].copy()
    if df.empty:
        raise ValueError("Aucun avis avec longueur >= 4 mots apres filtrage.")

    if "main_category" in df.columns:
        df = df.drop(columns=["main_category"])
        short_df = short_df.drop(columns=["main_category"], errors="ignore")

    df[text_column] = df[text_column].fillna("")
    _log(logs, f"{len(df)} reviews loaded.")
    _log(logs, f"Model: {MODEL_NAME} | device={_DEVICE} | batch={BATCH_SIZE}")

    hypothesis_template = _select_hypothesis_template(df)
    df["taxonomy_category"] = _classify_reviews_zero_shot(
        df,
        text_column,
        logs,
        hypothesis_template,
    )
    df["taxonomy_main_category"] = df["taxonomy_category"].apply(
        lambda x: x.split(" - ")[0] if " - " in x else "Autres / Indéterminé"
    )

    outputs = _build_taxonomy_outputs(df, logs, text_column)
    result_df = _append_short_rows(
        df,
        short_df,
        ["taxonomy_category", "taxonomy_main_category"],
    )
    return {
        "result_df": result_df,
        **outputs,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "device": _DEVICE,
        "source": "classified",
        "logs": logs,
    }


def summarize_taxonomy_csv(csv_path: Path, text_column: str = "text") -> Dict[str, Any]:
    logs: List[str] = []
    _log(logs, f"Reading CSV: {csv_path}")
    df = read_csv_safely(csv_path, logs)
    if "taxonomy_category" not in df.columns:
        raise ValueError("Missing required column 'taxonomy_category' in the CSV.")
    if text_column not in df.columns:
        raise ValueError(f"Missing required column '{text_column}'.")

    full_df = add_length_column(df, text_column=text_column, logs=logs)
    short_df = full_df[full_df["longueur"] == 0].copy()
    df = full_df[full_df["longueur"] == 1].copy()
    if df.empty:
        raise ValueError("Aucun avis avec longueur >= 4 mots apres filtrage.")

    if "taxonomy_main_category" in df.columns:
        if "main_category" in df.columns:
            df = df.drop(columns=["main_category"])
            short_df = short_df.drop(columns=["main_category"], errors="ignore")
    elif "main_category" in df.columns:
        df["taxonomy_main_category"] = df["main_category"]
        df = df.drop(columns=["main_category"])
        short_df = short_df.drop(columns=["main_category"], errors="ignore")
    else:
        _log(logs, "Missing 'main_category' column, deriving from taxonomy_category.")
        df["taxonomy_main_category"] = df["taxonomy_category"].apply(
            lambda x: x.split(" - ")[0] if " - " in x else "Autres / Indéterminé"
        )
    if "taxonomy_main_category" not in short_df.columns:
        short_df["taxonomy_main_category"] = None

    outputs = _build_taxonomy_outputs(df, logs, text_column)
    result_df = _append_short_rows(
        df,
        short_df,
        ["taxonomy_category", "taxonomy_main_category"],
    )
    return {
        "result_df": result_df,
        **outputs,
        "model_name": "Import CSV",
        "batch_size": None,
        "device": None,
        "source": "imported",
        "logs": logs,
    }


def release_resources() -> None:
    global _CLASSIFIER
    _CLASSIFIER = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
