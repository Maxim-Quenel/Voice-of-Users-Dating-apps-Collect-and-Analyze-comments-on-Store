from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd
import torch

from clustering_service import assign_coarse_group, read_csv_safely
from length_utils import filter_by_length


def _log(logs: List[str], message: str) -> None:
    logs.append(message)
    print(f"[rag] {message}", flush=True)

# Etat global simple pour partager l'index RAG entre les requêtes.
RAG_STATE: Dict[str, Any] = {
    "collection": None,
    "source": None,
    "count": 0,
    "preview": [],
    "columns": [],
    "chart_sample": [],
    "column_types": {},
    "chart_sample_size": 0,
    "logs": [],
}

def _build_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"[rag] Chargement du modèle {model_name} sur {device}...", flush=True)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )


_EMBEDDING_FUNCTION = None


def _get_embedding_function():
    global _EMBEDDING_FUNCTION
    if _EMBEDDING_FUNCTION is None:
        _EMBEDDING_FUNCTION = _build_embedding_function()
    return _EMBEDDING_FUNCTION


def _normalize_dataframe(df: pd.DataFrame, logs: List[str]) -> pd.DataFrame:
    """
    Nettoie le dataframe pour le RAG : texte obligatoire, colonnes ai_category / ai_group ajoutées si absentes.
    """
    if "text" not in df.columns:
        raise ValueError("La colonne 'text' est obligatoire pour indexer le RAG.")

    working = filter_by_length(df, logs=logs)
    if working.empty:
        raise ValueError("Aucun texte avec longueur >= 4 mots pour indexer le RAG.")

    working["text"] = working["text"].fillna("")
    if "ai_category" not in working.columns:
        working["ai_category"] = "Sans catégorie"
        _log(logs, "Colonne 'ai_category' absente: valeur par défaut appliquée.")
    if "ai_group" not in working.columns:
        working["ai_group"] = working["ai_category"].apply(assign_coarse_group)
        _log(logs, "Colonne 'ai_group' absente: dérivation à partir de ai_category.")
    return working


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


def _build_chart_payload(df: pd.DataFrame) -> Dict[str, Any]:
    sample_size = min(len(df), 1500)
    if sample_size <= 0:
        return {"columns": [], "chart_sample": [], "column_types": {}, "chart_sample_size": 0}
    sample_df = df.sample(n=sample_size, random_state=42, replace=False).copy()

    for col in ("text", "reply_content"):
        if col in sample_df.columns:
            sample_df[col] = sample_df[col].astype(str).str.slice(0, 160)

    sample_df = sample_df.where(pd.notna(sample_df), None)
    column_types = _infer_column_types(sample_df)
    return {
        "columns": sample_df.columns.tolist(),
        "chart_sample": sample_df.to_dict(orient="records"),
        "column_types": column_types,
        "chart_sample_size": sample_size,
    }


def _build_collection(df: pd.DataFrame, source_label: str, logs: List[str]) -> Dict[str, Any]:
    _log(logs, f"Création de l'index RAG depuis: {source_label}")
    embedding_fn = _get_embedding_function()
    _log(logs, "Initialisation du client Chroma (non persistant).")
    client = chromadb.Client(
        Settings(
            anonymized_telemetry=False,
            is_persistent=False,
            allow_reset=True,
        )
    )
    # Reset pour repartir sur un index propre à chaque upload (autorisé via allow_reset=True).
    client.reset()
    _log(logs, "Client réinitialisé, création de la collection.")
    collection = client.create_collection(
        name="reviews_rag",
        embedding_function=embedding_fn,
    )

    documents = []
    metadatas = []
    ids = []
    for idx, row in df.iterrows():
        enriched_text = f"{row['ai_category']} | {row['ai_group']} | {row['text']}"
        documents.append(enriched_text)
        metadatas.append(
            {
                "ai_category": str(row["ai_category"]),
                "ai_group": str(row["ai_group"]),
                "orig_text": str(row["text"]),
            }
        )
        ids.append(str(idx))

    # Chroma limite la taille des batchs (~5461). On découpe pour éviter ValueError.
    batch_size = 4000
    total = len(documents)
    _log(logs, f"Indexation en {((total - 1) // batch_size) + 1} batchs (taille {batch_size}).")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        _log(logs, f"Batch {start}-{end} indexé.")
    _log(logs, f"{len(documents)} avis indexés pour la recherche sémantique.")

    preview_cols = ["text", "ai_category", "ai_group"]
    preview = df[preview_cols].head(15).to_dict(orient="records")
    chart_payload = _build_chart_payload(df)

    RAG_STATE.update(
        {
            "collection": collection,
            "source": source_label,
            "count": len(documents),
            "preview": preview,
            "columns": chart_payload["columns"],
            "chart_sample": chart_payload["chart_sample"],
            "column_types": chart_payload["column_types"],
            "chart_sample_size": chart_payload["chart_sample_size"],
            "logs": logs.copy(),
        }
    )
    return {
        "source": source_label,
        "count": len(documents),
        "preview": preview,
        "columns": chart_payload["columns"],
        "chart_sample": chart_payload["chart_sample"],
        "column_types": chart_payload["column_types"],
        "chart_sample_size": chart_payload["chart_sample_size"],
        "logs": logs,
    }


def index_from_dataframe(df: pd.DataFrame, source_label: str) -> Dict[str, Any]:
    logs: List[str] = []
    working = _normalize_dataframe(df, logs)
    return _build_collection(working, source_label, logs)


def index_from_csv(csv_path: Path, original_name: str | None = None) -> Dict[str, Any]:
    logs: List[str] = []
    df = read_csv_safely(csv_path, logs)
    working = _normalize_dataframe(df, logs)
    label = f"Fichier uploadé: {original_name or csv_path.name}"
    return _build_collection(working, label, logs)


def current_state() -> Dict[str, Any]:
    state = RAG_STATE.copy()
    state.pop("collection", None)
    return state


def query_rag(query: str, top_n: int = 5, min_similarity: float = 0.35) -> Dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("La requête ne peut pas être vide.")
    if RAG_STATE["collection"] is None:
        raise ValueError("Aucune collection RAG construite. Merci d'indexer un dataset d'abord.")

    # S'assure que la fonction d'embedding est disponible (lazy load).
    _get_embedding_function()

    collection = RAG_STATE["collection"]
    raw_results = collection.query(
        query_texts=[query],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    # distances est une liste de listes, on accède à la 1ère requête via l'index 0.
    docs = raw_results.get("documents", [[]])[0]
    metas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1.0 - float(dist)
        if similarity < min_similarity:
            continue
        matches.append(
            {
                "similarity": round(similarity, 4),
                "ai_category": meta.get("ai_category", ""),
                "ai_group": meta.get("ai_group", ""),
                "text": meta.get("orig_text", doc),
            }
        )

    return {
        "query": query,
        "min_similarity": min_similarity,
        "matches": matches,
        "source": RAG_STATE["source"],
        "indexed_count": RAG_STATE["count"],
    }


def release_resources() -> None:
    global _EMBEDDING_FUNCTION
    _EMBEDDING_FUNCTION = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
