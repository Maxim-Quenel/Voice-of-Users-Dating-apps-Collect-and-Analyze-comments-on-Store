from pathlib import Path
from typing import Dict, Optional, List, Any
from unicodedata import normalize
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import umap
import hdbscan
from ai_categorizer import predict_category_from_reviews, release_resources as release_ai_resources
from length_utils import add_length_column

COARSE_CATEGORY_RULES = [
    ("Abonnements & Paiement", ["abonnement", "pay", "prix", "tarif", "factur", "premium", "paywall", "payer", "achat", "facture", "expensive", "money", "cost", "chat cost", "money farm", "suckery"]),
    ("Bannissement / Vérification", ["ban", "banni", "verification", "otp", "captcha", "shadow", "bloqu", "suspension"]),
    ("Bots / Faux Comptes / Spam", ["bot", "faux", "spam", "scam", "fake", "escort", "escroc", "fraud"]),
    ("Connexion / Technique", ["connexion", "login", "bug", "erreur", "crash", "lag", "upload", "otp", "technique", "serveur", "not working", "fonction", "functionality", "notifications", "rating"]),
    ("UI / Ergonomie", ["ui", "ux", "design", "interface", "dark mode", "mode sombre", "filter", "height", "architecture"]),
    ("Matching / Expérience", ["match", "like", "algorithme", "swipe", "experience", "profil", "recommandation", "dating", "boredom", "time wasting", "perfectionism", "friendship", "relationship"]),
    ("Service Client / Remboursement", ["service client", "support", "remboursement", "refund", "assistance", "customer service"]),
    ("Psychologie / Sécurité / Contenu", ["psychologique", "harcel", "haine", "secur", "urgence", "danger", "agression", "abusive", "nudity", "racist", "safety", "privacy"]),
    ("Localisation / Distance", ["distance", "géo", "geoloc", "localisation", "gps", "travel"]),
    ("Compte / Désabonnement", ["account", "deletion", "cancellation", "supprimer", "cancel"]),
    ("International / Langues", ["international", "langue", "francophone", "hindi", "inde", "es", "pt", "spanish", "portuguese"]),
    ("Suggestions / Fonctionnalités", ["suggest", "suggestion", "feature", "request", "add", "ajout", "option", "wish", "would like", "improve", "amelior", "enhancement", "idea", "fonctionnalite", "propose"]),
    ("Retours Positifs", ["love", "great", "amazing", "awesome", "excellent", "good", "super", "genial", "merci", "parfait", "top", "best", "nice", "cool", "fantastic", "wonderful"]),
    ("Divers / Autres", []),
]

_MODEL: Optional[SentenceTransformer] = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        print(f"[clustering] Chargement du modèle {_EMB_MODEL_NAME} sur {_DEVICE}...", flush=True)
        _MODEL = SentenceTransformer(_EMB_MODEL_NAME, device=_DEVICE)
    return _MODEL
#BAAI/bge-large-en-v1.5
#sentence-transformers/all-MiniLM-L6-v2


def assign_coarse_group(label: str) -> str:
    normalized = normalize("NFKD", label).encode("ASCII", "ignore").decode().lower()
    for group, keywords in COARSE_CATEGORY_RULES:
        if any(keyword in normalized for keyword in keywords):
            return group
    return "Autres"


def read_csv_safely(path: Path, logs: List[str]) -> pd.DataFrame:
    """
    Essaie UTF-8 puis latin-1 pour limiter les soucis d'encodage.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            logs.append(f"CSV lu avec encodage {enc}.")
            return df
        except UnicodeDecodeError:
            logs.append(f"Encodage {enc} invalide, tentative suivante...")
            continue
    logs.append("Aucun encodage valide trouvé, abandon.")
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Impossible de décoder le fichier")


def label_clusters_with_llm(data: pd.DataFrame, max_clusters: Optional[int] = None) -> Dict[int, str]:
    """
    Utilise Qwen pour synthétiser un nom de catégorie par cluster.
    max_clusters=None => tous les clusters (sauf bruit), sinon limite.
    """
    locale = _detect_locale_from_data(data)
    llm_labels: Dict[int, str] = {}
    clusters = [c for c in sorted(data["cluster"].unique()) if c != -1]
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
    for cluster_id in clusters:
        reviews = (
            data.loc[data["cluster"] == cluster_id, "text"]
            .fillna("")
            .head(10)
            .tolist()
        )
        llm_labels[cluster_id] = predict_category_from_reviews(reviews, locale=locale)
    return llm_labels


def _detect_locale_from_data(data: pd.DataFrame) -> str:
    if "country" not in data.columns:
        return "en"
    countries = data["country"].dropna().astype(str).str.lower()
    if countries.empty:
        return "en"
    top_country = countries.value_counts().idxmax()
    return "fr" if top_country == "fr" else "en"


def _build_category_pools(data: pd.DataFrame, sample_cap: int = 120) -> Dict[str, List[dict]]:
    """
    Prépare une structure pour afficher des extraits aléatoires par groupe/cluster.
    Les sous-catégories identiques (même ai_category) sont fusionnées même si plusieurs clusters partagent le nom.
    On limite les extraits gardés par cluster pour éviter de charger tout le dataset en mémoire.
    """
    pools: Dict[str, Dict[str, dict]] = {}
    clustered = data[data["cluster"] != -1].groupby("cluster")
    for cluster_id, subset in clustered:
        category = subset["ai_category"].iloc[0]
        group = subset["ai_group"].iloc[0]
        sample_size = min(sample_cap, len(subset))
        samples = (
            subset["text"]
            .sample(n=sample_size, random_state=42, replace=False)
            .tolist()
        )
        group_pool = pools.setdefault(group, {})
        if category not in group_pool:
            group_pool[category] = {"cluster_ids": [], "category": category, "samples": []}
        group_pool[category]["cluster_ids"].append(int(cluster_id))
        group_pool[category]["samples"].extend(samples)
    # Convertit en liste et tronque les samples à 200 max pour l'affichage
    final_pools: Dict[str, List[dict]] = {}
    for group, cats in pools.items():
        final_pools[group] = []
        for entry in cats.values():
            entry["samples"] = entry["samples"][:200]
            final_pools[group].append(entry)
    return final_pools


def _summarize(data: pd.DataFrame, llm_labels: Dict[int, str], logs: List[str]) -> Dict[str, Any]:
    """
    Construit les structures de sortie communes (comptages, pools d'extraits, aperçus).
    """
    data = data.copy()
    if "Topic_Label" in data.columns:
        data = data.drop(columns=["Topic_Label"])
    if "ai_category" not in data.columns:
        data["ai_category"] = data["cluster"].astype(str)

    cluster_sizes = data["ai_category"].value_counts().reset_index()
    cluster_sizes.columns = ["topic", "count"]

    # Si llm_labels vide mais ai_category présent (cas CSV annoté), on reconstruit le mapping
    if not llm_labels and "ai_category" in data.columns:
        llm_labels = (
            data.loc[data["cluster"] != -1]
            .groupby("cluster")[["ai_category"]]
            .first()["ai_category"]
            .to_dict()
        )

    if "ai_group" not in data.columns:
        data["ai_group"] = data["ai_category"].apply(assign_coarse_group)

    ai_group_sizes = data["ai_group"].value_counts().reset_index()
    ai_group_sizes.columns = ["group", "count"]

    ai_labels_preview = [
        {"cluster": int(cluster_id), "ai_category": label, "ai_group": assign_coarse_group(label)}
        for cluster_id, label in llm_labels.items()
    ]

    category_pools = _build_category_pools(data)
    logs.append(
        f"Pools d'extraits construits pour {sum(len(v) for v in category_pools.values())} clusters affichables."
    )

    sample_rows = (
        data.sort_values("cluster")
        .head(100)
        .to_dict(orient="records")
    )

    # Echantillon pour la visualisation UMAP 2D dans l'interface (limité pour la perf).
    umap_points = []
    if {"x", "y", "cluster"}.issubset(data.columns):
        sampled = data.sample(n=min(4000, len(data)), random_state=42, replace=False)
        umap_points = sampled[["x", "y", "cluster"]].to_dict(orient="records")
        logs.append(f"{len(umap_points)} points UMAP échantillonnés pour l'affichage.")

    return {
        "result_df": data,
        "cluster_sizes": cluster_sizes.to_dict(orient="records"),
        "ai_group_sizes": ai_group_sizes.to_dict(orient="records"),
        "ai_labels": ai_labels_preview,
        "num_clusters": data["cluster"].nunique() - (1 if -1 in data["cluster"].unique() else 0),
        "num_noise": int((data["cluster"] == -1).sum()),
        "preview": sample_rows,
        "category_pools": category_pools,
        "umap_points": umap_points,
        "logs": logs,
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


def run_clustering(csv_path: Path, label_clusters: bool = True) -> Dict[str, object]:
    """
    Embedding + UMAP + HDBSCAN sur un CSV uploadé (colonne 'text' requise).
    `label_clusters=False` permet de ne faire que le clustering sans LLM.
    """
    logs: List[str] = []

    def log(message: str) -> None:
        logs.append(message)
        print(f"[clustering] {message}", flush=True)

    log(f"Lecture du CSV: {csv_path}")
    data = read_csv_safely(csv_path, logs)
    if "text" not in data.columns:
        raise ValueError("Le CSV doit contenir une colonne 'text'.")

    full_data = add_length_column(data, logs=logs)
    short_df = full_data[full_data["longueur"] == 0].copy()
    data = full_data[full_data["longueur"] == 1].copy()
    if data.empty:
        raise ValueError("Aucun avis avec longueur >= 4 mots apres filtrage.")

    data["text"] = data["text"].fillna("")
    log(f"{len(data)} avis chargés.")

    model = get_model()
    log(f"Encodage des avis avec {_EMB_MODEL_NAME} sur {model.device}...")
    log("Progression encodage (barre fournie par sentence-transformers) en cours...")
    embeddings = model.encode(data["text"].tolist(), show_progress_bar=True)
    log(f"Embeddings calculés: shape={embeddings.shape}.")

    umap_5d = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)
    log("Projection UMAP 5D terminée.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=60,
        min_samples=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_5d)
    data["cluster"] = labels
    log(
        f"Clustering HDBSCAN: {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
        f"{list(labels).count(-1)} points bruit."
    )

    llm_labels: Dict[int, str] = {}
    if label_clusters:
        log("Nom automatique des clusters via LLM (tous les clusters hors bruit)...")
        llm_labels = label_clusters_with_llm(data, max_clusters=None)
        data["ai_category"] = data["cluster"].map(llm_labels).fillna("Autre / Bruit")
        data["ai_group"] = data["ai_category"].apply(assign_coarse_group)
        log(f"{len(llm_labels)} clusters renommés.")
    else:
        data["ai_category"] = "En attente de nom"
        data["ai_group"] = "Autres"
        log("Nom automatique désactivé (clustering uniquement).")

    umap_2d = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)
    data["x"] = umap_2d[:, 0]
    data["y"] = umap_2d[:, 1]
    log("Projection UMAP 2D terminée.")

    output = _summarize(data, llm_labels, logs)
    output["result_df"] = _append_short_rows(
        output["result_df"],
        short_df,
        ["cluster", "ai_category", "ai_group", "x", "y"],
    )
    return output


def summarize_annotated(data: pd.DataFrame) -> Dict[str, object]:
    """
    Recharge un CSV déjà annoté (ai_category, ai_group, cluster) sans refaire le clustering.
    """
    logs: List[str] = []

    def log(message: str) -> None:
        logs.append(message)
        print(f"[reload] {message}", flush=True)

    required = {"text", "cluster", "ai_category", "ai_group"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV annoté: {', '.join(sorted(missing))}")

    full_data = add_length_column(data, logs=logs)
    short_df = full_data[full_data["longueur"] == 0].copy()
    data = full_data[full_data["longueur"] == 1].copy()
    if data.empty:
        raise ValueError("Aucun avis avec longueur >= 4 mots apres filtrage.")

    log("CSV annote charge, reconstruction des metriques.")
    output = _summarize(data.copy(), {}, logs)
    output["result_df"] = _append_short_rows(
        output["result_df"],
        short_df,
        ["cluster", "ai_category", "ai_group", "x", "y"],
    )
    return output


def release_resources() -> None:
    global _MODEL
    _MODEL = None
    try:
        release_ai_resources()
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
