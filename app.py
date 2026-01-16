from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, request, send_file, abort, redirect, url_for

#BAAI/bge-large-en-v1.5
#sentence-transformers/all-MiniLM-L6-v2
#paraphrase-multilingual-MiniLM-L12-v2

#joeddav/xlm-roberta-base-xnli
#MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

# Modèles centralisés (modifiable via variables d'env)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
LLM_TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TAXONOMY_MODEL_NAME = os.getenv("TAXONOMY_MODEL_NAME", "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")
# Propagation pour les modules qui lisent l'env  #sentence-transformers/all-MiniLM-L6-v2
os.environ.setdefault("LLM_MODEL_NAME", LLM_MODEL_NAME)
os.environ.setdefault("LLM_TOKENIZER_NAME", LLM_TOKENIZER_NAME)
os.environ.setdefault("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME)
os.environ.setdefault("TAXONOMY_MODEL_NAME", TAXONOMY_MODEL_NAME)

from clustering_service import (
    run_clustering,
    summarize_annotated,
    read_csv_safely,
    release_resources as release_clustering_resources,
)
from rag_service import (
    current_state as rag_current_state,
    index_from_csv as rag_index_from_csv,
    index_from_dataframe as rag_index_from_dataframe,
    query_rag,
    release_resources as release_rag_resources,
)
from short_review_service import run_short_review_exploration
from sentiment_service import run_sentiment_analysis, release_resources as release_sentiment_resources
from taxonomy_service import (
    run_taxonomy_classification,
    summarize_taxonomy_csv,
    release_resources as release_taxonomy_resources,
)
from update_dataset_service import get_update_status, run_update, get_output_path
from visualization_service import build_chart_payload_from_csv


app = Flask(__name__)
SESSION_STATE: Dict[str, Any] = {"data": None}
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STEP_FILES = {
    "update-dataset": "STEP 1 mise_a_jour.csv",
    "short-reviews": "STEP 2 textes_courts.csv",
    "sentiment": "STEP 3 sentiment.csv",
    "taxonomies": "STEP 4 taxonomies.csv",
    "clustering": "STEP 5 clustering.csv",
}


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _save_step_csv(df, filename: str) -> Path:
    target = _ensure_data_dir() / filename
    df.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def _save_step_file(source: Path, filename: str) -> Path:
    target = _ensure_data_dir() / filename
    if source.resolve() != target.resolve():
        shutil.copyfile(source, target)
    return target


def _step_download_url(step_key: str) -> str:
    return f"/download/{step_key}"


def _largest_group(ai_group_sizes: List[Dict[str, Any]]) -> str:
    if not ai_group_sizes:
        return "Non déterminé"
    top = max(ai_group_sizes, key=lambda row: row.get("count", 0))
    return f"{top.get('group', 'Non déterminé')} ({top.get('count', 0)} avis)"


def _build_cluster_table(df) -> List[Dict[str, Any]]:
    if not {"cluster", "ai_category", "ai_group"}.issubset(df.columns):
        return []
    table = (
        df.groupby(["cluster", "ai_category", "ai_group"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )
    return table.to_dict(orient="records")


def _release_models() -> None:
    for fn in (
        release_clustering_resources,
        release_taxonomy_resources,
        release_rag_resources,
        release_sentiment_resources,
    ):
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("update_dataset_home"))


@app.route("/clustering", methods=["GET"])
def clustering_home():
    return render_template(
        "index.html",
        results=None,
        has_data=SESSION_STATE.get("data") is not None,
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    action = request.form.get("action", "cluster_labels")
    app_logs: List[str] = []
    clustering_output = None

    if action == "load_annotated":
        uploaded = request.files.get("annotated_file")
        if uploaded is None or uploaded.filename == "":
            _release_models()
            return render_template(
                "index.html",
                results={"error": "Merci de sélectionner un CSV annoté (ai_category / ai_group / cluster).", "logs": app_logs},
                has_data=SESSION_STATE.get("data") is not None,
            )
        app_logs.append(f"Fichier annoté reçu: {uploaded.filename}")
        print(f"[web] Fichier annoté reçu: {uploaded.filename}", flush=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        app_logs.append(f"Fichier annoté temporaire: {temp_path}")
        try:
            df = read_csv_safely(temp_path, app_logs)
            clustering_output = summarize_annotated(df)
        except Exception as exc:  # noqa: BLE001
            app_logs.append(f"Erreur pendant le chargement annoté: {exc}")
            print(f"[web] Erreur: {exc}", flush=True)
            _release_models()
            return render_template(
                "index.html",
                results={"error": f"Erreur pendant le chargement annoté : {exc}", "logs": app_logs},
                has_data=SESSION_STATE.get("data") is not None,
            )
        finally:
            temp_path.unlink(missing_ok=True)
            app_logs.append("Fichier annoté temporaire supprimé.")
    else:
        uploaded = request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            _release_models()
            return render_template(
                "index.html",
                results={"error": "Merci de sélectionner un fichier CSV contenant une colonne 'text'."},
                has_data=SESSION_STATE.get("data") is not None,
            )

        app_logs.append(f"Fichier reçu: {uploaded.filename}")
        print(f"[web] Fichier reçu: {uploaded.filename}", flush=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        app_logs.append(f"Fichier temporaire enregistré: {temp_path}")

        try:
            clustering_output = run_clustering(temp_path, label_clusters=True)
        except Exception as exc:  # noqa: BLE001
            app_logs.append(f"Erreur pendant le traitement: {exc}")
            print(f"[web] Erreur: {exc}", flush=True)
            _release_models()
            return render_template(
                "index.html",
                results={"error": f"Erreur pendant le traitement : {exc}", "logs": app_logs},
                has_data=SESSION_STATE.get("data") is not None,
            )
        finally:
            temp_path.unlink(missing_ok=True)
            app_logs.append("Fichier temporaire supprimé.")

    # Mémorise le dernier DF pour pouvoir relancer la génération de labels sans re-clusteriser.
    result_df = clustering_output["result_df"]
    SESSION_STATE["data"] = result_df
    _save_step_csv(result_df, STEP_FILES["clustering"])

    labels_ready = len(clustering_output.get("ai_labels", [])) > 0 and result_df["ai_category"].nunique() > 1
    results = {
        "ai_labels": clustering_output["ai_labels"],
        "cluster_sizes": clustering_output["cluster_sizes"],
        "ai_group_sizes": clustering_output["ai_group_sizes"],
        "num_clusters": clustering_output["num_clusters"],
        "num_noise": clustering_output["num_noise"],
        "largest_group": _largest_group(clustering_output["ai_group_sizes"]),
        "preview": clustering_output["preview"],
        "download_uri": _step_download_url("clustering"),
        "category_pools": clustering_output.get("category_pools", {}),
        "logs": app_logs + clustering_output.get("logs", []),
        "labels_ready": labels_ready,
        "cluster_table": _build_cluster_table(result_df),
        "umap_points": clustering_output.get("umap_points", []),
    }

    print("[web] Action terminée, rendu des résultats.", flush=True)

    _release_models()
    return render_template(
        "index.html",
        results=results,
        has_data=True,
    )


def _render_rag_page(rag_state=None, rag_results=None, rag_error=None):
    state = rag_state or rag_current_state()
    return render_template(
        "rag.html",
        rag_state=state,
        rag_results=rag_results,
        rag_error=rag_error,
        has_data=SESSION_STATE.get("data") is not None,
    )


@app.route("/rag", methods=["GET"])
def rag_home():
    return _render_rag_page()


@app.route("/rag/index", methods=["POST"])
def rag_index():
    rag_state = rag_current_state()
    rag_error = None
    temp_path = None

    try:
        action = request.form.get("rag_action", "memory")
        if action == "memory":
            if SESSION_STATE.get("data") is None:
                raise ValueError("Aucun dataset en mémoire. Lancez d'abord le clustering + labels.")
            rag_state = rag_index_from_dataframe(
                SESSION_STATE["data"],
                "Dataset en mémoire (clustering + labels)",
            )
        else:
            uploaded = request.files.get("rag_file")
            if uploaded is None or uploaded.filename == "":
                _release_models()
                raise ValueError("Merci de fournir un CSV pour construire l'index RAG.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                temp_path = Path(tmp.name)
                uploaded.save(temp_path)
            rag_state = rag_index_from_csv(temp_path, original_name=uploaded.filename)
    except Exception as exc:  # noqa: BLE001
        rag_error = str(exc)
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    _release_models()
    return _render_rag_page(rag_state=rag_state, rag_error=rag_error)


@app.route("/rag/search", methods=["POST"])
def rag_search():
    rag_state = rag_current_state()
    rag_results = None
    rag_error = None

    query = request.form.get("rag_query", "")
    try:
        top_n_raw = request.form.get("rag_top_n", "5")
        threshold_raw = request.form.get("rag_threshold", "0.35")
        top_n = max(1, min(50, int(top_n_raw)))
        threshold = max(0.0, min(1.0, float(threshold_raw)))
        rag_results = query_rag(query, top_n=top_n, min_similarity=threshold)
    except Exception as exc:  # noqa: BLE001
        rag_error = str(exc)

    _release_models()
    return _render_rag_page(rag_state=rag_state, rag_results=rag_results, rag_error=rag_error)


@app.route("/visualization", methods=["GET"])
def visualization_home():
    return render_template("visualization.html", results=None, viz_error=None, viz_share=20)


@app.route("/visualization/analyze", methods=["POST"])
def visualization_analyze():
    uploaded = request.files.get("viz_file")
    if uploaded is None or uploaded.filename == "":
        return render_template(
            "visualization.html",
            results=None,
            viz_error="Merci de selectionner un fichier CSV valide.",
            viz_share=20,
        )

    temp_path = None
    try:
        share_raw = request.form.get("viz_share", "20")
        try:
            share = float(share_raw)
        except ValueError:
            raise ValueError("Pourcentage invalide.")
        if not (0 < share <= 100):
            raise ValueError("Le pourcentage doit etre entre 1 et 100.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        output = build_chart_payload_from_csv(temp_path, sample_percent=share)
    except Exception as exc:  # noqa: BLE001
        return render_template(
            "visualization.html",
            results=None,
            viz_error=f"Erreur pendant le chargement : {exc}",
            viz_share=20,
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)

    return render_template("visualization.html", results=output, viz_error=None, viz_share=share)


@app.route("/short-reviews", methods=["GET"])
def short_reviews_home():
    return render_template("short_reviews.html", results=None, short_error=None)


@app.route("/short-reviews/analyze", methods=["POST"])
def short_reviews_analyze():
    logs: List[str] = []
    uploaded = request.files.get("short_file")
    if uploaded is None or uploaded.filename == "":
        _release_models()
        return render_template(
            "short_reviews.html",
            results=None,
            short_error="Merci de selectionner un fichier CSV valide.",
        )

    logs.append(f"Fichier recu: {uploaded.filename}")
    print(f"[short] Fichier recu: {uploaded.filename}", flush=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        logs.append(f"Fichier temporaire enregistre: {temp_path}")
        output = run_short_review_exploration(temp_path)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Erreur pendant le traitement: {exc}")
        print(f"[short] Erreur: {exc}", flush=True)
        _release_models()
        return render_template(
            "short_reviews.html",
            results=None,
            short_error=f"Erreur pendant le traitement : {exc}",
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
            logs.append("Fichier temporaire supprime.")

    result_df = output["result_df"]
    _save_step_csv(result_df, STEP_FILES["short-reviews"])
    results = {
        "download_uri": _step_download_url("short-reviews"),
        "preview": output["preview"],
        "preview_columns": output["preview_columns"],
        "length_counts": output["length_counts"],
        "short_label_counts": output["short_label_counts"],
        "short_word_counts": output["short_word_counts"],
        "short_rating_counts": output["short_rating_counts"],
        "short_app_counts": output["short_app_counts"],
        "short_samples": output["short_samples"],
        "num_reviews": output["num_reviews"],
        "short_count": output["short_count"],
        "long_count": output["long_count"],
        "short_share": output["short_share"],
        "logs": logs + output.get("logs", []),
    }

    _release_models()
    return render_template("short_reviews.html", results=results, short_error=None)


@app.route("/sentiment", methods=["GET"])
def sentiment_home():
    return render_template("sentiment.html", results=None, sentiment_error=None)


@app.route("/sentiment/analyze", methods=["POST"])
def sentiment_analyze():
    logs: List[str] = []
    uploaded = request.files.get("sentiment_file")
    if uploaded is None or uploaded.filename == "":
        _release_models()
        return render_template(
            "sentiment.html",
            results=None,
            sentiment_error="Merci de selectionner un fichier CSV valide.",
        )

    logs.append(f"Fichier recu: {uploaded.filename}")
    print(f"[sentiment] Fichier recu: {uploaded.filename}", flush=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        logs.append(f"Fichier temporaire enregistre: {temp_path}")
        output = run_sentiment_analysis(temp_path)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Erreur pendant le traitement: {exc}")
        print(f"[sentiment] Erreur: {exc}", flush=True)
        _release_models()
        return render_template(
            "sentiment.html",
            results=None,
            sentiment_error=f"Erreur pendant le traitement : {exc}",
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
            logs.append("Fichier temporaire supprime.")

    result_df = output["result_df"]
    _save_step_csv(result_df, STEP_FILES["sentiment"])
    results = {
        "download_uri": _step_download_url("sentiment"),
        "preview": output["preview"],
        "preview_columns": output["preview_columns"],
        "sentiment_counts": output["sentiment_counts"],
        "confidence_bins": output["confidence_bins"],
        "low_conf_samples": output["low_conf_samples"],
        "num_reviews": output["num_reviews"],
        "avg_conf": output["avg_conf"],
        "fallback_count": output["fallback_count"],
        "fallback_share": output["fallback_share"],
        "model_name": output["model_name"],
        "batch_size": output["batch_size"],
        "device": output["device"],
        "conf_threshold": output["conf_threshold"],
        "logs": logs + output.get("logs", []),
    }

    _release_models()
    return render_template("sentiment.html", results=results, sentiment_error=None)


@app.route("/taxonomies", methods=["GET"])
def taxonomies_home():
    return render_template("taxonomies.html", results=None, taxo_error=None)


@app.route("/taxonomies/analyze", methods=["POST"])
def taxonomies_analyze():
    logs: List[str] = []
    action = request.form.get("action", "classify")
    file_key = "taxonomy_file" if action == "classify" else "taxonomy_existing"
    uploaded = request.files.get(file_key)
    if uploaded is None or uploaded.filename == "":
        _release_models()
        return render_template(
            "taxonomies.html",
            results=None,
            taxo_error="Merci de selectionner un fichier CSV valide.",
        )

    logs.append(f"Fichier recu: {uploaded.filename}")
    print(f"[taxonomies] Fichier recu: {uploaded.filename}", flush=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            temp_path = Path(tmp.name)
            uploaded.save(temp_path)
        logs.append(f"Fichier temporaire enregistre: {temp_path}")
        if action == "load_existing":
            output = summarize_taxonomy_csv(temp_path)
        else:
            output = run_taxonomy_classification(temp_path)
    except Exception as exc:  # noqa: BLE001
        logs.append(f"Erreur pendant le traitement: {exc}")
        print(f"[taxonomies] Erreur: {exc}", flush=True)
        _release_models()
        return render_template(
            "taxonomies.html",
            results=None,
            taxo_error=f"Erreur pendant le traitement : {exc}",
        )
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)
            logs.append("Fichier temporaire supprime.")

    result_df = output["result_df"]
    _save_step_csv(result_df, STEP_FILES["taxonomies"])
    results = {
        "download_uri": _step_download_url("taxonomies"),
        "category_counts": output["category_counts"],
        "main_counts": output["main_counts"],
        "taxonomy_breakdown": output["taxonomy_breakdown"],
        "rating_breakdown": output["rating_breakdown"],
        "weekly_trends": output["weekly_trends"],
        "preview": output["preview"],
        "preview_columns": output["preview_columns"],
        "num_reviews": output["num_reviews"],
        "num_categories": output["num_categories"],
        "model_name": output["model_name"],
        "batch_size": output["batch_size"],
        "device": output["device"],
        "logs": logs + output.get("logs", []),
    }

    _release_models()
    return render_template("taxonomies.html", results=results, taxo_error=None)


@app.route("/update-dataset", methods=["GET"])
def update_dataset_home():
    status = get_update_status()
    return render_template("update_dataset.html", status=status, results=None, update_error=None)


@app.route("/update-dataset/run", methods=["POST"])
def update_dataset_run():
    update_error = None
    results = None
    status = get_update_status()
    action = request.form.get("update_action", "update")
    google_country = (request.form.get("google_country") or "").strip()
    google_lang = (request.form.get("google_lang") or "").strip()
    target_year_raw = (request.form.get("target_year") or "").strip()
    target_year = None
    if target_year_raw:
        try:
            target_year = int(target_year_raw)
        except ValueError:
            update_error = "Annee cible invalide. Merci de fournir une annee numerique."
            return render_template("update_dataset.html", status=status, results=None, update_error=update_error)
    uploaded = request.files.get("dataset_file")
    temp_path = None
    try:
        step_path = _ensure_data_dir() / STEP_FILES["update-dataset"]
        if action == "create":
            results = run_update(
                step_path,
                source_name=STEP_FILES["update-dataset"],
                google_country=google_country or None,
                google_lang=google_lang or None,
                target_year=target_year,
            )
        else:
            if uploaded is None or uploaded.filename == "":
                raise ValueError("Merci de selectionner un fichier CSV.")
            suffix = Path(uploaded.filename).suffix or ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)
                uploaded.save(temp_path)
            _save_step_file(temp_path, STEP_FILES["update-dataset"])
            results = run_update(
                step_path,
                source_name=uploaded.filename,
                google_country=google_country or None,
                google_lang=google_lang or None,
                target_year=target_year,
            )
        status = get_update_status()
    except Exception as exc:  # noqa: BLE001
        update_error = f"Erreur pendant la mise a jour : {exc}"
    finally:
        if temp_path is None and uploaded is not None:
            try:
                uploaded.close()
            except Exception:
                pass
    return render_template("update_dataset.html", status=status, results=results, update_error=update_error)


@app.route("/update-dataset/download", methods=["GET"])
def update_dataset_download():
    path = _ensure_data_dir() / STEP_FILES["update-dataset"]
    if not path.exists():
        fallback = get_output_path()
        if not fallback.exists():
            abort(404)
        path = _save_step_file(fallback, STEP_FILES["update-dataset"])
    return send_file(path, as_attachment=True, download_name=STEP_FILES["update-dataset"])


@app.route("/download/<step_key>", methods=["GET"])
def download_step(step_key: str):
    filename = STEP_FILES.get(step_key)
    if filename is None:
        abort(404)
    path = _ensure_data_dir() / filename
    if not path.exists():
        abort(404)
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == "__main__":
    # Désactive le reloader pour conserver les logs dans un seul process.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
