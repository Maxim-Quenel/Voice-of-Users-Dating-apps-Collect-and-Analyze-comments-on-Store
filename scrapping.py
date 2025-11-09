#!/usr/bin/env python3
"""
telecharger_avis_tinder_2025_pagine.py

But : Récupérer uniquement les avis Tinder (Play Store) publiés en 2025,
de façon efficace (scraping paginé avec arrêt dès qu'on quitte l'année 2025).

Prérequis :
    pip install google-play-scraper pandas

Usage :
    python3 telecharger_avis_tinder_2025_pagine.py
"""

import time
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from google_play_scraper import reviews, Sort
import pandas as pd

# --- Configuration ---
APP_PACKAGE = "com.tinder"
OUTPUT_DIR = Path(r"D:\5A\S1\PI2\scraping_3\avis_en")

OUTPUT_JSON = OUTPUT_DIR / "tinder_reviews_2025.json"
OUTPUT_CSV = OUTPUT_DIR / "tinder_reviews_2025.csv"

LANG = "gb"
COUNTRY = "gb"
SLEEP_MS = 1000
PAGE_SIZE = 200  # Nombre d'avis par requête (max autorisé)

# --- Fonctions utilitaires ---
def fetch_reviews_until_2025(app_package: str,
                             lang: str,
                             country: str,
                             sleep_milliseconds: int = 0,
                             page_size: int = 200) -> List[Dict[str, Any]]:
    """
    Récupère les avis page par page jusqu'à ce qu'on tombe sur un avis d'avant 2025.
    """
    all_reviews: List[Dict[str, Any]] = []
    continuation_token = None
    page_count = 0
    stop = False

    print(f"Début de la récupération paginée des avis Tinder (lang={lang}, country={country})...")
    start_time = time.time()

    while not stop:
        result, continuation_token = reviews(
            app_package,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=page_size,
            continuation_token=continuation_token
        )

        if not result:
            print("Aucun nouvel avis, arrêt.")
            break

        page_count += 1
        print(f"Page {page_count}: {len(result)} avis récupérés.")

        for r in result:
            if not isinstance(r.get("at"), datetime):
                continue
            review_date = r["at"]
            if review_date.year == 2025:
                all_reviews.append(r)
            else:
                # Dès qu'on trouve un avis avant 2025, on arrête
                stop = True
                print(f"Arrêt : avis de {review_date.date()} détecté (avant 2025).")
                break

        if continuation_token is None:
            print("Fin des pages.")
            break

        # Petite pause pour limiter le risque de blocage
        if sleep_milliseconds > 0:
            time.sleep(sleep_milliseconds / 1000)

    elapsed = time.time() - start_time
    print(f"Terminé après {page_count} pages, {len(all_reviews)} avis 2025 récupérés en {elapsed:.1f}s.")
    return all_reviews


def save_json(reviews: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2, default=str)
    print(f"Sauvegardé JSON -> {path}")


def save_csv(reviews: List[Dict[str, Any]], path: Path) -> None:
    if not reviews:
        print("Aucun avis à sauvegarder en CSV.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reviewId", "userName", "userImage", "content", "score",
        "thumbsUpCount", "reviewCreatedVersion", "at", "replyContent", "repliedAt"
    ]

    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in reviews:
            row = {k: r.get(k) for k in fieldnames}
            if isinstance(row.get("at"), datetime):
                row["at"] = row["at"].isoformat()
            if isinstance(row.get("repliedAt"), datetime):
                row["repliedAt"] = row["repliedAt"].isoformat()
            writer.writerow(row)
    print(f"Sauvegardé CSV -> {path}")


def main():
    try:
        reviews_2025 = fetch_reviews_until_2025(
            APP_PACKAGE,
            lang=LANG,
            country=COUNTRY,
            sleep_milliseconds=SLEEP_MS,
            page_size=PAGE_SIZE
        )
    except Exception as e:
        print("Erreur lors de la récupération :", e)
        reviews_2025 = []

    if reviews_2025:
        save_json(reviews_2025, OUTPUT_JSON)
        df = pd.json_normalize(reviews_2025)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    else:
        print("Aucun avis trouvé pour 2025.")


if __name__ == "__main__":
    main()
