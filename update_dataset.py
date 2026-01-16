import pandas as pd
import requests
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional
from google_play_scraper import Sort, reviews
import os

# --- CONFIGURATION ---
OUTPUT_FILE = 'INORBIT_Master_Dataset_Final_with_id.csv'
TARGET_YEAR = int(os.getenv("TARGET_YEAR", "2025"))
DEFAULT_START_DATE = datetime(2025, 1, 1)
GOOGLE_DEFAULT_COUNTRY = os.getenv("GOOGLE_COUNTRY", "us")
GOOGLE_DEFAULT_LANG = os.getenv("GOOGLE_LANG", "en")
GOOGLE_SLEEP_MIN = float(os.getenv("GOOGLE_SLEEP_MIN", "1.0"))
GOOGLE_SLEEP_MAX = float(os.getenv("GOOGLE_SLEEP_MAX", "3.0"))

# Google Play Apps
GOOGLE_APPS = [
    {'name': 'Tinder', 'id': 'com.tinder'},
    {'name': 'Bumble', 'id': 'com.bumble.app'},
    {'name': 'Hinge', 'id': 'co.hinge.app'},
]

# Apple Apps
APPLE_APPS = [
    {'name': 'Tinder', 'id': '547702041'},
    {'name': 'Hinge', 'id': '595287172'},
]
APPLE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def _log(message: str, logs: Optional[List[str]] = None) -> None:
    if logs is not None:
        logs.append(message)
    print(message, flush=True)

def _safe_row_count(filepath: str) -> int:
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, "rb") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except Exception:
        try:
            return int(pd.read_csv(filepath).shape[0])
        except Exception:
            return 0

def get_latest_date(filepath, logs: Optional[List[str]] = None):
    """Reads the existing dataset and returns the most recent date."""
    if not os.path.exists(filepath):
        _log(f"File {filepath} not found. Starting from {DEFAULT_START_DATE}.", logs)
        return DEFAULT_START_DATE
    
    try:
        df = pd.read_csv(filepath)
        if 'date' not in df.columns:
             _log("Column 'date' not found. Starting from default.", logs)
             return DEFAULT_START_DATE
        
        # Ensure UTC
        df['date'] = pd.to_datetime(df['date'], utc=True)
        max_date = df['date'].max()
        
        # If NaT, fallback
        if pd.isna(max_date):
            return DEFAULT_START_DATE
            
        # Return as python datetime (UTC)
        return max_date.to_pydatetime()
    except Exception as e:
        _log(f"Error reading existing file: {e}. Starting from default.", logs)
        return DEFAULT_START_DATE

def fetch_google_reviews(app_config, target_year, country, lang, logs: Optional[List[str]] = None):
    """Fetches Google Play reviews for a specific year."""
    _log(
        f"--- Google Play: Scraping {app_config['name']} ({app_config['id']}) "
        f"| country={country} lang={lang} year={target_year} ---",
        logs,
    )
    all_reviews = []
    continuation_token = None
    stop_scraping = False
    page_count = 0

    while not stop_scraping:
        page_count += 1
        try:
            result, continuation_token = reviews(
                app_config['id'],
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=100,
                continuation_token=continuation_token
            )
        except Exception as e:
            _log(f"Error fetching Google reviews: {e}", logs)
            break

        if not result:
            break

        for review in result:
            review_date = review['at']

            # 1) Ignore future dates
            if review_date.year > target_year:
                continue

            # 2) Stop when we reach older years
            if review_date.year < target_year:
                stop_scraping = True
                break

            # 3) Keep only target year
            row = {
                'review_id': review.get('reviewId'),
                'date': review_date,
                'source': 'Google Play',
                'app_name': app_config['name'],
                'country': country,
                'rating': review.get('score'),
                'text': review.get('content'),
                'version': review.get('reviewCreatedVersion'),
                'reply_content': review.get('replyContent'),
                'reply_date': review.get('repliedAt')
            }
            all_reviews.append(row)

        _log(f"   -> Page {page_count}: analyzed {len(result)} | kept {len(all_reviews)}.", logs)
        if not stop_scraping:
             time.sleep(random.uniform(GOOGLE_SLEEP_MIN, GOOGLE_SLEEP_MAX)) # Respectful pause

        if continuation_token is None:
            break

    _log(f"   -> Found {len(all_reviews)} reviews for {target_year}.", logs)
    return all_reviews

def fetch_apple_reviews(
    app_config,
    target_year,
    country,
    lang,
    logs: Optional[List[str]] = None,
):
    """Fetches Apple Store reviews for a specific year."""
    _log(
        f"--- Apple Store: Scraping {app_config['name']} ({app_config['id']}) "
        f"| country={country} lang={lang} year={target_year} ---",
        logs,
    )
    master_list = []
    page = 1
    stop_scraping = False

    while not stop_scraping:
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={app_config['id']}/sortby=mostrecent/json"
        )
        if lang:
            url = f"{url}?l={lang}"

        try:
            response = requests.get(url, headers=APPLE_HEADERS, timeout=10)
            if response.status_code != 200:
                break

            data = response.json()
            feed = data.get('feed', {})
            entries = feed.get('entry', [])

            if isinstance(entries, dict):
                entries = [entries]
            if not entries:
                break

            for entry in entries:
                try:
                    dt_str = entry['updated']['label']
                    dt_obj = pd.to_datetime(dt_str, utc=True).to_pydatetime()

                    if dt_obj.year > target_year:
                        continue
                    if dt_obj.year < target_year:
                        stop_scraping = True
                        break

                    review = {
                        'source': 'Apple Store',
                        'app_name': app_config['name'],
                        'country': country,
                        'review_id': entry['id']['label'],
                        'rating': int(entry['im:rating']['label']),
                        'text': entry['content']['label'],
                        'date': dt_obj,
                        'version': entry['im:version']['label'] if 'im:version' in entry else 'unknown',
                        'reply_content': None,
                        'reply_date': None
                    }
                    master_list.append(review)
                except KeyError:
                    continue

            page += 1
            time.sleep(random.uniform(0.5, 1.0))
        except Exception as e:
            _log(f"   Error: {e}", logs)
            break

    _log(f"   -> Found {len(master_list)} reviews for {target_year}.", logs)
    return master_list

def update_database(
    logs: Optional[List[str]] = None,
    google_country: Optional[str] = None,
    google_lang: Optional[str] = None,
    target_year: Optional[int] = None,
) -> Dict[str, Any]:
    if logs is None:
        logs = []
    _log("=== STARTING DATASET UPDATE ===", logs)

    resolved_country = (google_country or GOOGLE_DEFAULT_COUNTRY).strip().lower()
    if not resolved_country:
        resolved_country = GOOGLE_DEFAULT_COUNTRY
    resolved_lang = (google_lang or GOOGLE_DEFAULT_LANG).strip().lower()
    if not resolved_lang:
        resolved_lang = GOOGLE_DEFAULT_LANG
    resolved_year = int(target_year or TARGET_YEAR)
    _log(
        f"Google Play config: country={resolved_country} lang={resolved_lang} year={resolved_year}",
        logs,
    )
    
    # 1. Get Latest Date
    existing_rows = _safe_row_count(OUTPUT_FILE)
    last_date = get_latest_date(OUTPUT_FILE, logs)
    _log(f"Latest collected review date: {last_date}", logs)
    if existing_rows == 0:
        _log("No existing dataset detected. First run may take a while.", logs)
    
    new_data = []
    
    # 2. Fetch Google
    for app in GOOGLE_APPS:
        reviews_data = fetch_google_reviews(app, resolved_year, resolved_country, resolved_lang, logs)
        new_data.extend(reviews_data)
            
    # 3. Fetch Apple
    for app in APPLE_APPS:
        reviews_data = fetch_apple_reviews(
            app,
            resolved_year,
            resolved_country,
            resolved_lang,
            logs,
        )
        new_data.extend(reviews_data)
        
    _log(f"\nTotal new reviews found: {len(new_data)}", logs)
    
    if not new_data:
        _log("No new reviews to append.", logs)
        if os.path.exists(OUTPUT_FILE):
            try:
                existing_rows = len(pd.read_csv(OUTPUT_FILE))
            except Exception:
                pass
        return {
            "output_file": OUTPUT_FILE,
            "existing_rows": existing_rows,
            "new_reviews": 0,
            "duplicates_removed": 0,
            "final_rows": existing_rows,
            "net_change": 0,
            "last_date": last_date,
            "source_counts": {},
            "app_counts": {},
            "google_country": resolved_country,
            "google_lang": resolved_lang,
            "target_year": resolved_year,
            "apple_country": resolved_country,
            "apple_lang": resolved_lang,
            "apple_year": resolved_year,
            "updated": False,
        }

    # 4. Process New Data
    df_new = pd.DataFrame(new_data)
    
    # Keep only relevant columns
    cols_to_keep = ['review_id', 'date', 'source', 'app_name', 'country', 'rating', 'text', 'version', 'reply_content', 'reply_date']
    # Ensure they exist
    for col in cols_to_keep:
        if col not in df_new.columns:
            df_new[col] = None
            
    df_new = df_new[cols_to_keep].copy()
    
    # Normalize Dates
    df_new['date'] = pd.to_datetime(df_new['date'], utc=True)
    df_new['reply_date'] = pd.to_datetime(df_new['reply_date'], utc=True)

    source_counts = {}
    app_counts = {}
    if "source" in df_new.columns:
        source_counts = df_new["source"].value_counts().to_dict()
    if "app_name" in df_new.columns:
        app_counts = df_new["app_name"].value_counts().to_dict()
            
    # 6. Append to Master
    if os.path.exists(OUTPUT_FILE):
        df_old = pd.read_csv(OUTPUT_FILE)
        # Ensure dates are compatible for sorting
        df_old['date'] = pd.to_datetime(df_old['date'], utc=True)
        existing_rows = len(df_old)
        # Re-save reply_date as dt
        if 'reply_date' in df_old.columns:
            df_old['reply_date'] = pd.to_datetime(df_old['reply_date'], utc=True)
            
        df_final = pd.concat([df_new, df_old], ignore_index=True)
    else:
        df_final = df_new

    # 7. Final Clean & Save
    df_final = df_final.sort_values(by='date', ascending=False)
    
    # --- HANDLING DUPLICATES ---
    # We remove duplicates based on 'review_id'. 
    # If review_id is missing (older data), we fallback to 'date', 'app_name', 'text'.
    # keep='first' keeps the most recent/newest version (since we just sorted)
    if 'review_id' in df_final.columns:
        initial_len = len(df_final)
        # First pass: Deduplicate by ID where ID exists
        df_final = df_final.drop_duplicates(subset=['review_id'], keep='first')
        
        # Second pass: For rows without ID (if any remain from old data), or just safety
        # We can also dedup by content signature
        df_final = df_final.drop_duplicates(subset=['date', 'app_name', 'text'], keep='first')
        
        duplicates_removed = initial_len - len(df_final)
        _log(f"Duplicates removed: {duplicates_removed}", logs)
    else:
        # Fallback if review_id is somehow missing
        df_final = df_final.drop_duplicates(subset=['date', 'app_name', 'text'], keep='first')
        duplicates_removed = 0
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    final_rows = len(df_final)
    net_change = final_rows - existing_rows
    _log(f"Database updated. New total rows: {final_rows}", logs)

    return {
        "output_file": OUTPUT_FILE,
        "existing_rows": existing_rows,
        "new_reviews": len(df_new),
        "duplicates_removed": duplicates_removed,
        "final_rows": final_rows,
        "net_change": net_change,
        "last_date": last_date,
        "source_counts": source_counts,
        "app_counts": app_counts,
        "google_country": resolved_country,
        "google_lang": resolved_lang,
        "target_year": resolved_year,
        "apple_country": resolved_country,
        "apple_lang": resolved_lang,
        "apple_year": resolved_year,
        "updated": True,
    }

if __name__ == "__main__":
    update_database()
