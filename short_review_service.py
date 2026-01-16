from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import re

import pandas as pd

from clustering_service import read_csv_safely
from length_utils import add_length_column

POSITIVE_WORDS = {
    "good", "great", "awesome", "love", "lovely", "amazing", "perfect",
    "nice", "cool", "best", "super", "excellent", "excelent", "fun",
    "fantastic", "wonderful", "brilliant", "exceptional", "remarkable",
    "phenomenal", "splendid", "spectacular", "stunning", "superb",
    "flawless", "impressive", "terrific", "fabulous", "astonishing",
    "unbelievable", "beautiful", "charming", "delightful", "like",
    "liked", "enjoy", "enjoyed", "satisfying", "satisfied", "pleased",
    "happy", "joy", "joyful", "glad", "grateful", "thankful", "appreciate",
    "impressed", "cute", "pleasant", "positive", "optimistic", "reliable",
    "fast", "smooth", "responsive", "stable", "efficient", "effective",
    "powerful", "solid", "robust", "polished", "intuitive", "userfriendly",
    "seamless", "easy", "simple", "clean", "neat", "smart", "clever",
    "worth", "worthwhile", "worthit", "valuable", "affordable", "fair",
    "recommended", "recommend", "trusted", "trustworthy", "legit",
    "calm", "relaxed", "peaceful", "content", "improving", "improved",
    "fixed", "resolved", "success", "successful", "thanks", "thankyou",
    "thank-you", "thx", "ty", "secure", "safe",
    "not bad", "not too bad", "not terrible", "not awful", "not horrible",
    "no problem", "no problems", "no issue", "no issues", "no worries",
    "no complaints", "nothing wrong", "nothing bad", "cant complain",
    "not disappointed", "not unhappy", "not a scam", "not a problem",
    "no lag", "no crash", "no bugs", "without issues",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "worst", "bug", "bugs",
    "crash", "crashes", "lag", "laggy", "trash", "sucks", "suck",
    "poor", "broken", "error", "errors", "scam", "horrible",
    "dreadful", "abysmal", "pathetic", "useless", "worthless",
    "unusable", "unplayable", "garbage", "junk", "crap", "crappy",
    "atrocious", "nightmare", "horrendous", "ridiculous", "unacceptable",
    "unforgivable", "buggy", "glitch", "freeze", "stuck", "slow",
    "sluggish", "unresponsive", "delay", "disconnect", "timeout",
    "corrupt", "overpriced", "pricey", "expensive", "ripoff", "fraud",
    "misleading", "dishonest", "shady", "waste", "rude", "unhelpful",
    "ignored", "sad", "upset", "angry", "mad", "annoyed", "irritated",
    "regret", "unhappy", "stressful", "neveragain", "avoid", "skip",
    "stayaway", "worse", "downgrade", "broke", "ruined", "failed",
    "pointless", "disappointing", "disappointed", "mediocre", "lame",
    "refund", "refused", "canceled", "complain", "complaints",
    "not good", "not great", "not awesome", "not amazing", "not perfect",
    "not nice", "not ok", "not happy", "not satisfied", "not impressed",
    "not worth", "not recommended", "not reliable", "not responsive",
    "no help", "no support", "no refund", "no response", "no service",
    "not working", "doesnt work", "didnt work", "dont work", "cant use",
    "dont like", "not like", "not as expected", "no longer works",
}

POSITIVE_PHRASES = {token for token in POSITIVE_WORDS if " " in token}
NEGATIVE_PHRASES = {token for token in NEGATIVE_WORDS if " " in token}
POSITIVE_TOKENS = {token for token in POSITIVE_WORDS if " " not in token}
NEGATIVE_TOKENS = {token for token in NEGATIVE_WORDS if " " not in token}


def remove_emojis(text: str) -> str:
    if not isinstance(text, str):
        return text

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\u2600-\u26FF"
        u"\u2700-\u27BF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = remove_emojis(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\s\.,!?\'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def short_text_sentiment_label(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "uncertain"

    cleaned = basic_clean(text)
    if not cleaned:
        return "uncertain"

    words = cleaned.split()
    if len(words) >= 4:
        return "uncertain"

    normalized = f" {' '.join(words)} "
    pos = sum(1 for token in POSITIVE_TOKENS if token in words)
    neg = sum(1 for token in NEGATIVE_TOKENS if token in words)
    pos += sum(1 for token in POSITIVE_PHRASES if f" {token} " in normalized)
    neg += sum(1 for token in NEGATIVE_PHRASES if f" {token} " in normalized)

    if pos > 0 and neg == 0:
        return "positif"
    if neg > 0 and pos == 0:
        return "negatif"
    return "uncertain"


def _safe_counts(series: pd.Series, key_name: str) -> List[Dict[str, Any]]:
    return [
        {key_name: key, "count": int(value)}
        for key, value in series.items()
    ]


def run_short_review_exploration(
    csv_path: Path,
    text_column: str = "text",
) -> Dict[str, Any]:
    logs: List[str] = []
    logs.append(f"Lecture du CSV: {csv_path}")
    df = read_csv_safely(csv_path, logs)
    if text_column not in df.columns:
        raise ValueError(f"Missing required column '{text_column}'.")

    df[text_column] = df[text_column].fillna("")
    working = add_length_column(df, text_column=text_column, logs=logs)

    short_df = working[working["longueur"] == 0].copy()
    short_df["short_label"] = short_df[text_column].apply(short_text_sentiment_label)
    short_df["short_word_count"] = short_df[text_column].apply(
        lambda value: len(str(value).strip().split()) if isinstance(value, str) else 0
    )

    total_reviews = len(working)
    short_count = int((working["longueur"] == 0).sum())
    long_count = int((working["longueur"] == 1).sum())
    short_share = round((short_count / total_reviews) * 100, 2) if total_reviews else 0.0

    length_counts = [
        {"length": 0, "count": short_count},
        {"length": 1, "count": long_count},
    ]
    label_counts = _safe_counts(
        short_df["short_label"].value_counts().sort_index(), "label"
    )
    word_counts = _safe_counts(
        short_df["short_word_count"].value_counts().sort_index(), "words"
    )

    rating_counts = None
    if "rating" in short_df.columns:
        rating_counts = _safe_counts(
            short_df["rating"].value_counts().sort_index(), "rating"
        )

    app_counts = None
    if "app_name" in short_df.columns:
        app_counts = _safe_counts(
            short_df["app_name"].value_counts().head(12), "app_name"
        )

    preview_columns: List[str] = []
    for col in (text_column, "longueur", "rating", "app_name", "date"):
        if col in working.columns and col not in preview_columns:
            preview_columns.append(col)
    if not preview_columns:
        preview_columns = working.columns[:5].tolist()
    preview = working[preview_columns].head(100).to_dict(orient="records")

    sample_columns = [text_column, "short_label"]
    if "rating" in short_df.columns:
        sample_columns.append("rating")
    if "app_name" in short_df.columns:
        sample_columns.append("app_name")
    if "date" in short_df.columns:
        sample_columns.append("date")

    if len(short_df) > 0:
        sample_count = min(20, len(short_df))
        short_samples = (
            short_df.sample(n=sample_count, random_state=42, replace=False)[sample_columns]
            .to_dict(orient="records")
        )
    else:
        short_samples = []

    return {
        "result_df": working,
        "preview": preview,
        "preview_columns": preview_columns,
        "length_counts": length_counts,
        "short_label_counts": label_counts,
        "short_word_counts": word_counts,
        "short_rating_counts": rating_counts,
        "short_app_counts": app_counts,
        "short_samples": short_samples,
        "num_reviews": total_reviews,
        "short_count": short_count,
        "long_count": long_count,
        "short_share": short_share,
        "logs": logs,
    }
