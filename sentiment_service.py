from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from clustering_service import read_csv_safely


SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "nlptown/bert-base-multilingual-uncased-sentiment",
)
SENTIMENT_TOKENIZER_NAME = os.getenv("SENTIMENT_TOKENIZER_NAME", SENTIMENT_MODEL_NAME)
SENTIMENT_BATCH_SIZE = int(os.getenv("SENTIMENT_BATCH_SIZE", "32"))
SENTIMENT_MAX_LENGTH = int(os.getenv("SENTIMENT_MAX_LENGTH", "256"))
SENTIMENT_CONF_THRESHOLD = float(os.getenv("SENTIMENT_CONF_THRESHOLD", "0.55"))

_MODEL = None
_TOKENIZER = None
_VADER = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _log(logs: List[str], message: str) -> None:
    logs.append(message)
    print(f"[sentiment] {message}", flush=True)


def _get_model(logs: List[str] | None = None):
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        message = f"Loading sentiment model {SENTIMENT_MODEL_NAME} on {_DEVICE}..."
        if logs is not None:
            _log(logs, message)
        else:
            print(f"[sentiment] {message}", flush=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(SENTIMENT_TOKENIZER_NAME)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        _MODEL.to(_DEVICE)
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def _get_vader() -> SentimentIntensityAnalyzer:
    global _VADER
    if _VADER is None:
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"').replace("â€“", "-")
    text = text.replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _infer_label_groups(labels: List[str]) -> Dict[str, List[int]]:
    normalized = [re.sub(r"[^a-z0-9 ]+", "", label.lower()) for label in labels]

    star_map: Dict[int, int] = {}
    for idx, label in enumerate(normalized):
        if "star" in label:
            digits = re.findall(r"\d+", label)
            if digits:
                star_map[idx] = int(digits[0])
    if star_map:
        negatives = [idx for idx, star in star_map.items() if star <= 2]
        neutrals = [idx for idx, star in star_map.items() if star == 3]
        positives = [idx for idx, star in star_map.items() if star >= 4]
        if negatives and neutrals and positives:
            return {"negative": negatives, "neutral": neutrals, "positive": positives}

    negatives = [idx for idx, label in enumerate(normalized) if "neg" in label]
    neutrals = [idx for idx, label in enumerate(normalized) if "neu" in label]
    positives = [idx for idx, label in enumerate(normalized) if "pos" in label]

    if negatives and positives and not neutrals and len(labels) == 3:
        remaining = list({0, 1, 2} - set(negatives) - set(positives))
        if remaining:
            neutrals = remaining

    if negatives and neutrals and positives:
        return {"negative": negatives, "neutral": neutrals, "positive": positives}

    if len(labels) == 3:
        return {"negative": [0], "neutral": [1], "positive": [2]}

    raise ValueError("Unable to map model labels to negative/neutral/positive groups.")


def _predict_transformer(texts: List[str], logs: List[str]) -> Dict[str, List[float]]:
    model, tokenizer = _get_model(logs)
    id2label = getattr(model.config, "id2label", {}) or {}
    labels = [id2label.get(i, str(i)) for i in range(model.config.num_labels)]
    groups = _infer_label_groups(labels)
    _log(logs, f"Label groups -> neg={groups['negative']} neu={groups['neutral']} pos={groups['positive']}")
    _log(logs, f"Batch size={SENTIMENT_BATCH_SIZE}, max_len={SENTIMENT_MAX_LENGTH}, device={_DEVICE}.")

    all_conf: List[float] = []
    all_pred: List[int] = []

    for start in range(0, len(texts), SENTIMENT_BATCH_SIZE):
        batch = texts[start:start + SENTIMENT_BATCH_SIZE]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SENTIMENT_MAX_LENGTH,
        )
        encoded = {key: value.to(_DEVICE) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        probs = F.softmax(outputs.logits, dim=-1).cpu()

        neg = probs[:, groups["negative"]].sum(dim=1)
        neu = probs[:, groups["neutral"]].sum(dim=1)
        pos = probs[:, groups["positive"]].sum(dim=1)
        stacked = torch.stack([neg, neu, pos], dim=1)
        conf, idx = torch.max(stacked, dim=1)

        all_conf.extend(conf.tolist())
        all_pred.extend([-1 if i == 0 else 0 if i == 1 else 1 for i in idx.tolist()])

        if start == 0 or (start // SENTIMENT_BATCH_SIZE) % 10 == 0:
            _log(logs, f"Processed {min(start + SENTIMENT_BATCH_SIZE, len(texts))}/{len(texts)} reviews.")

    return {"pred": all_pred, "conf": all_conf}


def _vader_label(text: str) -> int:
    analyzer = _get_vader()
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return 1
    if compound <= -0.05:
        return -1
    return 0


def run_sentiment_analysis(csv_path: Path, text_column: str = "text") -> Dict[str, Any]:
    logs: List[str] = []
    _log(logs, f"Reading CSV: {csv_path}")
    df = read_csv_safely(csv_path, logs)
    if text_column not in df.columns:
        raise ValueError(f"Missing required column '{text_column}'.")

    df[text_column] = df[text_column].fillna("")
    texts = df[text_column].apply(_clean_text).tolist()
    if not texts:
        raise ValueError("No valid texts to analyze.")

    _log(logs, f"{len(texts)} reviews loaded for sentiment analysis.")
    transformer_out = _predict_transformer(texts, logs)
    bert_pred = transformer_out["pred"]
    bert_conf = transformer_out["conf"]

    if len(bert_pred) != len(df):
        raise RuntimeError("Mismatch between predictions and input rows.")

    conf_series = pd.Series(bert_conf)
    low_conf_mask = conf_series < SENTIMENT_CONF_THRESHOLD
    sentiment_pred = list(bert_pred)

    if low_conf_mask.any():
        _log(logs, f"Applying VADER fallback to {int(low_conf_mask.sum())} low-confidence rows.")
        for idx in low_conf_mask[low_conf_mask].index.tolist():
            sentiment_pred[idx] = _vader_label(texts[idx])

    label_map = {-1: "negatif", 0: "neutre", 1: "positif"}
    df["sentiment_pred"] = sentiment_pred
    df["sentiment_pred_label"] = [label_map.get(val, "neutre") for val in sentiment_pred]
    df["bert_conf"] = [round(val, 4) for val in bert_conf]

    sentiment_counts = []
    counts = df["sentiment_pred"].value_counts().to_dict()
    for key in (-1, 0, 1):
        sentiment_counts.append(
            {"sentiment": key, "label": label_map[key], "count": int(counts.get(key, 0))}
        )

    bins = [
        (0.0, 0.4, "0.00-0.40"),
        (0.4, 0.55, "0.40-0.55"),
        (0.55, 0.7, "0.55-0.70"),
        (0.7, 0.85, "0.70-0.85"),
        (0.85, 1.01, "0.85-1.00"),
    ]
    confidence_bins = []
    for low, high, label in bins:
        count = int(((conf_series >= low) & (conf_series < high)).sum())
        confidence_bins.append({"bucket": label, "count": count})

    preview_columns: List[str] = []
    for col in (text_column, "sentiment_pred", "sentiment_pred_label", "bert_conf", "rating", "app_name", "date"):
        if col in df.columns and col not in preview_columns:
            preview_columns.append(col)
    if not preview_columns:
        preview_columns = df.columns[:6].tolist()
    preview = df[preview_columns].head(100).to_dict(orient="records")

    low_conf_samples = []
    if low_conf_mask.any():
        sample_cols = [text_column, "sentiment_pred_label", "sentiment_pred", "bert_conf"]
        sample_df = df.loc[low_conf_mask, sample_cols]
        sample_count = min(20, len(sample_df))
        low_conf_samples = sample_df.sample(n=sample_count, random_state=42, replace=False).to_dict(orient="records")

    avg_conf = round(conf_series.mean(), 4) if len(conf_series) else 0.0
    fallback_count = int(low_conf_mask.sum())
    fallback_share = round((fallback_count / len(df)) * 100, 2) if len(df) else 0.0

    return {
        "result_df": df,
        "preview": preview,
        "preview_columns": preview_columns,
        "sentiment_counts": sentiment_counts,
        "confidence_bins": confidence_bins,
        "low_conf_samples": low_conf_samples,
        "num_reviews": len(df),
        "avg_conf": avg_conf,
        "fallback_count": fallback_count,
        "fallback_share": fallback_share,
        "model_name": SENTIMENT_MODEL_NAME,
        "batch_size": SENTIMENT_BATCH_SIZE,
        "device": _DEVICE,
        "conf_threshold": SENTIMENT_CONF_THRESHOLD,
        "logs": logs,
    }


def release_resources() -> None:
    global _MODEL, _TOKENIZER, _VADER
    _MODEL = None
    _TOKENIZER = None
    _VADER = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
