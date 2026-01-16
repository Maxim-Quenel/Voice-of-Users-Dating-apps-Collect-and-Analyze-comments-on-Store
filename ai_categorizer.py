from typing import List
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_TOKENIZER: AutoTokenizer | None = None
_MODEL: AutoModelForCausalLM | None = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
_LLM_TOKENIZER_NAME = os.getenv("LLM_TOKENIZER_NAME", "Qwen/Qwen2.5-1.5B-Instruct")


def _get_tokenizer() -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        print(f"[ai_categorizer] Chargement du tokenizer {_LLM_TOKENIZER_NAME}...", flush=True)
        _TOKENIZER = AutoTokenizer.from_pretrained(_LLM_TOKENIZER_NAME)
    return _TOKENIZER


def _get_model() -> AutoModelForCausalLM:
    global _MODEL
    if _MODEL is None:
        print(
            f"[ai_categorizer] Chargement du modèle {_LLM_MODEL_NAME} (device_map=auto, préf. {_DEVICE})...",
            flush=True,
        )
        _MODEL = AutoModelForCausalLM.from_pretrained(
            _LLM_MODEL_NAME,
            device_map="auto",  # envoie direct en VRAM si dispo, sinon bascule en CPU
            low_cpu_mem_usage=True,  # limite l'empreinte RAM lors du chargement
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return _MODEL


from typing import List

def _build_messages(reviews: List[str], locale: str) -> list[dict]:
    # Ajout de puces pour bien délimiter chaque avis
    reviews_block = "\n".join(f"- {text}" for text in reviews)
    
    if locale == "fr":
        instruction = """
Tu es un expert en UX Research et en classification de texte.
Ta tâche est d'étiqueter un groupe d'avis utilisateurs provenant d'une application de rencontre.
IMPORTANT : Ces avis ont déjà été regroupés car ils traitent du même sujet sémantique.

Instructions :
1. Identifie le problème technique ou l'insatisfaction spécifique commun à tous ces avis.
2. Génère un NOM DE CATÉGORIE qui résume ce problème.
3. Le nom doit être concis (2 à 5 mots maximum), objectif et professionnel.
4. N'utilise pas de phrases complètes, juste le label.

Exemples de format attendu :
- Problèmes de Connexion
- Faux Profils et Bots
- Prix de l'Abonnement Élevé
"""
        question = "Quel est le label précis pour ce groupe d'avis ? Renvoie UNIQUEMENT le label."
    
    else:
        instruction = """
You are an expert UX Researcher and Text Classifier.
Your task is to label a cluster of user reviews for a dating app.
IMPORTANT: These reviews have already been grouped because they share the same semantic topic.

Instructions:
1. Identify the specific technical issue or dissatisfaction common to all these reviews.
2. Generate a CATEGORY LABEL that summarizes this core issue.
3. The label must be concise (2-5 words max), objective, and professional.
4. Do not use full sentences, just the label.

Examples of expected format:
- Login Connectivity Issues
- Fake Profiles and Bots
- Subscription Cost High
"""
        question = "What is the precise label for this cluster? Return ONLY the label."

    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": f"Reviews Cluster:\n{reviews_block}\n\n{question}",
        },
    ]

def predict_category_from_reviews(
    reviews: List[str],
    max_new_tokens: int = 50,
    locale: str = "en",
) -> str:
    """
    Use Qwen2.5-1.5B-Instruct to propose a concise category name from a handful of reviews.
    The model is loaded once and reused.
    """
    clean_reviews = [text.strip() for text in reviews if isinstance(text, str) and text.strip()]
    if not clean_reviews:
        return "Category not found"

    sample = clean_reviews[:10]
    tokenizer = _get_tokenizer()
    model = _get_model()
    print(f"[ai_categorizer] Génération de catégorie sur {len(sample)} extraits (device={model.device})...", flush=True)
    messages = _build_messages(sample, locale)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip() or "Category not found"


def release_resources() -> None:
    global _MODEL, _TOKENIZER
    _MODEL = None
    _TOKENIZER = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
