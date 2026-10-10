"""
Central configuration for the Border Intelligence Multilingual Translator.

Defines:
  - Supported languages (name, ISO-ish code, script, resource tier)
  - Mapping to NLLB-200 FLORES codes (for languages the backbone model natively supports)
  - "Proxy" FLORES codes for languages NOT covered by NLLB-200 (Tulu, Kodava, Beary) so that
    the multilingual Transformer's language-tag embeddings can still be leveraged via
    transfer learning + LoRA adaptation from the linguistically closest high-resource language.
  - Paths used across the project.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GLOSSARY_DIR = os.path.join(DATA_DIR, "glossary")
SEED_CORPUS_DIR = os.path.join(DATA_DIR, "parallel_seed")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic")
VERIFICATION_LOG = os.path.join(DATA_DIR, "verification_log.csv")
ADAPTERS_DIR = os.path.join(BASE_DIR, "adapters")

BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Resource tiers:
#   "high"  -> natively supported by NLLB-200, strong zero/few-shot quality
#   "low"   -> NOT in NLLB-200 vocabulary; handled via proxy-tag transfer learning,
#              LoRA fine-tuning on seed corpora, back-translation and terminology retrieval.
LANGUAGES = {
    "English": {
        "code": "eng",
        "flores_code": "eng_Latn",
        "script": "Latin",
        "resource_tier": "high",
        "proxy_of": None,
    },
    "Hindi": {
        "code": "hin",
        "flores_code": "hin_Deva",
        "script": "Devanagari",
        "resource_tier": "high",
        "proxy_of": None,
    },
    "Kannada": {
        "code": "kan",
        "flores_code": "kan_Knda",
        "script": "Kannada",
        "resource_tier": "high",
        "proxy_of": None,
    },
    "Tamil": {
        "code": "tam",
        "flores_code": "tam_Taml",
        "script": "Tamil",
        "resource_tier": "high",
        "proxy_of": None,
    },
    "Malayalam": {
        "code": "mal",
        "flores_code": "mal_Mlym",
        "script": "Malayalam",
        "resource_tier": "high",
        "proxy_of": None,
    },
    "Konkani": {
        "code": "gom",
        "flores_code": "gom_Deva",
        "script": "Devanagari",
        "resource_tier": "high",
        "proxy_of": None,
    },
    # ---- Low-resource / unsupported-by-NLLB languages ----
    # These are handled via transfer learning: fine-tune the base model using the
    # FLORES tag of a linguistically related high-resource "proxy" language, with
    # LoRA adapters trained on small seed parallel corpora + back-translated data.
    "Tulu": {
        "code": "tcy",
        "flores_code": None,
        "proxy_flores_code": "kan_Knda",  # closest major literary/script neighbour
        "script": "Kannada",
        "resource_tier": "low",
        "proxy_of": "Kannada",
    },
    "Kodava": {
        "code": "kfa",
        "flores_code": None,
        "proxy_flores_code": "kan_Knda",
        "script": "Kannada",
        "resource_tier": "low",
        "proxy_of": "Kannada",
    },
    "Beary": {
        "code": "bary",
        "flores_code": None,
        "proxy_flores_code": "mal_Mlym",  # Beary bashe shares heavy Malayalam/Dakhini-Urdu lexicon
        "script": "Kannada",  # commonly written in Kannada script locally; Arabic script also used
        "resource_tier": "low",
        "proxy_of": "Malayalam",
    },
}

HIGH_RESOURCE_LANGS = [n for n, v in LANGUAGES.items() if v["resource_tier"] == "high"]
LOW_RESOURCE_LANGS = [n for n, v in LANGUAGES.items() if v["resource_tier"] == "low"]

# Target languages an analyst can request output in.
ANALYST_TARGETS = ["English", "Hindi"]

# Confidence thresholds for routing to human verification.
CONFIDENCE_HIGH = 0.75
CONFIDENCE_MEDIUM = 0.45

# LoRA / PEFT defaults for low-resource adapter fine-tuning.
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
}

# Demo-scale training defaults (kept small so the research pipeline runs end-to-end
# on a laptop/CPU for demonstration; scale epochs/data up for real research runs).
TRAIN_DEFAULTS = {
    "epochs": 3,
    "batch_size": 4,
    "lr": 1e-4,
    "max_length": 64,
}
