import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# LLM API configuration
EMBED_API_URI = os.getenv("EMBED_API_URI", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral")
LLM_API_URI = os.getenv("LLM_API_URI", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# Data path
TRAIN_DATA_PATH = Path(
    os.getenv("TRAIN_DATA_PATH", BASE_DIR / "data" / "macos_correlation_rules")
)
TEST_DATA_PATH = Path(
    os.getenv("TEST_DATA_PATH", BASE_DIR / "data" / "windows_correlation_rules")
)
TAXONOMY_RU_PATH = Path(
    os.getenv(
        "TAXONOMY_RU_PATH", BASE_DIR / "data" / "taxonomy_fields" / "i18n_ru.yaml"
    )
)
TAXONOMY_EN_PATH = Path(
    os.getenv(
        "TAXONOMY_EN_PATH", BASE_DIR / "data" / "taxonomy_fields" / "i18n_en.yaml"
    )
)

MITRE_CTI_PATH = Path(os.getenv("MITRE_CTI_PATH", BASE_DIR / "mitre" / "cti"))
MITRE_COMPACT_PATTERNS_PATH = Path(
    os.getenv(
        "MITRE_COMPACT_PATTERNS_PATH",
        BASE_DIR / "data" / "mitre" / "mitre_compact_patterns.json",
    )
)

# Langfuse configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
