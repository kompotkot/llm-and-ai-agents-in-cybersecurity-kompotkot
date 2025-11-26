import os
from pathlib import Path

IS_DEBUG = True  # TODO(kompotkot)

BASE_DIR = Path(__file__).resolve().parent.parent

# LLM API configuration
LLM_API_URI = os.getenv("LLM_API_URI", "http://localhost:11434")
LLM_EMBED_PATH = os.getenv("LLM_EMBED_PATH", "api/embed")
LLM_GENERATE_PATH = os.getenv("LLM_GENERATE_PATH", "api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

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
