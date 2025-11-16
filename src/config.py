import os

# LLM API configuration
LLM_API_URI = os.getenv("LLM_API_URI", "http://localhost:11434")
LLM_EMBED_PATH = os.getenv("LLM_EMBED_PATH", "api/embed")
LLM_GENERATE_PATH = os.getenv("LLM_GENERATE_PATH", "api/generate")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

# Directory path
TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "macos_correlation_rules")
TEST_DATA_PATH = os.getenv("TEST_DATA_PATH", "windows_correlation_rules")
