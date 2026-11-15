"""All tunable settings in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

DOCS_DIR = BASE_DIR / "data" / "docs"
DB_PATH = BASE_DIR / "data" / "sample.db"
FRONTEND_DIR = PROJECT_DIR / "frontend"

# Chunking defaults
FIXED_CHUNK_SIZE = 300      # characters
FIXED_CHUNK_OVERLAP = 50    # characters
SENTENCES_PER_CHUNK = 3

# Retrieval / re-ranking defaults
TOP_N = 8                   # candidates returned by retrieval
TOP_K = 3                   # chunks kept after re-ranking (sent to the LLM)
KMEANS_CLUSTERS = 5

# Models
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Azure AI Foundry (OpenAI model deployment)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-10-21"
