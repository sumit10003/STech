"""
S-TECH RAG Chatbot Configuration
All paths are relative to APP_ROOT - works anywhere the app is installed
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# APP ROOT - Automatically detects install location
# ==============================================================================
# Works whether running as:
# - Frozen exe (PyInstaller): APP_ROOT = directory containing the exe
# - Python script from _internal: APP_ROOT = parent of _internal
# - Python script: APP_ROOT = project root (parent of src/)

if getattr(sys, 'frozen', False):
    # Running as PyInstaller exe
    APP_ROOT = os.path.dirname(sys.executable)
else:
    # Running as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Check if we're in _internal/src structure (bundled deployment)
    if os.path.basename(parent_dir) == "_internal":
        APP_ROOT = os.path.dirname(parent_dir)  # Go up from _internal
    else:
        # Normal development mode - src/ is inside APP_ROOT
        APP_ROOT = parent_dir

# ==============================================================================
# PATHS - All relative to APP_ROOT
# ==============================================================================
ADMIN_DATA_DIR = os.path.join(APP_ROOT, "Data", "admin_data")
VECTOR_STORE_DIR = os.path.join(APP_ROOT, "Data", "vector_store")
MODELS_DIR = os.path.join(APP_ROOT, "Models")

# Sentence Transformer Model - relative path
# Model folder structure: Models/all-mpnet-base-v2/
EMBEDDING_MODEL = os.path.join(MODELS_DIR, "all-mpnet-base-v2")
print(f"[CONFIG] Embedding model path: {EMBEDDING_MODEL}")
#direct path to Embedding model (can be local path or Hugging Face repo)
#EMBEDDING_MODEL = r"C:\STech\Models\all-mpnet-base-v2"

# Allow override via environment variables
ADMIN_DATA_DIR = os.getenv("ADMIN_DATA_DIR", ADMIN_DATA_DIR)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", VECTOR_STORE_DIR)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

# ==============================================================================
# LLM Configuration (Ollama)
# ==============================================================================
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM Settings (for factual RAG responses)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))

# ==============================================================================
# Embedding Settings
# ==============================================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ==============================================================================
# Search Settings
# ==============================================================================
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))

# ==============================================================================
# Supported file formats
# ==============================================================================
SUPPORTED_FORMATS = ["pdf", "txt", "csv", "xlsx", "docx", "json"]

# ==============================================================================
# UI Settings
# ==============================================================================
UI_PAGE_WIDTH = "wide"
UI_INITIAL_SIDEBAR_STATE = "expanded"

# ==============================================================================
# Query Settings
# ==============================================================================
QUERY_CHAR_LIMIT = int(os.getenv("QUERY_CHAR_LIMIT", "500"))

# ==============================================================================
# Create directories if they don't exist
# ==============================================================================
os.makedirs(ADMIN_DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# ==============================================================================
# Debug output
# ==============================================================================
print(f"""
[CONFIG] S-TECH Configuration Loaded:
  - APP_ROOT: {APP_ROOT}
  - Data Directory: {ADMIN_DATA_DIR}
  - Vector Store: {VECTOR_STORE_DIR}
  - Embedding Model: {EMBEDDING_MODEL}
  - LLM Model: {LLM_MODEL}
  - LLM Temperature: {LLM_TEMPERATURE}
""")
