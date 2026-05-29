import os

# --- Default Paths ---
_DATA_DIR = os.getenv("BIOSEQ_DATA_DIR", "data")

DEFAULT_H5_PATH = os.getenv("BIOSEQ_H5_PATH", os.path.join(_DATA_DIR, "per-protein.h5"))
_h5_base = os.path.splitext(DEFAULT_H5_PATH)[0]

DEFAULT_INDEX_PATH = os.getenv("BIOSEQ_INDEX_PATH", f"{_h5_base}.index")
DEFAULT_CACHE_PATH = os.getenv("BIOSEQ_ACCESSIONS_CACHE_PATH", f"{_h5_base}.accessions.json")
SWISSPROT_CSV_PATH = os.getenv("BIOSEQ_SWISSPROT_CSV_PATH", os.path.join(_DATA_DIR, "swissprot_metadata.csv"))

# --- Security ---
ALLOWED_DATA_DIR = os.getenv("BIOSEQ_ALLOWED_DATA_DIR", _DATA_DIR)

# --- Fetcher & API ---
FETCH_TIMEOUT = float(os.getenv("BIOSEQ_FETCH_TIMEOUT", "300.0"))
MAX_RETRIES = int(os.getenv("BIOSEQ_MAX_RETRIES", "5"))
BACKOFF_FACTOR = float(os.getenv("BIOSEQ_BACKOFF_FACTOR", "2.0"))
SEARCH_PROBE_TIMEOUT = float(os.getenv("BIOSEQ_SEARCH_PROBE_TIMEOUT", "2.0"))

# --- Service Addresses ---
SEARCH_SERVICE_HOST = os.getenv("BIOSEQ_SEARCH_HOST", "0.0.0.0")
SEARCH_SERVICE_PORT = int(os.getenv("BIOSEQ_SEARCH_PORT", "8002"))
SEARCH_SERVICE_URL = os.getenv("BIOSEQ_SEARCH_SERVICE_URL", f"http://localhost:{SEARCH_SERVICE_PORT}")

# --- Retrieval Settings ---
RETRIEVAL_TOP_K = 75
REFINE_TOP_N = 5
USE_SERVICES = os.getenv("BIOSEQ_USE_SERVICES", "true").lower() == "true"
