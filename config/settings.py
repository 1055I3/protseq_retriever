import os

# --- Default Paths ---
_DATA_DIR = os.getenv("PROTSEQ_DATA_DIR", "data")

DEFAULT_H5_PATH = os.getenv("PROTSEQ_H5_PATH", os.path.join(_DATA_DIR, "per-protein.h5"))
_h5_base = os.path.splitext(DEFAULT_H5_PATH)[0]

DEFAULT_INDEX_PATH = os.getenv("PROTSEQ_INDEX_PATH", f"{_h5_base}.index")
DEFAULT_CACHE_PATH = os.getenv("PROTSEQ_ACCESSIONS_CACHE_PATH", f"{_h5_base}.accessions.json")
SWISSPROT_CSV_PATH = os.getenv("PROTSEQ_SWISSPROT_CSV_PATH", os.path.join(_DATA_DIR, "swissprot_metadata.csv"))
CONTEXT_H5_PATH = os.getenv("PROTSEQ_CONTEXT_H5_PATH", os.path.join(_DATA_DIR, "refinement_contexts.h5"))

# --- Security ---
ALLOWED_DATA_DIR = os.getenv("PROTSEQ_ALLOWED_DATA_DIR", _DATA_DIR)

# --- Fetcher & API ---
FETCH_TIMEOUT = float(os.getenv("PROTSEQ_FETCH_TIMEOUT", "300.0"))
MAX_RETRIES = int(os.getenv("PROTSEQ_MAX_RETRIES", "5"))
BACKOFF_FACTOR = float(os.getenv("PROTSEQ_BACKOFF_FACTOR", "2.0"))
SEARCH_PROBE_TIMEOUT = float(os.getenv("PROTSEQ_SEARCH_PROBE_TIMEOUT", "2.0"))

# --- Service Addresses ---
SEARCH_SERVICE_HOST = os.getenv("PROTSEQ_SEARCH_HOST", "0.0.0.0")
SEARCH_SERVICE_PORT = int(os.getenv("PROTSEQ_SEARCH_PORT", "8002"))
SEARCH_SERVICE_URL = os.getenv("PROTSEQ_SEARCH_SERVICE_URL", f"http://localhost:{SEARCH_SERVICE_PORT}")

# --- Retrieval Settings ---
RETRIEVAL_TOP_K = 75
REFINE_TOP_N = 5
USE_SERVICES = os.getenv("PROTSEQ_USE_SERVICES", "true").lower() == "true"
