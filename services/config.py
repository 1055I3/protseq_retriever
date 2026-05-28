import os

# --- Service Addresses ---
# The unified gateway handles all biological retrieval and refining
SEARCH_SERVICE_HOST = os.getenv("BIOSEQ_SEARCH_HOST", "0.0.0.0")
SEARCH_SERVICE_PORT = int(os.getenv("BIOSEQ_SEARCH_PORT", "8002"))

# --- Paths ---
# Default directory matches ``bootstrap.ensure_data``'s download target so
# search_service finds the files without extra env vars. Override any of
# the five vars below for non-standard layouts (HF Space, custom mounts).
_DATA_DIR = os.getenv("BIOSEQ_DATA_DIR", os.path.join("bioseq_retriever", "data"))

# Protein Specific Paths
DEFAULT_H5_PATH = os.getenv("BIOSEQ_H5_PATH", os.path.join(_DATA_DIR, "per-protein.h5"))
DEFAULT_INDEX_PATH = os.getenv("BIOSEQ_INDEX_PATH", os.path.join(_DATA_DIR, "per-protein.index"))
DEFAULT_CACHE_PATH = os.getenv("BIOSEQ_ACCESSIONS_CACHE_PATH", os.path.join(_DATA_DIR, "per-protein.accessions.json"))

# --- FAISS HNSW Tuning ---
HNSW_M = 128
HNSW_EF_CONSTRUCTION = 512
HNSW_EF_SEARCH = 2048
RANDOM_SEED = 42

# --- Model Settings ---
PROTEIN_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
REFINE_MODEL_NAME = os.getenv("BIOSEQ_REFINE_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Internal sensitivity constant for the refine signal (Maximum refiner authority)
REFINE_LAMBDA = 1

# Length limits
REFINE_MAX_LENGTH = 2048

# --- Default FAISS Threads ---
DEFAULT_FAISS_THREADS = int(os.getenv("FAISS_DEFAULT_THREADS", max(1, os.cpu_count())))

# --- HDF5 Loading Settings ---
# Batch size for reading embeddings from H5 files during index construction
H5_BATCH_SIZE = int(os.getenv("BIOSEQ_H5_BATCH_SIZE", "1000"))
