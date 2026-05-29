import os

# --- FAISS HNSW Tuning ---
HNSW_M = 128
HNSW_EF_CONSTRUCTION = 512
HNSW_EF_SEARCH = 2048
RANDOM_SEED = 42

# --- Model Settings ---
PROTEIN_MODEL_NAME = "esmc-300m-2024-12"
REFINE_MODEL_NAME = os.getenv("PROTSEQ_REFINE_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Internal sensitivity constant for the refine signal (Maximum refiner authority)
REFINE_LAMBDA = 1

# Length limits
REFINE_MAX_LENGTH = 2048
ESMC_MAX_LENGTH = 2048 # Model limit

# --- Default FAISS Threads ---
DEFAULT_FAISS_THREADS = int(os.getenv("FAISS_DEFAULT_THREADS", max(1, os.cpu_count() or 1)))

# --- HDF5 Loading Settings ---
# Batch size for reading embeddings from H5 files during index construction
H5_BATCH_SIZE = int(os.getenv("PROTSEQ_H5_BATCH_SIZE", "1000"))
