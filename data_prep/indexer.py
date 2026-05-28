import os
import sys
import faiss
import json
import h5py
import numpy as np
from tqdm import tqdm

# Add parent and grandparent dirs to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.config import (
    DEFAULT_H5_PATH, DEFAULT_INDEX_PATH, DEFAULT_CACHE_PATH,
    HNSW_M, HNSW_EF_CONSTRUCTION, H5_BATCH_SIZE
)

def build_faiss_index():
    """
    Builds a FAISS HNSW index from HDF5 embeddings.
    Ensures L2 normalization for cosine similarity.
    """
    print("Initializing FAISS Index Building...")
    
    if not os.path.exists(DEFAULT_H5_PATH):
        print(f"Error: H5 file not found at {DEFAULT_H5_PATH}. Run embedder first.")
        return

    # 1. Collect accessions and infer dimension
    with h5py.File(DEFAULT_H5_PATH, 'r') as f:
        accessions = list(f.keys())
        if not accessions:
            print("Error: H5 file is empty.")
            return
        
        # Infer dimension from first dataset
        dim = f[accessions[0]].shape[0]
        total_count = len(accessions)

    print(f"Found {total_count} embeddings with dimension {dim}.")
    print(f"Building HNSW index (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION})...")

    # Initialize Index
    # IndexHNSWIP uses Inner Product, so L2 normalization -> Cosine Similarity
    index = faiss.IndexHNSWIP(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    
    # 2. Add embeddings in batches
    with h5py.File(DEFAULT_H5_PATH, 'r') as f:
        for i in tqdm(range(0, total_count, H5_BATCH_SIZE), desc="Indexing"):
            batch_accs = accessions[i : i + H5_BATCH_SIZE]
            batch_vecs = np.zeros((len(batch_accs), dim), dtype=np.float32)
            
            for j, acc in enumerate(batch_accs):
                batch_vecs[j] = f[acc][:]
            
            # L2 Normalize for Cosine Similarity via IP
            faiss.normalize_L2(batch_vecs)
            
            # Add to index
            index.add(batch_vecs)

    # 3. Save Index and Accession Cache
    print(f"Saving index to {DEFAULT_INDEX_PATH}...")
    faiss.write_index(index, DEFAULT_INDEX_PATH)
    
    print(f"Saving accession cache to {DEFAULT_CACHE_PATH}...")
    with open(DEFAULT_CACHE_PATH, 'w') as f:
        json.dump(accessions, f)

    print("FAISS Indexing complete.")

if __name__ == "__main__":
    build_faiss_index()
