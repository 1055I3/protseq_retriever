import os
import sys
import faiss
import json
import numpy as np
import h5py
import asyncio
import torch
import traceback
import re
import polars as pl
import logging
from typing import List, Tuple, Generator, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
from transformers import AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add parent dir to path to import config
from config.settings import (
    DEFAULT_H5_PATH, DEFAULT_INDEX_PATH, DEFAULT_CACHE_PATH, SWISSPROT_CSV_PATH, CONTEXT_H5_PATH,
    HNSW_EF_SEARCH, RANDOM_SEED,
    SEARCH_SERVICE_HOST, SEARCH_SERVICE_PORT
)
from config.service_params import (
    PROTEIN_MODEL_NAME, REFINE_MODEL_NAME,
    H5_BATCH_SIZE, REFINE_LAMBDA, REFINE_MAX_LENGTH, ESMC_MAX_LENGTH
)
from protseq.common.utils import format_context_for_embedding

app = FastAPI(title="Unified Protseq Gateway Service (ESMC-300M)")
executor = ThreadPoolExecutor(max_workers=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set seed for reproducibility
np.random.seed(RANDOM_SEED)

# =============================================================================
# GLOBAL PARALLELISM CONFIGURATION
# =============================================================================
TOTAL_CORES = os.cpu_count() or 1
faiss.omp_set_num_threads(TOTAL_CORES)
torch.set_num_threads(TOTAL_CORES)
if device.type == 'cpu':
    try:
        torch.set_num_interop_threads(TOTAL_CORES)
    except RuntimeError:
        pass
logger.info(f"Parallelism Optimized: FAISS and PyTorch using {TOTAL_CORES} threads (Intra/Inter-op).")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SearchRequest(BaseModel):
    sequence: str
    k: int = 25

class Hit(BaseModel):
    accession: str
    score: float
    uncertainty_alpha: Optional[float] = None

class RefineRequest(BaseModel):
    hits: List[Hit]
    context_query: str  # This is the already formatted biological string
    top_n: int = 5

class HydrateRequest(BaseModel):
    accessions: List[str]

# =============================================================================
# SERVICE STATE (Models & Data)
# =============================================================================

protein_model = None
refine_model = None
refine_tokenizer = None
protein_index = None
protein_accessions = None
metadata_df = None

def validate_data_files():
    """Checks if all required data files exist."""
    missing = []
    if not os.path.exists(DEFAULT_H5_PATH): missing.append("Embeddings (H5)")
    if not os.path.exists(DEFAULT_INDEX_PATH): missing.append("FAISS Index")
    if not os.path.exists(DEFAULT_CACHE_PATH): missing.append("Accession Cache")
    if not os.path.exists(SWISSPROT_CSV_PATH): missing.append("Swiss-Prot Metadata (CSV)")
    if not os.path.exists(CONTEXT_H5_PATH): missing.append("Context Embeddings (H5)")
    
    if missing:
        msg = f"CRITICAL: Missing required data files: {', '.join(missing)}. " \
              "Please run 'python scripts/data_prep/orchestrator.py' to prepare the database."
        logger.error(msg)
        return False
    return True

def init_service():
    global protein_model, refine_model, refine_tokenizer, protein_index, protein_accessions, metadata_df
    
    if not validate_data_files():
        return

    try:
        # 1. Load ESMC Model
        logger.info(f"Loading ESMC Model: {PROTEIN_MODEL_NAME} to {device}...")
        protein_model = ESMC.from_pretrained(PROTEIN_MODEL_NAME).to(device)
        protein_model.eval()

        # 2. Load Refining model
        logger.info(f"Loading Refining model: {REFINE_MODEL_NAME}...")
        refine_tokenizer = AutoTokenizer.from_pretrained(REFINE_MODEL_NAME)
        refine_model = AutoModel.from_pretrained(REFINE_MODEL_NAME).to(device)
        refine_model.eval()

        # 3. Load FAISS Index and Accessions
        logger.info(f"Loading FAISS index from {DEFAULT_INDEX_PATH}...")
        protein_index = faiss.read_index(DEFAULT_INDEX_PATH)
        if hasattr(protein_index, "hnsw"):
            protein_index.hnsw.efSearch = HNSW_EF_SEARCH
            
        with open(DEFAULT_CACHE_PATH, 'r') as f:
            protein_accessions = json.load(f)

        # 4. Load Metadata CSV
        logger.info(f"Loading metadata from {SWISSPROT_CSV_PATH}...")
        metadata_df = pl.read_csv(SWISSPROT_CSV_PATH)
        metadata_df = metadata_df.with_columns(pl.col("accession").alias("_idx_acc"))
        
        logger.info("Service successfully initialized.")
        
    except Exception as e:
        logger.critical(f"Failed to initialize service: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

# Initialize on startup
init_service()

# =============================================================================
# INTERNAL LOGIC (CORE)
# =============================================================================

def _embed_protein(sequence: str) -> np.ndarray:
    """Generates mean-pooled ESMC-300M embedding."""
    if len(sequence) > ESMC_MAX_LENGTH:
        raise ValueError(f"Sequence length ({len(sequence)}) exceeds ESMC limit ({ESMC_MAX_LENGTH})")
    
    protein = ESMProtein(sequence=sequence)
    with torch.no_grad():
        output = protein_model(protein)
        emb = output.embeddings.mean(dim=1).cpu().numpy().flatten().astype(np.float32)
    return emb

def _embed_user_context(text: str) -> np.ndarray:
    """
    Generates semantic embedding for user context using ModernBERT-bio-large.
    Uses mean pooling to align with precomputed context embeddings.
    """
    inputs = refine_tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=REFINE_MAX_LENGTH, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = refine_model(**inputs)
        mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).cpu().numpy().flatten().astype(np.float32)
        
    return mean_pooled

def _load_precomputed_context_vectors(accessions: List[str]) -> np.ndarray:
    """Loads context vectors from HDF5 keyed by accession."""
    dim = refine_model.config.hidden_size
    vectors = np.zeros((len(accessions), dim), dtype=np.float32)
    
    with h5py.File(CONTEXT_H5_PATH, 'r') as f:
        for i, acc in enumerate(accessions):
            if acc in f:
                vectors[i] = f[acc][:]
            else:
                logger.warning(f"Context vector missing for {acc}. Using zero vector.")
                
    return vectors

def _normalize_z_score(scores: np.ndarray) -> np.ndarray:
    mu = np.mean(scores)
    sigma = np.std(scores) + 1e-9
    return (scores - mu) / sigma

def _compute_conformal_uncertainty(scores: np.ndarray) -> np.ndarray:
        """
    Estimates local posterior uncertainty using local conformal nonconformity.
    
    Mathematical Principle:
    1. Ranking instability: Uncertainty is induced when local score distances are small,
       implying candidates are exchangeable under local score permutations.
    2. Nonconformity score (a_i): Defined as the minimum distance to the nearest neighbor.
       Small a_i -> high exchangeability -> high uncertainty.
    3. Empirical Conformal p-value (p_i): Measures where a_i sits in the empirical 
       distribution of all nonconformity scores.
    4. Uncertainty (alpha_i): Defined as (1 - p_i), bounding it in [0, 1] without parameters.
    
    This model is Parameter-Free, Distribution-Free, and statistically rigorous.
    """
    n = len(scores)
    if n < 2:
        return np.array([0.5] * n) # Degenerate case safety
        
    # 1. Compute local nonconformity scores (nearest-neighbor gaps)
    a = np.zeros(n)
    for i in range(n):
        if i == 0:
            a[i] = abs(scores[0] - scores[1])
        elif i == n - 1:
            a[i] = abs(scores[n-1] - scores[n-2])
        else:
            # Geometric isolation defines nonconformity
            a[i] = min(abs(scores[i] - scores[i-1]), abs(scores[i] - scores[i+1]))
            
    # 2. Compute empirical conformal p-values
    # p_i represents the probability that a randomly sampled candidate is 
    # as isolated as candidate i.
    p = np.zeros(n)
    for i in range(n):
        p[i] = (1 + np.sum(a <= a[i])) / (n + 1)
        
    # 3. Transform to uncertainty (alpha)
    # α_i -> 1 means the candidate is statistically exchangeable (high uncertainty)
    # α_i -> 0 means the candidate is isolated (high confidence)
    return 1 - p

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/search")
async def search(request: SearchRequest):
    if protein_index is None:
        raise HTTPException(status_code=503, detail="Search service not initialized.")
        
    try:
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(executor, _embed_protein, request.sequence)
        
        query_vec = emb.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        dist, idxs = await loop.run_in_executor(executor, protein_index.search, query_vec, request.k)
        
        results = [{"accession": protein_accessions[i], "score": float(dist[0][j])} 
                   for j, i in enumerate(idxs[0]) if i != -1]
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine")
async def refine(request: RefineRequest):
    if refine_model is None:
        raise HTTPException(status_code=503, detail="Refining model not loaded.")
        
    try:
        loop = asyncio.get_event_loop()
        
        # 1. Accession list from request
        accessions = [h.accession for h in request.hits]
        retrieval_raw = np.array([h.score for h in request.hits])

        # 2. Vector computation (Runtime embedding of user context vs. Precomputed database contexts)
        query_vec = await loop.run_in_executor(executor, _embed_user_context, request.context_query)
        doc_vecs = await loop.run_in_executor(executor, _load_precomputed_context_vectors, accessions)
        
        # 3. Raw Semantic Scores (Cosine Similarity via Inner Product of normalized vectors)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
        semantic_scores = np.dot(doc_vecs, query_vec.T).flatten()

        # 4. Z-Score Calibration
        retrieval_z = _normalize_z_score(retrieval_raw)
        semantic_z = _normalize_z_score(semantic_scores)

        # 5. Conformal Ranking Uncertainty Estimation (α_i)
        alpha = _compute_conformal_uncertainty(retrieval_z)

        # 6. Principled Confidence-Aware Fusion
        # f_i = s_i + α_i * λ * (r_i - s_i)
        # Correction is applied only where the retrieval ranking is statistically exchangeable.
        refined_list = []
        for i, hit in enumerate(request.hits):
            final_fused_score = retrieval_z[i] + (alpha[i] * REFINE_LAMBDA * (semantic_z[i] - retrieval_z[i]))
            
            refined_list.append({
                "accession": hit.accession,
                "score": float(final_fused_score),
                "uncertainty_alpha": float(alpha[i])
            })

        # 7. Stable Rank Sort
        refined_list.sort(key=lambda x: x["score"], reverse=True)

        return {"results": refined_list[:request.top_n]}
        
    except Exception as e:
        logger.error(f"Refining failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hydrate")
async def hydrate(request: HydrateRequest):
    """Enriches accession list with full Swiss-Prot records from local memory cache."""
    if metadata_df is None:
        raise HTTPException(status_code=503, detail="Metadata cache unavailable.")
        
    try:
        # Filter metadata for requested accessions
        hits_df = metadata_df.filter(pl.col("accession").is_in(request.accessions))
        
        # Maintain order of requested accessions
        records = hits_df.to_dicts()
        acc_to_idx = {acc: i for i, acc in enumerate(request.accessions)}
        records.sort(key=lambda x: acc_to_idx.get(x["accession"], 999))
        
        return {"records": records}
    except Exception as e:
        logger.error(f"Hydration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SEARCH_SERVICE_HOST, port=SEARCH_SERVICE_PORT)
