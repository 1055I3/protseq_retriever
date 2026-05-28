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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.config import (
    DEFAULT_H5_PATH, DEFAULT_INDEX_PATH, DEFAULT_CACHE_PATH, SWISSPROT_CSV_PATH,
    HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, RANDOM_SEED,
    SEARCH_SERVICE_HOST, SEARCH_SERVICE_PORT,
    PROTEIN_MODEL_NAME, REFINE_MODEL_NAME,
    DEFAULT_FAISS_THREADS, H5_BATCH_SIZE,
    REFINE_LAMBDA, REFINE_MAX_LENGTH, ESMC_MAX_LENGTH
)

app = FastAPI(title="Unified BioSeq Gateway Service (ESMC-300M)")
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
torch.set_num_interop_threads(TOTAL_CORES)
logger.info(f"Parallelism Optimized: FAISS and PyTorch using {TOTAL_CORES} threads (Intra/Inter-op).")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SearchRequest(BaseModel):
    sequence: str
    k: int = 25

class RefineRequest(BaseModel):
    records: List[Dict[str, Any]]
    context_query: str
    top_n: int = 5

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
    
    if missing:
        msg = f"CRITICAL: Missing required data files: {', '.join(missing)}. " \
              "Please run 'python data_prep/orchestrator.py' to prepare the database."
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
        refine_tokenizer = AutoTokenizer.from_pretrained(REFINE_MODEL_NAME, padding_side="left")
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

def _enrich_results(accessions: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """Enriches accession hits with local Swiss-Prot metadata using Polars."""    
    # Efficient lookup using polars
    hits_df = metadata_df.filter(pl.col("accession").is_in(accessions))
    
    # Map scores back
    score_map = {acc: score for acc, score in zip(accessions, scores)}
    
    records = hits_df.to_dicts()
    for rec in records:
        rec["_search_score"] = score_map.get(rec["accession"])
        
    # Maintain FAISS order
    acc_to_idx = {acc: i for i, acc in enumerate(accessions)}
    records.sort(key=lambda x: acc_to_idx.get(x["accession"], 999))
    return records

def _embed_refine_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """Generates semantic embeddings using Qwen3-Embedding."""
    if is_query:
        instruction = (
            "Given a bioinformatics context or sequence retrieval prompt, identify relevant biological "
            "entities, molecular functions, biological processes, protein families and domains, "
            "subcellular localizations, taxonomic and evolutionary constraints, ontology-related terms, "
            "and structural or functional relationships to retrieve matching entries from the Swiss-Prot database."
        )
        processed_texts = [f"{instruction}\nQuery: {t}" for t in texts]
    else:
        processed_texts = texts
        
    inputs = refine_tokenizer(
        processed_texts, 
        padding=True, 
        truncation=True, 
        max_length=REFINE_MAX_LENGTH, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = refine_model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, -1].to(torch.float32)
        
    return embeddings.cpu().numpy()

def _format_record_for_embedding(record: Dict[str, Any]) -> str:
    """
    Creates a biologically dense text summary of a UniProt record.
    Matches the entities and constraints mentioned in the Qwen3 instruction.
    """
    name = record.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
    organism = record.get('organism', {}).get('scientificName', 'N/A')
    
    # Extract Lineage (Taxonomic constraints)
    lineage = [t if isinstance(t, str) else t.get('scientificName', '') 
               for t in record.get('organism', {}).get('lineage', [])]
    lineage_text = " > ".join(lineage)

    # Extract function and localization comments
    functions = []
    locations = []
    for comment in record.get('comments', []):
        ctype = comment.get('commentType')
        if ctype == 'FUNCTION':
            functions.extend([t.get('value', '') for t in comment.get('texts', [])])
        elif ctype == 'SUBCELLULAR_LOCATION':
            locations.extend([l.get('location', {}).get('value', '') for l in comment.get('locations', [])])
    
    # Extract GO terms and Domains from cross-references (Structural/Functional relationships)
    go_terms = []
    domains = []
    for xref in record.get('uniProtKBCrossReferences', []):
        db = xref.get('database')
        if db == 'GO':
            props = xref.get('properties', [])
            if props: go_terms.append(props[0].get('value', ''))
        elif db in ['Pfam', 'InterPro']:
            props = xref.get('properties', [])
            if props: domains.append(props[0].get('value', ''))
    
    # Extract keywords
    keywords = ", ".join([k.get('value', '') for k in record.get('keywords', [])])
    
    # Construct dense biological profile
    profile_parts = [
        f"Protein: {name}",
        f"Organism: {organism} (Lineage: {lineage_text})",
        f"Function: {' '.join(functions)}",
        f"Subcellular Location: {', '.join(locations)}",
        f"Gene Ontology: {', '.join(go_terms[:15])}",
        f"Domains/Families: {', '.join(domains[:10])}",
        f"Keywords: {keywords}"
    ]
    
    return ". ".join(profile_parts)

# =============================================================================
# CONFORMAL UNCERTAINTY-AWARE FUSION LOGIC
# =============================================================================

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
    try:
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(executor, _embed_protein, request.sequence)
        
        query_vec = emb.reshape(1, -1)
        faiss.normalize_L2(query_vec)
        dist, idxs = await loop.run_in_executor(executor, protein_index.search, query_vec, request.k)
        
        accessions = [protein_accessions[i] for i in idxs[0] if i != -1]
        scores = [float(s) for s in dist[0][:len(accessions)]]
        
        results = await loop.run_in_executor(executor, _enrich_results, accessions, scores)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine")
async def refine(request: RefineRequest):
    try:
        loop = asyncio.get_event_loop()
        
        # 1. Semantic Embedding
        passages = [_format_record_for_embedding(rec) for rec in request.records]
        query_vec = await loop.run_in_executor(executor, _embed_refine_texts, [request.context_query], True)
        doc_vecs = await loop.run_in_executor(executor, _embed_refine_texts, passages, False)
        
        # 2. Raw Semantic Scores (Cosine Similarity)
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)
        doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
        semantic_scores = np.dot(doc_vecs, query_vec.T).flatten()

        # 3. Z-Score Calibration
        retrieval_raw = np.array([r.get("_search_score", 0.0) for r in request.records])
        retrieval_z = _normalize_z_score(retrieval_raw)
        semantic_z = _normalize_z_score(semantic_scores)

        # 4. Conformal Ranking Uncertainty Estimation (α_i)
        alpha = _compute_conformal_uncertainty(retrieval_z)

        # 5. Principled Confidence-Aware Fusion
        # f_i = s_i + α_i * λ * (r_i - s_i)
        # Correction is applied only where the retrieval ranking is statistically exchangeable.
        refined_list = []
        for i, record in enumerate(request.records):
            final_fused_score = retrieval_z[i] + (alpha[i] * REFINE_LAMBDA * (semantic_z[i] - retrieval_z[i]))

            # Store metadata for transparency
            record["_search_score"] = float(final_fused_score)
            record["_uncertainty_alpha"] = float(alpha[i])
            refined_list.append(record)

        # 6. Stable Rank Sort
        refined_list.sort(key=lambda x: x["_search_score"], reverse=True)

        return {"results": refined_list[:request.top_n]}
        
    except Exception as e:
        logger.error(f"Refining failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SEARCH_SERVICE_HOST, port=SEARCH_SERVICE_PORT)
