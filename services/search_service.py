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
from typing import List, Tuple, Generator, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5EncoderModel, T5Tokenizer, AutoModel, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.config import (
    DEFAULT_H5_PATH, DEFAULT_INDEX_PATH, DEFAULT_CACHE_PATH,
    DNA_H5_PATH, DNA_INDEX_PATH, DNA_CACHE_PATH,
    HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, RANDOM_SEED,
    SEARCH_SERVICE_HOST, SEARCH_SERVICE_PORT,
    PROTEIN_MODEL_NAME, DNA_MODEL_NAME, RERANK_MODEL_NAME,
    DNA_MAX_LENGTH, DEFAULT_FAISS_THREADS, H5_BATCH_SIZE,
    RERANK_LAMBDA, RERANK_MAX_LENGTH
)

app = FastAPI(title="Unified BioSeq Gateway Service")
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
print(f"Parallelism Optimized: FAISS and PyTorch using {TOTAL_CORES} threads.")

# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SearchRequest(BaseModel):
    sequence: str
    k: int = 25

class RerankRequest(BaseModel):
    records: List[Dict[str, Any]]
    context_query: str
    top_n: int = 5

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

# Global references
protein_model = None
protein_tokenizer = None
dna_model = None
dna_tokenizer = None
rerank_model = None
rerank_tokenizer = None

def init_models():
    global protein_model, protein_tokenizer, dna_model, dna_tokenizer, rerank_model, rerank_tokenizer
    
    print(f"Loading Protein model: {PROTEIN_MODEL_NAME}...")
    protein_tokenizer = T5Tokenizer.from_pretrained(PROTEIN_MODEL_NAME, do_lower_case=False)
    protein_model = T5EncoderModel.from_pretrained(PROTEIN_MODEL_NAME).to(device)
    protein_model.eval()

    print(f"Loading DNA model: {DNA_MODEL_NAME}...")
    dna_tokenizer = AutoTokenizer.from_pretrained(DNA_MODEL_NAME, trust_remote_code=True)
    dna_model = AutoModel.from_pretrained(DNA_MODEL_NAME, trust_remote_code=True).to(device)
    dna_model.eval()

    print(f"Loading Reranking model: {RERANK_MODEL_NAME}...")
    rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME, padding_side="left")
    rerank_model = AutoModel.from_pretrained(RERANK_MODEL_NAME).to(device)
    rerank_model.eval()
    
    print(f"All models loaded on {device}")

init_models()

# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

def iter_embeddings(h5_path: str, batch_size: int = H5_BATCH_SIZE) -> Generator[Tuple[np.ndarray, List[str]], None, None]:
    # Use libver='latest' to handle modern HDF5 layout messages
    with h5py.File(h5_path, 'r', libver='latest') as f:
        # Filter keys to ensure we only process actual datasets
        all_keys = list(f.keys())
        accessions = [k for k in all_keys if isinstance(f[k], h5py.Dataset)]

        if not accessions:
            raise ValueError(f"HDF5 file {h5_path} contains no valid datasets.")

        # Infer dimension from the first valid dataset
        dim = f[accessions[0]].shape[0]
        
        for i in range(0, len(accessions), batch_size):
            batch_accs = accessions[i : i + batch_size]
            batch_embeddings = np.zeros((len(batch_accs), dim), dtype=np.float32)
            for j, acc in enumerate(batch_accs):
                batch_embeddings[j] = f[acc][:]
            yield batch_embeddings, batch_accs

def build_index(h5_path: str, index_path: str, name: str) -> faiss.IndexFlatIP:
    index = None
    for batch_embeddings, _ in iter_embeddings(h5_path):
        dim = batch_embeddings.shape[1]
        if index is None:
            print(f"Building {name} Exhaustive Flat index...")
            index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(batch_embeddings)
        index.add(batch_embeddings)
    if index_path:
        faiss.write_index(index, index_path)
    return index

def load_or_create_index(h5_path: str, index_path: str, cache_path: str, name: str) -> Tuple[faiss.IndexFlatIP, List[str]]:
    if os.path.exists(index_path) and os.path.exists(cache_path):
        print(f"Loading existing {name} index...")
        index = faiss.read_index(index_path)
        with open(cache_path, 'r') as f:
            accessions = json.load(f)
        return index, accessions
    
    # If index or cache missing, build it
    index = build_index(h5_path, index_path, name)

    # Extract and cache accessions (ensuring we only cache dataset keys)
    with h5py.File(h5_path, 'r', libver='latest') as f:
        accessions = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]

    with open(cache_path, 'w') as f:
        json.dump(accessions, f)

    return index, accessions

print("Initializing FAISS indices...")
protein_index, protein_accessions = load_or_create_index(DEFAULT_H5_PATH, DEFAULT_INDEX_PATH, DEFAULT_CACHE_PATH, "Protein")
dna_index, dna_accessions = load_or_create_index(DNA_H5_PATH, DNA_INDEX_PATH, DNA_CACHE_PATH, "DNA")

print("Indices ready.")

# =============================================================================
# INTERNAL LOGIC (CORE)
# =============================================================================

# --- Embedding ---

def _embed_protein(sequence: str) -> np.ndarray:
    # ProtT5 reference recipe: substitute rare/ambiguous residues with X
    seq = re.sub(r"[UZOB]", "X", sequence.upper())
    processed_seq = " ".join(list(seq))
    
    inputs = protein_tokenizer(processed_seq, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = protein_model(**inputs, return_dict=True)
        # Exclude the trailing </s> (EOS) token from the mean pool to match bio_embeddings distribution
        residue_embeddings = outputs.last_hidden_state[0, :len(seq), :]
        
    return residue_embeddings.mean(dim=0).cpu().numpy().astype(np.float32)

def _embed_dna(sequence: str) -> np.ndarray:
    inputs = dna_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=DNA_MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = dna_model(**inputs, return_dict=True)
        mean_pooled = outputs.last_hidden_state.mean(dim=1).squeeze()
    return mean_pooled.cpu().numpy().astype(np.float32)

def _embed_rerank_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Generates semantic embeddings using Qwen3-Embedding.
    1. Uses last-token pooling (standard for decoder-based embeddings).
    2. Uses explicit max_length and truncation.
    3. Distinct paths for query (with instructions) and documents (plain text).
    """
    if is_query:
        # Instruction-aware path for queries
        instruction = (
            "Given a bioinformatics context or sequence retrieval prompt, identify relevant biological "
            "entities, molecular functions, biological processes, protein families and domains, "
            "subcellular localizations, taxonomic and evolutionary constraints, ontology-related terms, "
            "and structural or functional relationships to retrieve matching entries from the Swiss-Prot database."
        )
        processed_texts = [f"{instruction}\nQuery: {t}" for t in texts]
    else:
        # Plain text path for documents
        processed_texts = texts
        
    inputs = rerank_tokenizer(
        processed_texts, 
        padding=True, 
        truncation=True, 
        max_length=RERANK_MAX_LENGTH, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = rerank_model(**inputs, return_dict=True)
        # Use last-token pooling of the last hidden state
        # Explicitly cast to float32 before numpy conversion (numpy does not support bfloat16)
        embeddings = outputs.last_hidden_state[:, -1].to(torch.float32)
        
    return embeddings.cpu().numpy()

# --- Search ---

def _perform_vector_search(index, query_emb: np.ndarray, k: int):
    query_vec = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_vec)
    # Enable full parallel computation as requested (removed omp_set_num_threads(1))
    return index.search(query_vec, k)

# --- Reranking Helpers ---

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
    """
    Performs Z-score normalization for scale robustness and translation invariance.
    Ensures both retrieval and semantic signals are in a comparable space.
    """
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

@app.post("/search/protein")
async def search_protein(request: SearchRequest):
    try:
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(executor, _embed_protein, request.sequence)
        dist, idxs = await loop.run_in_executor(executor, _perform_vector_search, protein_index, emb, request.k)
        results = [{"accession": protein_accessions[i], "score": float(dist[0][j])} for j, i in enumerate(idxs[0]) if i != -1]
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/dna")
async def search_dna(request: SearchRequest):
    try:
        loop = asyncio.get_event_loop()
        emb = await loop.run_in_executor(executor, _embed_dna, request.sequence)
        dist, idxs = await loop.run_in_executor(executor, _perform_vector_search, dna_index, emb, request.k)
        results = [{"accession": dna_accessions[i], "score": float(dist[0][j])} for j, i in enumerate(idxs[0]) if i != -1]
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rerank")
async def rerank(request: RerankRequest):
    try:
        loop = asyncio.get_event_loop()
        
        # 1. Semantic Embedding
        passages = [_format_record_for_embedding(rec) for rec in request.records]
        query_vec = await loop.run_in_executor(executor, _embed_rerank_texts, [request.context_query], True)
        doc_vecs = await loop.run_in_executor(executor, _embed_rerank_texts, passages, False)
        
        # 2. Raw Semantic Scores (Cosine Similarity)
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)
        doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
        semantic_scores = np.dot(doc_vecs, query_vec.T).flatten()

        # 3. Z-Score Calibration
        retrieval_raw = np.array([r.get("_search_score", 0.0) for r in request.records])
        retrieval_z = _normalize_z_score(retrieval_raw)
        semantic_z = _normalize_z_score(semantic_scores)

        # 4. Conformal Ranking Uncertainty Estimation (α_i)
        # Parameter-free and distribution-free exchangeability estimation.
        alpha = _compute_conformal_uncertainty(retrieval_z)

        # 5. Principled Confidence-Aware Fusion
        # f_i = s_i + α_i * λ * (r_i - s_i)
        # Correction is applied only where the retrieval ranking is statistically exchangeable.
        LAMBDA = RERANK_LAMBDA
        
        reranked_list = []
        for i, record in enumerate(request.records):
            final_fused_score = retrieval_z[i] + (alpha[i] * LAMBDA * (semantic_z[i] - retrieval_z[i]))
            
            # Store metadata for transparency
            record["_search_score"] = float(final_fused_score)
            record["_uncertainty_alpha"] = float(alpha[i])
            reranked_list.append(record)

        # 6. Stable Rank Sort
        reranked_list.sort(key=lambda x: x["_search_score"], reverse=True)
        
        return {"results": reranked_list[:request.top_n]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SEARCH_SERVICE_HOST, port=SEARCH_SERVICE_PORT)
