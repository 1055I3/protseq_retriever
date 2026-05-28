#!/usr/bin/env python3
import os
import logging
import h5py
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BigBirdForMaskedLM
from CodonTransformer.CodonPrediction import predict_dna_sequence
from CodonTransformer.CodonUtils import ORGANISM2ID
from typing import List, Tuple, Dict

# Import configuration
import config

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("BioSeqPipeline")

# =============================================================================
# NON-NEGOTIABLE CONSTANTS
# =============================================================================
HYENA_MODEL_NAME = "LongSafari/hyenadna-medium-160k-seqlen-hf"
CODON_MODEL_NAME = "adibvafa/CodonTransformer"

# Configure PyTorch threads
torch.set_num_threads(config.THREAD_COUNT)

# =============================================================================
# TAXONOMY & HOST CONFIGURATION
# =============================================================================
DEFAULT_HOST = "Escherichia coli general"

# =============================================================================
# FUNCTIONAL COMPONENTS
# =============================================================================

def load_refseq_lookup(path: str) -> Dict[str, str]:
    """Loads existing RefSeq CDS mappings into a fast lookup dictionary."""
    if not os.path.exists(path):
        logger.warning(f"RefSeq lookup file not found at {path}. Proceeding with full synthetic generation.")
        return {}
    
    logger.info(f"Loading RefSeq lookup from {path}...")
    df = pl.read_csv(path)
    return dict(zip(df["swissprot_id"], df["cds_sequence"]))

def initialize_stack() -> Tuple[AutoModel, AutoTokenizer, BigBirdForMaskedLM, AutoTokenizer, torch.device]:
    """Initializes CodonTransformer and HyenaDNA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing models on {device}...")

    # 1. CodonTransformer (Back-translation)
    codon_tok = AutoTokenizer.from_pretrained(CODON_MODEL_NAME)
    codon_mod = BigBirdForMaskedLM.from_pretrained(CODON_MODEL_NAME).to(device)
    codon_mod.eval()
    
    # 2. HyenaDNA (Embedding)
    hyena_tok = AutoTokenizer.from_pretrained(HYENA_MODEL_NAME, trust_remote_code=True)
    hyena_mod = AutoModel.from_pretrained(HYENA_MODEL_NAME, trust_remote_code=True).to(device)
    hyena_mod.eval()
    
    return hyena_mod, hyena_tok, codon_mod, codon_tok, device

def resolve_host_dynamically(organism_name: str) -> str:
    """
    Robustly maps raw organism name to supported CodonTransformer host.
    Uses model's internal ORGANISM2ID mapping for fuzzy keyword matching.
    """
    try:
        org_lower = organism_name.lower()
        
        # 1. Look for significant substring matches
        for host in ORGANISM2ID.keys():
            host_lower = host.lower()
            if host_lower in org_lower or org_lower in host_lower:
                return host
                
        # 2. Heuristic word matching
        org_words = set(org_lower.replace("(", "").replace(")", "").replace("/", " ").replace("-", " ").split())
        for host in ORGANISM2ID.keys():
            host_words = set(host.lower().replace("(", "").replace(")", "").replace("/", " ").replace("-", " ").split())
            if org_words.intersection(host_words):
                return host
                
        return organism_name # Try raw if no obvious match
    except Exception:
        return organism_name

def back_translate_with_model(aa_sequence: str, organism: str, model: BigBirdForMaskedLM, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """
    Generates DNA from protein using CodonTransformer.
    Uses dynamic host resolution and proper attribute access for robustness.
    """
    target_host = resolve_host_dynamically(organism)
    
    try:
        output = predict_dna_sequence(
            protein=aa_sequence,
            organism=target_host,
            device=device,
            tokenizer=tokenizer,
            model=model,
            deterministic=True
        )
        # FIX: proper attribute access (.predicted_dna)
        return output.predicted_dna.upper()
    except Exception:
        # Fallback to DEFAULT_HOST
        try:
            output = predict_dna_sequence(
                protein=aa_sequence,
                organism=DEFAULT_HOST,
                device=device,
                tokenizer=tokenizer,
                model=model,
                deterministic=True
            )
            return output.predicted_dna.upper()
        except Exception as e_inner:
            logger.error(f"Back-translation failed for {organism} (resolved as {target_host}) and fallback {DEFAULT_HOST}: {e_inner}")
            return aa_sequence

def compute_masked_mean_embedding(dna_batch: List[str], model: AutoModel, tokenizer: AutoTokenizer, device: torch.device) -> np.ndarray:
    """Computes mean embeddings, handling cases where attention_mask is missing."""
    inputs = tokenizer(
        dna_batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.EMBEDDING_MAX_LENGTH
    ).to(device)
    
    with torch.inference_mode():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state # [B, L, D]
        
        # Check if attention_mask is available
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1) # [B, L, 1]
            # Mean pooling ignoring padding
            sum_embeddings = (hidden * mask).sum(dim=1)
            num_tokens = mask.sum(dim=1)
            embeddings = sum_embeddings / num_tokens
        else:
            # Fallback to standard mean pooling if mask is missing
            embeddings = hidden.mean(dim=1)
        
        # Ensure Batch dimension exists
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
            
    return embeddings.cpu().numpy().astype(np.float32)

def save_to_h5(h5_path: str, accessions: List[str], embeddings: np.ndarray):
    """Writes 1D embeddings to HDF5 keyed by accession. Compatible with search_service."""
    with h5py.File(h5_path, "a", libver='latest') as f:
        for acc, emb in zip(accessions, embeddings):
            if acc in f:
                del f[acc]
            f.create_dataset(acc, data=emb, compression="gzip", compression_opts=4)
        f.flush()

class BioDataset(Dataset):
    """Provides access to UniProt sequences and organism metadata."""
    def __init__(self, df: pl.DataFrame):
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.row(idx, named=True)
        return row["Entry"], row["Sequence"], row["Organism"]

def run_pipeline():
    """Orchestrates the data-to-embedding pipeline using SwissProt metadata."""
    try:
        # 1. Initialization
        df = pl.read_csv(config.SWISSPROT_TSV, separator="\t")
        lookup = load_refseq_lookup(config.REFSEQ_CDS_CSV)
        h_mod, h_tok, c_mod, c_tok, device = initialize_stack()
        
        dataset = BioDataset(df)
        dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        logger.info(f"Pipeline started: {len(df)} total, {len(lookup)} cached RefSeq.")

        # 2. Main Loop
        for i, (batch_acc, batch_aa, batch_org) in enumerate(dataloader):
            final_dna_batch = []
            
            for acc, aa, org in zip(batch_acc, batch_aa, batch_org):
                # Step A: Priority Lookup
                dna = lookup.get(acc)
                
                if not dna:
                    # Step B: Back-translation with organism metadata
                    dna = back_translate_with_model(aa, org, c_mod, c_tok, device)
                
                final_dna_batch.append(dna)
            
            # Step C: Embedding
            try:
                embeddings = compute_masked_mean_embedding(final_dna_batch, h_mod, h_tok, device)
                save_to_h5(config.EMBEDDING_OUTPUT_FILE, batch_acc, embeddings)
                
                if i % 5 == 0:
                    processed = min((i + 1) * config.BATCH_SIZE, len(df))
                    logger.info(f"Processed {processed}/{len(df)} ({(processed/len(df))*100:.1f}%)")
            except Exception as e:
                logger.error(f"Batch {i} embedding failed: {e}")
                continue

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Fatal pipeline crash: {e}")

if __name__ == "__main__":
    run_pipeline()
