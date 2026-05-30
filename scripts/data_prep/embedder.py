import os
import sys
import h5py
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config.settings import DEFAULT_H5_PATH, SWISSPROT_CSV_PATH, CONTEXT_H5_PATH
from config.service_params import PROTEIN_MODEL_NAME, REFINE_MODEL_NAME, ESMC_MAX_LENGTH, REFINE_MAX_LENGTH
from protseq.common.utils import format_context_for_embedding

def embed_sequences():
    """
    Generates embeddings for Swiss-Prot sequences using ESMC-300M.
    Implements checkpointing and OOM protection.
    """
    print(f"\n--- Phase 1: Sequence Embedding ({PROTEIN_MODEL_NAME}) ---")
    
    if not os.path.exists(SWISSPROT_CSV_PATH):
        print(f"Error: Metadata CSV not found at {SWISSPROT_CSV_PATH}. Run downloader first.")
        return

    # Load ESMC Model
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein
    except ImportError:
        print("Error: 'esm' library not found. Install it with: pip install esm@git+https://github.com/Biohub/esm.git")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")
    model = ESMC.from_pretrained(PROTEIN_MODEL_NAME).to(device)
    model.eval()

    # Load metadata
    df = pl.read_csv(SWISSPROT_CSV_PATH)
    total_records = df.height
    print(f"Loaded {total_records} records from CSV.")

    # Check for existing H5 file to resume
    existing_keys = set()
    if os.path.exists(DEFAULT_H5_PATH):
        with h5py.File(DEFAULT_H5_PATH, 'r') as f:
            existing_keys = set(f.keys())
    
    print(f"Found {len(existing_keys)} existing sequence embeddings.")

    # Open H5 for appending
    with h5py.File(DEFAULT_H5_PATH, 'a') as f:
        processed_count = 0
        skipped_count = 0
        
        for row in tqdm(df.iter_rows(named=True), total=total_records, desc="Seq Embedding"):
            acc = row['accession']
            seq = row['sequence']
            
            if acc in existing_keys:
                continue
            
            if not seq or len(seq) > ESMC_MAX_LENGTH:
                skipped_count += 1
                continue

            try:
                protein = ESMProtein(sequence=seq)
                with torch.no_grad():
                    output = model(protein)
                    # Mean pool over residue dimension
                    emb = output.embeddings.mean(dim=1).cpu().numpy().flatten()
                
                f.create_dataset(acc, data=emb, compression=None)
                processed_count += 1
                
                if processed_count % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM Error for accession {acc} (len={len(seq)}). Skipping.")
                torch.cuda.empty_cache()
                skipped_count += 1
                continue
            except Exception as e:
                print(f"\nError embedding {acc}: {e}")
                skipped_count += 1
                continue

    print(f"Newly processed: {processed_count}, Skipped: {skipped_count}")

def embed_contexts():
    """
    Generates semantic embeddings for biological contexts using ModernBERT-bio-large.
    Used for precomputing refinement-context embeddings.
    """
    print(f"\n--- Phase 2: Context Embedding ({REFINE_MODEL_NAME}) ---")
    
    if not os.path.exists(SWISSPROT_CSV_PATH):
        print(f"Error: Metadata CSV not found at {SWISSPROT_CSV_PATH}.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {REFINE_MODEL_NAME} to {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(REFINE_MODEL_NAME)
    model = AutoModel.from_pretrained(REFINE_MODEL_NAME).to(device)
    model.eval()

    df = pl.read_csv(SWISSPROT_CSV_PATH)
    total_records = df.height

    # Check for checkpoint
    existing_keys = set()
    if os.path.exists(CONTEXT_H5_PATH):
        with h5py.File(CONTEXT_H5_PATH, 'r') as f:
            existing_keys = set(f.keys())
    
    print(f"Found {len(existing_keys)} existing context embeddings.")

    with h5py.File(CONTEXT_H5_PATH, 'a') as f:
        processed_count = 0
        skipped_count = 0

        for row in tqdm(df.iter_rows(named=True), total=total_records, desc="Ctx Embedding"):
            acc = row['accession']
            
            if acc in existing_keys:
                continue

            try:
                # 1. Align biological context using unified formatting
                formatted_text = format_context_for_embedding(dict(row))
                
                # 2. Tokenize
                inputs = tokenizer(
                    formatted_text, 
                    padding=True, 
                    truncation=True, 
                    max_length=REFINE_MAX_LENGTH, 
                    return_tensors="pt"
                ).to(device)

                # 3. Embed with mean pooling
                with torch.no_grad():
                    outputs = model(**inputs)
                    # ModernBERT mean pooling
                    # Using attention_mask to avoid padding tokens
                    mask = inputs['attention_mask']
                    embeddings = outputs.last_hidden_state
                    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    mean_pooled = (sum_embeddings / sum_mask).cpu().numpy().flatten().astype(np.float32)

                # 4. Save
                f.create_dataset(acc, data=mean_pooled, compression=None)
                processed_count += 1

                if processed_count % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM Error for context {acc}. Skipping.")
                torch.cuda.empty_cache()
                skipped_count += 1
                continue
            except Exception as e:
                print(f"\nError embedding context {acc}: {e}")
                skipped_count += 1
                continue

    print(f"Newly processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    embed_sequences()
    embed_contexts()
