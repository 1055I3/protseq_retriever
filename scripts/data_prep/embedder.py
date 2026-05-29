import os
import sys
import h5py
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from config.settings import DEFAULT_H5_PATH, SWISSPROT_CSV_PATH
from config.service_params import PROTEIN_MODEL_NAME, ESMC_MAX_LENGTH

def embed_sequences():
    """
    Generates embeddings for Swiss-Prot sequences using ESMC-300M.
    Implements checkpointing and OOM protection.
    """
    print(f"Initializing Embedding Pipeline with {PROTEIN_MODEL_NAME}...")
    
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
    
    print(f"Found {len(existing_keys)} existing embeddings. Resuming...")

    # Filter out already processed and sequences that are too long
    # We'll do this in the loop for memory efficiency

    # Open H5 for appending
    with h5py.File(DEFAULT_H5_PATH, 'a') as f:
        processed_count = 0
        skipped_count = 0
        
        for row in tqdm(df.iter_rows(named=True), total=total_records, desc="Embedding"):
            acc = row['accession']
            seq = row['sequence']
            
            if acc in existing_keys:
                continue
            
            if not seq or len(seq) > ESMC_MAX_LENGTH:
                skipped_count += 1
                continue

            try:
                # Prepare protein
                protein = ESMProtein(sequence=seq)
                
                # Run inference
                with torch.no_grad():
                    # The model expects a single ESMProtein or list
                    output = model(protein)
                    # output.embeddings is [1, seq_len, 960]
                    # Mean pooling over residue dimension
                    # Note: check if output.embeddings includes special tokens
                    emb = output.embeddings.mean(dim=1).cpu().numpy().flatten()
                
                # Save to H5
                f.create_dataset(acc, data=emb, compression=None)
                processed_count += 1
                
                # Clear cache occasionally for OOM safety
                if processed_count % 100 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                print(f"\nOOM Error for accession {acc} (len={len(seq)}). Skipping and clearing cache.")
                torch.cuda.empty_cache()
                skipped_count += 1
                continue
            except Exception as e:
                print(f"\nError embedding {acc}: {e}")
                skipped_count += 1
                continue

    print(f"\nEmbedding complete.")
    print(f"Newly processed: {processed_count}")
    print(f"Skipped/Failed: {skipped_count}")
    print(f"Total in H5: {len(h5py.File(DEFAULT_H5_PATH, 'r').keys())}")

if __name__ == "__main__":
    embed_sequences()
