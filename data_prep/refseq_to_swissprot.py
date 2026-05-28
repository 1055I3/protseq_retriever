#!/usr/bin/env python3
import os
import time
import random
import ssl
import pandas as pd
from Bio import Entrez, SeqIO
from urllib.error import HTTPError, URLError
from typing import List, Dict, Optional, Generator, Tuple

# =============================================================================
# SSL MONKEY-PATCH (Global workaround for local certificate errors)
# =============================================================================
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Import configuration
import config

# Apply NCBI Configuration
Entrez.email = config.EMAIL
# Securely load API Key from environment if present
ncbi_api_key = os.environ.get("NCBI_API_KEY")
if ncbi_api_key:
    Entrez.api_key = ncbi_api_key

# Local aliases for convenience
BATCH_SIZE = config.BATCH_SIZE
MAX_RETRIES = config.MAX_RETRIES
BASE_DELAY = config.BASE_DELAY
OUTPUT_FILE = config.OUTPUT_FILE

# Non-negotiable search query
SEARCH_QUERY = 'srcdb_refseq[PROP] AND biomol_mrna[PROP] AND "UniProtKB/Swiss-Prot"[db_xref]'

# =============================================================================
# ROBUST API UTILS
# =============================================================================

def get_ssl_context():
    """Creates a custom SSL context to bypass certain local ASN1/certificate errors."""
    context = ssl._create_unverified_context()
    return context

def retry_call(func, *args, **kwargs):
    """Executes a function with exponential backoff and custom SSL context."""
    # Inject custom SSL context for all network calls
    if "context" not in kwargs:
        kwargs["context"] = get_ssl_context()
        
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (HTTPError, URLError, Exception) as e:
            # Handle specific Bio.Entrez handle closing if it's an HTTPError
            if attempt == MAX_RETRIES - 1:
                print(f"\n[ERROR] Final attempt failed: {e}")
                raise e
            
            # Exponential backoff with jitter
            wait = (BASE_DELAY ** attempt) + random.uniform(0, 1)
            print(f"\n[WARN] Attempt {attempt+1} failed ({e}). Retrying in {wait:.2f}s...")
            time.sleep(wait)

# =============================================================================
# FUNCTIONAL LOGIC
# =============================================================================

def get_search_history(query: str) -> Tuple[int, str, str]:
    """Performs search and returns total count, WebEnv, and QueryKey."""
    print(f"Searching NCBI for: {query}")
    handle = retry_call(Entrez.esearch, db="nucleotide", term=query, usehistory="y")
    results = Entrez.read(handle)
    handle.close()
    
    return int(results["Count"]), results["WebEnv"], results["QueryKey"]

def fetch_batch_records(webenv: str, query_key: str, start: int, size: int) -> List:
    """Fetches a batch of GenBank records from the history server."""
    handle = retry_call(
        Entrez.efetch,
        db="nucleotide",
        rettype="gb",
        retmode="text",
        retstart=start,
        retmax=size,
        webenv=webenv,
        query_key=query_key,
        timeout=30
    )
    records = list(SeqIO.parse(handle, "genbank"))
    handle.close()
    return records

def extract_swissprot_id(db_xrefs: List[str]) -> Optional[str]:
    """Extracts the Swiss-Prot/UniProtKB accession from db_xref qualifiers."""
    for xref in db_xrefs:
        if "UniProtKB/Swiss-Prot" in xref or "Swiss-Prot" in xref:
            return xref.split(":")[-1].strip()
    return None

def parse_cds_entries(record) -> Generator[Dict[str, str], None, None]:
    """Parses CDS features from a record and yields mapping dicts."""
    refseq_id = record.id
    for feature in record.features:
        if feature.type == "CDS":
            sp_id = extract_swissprot_id(feature.qualifiers.get("db_xref", []))
            if sp_id:
                try:
                    # Extract the nucleotide sequence for this CDS
                    cds_seq = str(feature.extract(record.seq)).upper()
                    if cds_seq:
                        yield {
                            "swissprot_id": sp_id,
                            "refseq_id": refseq_id,
                            "cds_sequence": cds_seq
                        }
                except Exception:
                    continue

def process_pipeline(total_count: int, webenv: str, query_key: str) -> List[Dict[str, str]]:
    """Orchestrates fetching and extraction across all batches."""
    all_data = []
    
    print(f"Processing {total_count} potential records in batches of {BATCH_SIZE}...")
    
    for start in range(0, total_count, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_count)
        print(f"\rFetching records {start} to {end}...", end="", flush=True)
        
        try:
            records = fetch_batch_records(webenv, query_key, start, BATCH_SIZE)
            for rec in records:
                all_data.extend(list(parse_cds_entries(rec)))
            
            # Respect NCBI's rate limit (3 requests/sec without API key)
            time.sleep(0.34)
            
        except Exception as e:
            print(f"\nSkipping batch {start}-{end} due to persistent error: {e}")
            continue

    print("\nExtraction complete.")
    return all_data

def display_stats_and_save(data: List[Dict[str, str]]):
    """Calculates statistics using Pandas and saves to CSV."""
    if not data:
        print("No valid Swiss-Prot/RefSeq CDS pairs found.")
        return

    df = pd.DataFrame(data)
    
    # Calculate lengths for statistics
    lengths = df["cds_sequence"].str.len()
    
    stats = {
        "Total Sequences": len(df),
        "Shortest CDS (bp)": lengths.min(),
        "Longest CDS (bp)": lengths.max(),
        "Average CDS (bp)": round(lengths.mean(), 2)
    }
    
    print("\n" + "="*30)
    print("      BASIC STATISTICS")
    print("="*30)
    for k, v in stats.items():
        print(f"{k:<20}: {v}")
    print("="*30 + "\n")
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data successfully saved to {OUTPUT_FILE}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    try:
        total, webenv, qkey = get_search_history(SEARCH_QUERY)
        
        if total == 0:
            print("No records matched the search criteria.")
            return

        # Start the processing pipeline
        extracted_data = process_pipeline(total, webenv, qkey)
        
        # Analyze and save
        display_stats_and_save(extracted_data)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
