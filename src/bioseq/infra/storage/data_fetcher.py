import os
import polars as pl
from typing import List, Dict, Any, Optional
from config.settings import SWISSPROT_CSV_PATH

class DataFetcher:
    """
    Handles retrieval of protein metadata from the local Swiss-Prot cache.
    Replaces the UniProt REST API client for production efficiency.
    """
    def __init__(self):
        self._df = None
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(SWISSPROT_CSV_PATH):
            self._df = pl.read_csv(SWISSPROT_CSV_PATH)
        else:
            print(f"Warning: Metadata cache not found at {SWISSPROT_CSV_PATH}")

    def get_records(self, accessions: List[str]) -> List[Dict[str, Any]]:
        """Retrieves rich records for the given accessions from local CSV."""
        if self._df is None or not accessions:
            return []
        
        # Filter and convert to list of dicts
        hits = self._df.filter(pl.col("accession").is_in(accessions))
        return hits.to_dicts()

# Global singleton-like instance
local_fetcher = DataFetcher()

def get_uniprot_records(accessions: List[str]) -> List[Dict[str, Any]]:
    """Shim for backward compatibility with the new local fetcher."""
    return local_fetcher.get_records(accessions)
