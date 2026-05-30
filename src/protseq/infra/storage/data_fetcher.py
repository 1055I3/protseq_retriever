import os
import sys
from typing import List, Dict, Any

from config.settings import SEARCH_SERVICE_URL
from protseq.infra.clients.api_client import default_api_client

def get_uniprot_records(accessions: List[str]) -> List[Dict[str, Any]]:
    """
    Retrieves full Swiss-Prot records for a list of accessions.
    Calls the unified search gateway's hydration endpoint.
    """
    if not accessions:
        return []

    print(f"Hydrating {len(accessions)} records via gateway...")
    
    response = default_api_client.request_with_retry(
        "POST", f"{SEARCH_SERVICE_URL}/hydrate",
        json={"accessions": accessions}
    )
    
    return response.json()["records"]
