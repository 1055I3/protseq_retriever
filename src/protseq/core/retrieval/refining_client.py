from typing import List, Dict, Any
from config.settings import SEARCH_SERVICE_URL
from protseq.infra.clients.api_client import default_api_client

class LocalRefiner:
    """
    Client for the Unified Protseq Gateway Refining service.
    Sends candidate records and context query to the remote service for biological refining.
    """
    def __init__(self):
        # No local model initialization required
        print("Connected to Unified Refining Service.")

    def refine_by_context(
        self, 
        hits: List[Dict[str, Any]], 
        context_query: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Refines candidates by calling the unified /refine service.
        Expects hits with 'accession' and 'score'.
        """
        if not hits:
            return []

        print(f"Refining {len(hits)} candidates via gateway...")
        
        response = default_api_client.request_with_retry(
            "POST", f"{SEARCH_SERVICE_URL}/refine",
            json={
                "hits": hits,
                "context_query": context_query,
                "top_n": top_n
            }
        )
        
        return response.json()["results"]
