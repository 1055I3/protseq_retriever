from typing import List, Dict, Any
from src.config import SEARCH_SERVICE_URL
from src.api_client import default_api_client

class LocalReranker:
    """
    Client for the Unified BioSeq Gateway Reranking service.
    Sends candidate records and context query to the remote service for biological reranking.
    """
    def __init__(self):
        # No local model initialization required
        print("Connected to Unified Reranking Service.")

    def rerank_by_context(
        self, 
        records: List[Dict[str, Any]], 
        context_query: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks Swiss-Prot candidates by calling the unified /rerank service.
        """
        if not records:
            return []

        print(f"Reranking {len(records)} candidates via gateway...")
        
        response = default_api_client.request_with_retry(
            "POST", f"{SEARCH_SERVICE_URL}/rerank",
            json={
                "records": records,
                "context_query": context_query,
                "top_n": top_n
            }
        )
        
        return response.json()["results"]
