from typing import List, Dict, Any
from src.api_client import default_api_client

def get_uniprot_records(accessions: List[str]) -> List[Dict[str, Any]]:
    if not accessions:
        return []
        
    print(f"Fetching UniProt records for: {', '.join(accessions)}")
    
    ids_query = " OR ".join([f"accession:{acc}" for acc in accessions])
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": ids_query,
        "format": "json",
        "size": len(accessions)
    }
    
    response = default_api_client.request_with_retry("GET", url, params=params)
    return response.json().get('results', [])
