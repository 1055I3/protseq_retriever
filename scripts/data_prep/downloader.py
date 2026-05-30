import os
import sys
import polars as pl
import time
from typing import List, Dict, Any, Optional

from protseq.infra.clients.api_client import default_api_client
from config.settings import SWISSPROT_CSV_PATH

def download_swissprot():
    """
    Downloads all reviewed UniProtKB (Swiss-Prot) records and saves to CSV.
    Uses UniProt REST API with pagination.
    """
    print("Starting Swiss-Prot metadata download...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(SWISSPROT_CSV_PATH), exist_ok=True)
    
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = "reviewed:true"
    fields = "accession,id,protein_name,gene_names,organism_name,lineage,sequence,comments,xref_go,xref_pfam,keywords"
    
    params = {
        "query": query,
        "format": "json",
        "fields": fields,
        "size": 500  # Batch size
    }
    
    all_data = []
    total_downloaded = 0
    next_url = base_url
    
    start_time = time.time()
    
    try:
        while next_url:
            response = default_api_client.request_with_retry("GET", next_url, params=params if next_url == base_url else None)
            data = response.json()
            
            results = data.get("results", [])
            for res in results:
                locations = []
                for comment in res.get('comments', []):
                    if comment.get('commentType') == 'SUBCELLULAR_LOCATION':
                        locations.extend([l.get('location', {}).get('value', '') for l in comment.get('locations', [])])

                # Flatten complex structures for CSV compatibility
                record = {
                    "accession": res.get("primaryAccession"),
                    "id": res.get("uniProtkbId"),
                    "protein_name": res.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A"),
                    "gene_names": ", ".join([g.get("geneName", {}).get("value", "") for g in res.get("genes", [])]),
                    "organism_name": res.get("organism", {}).get("scientificName", "N/A"),
                    "lineage": " > ".join([t.get("scientificName", "") for t in res.get("organism", {}).get("lineage", [])]),
                    "sequence": res.get("sequence", {}).get("value", ""),
                    "functions": " | ".join([c.get("texts", [{}])[0].get("value", "") for c in res.get("comments", []) if c.get("commentType") == "FUNCTION"]),
                    "subcellular_locations": ", ".join(locations),
                    "go_terms": ", ".join([x.get("id") for x in res.get("uniProtKBCrossReferences", []) if x.get("database") == "GO"]),
                    "domains_families": ", ".join([x.get("id") for x in res.get("uniProtKBCrossReferences", []) if x.get("database") in ["Pfam", "InterPro"]]),
                    "keywords": ", ".join([k.get("value") for k in res.get("keywords", [])])
                }
                all_data.append(record)
            
            total_downloaded += len(results)
            print(f"Downloaded {total_downloaded} records...", end="\r")
            
            # Check for next page link in headers
            link_header = response.headers.get("Link")
            next_url = None
            if link_header:
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip("< >")
                        break
            
            if len(all_data) >= 50000:
                print(f"\nCheckpointing {len(all_data)} records to CSV...")
        
        print(f"\nDownload complete. Total: {total_downloaded}")
        print("Finalizing CSV file...")
        df = pl.from_dicts(all_data)
        df.write_csv(SWISSPROT_CSV_PATH)
        print(f"Metadata saved to {SWISSPROT_CSV_PATH}")
        
    except Exception as e:
        print(f"\nFatal error during download: {e}")
        if all_data:
            print("Attempting to save partial data...")
            pl.from_dicts(all_data).write_csv(f"{SWISSPROT_CSV_PATH}.partial")
        raise

if __name__ == "__main__":
    download_swissprot()
