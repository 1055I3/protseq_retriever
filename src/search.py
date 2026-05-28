import os
import socket
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

from src.config import SEARCH_SERVICE_URL, SEARCH_PROBE_TIMEOUT
from src.api_client import default_api_client


def _search_service_alive(url: str = SEARCH_SERVICE_URL, timeout: float = SEARCH_PROBE_TIMEOUT) -> bool:
    """Quick TCP liveness probe for the local search gateway.

    The api_client's retry loop waits ~31s + jitter across 5 attempts before
    giving up — meaningful for transient 429 / 5xx but pure overhead when the
    port is simply closed. A 0.5s connect probe lets us fail fast with a
    clear message instead of making the user wait a minute for the inevitable.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def search_protein_top_k(
    query_sequence: str,
    k: int = 25
) -> List[Tuple[str, float]]:
    """
    Client function to call the unified protein search service.
    """
    print(f"Searching protein index for top {k} matches...")
    if not _search_service_alive():
        raise ConnectionError(
            f"Search service at {SEARCH_SERVICE_URL} is not reachable — "
            "is the gateway (services/search_service.py) running on the "
            "expected port?"
        )

    response = default_api_client.request_with_retry(
        "POST", f"{SEARCH_SERVICE_URL}/search",
        json={"sequence": query_sequence, "k": k}
    )
    results = response.json()["results"]
    return [(r["accession"], r["score"]) for r in results]
