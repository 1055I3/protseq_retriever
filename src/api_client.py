import httpx
import time
import random
from typing import Callable, Any
from src.config import MAX_RETRIES, BACKOFF_FACTOR, FETCH_TIMEOUT

class APIClient:
    """
    Centralized API client with connection pooling, retries, and exponential backoff.
    """
    def __init__(self, timeout: float = FETCH_TIMEOUT):
        self.limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self.timeout = timeout
        self.client = httpx.Client(limits=self.limits, timeout=self.timeout)

    def request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Executes a request with exponential backoff for 429 and 5xx errors."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.request(method, url, **kwargs)
                
                # Success
                if response.status_code == 200:
                    return response
                
                # Rate limited or Server Error
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    wait_time = (BACKOFF_FACTOR ** attempt) + random.uniform(0, 1)
                    print(f"API Error {response.status_code}. Retrying in {wait_time:.2f}s (Attempt {attempt+1}/{MAX_RETRIES})...")
                    time.sleep(wait_time)
                    continue
                
                # Other errors (4xx) - don't retry by default unless specified
                response.raise_for_status()
                
            except httpx.RequestError as exc:
                wait_time = (BACKOFF_FACTOR ** attempt) + random.uniform(0, 1)
                print(f"Request Error {exc}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
        
        # If all retries fail
        raise Exception(f"Failed to execute {method} request to {url} after {MAX_RETRIES} attempts.")

    def close(self):
        self.client.close()

# Singleton-like instance for internal modules
default_api_client = APIClient()
