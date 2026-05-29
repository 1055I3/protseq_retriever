import pytest
from protseq.infra.clients.api_client import APIClient
import httpx
from unittest.mock import MagicMock

def test_api_client_retries_on_500():
    mock_client = MagicMock()
    # Simulate 500 error then 200
    mock_response_500 = MagicMock(status_code=500)
    mock_response_200 = MagicMock(status_code=200)
    mock_client.request.side_effect = [mock_response_500, mock_response_200]
    
    client = APIClient()
    client.client = mock_client
    
    # We need to mock time.sleep to avoid waiting
    import time
    with MagicMock(spec=time.sleep) as mock_sleep:
        response = client.request_with_retry("GET", "http://test.com")
        assert response.status_code == 200
        assert mock_client.request.call_count == 2
        assert mock_sleep.called
