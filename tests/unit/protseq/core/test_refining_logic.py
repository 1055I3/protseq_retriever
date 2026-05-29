import pytest
import numpy as np
from protseq.app.gateway.search_service import _compute_conformal_uncertainty

def test_conformal_uncertainty_basic_distribution():
    scores = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    uncertainty = _compute_conformal_uncertainty(scores)
    
    # Check bounds
    assert np.all(uncertainty >= 0) and np.all(uncertainty <= 1)
    
    # Isolated/high score (index 0) should have lower uncertainty than middle
    assert uncertainty[0] < uncertainty[2]

def test_conformal_uncertainty_degenerate_case():
    scores = np.array([0.5])
    uncertainty = _compute_conformal_uncertainty(scores)
    assert len(uncertainty) == 1
    assert uncertainty[0] == 0.5
