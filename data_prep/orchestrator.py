import os
import sys
import logging
import traceback
from datetime import datetime

# Setup logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"data_prep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add parent and grandparent dirs to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_prep.downloader import download_swissprot
from data_prep.embedder import embed_sequences
from data_prep.indexer import build_faiss_index
from data_prep.convert_h5_layout import convert_h5_to_compatible
from services.config import DEFAULT_H5_PATH

def run_pipeline():
    """
    Orchestrates the entire Swiss-Prot Data Preparation Pipeline.
    Designed to be robust and resume-able.
    """
    logger.info("=== Starting BioSeq Data Preparation Pipeline ===")
    
    # 1. Download Metadata
    try:
        logger.info("Phase 1: Downloading Swiss-Prot Metadata...")
        download_swissprot()
        logger.info("Phase 1 Complete.")
    except Exception as e:
        logger.error(f"Phase 1 Failed: {e}")
        logger.error(traceback.format_exc())
        # We might continue if partial data exists, but usually better to fix and rerun
        return

    # 2. Generate Embeddings
    try:
        logger.info("Phase 2: Generating ESMC-300M Embeddings...")
        embed_sequences()
        logger.info("Phase 2 Complete.")
    except Exception as e:
        logger.error(f"Phase 2 Failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("Note: Embedder supports checkpointing. Fix the issue and rerun orchestrator.")
        return

    # 3. Optimize Layout (Optional but recommended for compatibility)
    try:
        logger.info("Phase 3: Optimizing HDF5 Layout for compatibility...")
        temp_target = f"{DEFAULT_H5_PATH}.compatible"
        convert_h5_to_compatible(DEFAULT_H5_PATH, temp_target)
        
        # Replace original with optimized
        backup = f"{DEFAULT_H5_PATH}.bak"
        if os.path.exists(backup): os.remove(backup)
        os.rename(DEFAULT_H5_PATH, backup)
        os.rename(temp_target, DEFAULT_H5_PATH)
        
        logger.info("Phase 3 Complete.")
    except Exception as e:
        logger.warning(f"Phase 3 Failed (Non-critical): {e}")
        # Not fatal, search_service can still read original if needed

    # 4. Build FAISS Index
    try:
        logger.info("Phase 4: Building FAISS HNSW Index...")
        build_faiss_index()
        logger.info("Phase 4 Complete.")
    except Exception as e:
        logger.error(f"Phase 4 Failed: {e}")
        logger.error(traceback.format_exc())
        return

    logger.info("=== BioSeq Data Preparation Pipeline Successfully Completed ===")

if __name__ == "__main__":
    run_pipeline()
