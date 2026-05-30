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

from scripts.data_prep.downloader import download_swissprot
from scripts.data_prep.embedder import embed_sequences, embed_contexts
from scripts.data_prep.indexer import build_faiss_index
from scripts.data_prep.convert_h5_layout import convert_h5_to_compatible
from config.settings import DEFAULT_H5_PATH, CONTEXT_H5_PATH

def run_pipeline():
    """
    Orchestrates the entire Swiss-Prot Data Preparation Pipeline.
    Designed to be robust and resume-able.
    """
    logger.info("=== Starting Protseq Data Preparation Pipeline ===")
    
    # 1. Download Metadata
    try:
        logger.info("Phase 1: Downloading Swiss-Prot Metadata...")
        download_swissprot()
        logger.info("Phase 1 Complete.")
    except Exception as e:
        logger.error(f"Phase 1 Failed: {e}")
        logger.error(traceback.format_exc())
        return

    # 2. Generate Protein Embeddings (Phase 2a)
    try:
        logger.info("Phase 2a: Generating ESMC-300M Sequence Embeddings...")
        embed_sequences()
        logger.info("Phase 2a Complete.")
    except Exception as e:
        logger.error(f"Phase 2a Failed: {e}")
        logger.error(traceback.format_exc())
        return

    # 3. Generate Context Embeddings (Phase 2b)
    try:
        logger.info("Phase 2b: Generating ModernBERT-bio-large Context Embeddings...")
        embed_contexts()
        logger.info("Phase 2b Complete.")
    except Exception as e:
        logger.error(f"Phase 2b Failed: {e}")
        logger.error(traceback.format_exc())
        return

    # 4. Optimize Layout (Optional but recommended for compatibility)
    try:
        logger.info("Phase 3: Optimizing HDF5 Layouts for compatibility...")
        for h5_path in [DEFAULT_H5_PATH, CONTEXT_H5_PATH]:
            if not os.path.exists(h5_path): continue
            
            temp_target = f"{h5_path}.compatible"
            convert_h5_to_compatible(h5_path, temp_target)
            
            backup = f"{h5_path}.bak"
            if os.path.exists(backup): os.remove(backup)
            os.rename(h5_path, backup)
            os.rename(temp_target, h5_path)
        
        logger.info("Phase 3 Complete.")
    except Exception as e:
        logger.warning(f"Phase 3 Failed (Non-critical): {e}")

    # 5. Build FAISS Index
    try:
        logger.info("Phase 4: Building FAISS HNSW Index...")
        build_faiss_index()
        logger.info("Phase 4 Complete.")
    except Exception as e:
        logger.error(f"Phase 4 Failed: {e}")
        logger.error(traceback.format_exc())
        return

    logger.info("=== Protseq Data Preparation Pipeline Successfully Completed ===")

if __name__ == "__main__":
    run_pipeline()
