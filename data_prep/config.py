# =============================================================================
# CONFIGURATION
# =============================================================================

from multiprocessing import cpu_count

THREAD_COUNT = cpu_count() # Total logical cores available

# REQUIRED by NCBI: Set this to your active email address
EMAIL = "your@email.here"

# Retry logic parameters
MAX_RETRIES = 5
BASE_DELAY = 2

# Swissprot data source
SWISSPROT_TSV = "swissprot.tsv"

# RefSeq Mapped Data
REFSEQ_CDS_CSV = "refseq_swissprot_cds.csv"

# Pipeline parameters
BATCH_SIZE = 256 # Optimal batch size for high-core CPU inference

# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
EMBEDDING_OUTPUT_FILE = "per-gene.h5"
EMBEDDING_MAX_LENGTH = 160_000
