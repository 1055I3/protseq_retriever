# BioSeq Retriever

BioSeq Retriever is an advanced bioinformatics pipeline designed for context-aware biological sequence search. It leverages Large Language Models (LLMs), LangGraph, and FAISS to provide a highly flexible system that can interpret natural language queries, classify sequence types, and perform multi-stage similarity searches.

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n bioseq python=3.12 -y
conda activate bioseq
```

### 2. Install Dependencies
Install the required packages using Conda where available, and pip for others:
```bash
conda install -c conda-forge h5py faiss-cpu numpy httpx pyfaidx transformers pytorch fastapi uvicorn -y
pip install langchain-mistralai langchain-openai langgraph tiktoken sentencepiece protobuf
```

*Note: If you have a GPU, you might prefer `faiss-gpu`.*

### 3. Configuration & API Keys
The pipeline requires either a **Mistral AI API Key** or an **OpenAI API Key**.

#### Security & Paths
Configure the data environment using the following variables:
- `BIOSEQ_H5_PATH`: Path to the .h5 embeddings file (default: `data/per-protein.h5`).
- `BIOSEQ_INDEX_PATH`: Path to the FAISS index (default: derived from H5 path).
- `BIOSEQ_ACCESSIONS_CACHE_PATH`: Path to the accession JSON cache (default: derived from H5 path).
- `BIOSEQ_ALLOWED_DATA_DIR`: Directory allowed for FASTA file resolution (default: `data`). This prevents directory traversal attacks.
- `BIOSEQ_FETCH_TIMEOUT`: Timeout for UniProt API calls in seconds (default: `10.0`).

#### AI Providers
To force a provider or model:
```bash
export BIOSEQ_LLM_PROVIDER=mistral # or 'openai'
export BIOSEQ_EMBEDDINGS_PROVIDER=mistral # or 'openai'
export MISTRAL_API_KEY='your-key'
```

## What This Code Does
- **Intelligent Input Parsing**: Uses LLMs to extract sequences, file paths, and semantic context from natural language prompts.
- **Automated Sequence Classification**: Employs Chain-of-Thought reasoning to determine if a sequence is DNA or Protein.
- **Secure File Resolution**: Resolves sequences from file paths while enforcing directory restrictions to prevent unauthorized access.
- **Memory-Efficient Indexing**: Uses HDF5 generators and HNSW indexing to handle large-scale embedding datasets without exceeding memory limits.
- **High-Dimensional Similarity Search**: Performs initial ranking of protein sequences using ProtT5 embeddings.
- **Cloud-Powered Semantic Reranking**: Refines results using semantic context-aware reranking via `httpx`.
- **UniProt Data Integration**: Fetches rich biological metadata with built-in timeouts and error handling.

## Integration: Using the Pipeline
### Command Line Interface
You can run the pipeline directly from the terminal:
```bash
python pipeline_interface.py "I have a sequence: MALW... find matches involved in insulin signaling."
```

### Python API
```python
from src.pipeline import run_bioseq_pipeline

# Invoke the pipeline
result = run_bioseq_pipeline("Compare this sequence: MKTLL... against human insulin markers.")
```

## Execution Flow
1. **Extraction & Classification**: The LLM parses the prompt and classifies the molecule.
2. **Short-Circuit Error Handling**: If any node fails, the graph immediately terminates and returns the error in the state.
3. **Dynamic Resolution**: File paths are validated for security and resolved via `pyfaidx`.
4. **Microservice Processing**: Sequence embedding and similarity search are performed by dedicated services if enabled.
5. **Contextual Refinement (Reranking)**: Top 5 matches are selected based on semantic alignment with the user's query context.

## Running the System
The system relies on the **Unified BioSeq Gateway Service** to handle all sequence embedding, similarity search, and contextual reranking.

### Start the Unified Gateway Service
```bash
python bioseq_retriever/services/search_service.py
```
This service loads the required models (ProtT5, HyenaDNA, BioE5) and FAISS indices, exposing endpoints for protein search, DNA search, and biological reranking.

### Run Pipeline
The pipeline now operates asynchronously:
```bash
python pipeline_interface.py "I have a sequence: MALW..."
```

## Project & File Structure
- `src/`: Core logic and pipeline modules.
  - pipeline.py: LangGraph workflow and LLM node orchestration.
  - reranking.py: Semantic similarity logic using Mistral cloud embeddings.
  - utils.py: DNA translation tables, API environment setup, FASTA parsing, and sequence cleaning.
  - search.py: Unified Search Service client.
  - api_client.py: Centralized API client with pooling and exponential backoff.
  - config.py: Environment configuration and service settings.
  - data_fetcher.py: REST client for UniProt using `httpx`.
- services/: Unified Search Service.
  - search_service.py: Unified gateway for Protein/DNA embeddings, FAISS indices, and biological reranking.
  - config.py: Service-specific configuration (ports, FAISS params).
- `data/`: Directory for embeddings and FAISS indexes.
- `../../../tests/backend/bioseq_retriever/`: Automated unit and pipeline tests.
- `pipeline_interface.py`: CLI entry point script.

## Limitations and Remarks
- **API Dependency**: Requires an active Mistral AI or OpenAI API key.
- **Memory Usage**: ProtT5 loading requires significant RAM (~8GB+ recommended).
- **Sequence Length**: Assumes DNA is in-frame and divisible by 3.
- **Data Source**: Dependent on UniProt database coverage and pre-computed embedding quality.
- **Local Data Dependency**: Requires pre-existing data/ directory with embeddings and FAISS indexes.
