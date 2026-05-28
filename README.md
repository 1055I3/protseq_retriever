# BioSeq Retriever

BioSeq Retriever is an advanced bioinformatics pipeline designed for context-aware protein sequence search. It leverages Large Language Models (LLMs), LangGraph, and FAISS to provide a highly flexible system that can interpret natural language queries and perform multi-stage similarity searches.

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
- `BIOSEQ_FETCH_TIMEOUT`: Timeout for UniProt API calls in seconds (default: `300.0`).

#### AI Providers
To force a provider or model:
```bash
export BIOSEQ_LLM_PROVIDER=mistral # or 'openai'
export BIOSEQ_EMBEDDINGS_PROVIDER=mistral # or 'openai'
export MISTRAL_API_KEY='your-key'
```

## What This Code Does
- **Intelligent Sequence Extraction**: Uses LLMs with schema-guided reasoning to extract protein sequences and biological context from natural language prompts.
- **Hardened Security**: Implements multi-layer defense against prompt injection and unauthorized command execution.
- **High-Dimensional Similarity Search**: Performs initial ranking of protein sequences using ProtT5 embeddings.
- **Contextual Refining**: Refines results using semantic context-aware refining (previously known as reranking) via a cross-encoder-style embedding fusion.
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
1. **Extraction & Hardening**: The LLM parses the prompt, extracts the sequence, and validates the request against security protocols.
2. **Short-Circuit Error Handling**: If any node fails or a security breach is detected, the graph immediately terminates.
3. **Similarity Search**: Performs high-speed vector search in the FAISS index.
4. **Contextual Refining**: Top matches are refined based on semantic alignment with the user's biological query context.

## Running the System
The system relies on the **Unified BioSeq Gateway Service** to handle all sequence embedding, similarity search, and contextual refining.

### Start the Unified Gateway Service
```bash
python services/search_service.py
```
This service loads the required models (ProtT5, Qwen3) and FAISS indices, exposing endpoints for protein search and biological refining.

### Run Pipeline
The pipeline now operates asynchronously:
```bash
python pipeline_interface.py "I have a sequence: MALW..."
```

## Project & File Structure
- `src/`: Core logic and pipeline modules.
  - pipeline.py: LangGraph workflow and LLM node orchestration.
  - refining.py: Semantic similarity logic using instruction-aware embeddings.
  - utils.py: API environment setup, FASTA parsing, and sequence cleaning.
  - search.py: Unified Search Service client.
  - api_client.py: Centralized API client with pooling and exponential backoff.
  - config.py: Environment configuration and service settings.
  - data_fetcher.py: REST client for UniProt using `httpx`.
- services/: Unified Search Service.
  - search_service.py: Unified gateway for Protein embeddings, FAISS indices, and biological refining.
  - config.py: Service-specific configuration (ports, FAISS params).
- `data/`: Directory for embeddings and FAISS indexes.
- `pipeline_interface.py`: CLI entry point script.
- `e2e_eval.py`: Hardened protein-only evaluation suite.

## Limitations and Remarks
- **API Dependency**: Requires an active Mistral AI or OpenAI API key.
- **Memory Usage**: ProtT5 loading requires significant RAM (~8GB+ recommended).
- **No DNA Support**: This version is strictly optimized for protein sequences. Filepath resolution has been removed in favor of direct sequence input.
