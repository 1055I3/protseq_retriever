[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Windows](https://img.shields.io/badge/Windows-0078D6)](https://en.wikipedia.org/wiki/Windows_10)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

# ProtSeq Retriever

ProtSeq Retriever is a high-performance, context-aware protein sequence retrieval and refinement system. It utilizes state-of-the-art embedding models (ESMC-300M) for sequence similarity and modern language models (ModernBERT-bio-large) to refine retrieval  via cross-encoder fusion to identify and rank Swiss-Prot entries based on natural language queries.

## Background
The BLAST algorithm, developed in the early 1990s, has long been the standard tool for comparing biological sequences and querying genomic databases. Since then, advances in natural language processing have introduced embeddings — vector representations capable of capturing semantic relationships in data. More recently, similar approaches have been applied to biological sequences, treating proteins as a biological language. Unlike traditional alignment-based methods, embeddings can capture higher-level relationships between sequences, opening up new possibilities for faster and more scalable similarity searches.

## Project Origins

This project originated from the `bio_seq_project` repository and was initially developed during the "Intro to AI Agents" course held at the University of Belgrade (March 21st – May 23rd, 2026).

Reference: [https://datasanity.dev/intro-to-ai-agents-belgrade-2026.html](https://datasanity.dev/intro-to-ai-agents-belgrade-2026.html)

---

## 🏗️ Architecture Overview

The system is a modular, asynchronous LCEL pipeline:

1.  **Extraction Pipeline**: Uses LLM-driven structured reasoning to secure, parse, and distill broad semantic context from user queries.
2.  **Sequence Retrieval**: Performs high-speed HNSW vector search using ESMC-300M sequence embeddings.
3.  **Refinement Stage**: Efficiently reranks top hits by fusing retrieval scores with semantic similarity scores calculated from precomputed context embeddings (ModernBERT-bio-large).
4.  **Embedding Generation**: Offline pipeline (`scripts/data_prep/`) for generating and indexing ESMC-300M sequence embeddings and ModernBERT context embeddings.
5.  **Vector Search**: FAISS-based HNSW indexing over normalized sequence embeddings.
6.  **Summary Generation**: Final-stage hydration of protein records followed by LLM-based plain-language briefing.

---

## 🛠️ Technology Stack
- **Languages**: Python 3.12+
- **Machine Learning**: `biohub/esm` (ESMC-300M), `transformers` (ModernBERT-bio-large), `PyTorch`, `FAISS`.
- **API & Concurrency**: `FastAPI`, `uvicorn`, `asyncio`, `httpx`.
- **Data Engineering**: `Polars`, `h5py`.
- **Pipeline**: `LangChain` (LCEL), `Pydantic` (structured output).
- **Testing**: `pytest`, `pytest-asyncio`.

---

## 💻 Hardware Support

The ProtSeq Retriever is designed for portability.

*   **CPU execution** is fully supported and optimized.
*   **AMD GPU execution** via ROCm is the preferred accelerated platform.
*   **CUDA compatibility** exists for users running NVIDIA hardware (detected via PyTorch's backend).

Hardware acceleration is automatically detected. No vendor-specific code modifications are required for deployment.

---

## 🚀 Setup & Installation

### 1. Environment Setup
```bash
# Create the environment
conda create -n protseq python=3.12 -y
conda activate protseq

# Install primary dependencies via conda
conda install -c conda-forge pytorch transformers faiss-cpu polars h5py uvicorn fastapi httpx pydantic langchain -y

# Install git-based dependencies (pip required for non-conda-packaged source)
pip install esm@git+https://github.com/Biohub/esm.git
```

### 2. Configuration
The system is configured via files in the `config/` directory.
- `config/settings.py`: Global environment, paths, and service URLs.
- `config/service_params.py`: Model names, FAISS HNSW tuning, and length constraints.

---

## ⚙️ Data Preparation
To populate the system with protein data, run the complete preparation pipeline:
```bash
python -m scripts.data_prep.orchestrator
```
*Note: This process is robust, checkpoint-aware, and OOM-protected.*

---

## 🔌 Running the System

### 1. Start the Search Gateway
```bash
python -m src.protseq.app.gateway.search_service
```

### 2. Run the Interface
```bash
# Query the system via CLI
python -m src.protseq.app.cli.retriever_interface "Identify sequence: MKTLL... related to insulin."
```

---

## 🧪 Testing and Benchmarks
The repository includes an E2E evaluation suite:
```bash
python tests/benchmarks/e2e_eval.py
```

---

## 🔐 Security Considerations
- **Prompt Injection**: Implements a regex-based security guard and an LLM-guided security scan in `security_scan_node`.
- **Input Validation**: Sequences are sanitized via `clean_sequence` to prevent non-biological characters.
- **Service Isolation**: The Gateway Service validates local data integrity on startup.
