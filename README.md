# ProtSeq Retriever

ProtSeq Retriever is a high-performance, context-aware protein sequence retrieval and refining system. It leverages state-of-the-art protein embedding models (ESMC-300M) and semantic refining via cross-encoder fusion to identify and rank Swiss-Prot entries based on natural language queries.

## Background
The BLAST algorithm, developed in the early 1990s, has long been the standard tool for comparing biological sequences and querying genomic databases. Since then, advances in natural language processing have introduced embeddings — vector representations capable of capturing semantic relationships in data. More recently, similar approaches have been applied to biological sequences, treating proteins as a biological language. Unlike traditional alignment-based methods, embeddings can capture higher-level relationships between sequences, opening up new possibilities for faster and more scalable similarity searches.

---

## 🏗️ Architecture Overview

The system is designed as a modular, asynchronous pipeline:
1.  **Orchestration Layer**: Uses LangChain LCEL to manage extraction, security, and data flow.
2.  **Gateway Service**: A high-performance FastAPI backend serving vector similarity search (FAISS HNSW) and contextual refining (Qwen3).
3.  **Data Engineering**: Robust, modular pipeline for Swiss-Prot ingestion, ESMC-300M embedding, and HNSW index construction.

### Data Flow
`User Prompt` → `Security Guard` → `Extraction (LLM)` → `Vector Search (FAISS)` → `Refining (Semantic Fusion)` → `Summary (Plain Language)`

---

## 🛠️ Technology Stack
- **Languages**: Python 3.12+
- **Machine Learning**: `biohub/esm` (ESMC-300M), `transformers` (Qwen3-Embedding), `PyTorch`, `FAISS`.
- **API & Concurrency**: `FastAPI`, `uvicorn`, `asyncio`, `httpx`.
- **Data Engineering**: `Polars`, `h5py`.
- **Pipeline**: `LangChain` (LCEL), `Pydantic` (structured output).
- **Testing**: `pytest`, `pytest-asyncio`.

---

## 🚀 Setup & Installation

### 1. Environment Setup
```bash
conda create -n protseq python=3.12 -y
conda activate protseq
# Install esm from source as required by biohub
pip install esm@git+https://github.com/Biohub/esm.git
pip install -r requirements.txt # See requirements.txt for full dependency list
```

### 2. Configuration
The system is configured via files in the `config/` directory. Environment variables override these settings.
- `config/settings.py`: Global environment, API keys, and service URLs.
- `config/service_params.py`: Model names, FAISS HNSW tuning, and length constraints.

---

## ⚙️ Data Preparation
To populate the system with protein data, run the data prep pipeline:
```bash
python -m scripts.data_prep.orchestrator
```
*Note: This process is robust with checkpointing. If interrupted, simply rerun it.*

---

## 🔌 Running the System

### 1. Start the Search Gateway
```bash
python -m src.bioseq.app.gateway.search_service
```

### 2. Run the Interface
```bash
# Query the system
python -m src.bioseq.app.cli.retriever_interface "Identify sequence: MKTLL... related to insulin."
```

---

## 🧪 Testing and Benchmarks
The repository includes an E2E evaluation suite designed to validate behavioral correctness:
```bash
python tests/benchmarks/e2e_eval.py
```

---

## 🔐 Security Considerations
- **Prompt Injection**: The pipeline implements a regex-based security guard and LLM-guided security scan in `security_scan_node`.
- **Input Validation**: Sequences are sanitized via `clean_sequence` to prevent non-biological characters from entering the model.
- **Service Isolation**: The Gateway Service validates local data integrity on startup to prevent operational risks from malformed indexes.

---

## 📖 Contributor Guide
- **Commit Conventions**: Conventional commits required (`feat:`, `fix:`, `refactor:`, `docs:`).
- **Architectural Constraints**: Never import outside the `bioseq.` or `config.` namespace.
- **Review Expectations**: All changes must maintain deterministic behavior. Avoid mocking unless external systems are involved.
