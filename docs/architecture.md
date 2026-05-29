# BioSeq Retriever: Architectural Documentation

## 1. Design Philosophy
The system prioritizes **determinism**, **production-grade robustness**, and **modular separation of concerns**. The pipeline uses **LCEL (LangChain Expression Language)** to enforce a strictly linear, short-circuiting flow for biological data retrieval.

## 2. Module Boundary Ownership
- `config/`: Single source of truth for runtime, service, and infrastructure parameters.
- `src/bioseq/app/`: The "Application" layer. Handles orchestrations and entry points.
- `src/bioseq/core/`: The "Domain" layer. Contains pure bioinformatics business logic (e.g., how we calculate relevance).
- `src/bioseq/infra/`: The "Infrastructure" layer. Contains technology-specific adapters (FAISS, UniProt API Client, Polars metadata storage).
- `scripts/`: Operational utilities (data preparation, indexing).
- `tests/`: Organized by test scope (Unit, Integration, Benchmarks).

## 3. Runtime Lifecycle
1. **Gateway Initialization**: `services/search_service.py` initializes upon startup, verifying the presence of indexed embedding and metadata files. If critical assets are missing, it shuts down with actionable feedback.
2. **Pipeline Invocation**: The `bioseq.app.cli.retriever_interface` invokes `run_bioseq_pipeline`.
3. **LCEL Cascade**:
    - **Security Guard**: Regex scan for injection patterns.
    - **Extraction (LLM)**: Structured extraction of the protein sequence and biological context.
    - **Vector Search (FAISS)**: HNSW-based retrieval via the gateway.
    - **Refining (Cross-Encoder)**: Semantic similarity enhancement using conformal uncertainty fusion.
    - **Summary (LLM)**: Generation of a plain-language biological briefing for non-experts.

## 4. Operational Considerations
- **Memory**: Running ESMC-300M and holding the FAISS index in memory requires high-RAM environments.
- **Data Integrity**: The data preparation pipeline in `scripts/data_prep/` is designed to be fully idempotent and resume-able.
- **Scaling**: The Gateway Service uses `ThreadPoolExecutor` for asynchronous I/O and vector operations, allowing it to handle concurrent request loads effectively.
