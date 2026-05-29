# Test Coverage Strategy

## 1. Scope and Coverage Priorities
The testing strategy is organized by architectural layer to ensure deterministic, isolated, and contract-based validation.

### Priority 1: Core Domain Logic (`tests/unit/bioseq/core/`)
- **Retrieval Logic**: Validate `LocalRefiner` and `search_client`. Focus on mathematical correctness of fusion algorithms and interaction contracts.
- **Summary Logic**: Verify `summary_node` output quality for non-experts.

### Priority 2: Infrastructure Adapters (`tests/unit/bioseq/infra/` & `tests/integration/infra/`)
- **API Client**: Unit test retries, exponential backoff, and exception handling.
- **Storage**: Unit test `DataFetcher` (metadata CSV loading).
- **Vector Store**: Unit test index loading (mocking FAISS index file interactions).

### Priority 3: Application Layer (`tests/unit/bioseq/app/` & `tests/integration/app/`)
- **Pipeline**: Test the linear LCEL chain (security scan -> extraction -> search -> refine -> summary). Use mocks for LLM and Search Service.
- **API Gateway**: Integration tests for FastAPI endpoints (protein search, refine).

## 2. Testing Principles
- **No Mocking Unnecessarily**: We will use real code for core logic, only mocking external network dependencies or expensive ML models.
- **Deterministic**: Seed-based RNG and fixed datasets in `tests/fixtures/`.
- **Isolation**: Each test runs with a pristine setup.

## 3. Untested Critical Paths Identified
- **Security Guard**: Regex injection pattern handling.
- **Refinement Fallback**: Behavior when the refining service fails.
- **Search Service Initialization**: Data file validation logic.
- **Pipeline Short-Circuiting**: Verifying that errors in one node stop execution.

## 4. Test Structure
```text
tests/
├── unit/
│   ├── bioseq/
│   │   ├── app/      # Pipeline orchestration logic
│   │   ├── core/     # Domain retrieval/refinement
│   │   └── infra/    # Low-level infrastructure adapters
├── integration/
│   ├── app/          # API endpoint tests
│   └── infra/        # Data persistence/CSV loading tests
└── fixtures/         # Mock data, CSV files, H5 samples
```
