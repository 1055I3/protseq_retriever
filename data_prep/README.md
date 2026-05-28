# DNA Back-Translation and Embedding Pipeline

This project provides a high-throughput, production-grade pipeline to generate DNA gene sequences from protein amino acid sequences and compute embeddings using genomic language models.

## Workflow

1.  **RefSeq Mapping Pipeline (`refseq_to_swissprot.py`)**:
    *   **Phase 1 (Bulk Scan)**: Scans NCBI GenBank for existing `UniProtKB/Swiss-Prot` cross-references.
    *   **Phase 2 (Targeted Search)**: For remaining proteins, performs targeted metadata searches using SwissProt `Gene Name` and `Organism` to find high-quality RefSeq curated records (`NG_`, `NM_`).
    *   **Standardization**: Automatically converts all RNA sequences to DNA by replacing Uracil (U) with Thymine (T).
2.  **Protein Data Input**: Reads protein UniProt accessions and sequences from `swissprot.tsv`.
3.  **Taxonomy-Informed Synthesis**: For proteins without RefSeq mappings, the pipeline (`backtranslate_and_embed.py`) uses **CodonTransformer** to generate plausible synthetic DNA sequences based on host-specific codon bias.
4.  **Genomic Embedding**: Sequences are embedded using the **HyenaDNA** model for high-throughput inference.
5.  **HDF5 Output**: Embeddings are stored in a high-performance HDF5 file (`per-gene.h5`), ensuring strict compatibility with downstream FAISS retrieval systems.
6.  **Compatibility Utility (`convert_h5_layout.py`)**: A post-processing script used to convert HDF5 file layouts for maximum backward compatibility with older retrieval services.

## Environment Setup

1.  **Create Conda Environment**:
    ```bash
    conda create -n bioprep-env -c conda-forge python=3.10 numpy pytorch transformers biopython pandas polars h5py -y
    conda activate bioprep-env
    pip install CodonTransformer
    ```

2.  **Configuration**:
    Adjust pipeline parameters (e.g., input file paths) in `config.py`.

3.  **Input Data**:
    Ensure `swissprot.tsv` (protein data) and `refseq_swissprot_cds.csv` (optional RefSeq cache) are present in the project root directory.

## Usage

1.  **Generate Mappings**:
    ```bash
    python refseq_to_swissprot.py
    ```

2.  **Run Pipeline**:
    ```bash
    python backtranslate_and_embed.py
    ```

3.  **Optimize Layout (Optional)**:
    If required for legacy service compatibility, run:
    ```bash
    python convert_h5_layout.py
    ```
