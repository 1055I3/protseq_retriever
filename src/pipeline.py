from typing import List, Dict, Any, Optional, TypedDict, Literal, Union
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

import re

from src.utils import get_llm, get_first_fasta_entry, is_secure_path, clean_sequence
from src.data_fetcher import get_uniprot_records
from src.search import search_protein_top_k
from src.reranking import LocalReranker

from src.config import ALLOWED_DATA_DIR, RETRIEVAL_TOP_K, RERANK_TOP_N


# =============================================================================
# SCHEMA-GUIDED REASONING ROUTER DEFINITIONS
# =============================================================================

class AnalysisFailure(BaseModel):
    """Unified error state for all failure points in the analytical cascade."""
    kind: Literal["error"] = Field(description="Discriminator for error outcomes.")
    failure_stage: Literal["INITIAL_ROUTING", "SEQUENCE_VALIDATION", "PATH_VALIDATION"] = Field(description="The logical stage where the analysis failed.")
    error_message: str = Field(description="A clear, professional error message to be returned to the user.")
    technical_root_cause: str = Field(description="Elaborate technical explanation of the failure for debugging.")

# --- Success Outcome Schemas ---

class SequenceSuccess(BaseModel):
    """Final state for a successfully identified protein sequence with extracted context."""
    kind: Literal["sequence_success"]
    raw_sequence: str = Field(description="The extracted raw protein sequence string.")
    context: str = Field(description="Extract the broadest possible biologically relevant semantic context from the query, including biological entities, genes, proteins, domains, functions, pathways, processes, molecular interactions, structural features, localization, taxonomy, evolutionary relationships, diseases, phenotypes, experimental evidence, ontology terms, synonyms, aliases, regulatory relationships, host-pathogen context, biochemical activities, cellular context, and inferred biological associations. Preserve broad contextual and relational information with high semantic recall, including weakly implied or partially related concepts, without aggressive filtering or compression, since downstream instruction-aware embedding will refine and prioritize the signal.")

class FilePathSuccess(BaseModel):
    """Final state for a successfully identified filesystem path with extracted context."""
    kind: Literal["filepath_success"]
    path: str = Field(description="The extracted filesystem path to a protein FASTA file.")
    context: str = Field(description="Extract the broadest possible biologically relevant semantic context from the query, including biological entities, genes, proteins, domains, functions, pathways, processes, molecular interactions, structural features, localization, taxonomy, evolutionary relationships, diseases, phenotypes, experimental evidence, ontology terms, synonyms, aliases, regulatory relationships, host-pathogen context, biochemical activities, cellular context, and inferred biological associations. Preserve broad contextual and relational information with high semantic recall, including weakly implied or partially related concepts, without aggressive filtering or compression, since downstream instruction-aware embedding will refine and prioritize the signal.")

# --- Reasoning Cascade Branches ---

class SequenceAnalysis(BaseModel):
    """Reasoning cascade for protein sequence strings. Implements Step-by-Step validation."""
    kind: Literal["sequence"] = Field(description="Discriminator for sequence-based routing.")
    
    step_1_alphabet_validation: str = Field(description="Examine the unique characters. Search for protein-specific residues like 'M,W,Y,K' and ensure it's not a nucleotide string.")
    step_2_functional_clues_from_context: str = Field(description="Analyze the prompt for functional mentions (e.g. 'enzyme', 'receptor', 'protein') to support the protein classification.")
    step_3_certainty_validation: str = Field(description="Synthesize steps 1 & 2. Are you 100% certain this is a protein sequence? If any ambiguity exists, route to 'error' in the final outcome.")
    
    final_outcome: Union[SequenceSuccess, AnalysisFailure] = Field(description="The terminal result of the sequence reasoning path.")

class FilePathAnalysis(BaseModel):
    """Reasoning cascade for filesystem paths. Implements Step-by-Step validation."""
    kind: Literal["filepath"] = Field(description="Discriminator for path-based routing.")
    
    step_1_extension_integrity_check: str = Field(description="Evaluate the file extension (.faa, .fasta). Determine if it is explicit or ambiguous.")
    step_2_contextual_verification: str = Field(description="Does the user refer to this path as a 'protein file' or 'FASTA'? Match extension to context.")
    step_3_certainty_validation: str = Field(description="Are the extension and context consistent and sufficient for a 100% certain protein classification? If not, route to 'error' in the final outcome.")
    
    final_outcome: Union[FilePathSuccess, AnalysisFailure] = Field(description="The terminal result of the path reasoning path.")

# --- Master Router Root ---

class PipelineRouter(BaseModel):
    """Master router and orchestrator for schema-guided protein analysis. Entry point of the cascade."""
    step_1_data_extraction_and_mapping: str = Field(description="Initial extraction of candidate protein sequences, paths, and raw intent metadata.")
    step_2_initial_routing_logic: str = Field(description="Decide which analytical path is supported by the data (sequence, filepath, or immediate failure).")
    
    route: Union[SequenceAnalysis, FilePathAnalysis, AnalysisFailure] = Field(description="The analytical path chosen by the router based on initial evidence.")

# =============================================================================
# GRAPH STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    prompt: str
    sequence_or_path: Optional[str]
    input_type: Optional[str]
    context: Optional[str]
    sequence: Optional[str]
    results: Optional[List[Dict[str, Any]]]
    error: Optional[str]

# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def extract_and_classify_node(state: GraphState) -> Dict[str, Any]:
    """
    Implements the Schema-Guided Reasoning Cascade for Proteins.
    Forces the model through sequential logic gates before finalizing
    extraction.
    """
    if state.get("error"): return {}

    system_message = (
        "You are an elite protein bioinformatics data architect and routing engine. Your mission is to process raw user prompts "
        "and route them into a high-precision protein search pipeline with absolute scientific accuracy. "
        "You MUST follow the schema-guided reasoning process. Each field in the schema represents a mandatory logical checkpoint.\n\n"
        
        "### YOUR ARCHITECTURAL PROTOCOL:\n"
        "1. **INITIAL SCAN**: Identify strings resembling protein sequences (IUPAC amino acid codes) or filesystem paths.\n"
        "2. **ROUTING**: Select the analytical branch based on the strongest initial evidence.\n"
        "   - Select `SequenceAnalysis` if a raw protein sequence is found.\n"
        "   - Select `FilePathAnalysis` if a path is found.\n"
        "   - Select `AnalysisFailure` if data is missing, the sequence appears to be DNA, or extraction is impossible.\n"
        "3. **CASCADING REASONING**: Within the chosen branch, perform mandatory logic steps:\n"
        "   - **FOR SEQUENCES**: Analyze character distributions (searching for 'M', 'W', 'Y', etc.) and match with protein functional context.\n"
        "   - **FOR PATHS**: Analyze extensions (.faa, .fasta) and verify against user instructions.\n"
        "4. **VALIDATION & ERROR ROUTING**: If, during your reasoning steps, you find the classification uncertain or the data invalid, "
        "you MUST route the `final_outcome` field to an `AnalysisFailure` object. Uncertainty is unacceptable.\n\n"
        
        "Your reasoning must be generous, elaborate, and demonstrate a profound mastery of protein sequence signals."
    )

    try:
        llm = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(PipelineRouter)
        result = structured_llm.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=state['prompt'])
        ])

        terminal = result.route
        if isinstance(terminal, (SequenceAnalysis, FilePathAnalysis)):
            terminal = terminal.final_outcome

        if terminal.kind == "error":
            return {"error": f"[{terminal.failure_stage}] {terminal.error_message} (Root Cause: {terminal.technical_root_cause})"}

        if terminal.kind == "sequence_success":
            return {
                "sequence_or_path": terminal.raw_sequence,
                "input_type": "SEQUENCE",
                "context": terminal.context,
                "error": None,
            }
        else: # filepath_success
            return {
                "sequence_or_path": terminal.path,
                "input_type": "FILEPATH",
                "context": terminal.context,
                "error": None,
            }

    except Exception as e:
        exc_text = f"{type(e).__name__}: {e}"
        return {"error": f"Guided Extraction Pipeline Failure: {exc_text}"}

def resolve_filepath_node(state: GraphState) -> Dict[str, Any]:
    """Node to resolve sequence from a file path with security check."""
    if state.get("error"): return {}
    path = state['sequence_or_path']
    if not is_secure_path(path):
        return {"error": f"Security violation: path {path} is not in {ALLOWED_DATA_DIR}"}
        
    try:
        header, sequence = get_first_fasta_entry(path)
        new_context = f"{state.get('context') or ''}\nFASTA Header: {header}".strip()
        return {
            "sequence": sequence,
            "context": new_context
        }
    except Exception as e:
        return {"error": f"File resolution failed: {str(e)}"}

def use_raw_sequence_node(state: GraphState) -> Dict[str, Any]:
    """Node to handle raw sequence input with cleanup."""
    if state.get("error"): return {}
    seq = state['sequence_or_path']
    cleaned_seq = clean_sequence(seq)
    return {"sequence": cleaned_seq}

def search_node(state: GraphState) -> Dict[str, Any]:
    """Performs protein sequence similarity search via the embedding backend."""
    if state.get('error'): return {}
    try:
        matches = search_protein_top_k(state['sequence'], k=RETRIEVAL_TOP_K)
        records = get_uniprot_records([m[0] for m in matches])

        score_map = {m[0]: m[1] for m in matches}
        for rec in records:
            score = score_map.get(rec.get("primaryAccession"))
            rec["_search_score"] = score

        return {"results": records}
    except Exception as e:
        return {"error": f"Protein Search failed: {str(e)}"}

def rerank_node(state: GraphState) -> Dict[str, Any]:
    """Performs contextual reranking."""
    if state.get('error'): return {}
    results = state.get('results') or []

    try:
        reranker = LocalReranker()
        final_records = reranker.rerank_by_context(results, state['context'], top_n=RERANK_TOP_N)
        return {"results": final_records}
    except Exception as e:
        print(f"Rerank skipped ({e}); falling back to top-{RERANK_TOP_N} of initial results.")
        return {"results": results[:RERANK_TOP_N]}

# --- Conditional Routing Logic ---

def check_error(state: GraphState) -> Literal["error", "continue"]:
    return "error" if state.get("error") else "continue"

def should_resolve_filepath(state: GraphState) -> Literal["resolve", "raw", "error"]:
    if state.get('error'): return "error"
    return "resolve" if state['input_type'] == "FILEPATH" else "raw"

# --- Graph Construction ---

def create_pipeline():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("extract", extract_and_classify_node)
    workflow.add_node("resolve_file", resolve_filepath_node)
    workflow.add_node("use_raw", use_raw_sequence_node)
    workflow.add_node("search", search_node)
    workflow.add_node("rerank", rerank_node)
    
    workflow.set_entry_point("extract")
    
    workflow.add_conditional_edges("extract", should_resolve_filepath, {"resolve": "resolve_file", "raw": "use_raw", "error": END})
    
    workflow.add_edge("resolve_file", "search")
    workflow.add_edge("use_raw", "search")
    
    workflow.add_conditional_edges("search", check_error, {"error": END, "continue": "rerank"})
    
    workflow.add_edge("rerank", END)
    
    return workflow.compile()

async def run_bioseq_pipeline(prompt: str):
    pipeline = create_pipeline()
    initial_state = {
        "prompt": prompt,
        "sequence_or_path": None,
        "input_type": None,
        "context": None,
        "sequence": None,
        "results": None,
        "error": None
    }
    return await pipeline.ainvoke(initial_state)

