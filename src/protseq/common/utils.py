import os
import re
from config.settings import ALLOWED_DATA_DIR

def is_secure_path(path: str) -> bool:
    """Verifies if the path is within the allowed directory."""
    abs_allowed = os.path.abspath(ALLOWED_DATA_DIR)
    abs_path = os.path.abspath(path)
    return abs_path.startswith(abs_allowed)

def clean_sequence(sequence: str) -> str:
    """
    Cleans up a sequence by removing FASTA headers, whitespace, and non-letter characters.
    """
    sequence = sequence.strip()
    if sequence.startswith(">"):
        lines = sequence.splitlines()
        # Find where the sequence starts (after the header line)
        sequence = "".join(lines[1:])
    
    # Remove any non-letter characters (e.g., numbers, whitespace, punctuation)
    cleaned = re.sub(r'[^A-Za-z]', '', sequence)
    return cleaned.upper()

def _select_provider(provider: str | None = None) -> str:
    requested = (provider or os.getenv("PROTSEQ_LLM_PROVIDER") or "").strip().lower()
    if requested:
        if requested not in {"mistral", "openai"}:
            raise ValueError("PROTSEQ_LLM_PROVIDER must be either 'mistral' or 'openai'.")
        return requested
    if os.getenv("MISTRAL_API_KEY"):
        return "mistral"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mistral"


def setup_environment(provider: str | None = None):
    """Sets up API keys and environment variables."""
    selected_provider = _select_provider(provider)
    env_key = "OPENAI_API_KEY" if selected_provider == "openai" else "MISTRAL_API_KEY"
    api_key = os.getenv(env_key)
    if not api_key:
        raise ValueError("Set MISTRAL_API_KEY or OPENAI_API_KEY before running the pipeline.")
    return api_key

def get_llm(temperature=0):
    """Returns a configured chat LLM instance."""
    provider = _select_provider()
    setup_environment(provider)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
            temperature=temperature
        )

    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=temperature
    )

def get_text_embedder():
    """Returns a configured embeddings instance for text/context."""
    provider = _select_provider(os.getenv("PROTSEQ_EMBEDDINGS_PROVIDER"))
    setup_environment(provider)
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"))

    from langchain_mistralai import MistralAIEmbeddings

    return MistralAIEmbeddings(model=os.getenv("MISTRAL_EMBEDDINGS_MODEL", "mistral-embed"))

def get_first_fasta_entry(fasta_path: str) -> tuple[str, str]:
    """
    Extracts the first header and the full sequence from a FASTA file using pyfaidx.
    Returns a tuple (header, sequence).
    """
    from pyfaidx import Fasta
    fasta = Fasta(fasta_path)
    first_record = fasta[0]
    header = f">{first_record.long_name}"
    sequence = str(first_record)
    return header, clean_sequence(sequence)
