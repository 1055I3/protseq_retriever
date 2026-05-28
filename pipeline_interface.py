import os
import json
import argparse
from src.pipeline import run_bioseq_pipeline
from src.utils import setup_environment

import asyncio

def run_pipeline_interface(user_prompt: str):
    """
    Interface to run the bioseq pipeline with a custom prompt.
    Returns the results dictionary.
    """
    setup_environment()

    print(f"Executing pipeline for prompt: {user_prompt[:50]}...")
    # run_bioseq_pipeline is now async and simplified
    result = asyncio.run(run_bioseq_pipeline(user_prompt))

    if result.get("error"):
        print(f"Pipeline Error: {result['error']}")

    return result

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="BioSeq Investigator: Advanced LangGraph Pipeline")
    parser.add_argument(
        "prompt", 
        type=str, 
        nargs="?",
        default=(
            "I have a sequence: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN. "
            "I am looking for sequences involved in glucose metabolism or structurally related to human insulin."
        ),
        help="The natural language prompt containing a biological protein sequence."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    user_prompt = args.prompt

    print("--- BioSeq Investigator: Advanced LangGraph Pipeline ---")
    print(f"User Prompt: {user_prompt}\n")

    try:
        result = run_pipeline_interface(user_prompt)

        if result.get("error"):
            return

        print("\n--- Pipeline Summary ---")
        print(f"Sequence Length: {len(result.get('sequence') or '')}")

        print("\n--- Top Matches (UniProt JSON) ---")
        # result['results'] contains the finalized top matches
        final_results = result.get("results", [])

        # Output results
        output = {
            "top_matches": final_results
        }

        print(json.dumps(output, indent=2))

        print("\n--- Quick View ---")
        for i, record in enumerate(final_results, 1):
            acc = record.get('primaryAccession')
            name = record.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
            # Show the search score if available
            score = record.get('_search_score', 'N/A')
            print(f"{i}. [{acc}] (Score: {score}) {name}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Critical Failure: {e}")

if __name__ == "__main__":
    main()
