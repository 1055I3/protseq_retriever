import os
import json
import argparse
import asyncio
from typing import Dict, Any

from bioseq.app.pipeline.retrieval_pipeline import run_bioseq_pipeline
from bioseq.common.utils import setup_environment

async def run_retriever(user_prompt: str) -> Dict[str, Any]:
    """
    Asynchronous interface for the BioSeq Retriever.
    Executes the pipeline and masks internal state/computation details 
    before returning the final biological results and summary.
    """
    setup_environment()
    
    raw_result = await run_bioseq_pipeline(user_prompt)

    masked_result = {
        "results": raw_result.get("results"),
        "summary": raw_result.get("summary"),
        "error": raw_result.get("error")
    }
    
    return masked_result

def main():
    """Synchronous entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="BioSeq Retriever: ESMC-300M Powered Retrieval")
    parser.add_argument(
        "prompt", 
        type=str, 
        nargs="?",
        default=(
            "I have a sequence: MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN. "
            "I am looking for sequences involved in glucose metabolism or structurally related to human insulin."
        ),
        help="The biological protein query prompt."
    )
    args = parser.parse_args()
    
    print(f"\nAnalyzing biological query...")
    result = asyncio.run(run_retriever(args.prompt))

    if result.get("error"):
        print(f"Error: {result['error']}")
        return

    # 1. Biological Summary
    summary = result.get("summary")
    if summary:
        print("\n" + "="*20 + " BIOLOGICAL ANALYSIS " + "="*20)
        print(f"\nOVERVIEW:\n{summary.get('overview')}")
        print(f"\nSIGNIFICANCE:\n{summary.get('biological_significance')}")
        print("\nDETAILED FINDINGS:")
        for i, expl in enumerate(summary.get('detailed_explanations', []), 1):
            print(f"  {i}. {expl}")

    # 2. Technical Matches
    final_results = result.get("results", [])
    if final_results:
        print("\n" + "="*20 + " TOP RETRIEVED ENTRIES " + "="*20)
        for i, record in enumerate(final_results, 1):
            acc = record.get('accession')
            name = record.get('protein_name')
            org = record.get('organism_name')
            score = record.get('_search_score', 'N/A')
            score_str = f"{score:.2f}" if isinstance(score, (float, int)) else str(score)
            print(f"{i}. [{acc}] (Match: {score_str}) {name} - {org}")
        print("="*63 + "\n")

if __name__ == "__main__":
    main()
