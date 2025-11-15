"""
Batch Narrative Extraction from Multiple Trade Ideas

This script reads a file containing multiple trade analyses separated by
separator lines, extracts narratives for each one, and saves all results.

Usage:
    python batch_process_narratives.py           # Process all analyses
    python batch_process_narratives.py --test    # Process only first 3 (test mode)
    python batch_process_narratives.py --limit 5 # Process only first 5
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from ExtractionAgent import extract_narratives, ExtractionConfig, NarrativeItem
import json


def split_analyses(file_path: str, separator: str = "-" * 80) -> List[str]:
    """
    Split a text file containing multiple analyses into separate texts.
    
    Args:
        file_path: Path to the input file
        separator: String pattern that separates different analyses
        
    Returns:
        List of text strings, one for each analysis
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator and filter out empty strings
    analyses = [text.strip() for text in content.split(separator) if text.strip()]
    
    print(f"📄 Found {len(analyses)} analyses in file")
    return analyses


def extract_metadata_from_text(text: str) -> dict:
    """
    Try to extract metadata from text if it follows the structured format.
    
    Looks for patterns like:
    ticker: TSMC
    stance: long
    """
    metadata = {}
    lines = text.split('\n')[:20]  # Check first 20 lines
    
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                if key in ['ticker', 'stance', 'source_id', 'source_title']:
                    metadata[key] = value
    
    return metadata


def batch_extract_narratives(
    analyses: List[str],
    config: Optional[ExtractionConfig] = None,
    verbose: bool = True,
    limit: Optional[int] = None
) -> List[Optional[NarrativeItem]]:
    """
    Extract narratives from multiple analyses.
    
    Args:
        analyses: List of analysis texts
        config: Configuration for extraction (uses defaults if None)
        verbose: Whether to print progress
        limit: Maximum number of analyses to process (None = all)
        
    Returns:
        List of NarrativeItem objects (one per analysis)
    """
    if config is None:
        config = ExtractionConfig()
    
    # Apply limit if specified
    total_available = len(analyses)
    if limit is not None and limit < total_available:
        analyses = analyses[:limit]
        print(f"\n⚠️  TEST MODE: Processing only first {limit} of {total_available} analyses")
    
    results = []
    
    for idx, text in enumerate(analyses, 1):
        print("\n" + "=" * 70)
        print(f"PROCESSING ANALYSIS {idx}/{len(analyses)}")
        print("=" * 70)
        
        # Try to extract metadata for better context
        metadata = extract_metadata_from_text(text)
        if metadata:
            print(f"📋 Detected metadata: {metadata}")
        
        # Extract narrative
        try:
            narrative = extract_narratives(text, config=config)
            
            if narrative:
                print(f"\n✅ Successfully extracted narrative {idx}:")
                print(f"   Ticker: {narrative.ticker}")
                print(f"   Position: {narrative.position.value}")
                print(f"   Nodes: {len(narrative.narrative_graph.nodes)}")
                print(f"   Edges: {len(narrative.narrative_graph.edges)}")
                print(f"   Convergence: {narrative.narrative_graph.convergence_score:.3f}")
                results.append(narrative)
            else:
                print(f"\n⚠️  Failed to extract narrative {idx}")
                results.append(None)
                
        except Exception as e:
            print(f"\n❌ Error processing analysis {idx}: {str(e)}")
            results.append(None)
    
    return results


def save_batch_results(
    narratives: List[Optional[NarrativeItem]],
    output_path: str
):
    """
    Save multiple narratives to a single JSON file.
    
    Args:
        narratives: List of NarrativeItem objects
        output_path: Path to output JSON file
    """
    # Filter out None values
    valid_narratives = [n for n in narratives if n is not None]
    
    if not valid_narratives:
        print("\n⚠️  No valid narratives to save!")
        return
    
    # Create output structure
    # Use mode='json' to properly serialize datetime objects
    output = {
        "extraction_date": datetime.now().isoformat(),
        "num_narratives": len(valid_narratives),
        "narratives": [n.model_dump(mode='json') for n in valid_narratives]
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved {len(valid_narratives)} narrative(s) to: {output_path}")


def print_summary(narratives: List[Optional[NarrativeItem]]):
    """Print a summary table of all extracted narratives."""
    valid = [n for n in narratives if n is not None]
    
    print("\n" + "=" * 100)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 100)
    print(f"Total Analyses: {len(narratives)}")
    print(f"Successful: {len(valid)}")
    print(f"Failed: {len(narratives) - len(valid)}")
    print()
    
    if valid:
        print(f"{'#':<4} {'Ticker':<8} {'Position':<8} {'Nodes':<7} {'Edges':<7} {'Convergence':<12} {'Prediction':<10}")
        print("-" * 100)
        
        for idx, narrative in enumerate(valid, 1):
            ticker = narrative.ticker or "N/A"
            position = narrative.position.value if narrative.position else "N/A"
            nodes = len(narrative.narrative_graph.nodes)
            edges = len(narrative.narrative_graph.edges)
            conv = narrative.narrative_graph.convergence_score
            pred = narrative.prediction.direction if narrative.prediction else "N/A"
            
            print(f"{idx:<4} {ticker:<8} {position:<8} {nodes:<7} {edges:<7} {conv:<12.3f} {pred:<10}")
    
    print("=" * 100)


def parse_args():
    """Parse command line arguments."""
    test_mode = False
    limit = None
    
    if "--test" in sys.argv:
        test_mode = True
        limit = 3
    elif "--limit" in sys.argv:
        try:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                limit = int(sys.argv[idx + 1])
        except (ValueError, IndexError):
            print("⚠️  Invalid --limit argument, using default (all)")
            limit = None
    
    return test_mode, limit


def main():
    """Main batch processing workflow."""
    
    # Parse command line arguments
    test_mode, limit = parse_args()
    
    print("\n" + "=" * 70)
    if test_mode:
        print("BATCH NARRATIVE EXTRACTION (TEST MODE - First 3 Only)")
    elif limit:
        print(f"BATCH NARRATIVE EXTRACTION (Limited to First {limit})")
    else:
        print("BATCH NARRATIVE EXTRACTION")
    print("=" * 70)
    
    # Configuration
    input_file = "Data/test_data.txt"
    if test_mode or limit:
        output_file = f"output/batch_narratives_test.json"
    else:
        output_file = "output/batch_narratives.json"
    
    # Check if input exists
    if not os.path.exists(input_file):
        print(f"\n❌ Input file not found: {input_file}")
        return
    
    print(f"\n📂 Input: {input_file}")
    print(f"📂 Output: {output_file}")
    
    # Step 1: Split the file into separate analyses
    print("\n" + "-" * 70)
    print("STEP 1: Splitting analyses")
    print("-" * 70)
    analyses = split_analyses(input_file)
    
    if not analyses:
        print("❌ No analyses found in file!")
        return
    
    # Step 2: Extract narratives with default config
    print("\n" + "-" * 70)
    print("STEP 2: Extracting narratives")
    print("-" * 70)
    
    config = ExtractionConfig(
        enable_safety_limits=True,
        max_nodes_per_narrative=50,
        max_edges_per_narrative=100
    )
    
    narratives = batch_extract_narratives(analyses, config=config, limit=limit)
    
    # Step 3: Save results
    print("\n" + "-" * 70)
    print("STEP 3: Saving results")
    print("-" * 70)
    save_batch_results(narratives, output_file)
    
    # Step 4: Print summary
    print_summary(narratives)
    
    print("\n" + "=" * 70)
    print("✅ BATCH PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

