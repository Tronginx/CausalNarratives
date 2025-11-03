"""
Example: How to use JSON export functionality

This script demonstrates:
1. Extracting narratives from text
2. Saving to a single JSON file
3. Saving to separate JSON files
4. Loading narratives back from JSON
"""

from ExtractionAgent import extract_narratives, save_narratives_to_json, load_narratives_from_json
from pathlib import Path

def main():
    # Read the test data
    data_file = Path("Data/TSMC: A Strong Buy At All-Time Highs")
    
    if not data_file.exists():
        print(f"❌ Test data file not found: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("="*70)
    print("EXAMPLE 1: Extract narratives and save to a single JSON file")
    print("="*70)
    
    # Method 1: Automatic saving during extraction
    narratives = extract_narratives(
        text, 
        save_json=True,
        json_output_path="output/tsmc_narratives.json"
    )
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract first, then save to separate files")
    print("="*70)
    
    # Method 2: Manual saving after extraction
    # (narratives are already extracted above, so we'll just save them differently)
    save_narratives_to_json(
        narratives,
        output_path="output/individual/",
        separate_files=True
    )
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Load narratives from JSON file")
    print("="*70)
    
    # Load the narratives back
    loaded_narratives = load_narratives_from_json("output/tsmc_narratives.json")
    
    print(f"\n✅ Successfully loaded {len(loaded_narratives)} narratives")
    for i, item in enumerate(loaded_narratives, 1):
        print(f"\nNarrative {i}:")
        print(f"  Ticker: {item.ticker}")
        print(f"  Position: {item.position.value}")
        print(f"  Convergence Score: {item.narrative_graph.convergence_score:.3f}")
        print(f"  Nodes: {len(item.narrative_graph.nodes)}")
        print(f"  Edges: {len(item.narrative_graph.edges)}")
        if item.prediction:
            print(f"  Prediction: {item.prediction.direction} ({item.prediction.confidence:.2f} confidence)")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print(f"\nJSON files created in:")
    print(f"  - output/tsmc_narratives.json (single file)")
    print(f"  - output/individual/*.json (separate files)")


if __name__ == "__main__":
    main()

