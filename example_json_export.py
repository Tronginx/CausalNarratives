"""
Example: How to use JSON export functionality

This script demonstrates:
1. Extracting a narrative from text (single unified graph)
2. Saving to a JSON file
3. Loading narrative back from JSON
4. Using ExtractionConfig to control safety limits
"""

from ExtractionAgent import (
    extract_narratives, 
    save_narratives_to_json, 
    load_narratives_from_json,
    ExtractionConfig
)
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
    print("EXAMPLE 1: Extract narrative with default safety limits")
    print("="*70)
    
    # Method 1: Automatic saving during extraction (with default config)
    narrative = extract_narratives(
        text, 
        save_json=True,
        json_output_path="output/tsmc_narrative.json"
    )
    
    if narrative:
        print(f"\n✅ Successfully extracted narrative:")
        print(f"  Ticker: {narrative.ticker}")
        print(f"  Position: {narrative.position.value}")
        print(f"  Convergence Score: {narrative.narrative_graph.convergence_score:.3f}")
        print(f"  Nodes: {len(narrative.narrative_graph.nodes)}")
        print(f"  Edges: {len(narrative.narrative_graph.edges)}")
        if narrative.prediction:
            print(f"  Prediction: {narrative.prediction.direction} ({narrative.prediction.confidence:.2f} confidence)")
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Extract with safety limits disabled")
    print("="*70)
    
    # Method 2: Extract with custom configuration
    config = ExtractionConfig(
        enable_safety_limits=False  # Disable limits to see full extraction
    )
    narrative_unlimited = extract_narratives(text, config=config)
    
    if narrative_unlimited:
        print(f"\n✅ Extracted without limits:")
        print(f"  Nodes: {len(narrative_unlimited.narrative_graph.nodes)}")
        print(f"  Edges: {len(narrative_unlimited.narrative_graph.edges)}")
        print(f"  Convergence: {narrative_unlimited.narrative_graph.convergence_score:.3f}")
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Load narrative from JSON file")
    print("="*70)
    
    # Load the narrative back
    loaded_narratives = load_narratives_from_json("output/tsmc_narrative.json")
    
    if loaded_narratives:
        item = loaded_narratives[0] if isinstance(loaded_narratives, list) else loaded_narratives
        print(f"\n✅ Successfully loaded narrative")
        print(f"  Ticker: {item.ticker}")
        print(f"  Position: {item.position.value}")
        print(f"  Convergence Score: {item.narrative_graph.convergence_score:.3f}")
        print(f"  Nodes: {len(item.narrative_graph.nodes)}")
        print(f"  Edges: {len(item.narrative_graph.edges)}")
        if item.prediction:
            print(f"  Prediction: {item.prediction.direction} ({item.prediction.confidence:.2f} confidence)")
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Configure custom safety limits")
    print("="*70)
    
    # Method 4: Custom limits
    custom_config = ExtractionConfig(
        enable_safety_limits=True,
        max_nodes_per_narrative=30,  # Lower limit
        max_edges_per_narrative=50,  # Lower limit
        min_convergence_score=0.5    # Higher threshold
    )
    narrative_custom = extract_narratives(text, config=custom_config)
    
    if narrative_custom:
        print(f"\n✅ Extracted with custom limits:")
        print(f"  Nodes: {len(narrative_custom.narrative_graph.nodes)} (max: 30)")
        print(f"  Edges: {len(narrative_custom.narrative_graph.edges)} (max: 50)")
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print(f"\nJSON file created:")
    print(f"  - output/tsmc_narrative.json")


if __name__ == "__main__":
    main()

