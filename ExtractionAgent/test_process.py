"""
Test Program for ExtractionAgent/process.py

This program tests the business logic functions in process.py using the TSMC article.
It tests both individual functions and the complete workflow.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ExtractionAgent import (
    extract_narratives,
    calculate_convergence_score,
    calculate_graph_metrics,
    validate_graph,
    print_graph_summary,
    Node,
    Edge,
    NodeType,
    RelationshipType,
    Position
)

from ExtractionAgent.process import (
    process_extract_metadata,
    process_extract_conclusions,
    process_extract_nodes,
    process_extract_edges,
    process_build_narrative_graph,
    process_extract_prediction,
    build_networkx_graph,
    find_disconnected_nodes,
    merge_duplicate_nodes,
    get_fact_nodes,
    get_conclusion_nodes
)


def load_test_data():
    """Load the TSMC article for testing"""
    data_path = "/Users/tron/RealTron/UIUC/25Fall/CS546/Project/CausalNarratives/Data/TSMC: A Strong Buy At All-Time Highs"
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"✓ Loaded test data: {len(text)} characters")
    print(f"  First 200 chars: {text[:200]}...")
    return text


def test_metadata_extraction(text: str):
    """Test 1: Metadata Extraction"""
    print("\n" + "="*70)
    print("TEST 1: Metadata Extraction (ticker, position)")
    print("="*70)
    
    ticker, position, error = process_extract_metadata(text)
    
    if error:
        print(f"❌ Error: {error}")
    else:
        print(f"✓ Ticker: {ticker}")
        print(f"✓ Position: {position.value}")
    
    return ticker, position


def test_conclusion_extraction(text: str):
    """Test 2: Conclusion Extraction"""
    print("\n" + "="*70)
    print("TEST 2: Conclusion Extraction")
    print("="*70)
    
    conclusions, error = process_extract_conclusions(text)
    
    if error:
        print(f"❌ Error: {error}")
        return []
    
    print(f"✓ Found {len(conclusions)} conclusion(s)")
    for i, conclusion in enumerate(conclusions, 1):
        print(f"\n  Conclusion {i}:")
        print(f"    ID: {conclusion.id}")
        print(f"    Type: {conclusion.node_type.value}")
        print(f"    Content: \"{conclusion.content[:100]}...\"")
    
    return conclusions


def test_node_extraction(text: str, conclusion: Node):
    """Test 3: Node Extraction"""
    print("\n" + "="*70)
    print("TEST 3: Node Extraction (facts, events, opinions, assumptions)")
    print("="*70)
    print(f"Processing conclusion: \"{conclusion.content[:80]}...\"")
    
    nodes, error = process_extract_nodes(text, conclusion)
    
    if error:
        print(f"❌ Error: {error}")
        return [conclusion]
    
    # Count by type
    type_counts = {}
    for node in nodes:
        node_type = node.node_type.value
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print(f"\n✓ Extracted {len(nodes)} total nodes:")
    for node_type, count in sorted(type_counts.items()):
        print(f"    {node_type}: {count}")
    
    # Show sample nodes
    print("\n  Sample nodes:")
    for node_type in [NodeType.FACT, NodeType.EVENT, NodeType.OPINION, NodeType.ASSUMPTION]:
        sample = next((n for n in nodes if n.node_type == node_type), None)
        if sample:
            print(f"    [{node_type.value}] \"{sample.content[:80]}...\"")
    
    return nodes


def test_edge_extraction(text: str, nodes: list, conclusion: Node):
    """Test 4: Edge Extraction"""
    print("\n" + "="*70)
    print("TEST 4: Edge Extraction (relationships)")
    print("="*70)
    print(f"Finding relationships among {len(nodes)} nodes")
    
    edges, error = process_extract_edges(nodes, conclusion, text)
    
    if error:
        print(f"❌ Error: {error}")
        return []
    
    # Count by type
    type_counts = {}
    for edge in edges:
        edge_type = edge.relationship_type.value
        type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
    
    print(f"\n✓ Extracted {len(edges)} relationships:")
    for edge_type, count in sorted(type_counts.items()):
        print(f"    {edge_type}: {count}")
    
    # Show sample edges
    print("\n  Sample relationships:")
    for i, edge in enumerate(edges[:5], 1):
        source = next((n for n in nodes if n.id == edge.source_id), None)
        target = next((n for n in nodes if n.id == edge.target_id), None)
        
        if source and target:
            print(f"\n    {i}. [{edge.relationship_type.value}]")
            print(f"       From: \"{source.content[:60]}...\"")
            print(f"       To:   \"{target.content[:60]}...\"")
            if edge.description:
                print(f"       Desc: {edge.description[:80]}")
    
    return edges


def test_graph_validation(nodes: list, edges: list):
    """Test 5: Graph Validation"""
    print("\n" + "="*70)
    print("TEST 5: Graph Validation")
    print("="*70)
    
    is_valid, errors = validate_graph(nodes, edges)
    
    if is_valid:
        print("✓ Graph is valid")
    else:
        print(f"❌ Graph validation failed:")
        for error in errors:
            print(f"    - {error}")
    
    # Check for disconnected nodes
    disconnected = find_disconnected_nodes(nodes, edges)
    if disconnected:
        print(f"\n⚠ Found {len(disconnected)} disconnected nodes:")
        for node_id in disconnected[:3]:
            node = next((n for n in nodes if n.id == node_id), None)
            if node:
                print(f"    - {node_id}: \"{node.content[:60]}...\"")
    else:
        print("\n✓ All nodes are connected")
    
    return is_valid, errors


def test_graph_metrics(nodes: list, edges: list, conclusion_id: str):
    """Test 6: Graph Metrics Calculation"""
    print("\n" + "="*70)
    print("TEST 6: Graph Metrics Calculation")
    print("="*70)
    
    metrics = calculate_graph_metrics(nodes, edges, conclusion_id)
    
    print("✓ Calculated metrics:")
    print(f"    Total nodes: {metrics['total_nodes']}")
    print(f"    Total edges: {metrics['total_edges']}")
    print(f"    Facts: {metrics['num_facts']}")
    print(f"    Events: {metrics['num_events']}")
    print(f"    Opinions: {metrics['num_opinions']}")
    print(f"    Assumptions: {metrics['num_assumptions']}")
    print(f"    Convergence score: {metrics['convergence_score']:.3f}")
    print(f"    Avg path length: {metrics['avg_path_length']}")
    print(f"    Max depth: {metrics['max_convergence_depth']}")
    
    return metrics


def test_narrative_graph_building(nodes: list, edges: list, conclusion: Node):
    """Test 7: Narrative Graph Building"""
    print("\n" + "="*70)
    print("TEST 7: Build Complete Narrative Graph")
    print("="*70)
    
    graph, errors = process_build_narrative_graph(nodes, edges, conclusion)
    
    if errors:
        print(f"❌ Graph building failed:")
        for error in errors:
            print(f"    - {error}")
        return None
    
    print(f"✓ Successfully built narrative graph")
    print(f"    Convergence score: {graph.convergence_score:.3f}")
    print(f"    Nodes: {len(graph.nodes)}")
    print(f"    Edges: {len(graph.edges)}")
    
    return graph


def test_prediction_extraction(text: str, conclusion: Node, position: Position):
    """Test 8: Prediction Extraction"""
    print("\n" + "="*70)
    print("TEST 8: Prediction Extraction")
    print("="*70)
    
    prediction, error = process_extract_prediction(text, conclusion, position)
    
    if error:
        print(f"⚠ Warning: {error}")
    
    print(f"✓ Extracted prediction:")
    print(f"    Direction: {prediction.direction}")
    print(f"    Magnitude: {prediction.magnitude}%")
    print(f"    Time horizon: {prediction.time_horizon_days} days")
    print(f"    Confidence: {prediction.confidence:.2f}")
    
    return prediction


def test_helper_functions(nodes: list):
    """Test 9: Helper Functions"""
    print("\n" + "="*70)
    print("TEST 9: Helper Functions")
    print("="*70)
    
    # Test get_fact_nodes
    fact_nodes = get_fact_nodes(nodes)
    print(f"✓ get_fact_nodes(): Found {len(fact_nodes)} facts")
    
    # Test get_conclusion_nodes
    conclusion_nodes = get_conclusion_nodes(nodes)
    print(f"✓ get_conclusion_nodes(): Found {len(conclusion_nodes)} conclusions")
    
    # Test merge_duplicate_nodes
    original_count = len(nodes)
    unique_nodes = merge_duplicate_nodes(nodes)
    duplicates_removed = original_count - len(unique_nodes)
    print(f"✓ merge_duplicate_nodes(): Removed {duplicates_removed} duplicates")
    
    # Test build_networkx_graph
    from ExtractionAgent.process import build_networkx_graph
    edges = []  # Empty for this test
    G = build_networkx_graph(nodes, edges)
    print(f"✓ build_networkx_graph(): Created graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def test_full_workflow(text: str):
    """Test 10: Full Workflow Integration"""
    print("\n" + "="*70)
    print("TEST 10: Full Workflow Integration")
    print("="*70)
    print("Running complete extraction workflow via extract_narratives()...")
    
    try:
        narrative_items = extract_narratives(text)
        
        print(f"\n✓ Extraction complete!")
        print(f"    Generated {len(narrative_items)} narrative item(s)")
        
        for i, item in enumerate(narrative_items, 1):
            print(f"\n  Narrative Item {i}:")
            print(f"    Ticker: {item.ticker}")
            print(f"    Position: {item.position.value}")
            print(f"    Convergence score: {item.narrative_graph.convergence_score:.3f}")
            print(f"    Argument quality: {item.argument_quality_score:.3f}")
            print(f"    Nodes: {len(item.narrative_graph.nodes)}")
            print(f"    Edges: {len(item.narrative_graph.edges)}")
            
            if item.prediction:
                print(f"    Prediction: {item.prediction.direction} ({item.prediction.confidence:.2f} confidence)")
        
        return narrative_items
        
    except Exception as e:
        print(f"❌ Full workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_all_tests():
    """Run all tests in sequence"""
    print("\n" + "🧪"*35)
    print("TESTING ExtractionAgent/process.py")
    print("🧪"*35)
    
    try:
        # Load test data
        text = load_test_data()
        
        # Test individual processing steps
        ticker, position = test_metadata_extraction(text)
        conclusions = test_conclusion_extraction(text)
        
        if not conclusions:
            print("\n❌ Cannot continue: No conclusions extracted")
            return
        
        # Use the first conclusion for testing
        conclusion = conclusions[0]
        
        nodes = test_node_extraction(text, conclusion)
        edges = test_edge_extraction(text, nodes, conclusion)
        
        is_valid, validation_errors = test_graph_validation(nodes, edges)
        
        if is_valid:
            metrics = test_graph_metrics(nodes, edges, conclusion.id)
            graph = test_narrative_graph_building(nodes, edges, conclusion)
            prediction = test_prediction_extraction(text, conclusion, position)
        
        # Test helper functions
        test_helper_functions(nodes)
        
        # Test full integrated workflow
        test_full_workflow(text)
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETED")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()

