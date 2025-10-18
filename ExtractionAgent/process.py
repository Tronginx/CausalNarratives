"""
Business Logic for Narrative Processing

This module contains all business logic for narrative extraction:
- Extraction step processing (delegates to chains.py for LLM calls)
- Graph analysis and metrics calculation
- Validation and helper functions

Architecture:
- chains.py: LLM prompts and interactions
- process.py: Business logic (this file)
- graph.py: Workflow orchestration
"""

from typing import List, Tuple, Optional, Dict, Set
import networkx as nx
from datetime import datetime

from data_types import (
    Node, Edge, NarrativeGraph, NodeType, 
    RelationshipType, NarrativeItem, Position, Prediction
)
from chains import NarrativeChains


# ============================================================================
# BUSINESS LOGIC FOR EXTRACTION WORKFLOW STEPS
# ============================================================================

def process_extract_metadata(source_text: str) -> Tuple[str, Position, Optional[str]]:
    """
    Extract ticker symbol and investment position from text
    
    Args:
        source_text: The financial analysis text
        
    Returns:
        Tuple of (ticker, position, error_message)
    """
    chains = NarrativeChains()
    
    try:
        result = chains.extract_metadata(source_text)
        return result.ticker, Position(result.position), None
    except Exception as e:
        return 'UNKNOWN', Position.NEUTRAL, f"Metadata extraction error: {str(e)}"


def process_extract_conclusions(source_text: str) -> Tuple[List[Node], Optional[str]]:
    """
    Identify ALL conclusion statements in the text
    
    Args:
        source_text: The financial analysis text
        
    Returns:
        Tuple of (conclusions_list, error_message)
    """
    chains = NarrativeChains()
    
    try:
        conclusions = chains.extract_conclusions(source_text)
        
        if not conclusions:
            return [], "No conclusions found in text"
        
        return conclusions, None
    except Exception as e:
        return [], f"Conclusion extraction error: {str(e)}"


def process_extract_nodes(source_text: str, conclusion: Node) -> Tuple[List[Node], Optional[str]]:
    """
    Extract nodes (facts, events, opinions, assumptions) related to a conclusion
    
    Args:
        source_text: The financial analysis text
        conclusion: The conclusion node
        
    Returns:
        Tuple of (all_nodes_including_conclusion, error_message)
    """
    chains = NarrativeChains()
    
    try:
        # Extract supporting nodes
        nodes = chains.extract_nodes(source_text, conclusion)
        
        # Add the conclusion to the node list
        all_nodes = nodes + [conclusion]
        
        return all_nodes, None
    except Exception as e:
        return [conclusion], f"Node extraction error: {str(e)}"


def process_extract_edges(
    nodes: List[Node],
    conclusion: Node,
    source_text: str
) -> Tuple[List[Edge], Optional[str]]:
    """
    Extract causal and logical relationships between nodes
    
    Args:
        nodes: All nodes including the conclusion
        conclusion: The target conclusion node
        source_text: The financial analysis text
        
    Returns:
        Tuple of (edges_list, error_message)
    """
    chains = NarrativeChains()
    
    try:
        edges = chains.extract_edges(nodes, conclusion, source_text)
        return edges, None
    except Exception as e:
        return [], f"Edge extraction error: {str(e)}"


def process_build_narrative_graph(
    nodes: List[Node],
    edges: List[Edge],
    conclusion: Node
) -> Tuple[Optional[NarrativeGraph], List[str]]:
    """
    Build and validate the NarrativeGraph with calculated metrics
    
    Args:
        nodes: All nodes
        edges: All edges
        conclusion: The conclusion node
        
    Returns:
        Tuple of (narrative_graph, error_messages)
    """
    try:
        # Validate graph
        is_valid, errors = validate_graph(nodes, edges)
        
        if not is_valid:
            return None, errors
        
        # Create narrative graph with metrics
        graph = create_narrative_graph(nodes, edges, conclusion.id)
        
        return graph, []
    except Exception as e:
        return None, [f"Graph building error: {str(e)}"]


def process_extract_prediction(
    source_text: str,
    conclusion: Node,
    position: Position
) -> Tuple[Prediction, Optional[str]]:
    """
    Extract price prediction from the analysis
    
    Args:
        source_text: The financial analysis text
        conclusion: The conclusion node
        position: The author's position
        
    Returns:
        Tuple of (prediction, error_message)
    """
    chains = NarrativeChains()
    
    try:
        prediction = chains.extract_prediction(source_text, conclusion, position)
        return prediction, None
    except Exception as e:
        # Create default prediction on error
        default_pred = Prediction(
            direction="neutral",
            magnitude=None,
            time_horizon_days=30,
            confidence=0.5
        )
        return default_pred, f"Prediction extraction error: {str(e)}"


def process_create_narrative_item(
    ticker: str,
    position: Position,
    source_text: str,
    narrative_graph: NarrativeGraph,
    prediction: Prediction
) -> Tuple[NarrativeItem, Optional[str]]:
    """
    Create the final NarrativeItem
    
    Args:
        ticker: Stock ticker
        position: Investment position
        source_text: Original text
        narrative_graph: Constructed narrative graph
        prediction: Price prediction
        
    Returns:
        Tuple of (narrative_item, error_message)
    """
    try:
        item = NarrativeItem(
            ticker=ticker,
            position=position,
            created_at=datetime.now(),
            source_text=source_text,
            narrative_graph=narrative_graph,
            prediction=prediction,
            actual_outcome=None,
            argument_quality_score=narrative_graph.convergence_score
        )
        
        # Print summary
        print_graph_summary(narrative_graph)
        
        return item, None
    except Exception as e:
        return None, f"Item creation error: {str(e)}"


# ============================================================================
# GRAPH ANALYSIS FUNCTIONS
# ============================================================================

def build_networkx_graph(nodes: List[Node], edges: List[Edge]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from nodes and edges
    
    Args:
        nodes: List of Node objects
        edges: List of Edge objects
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(
            node.id,
            content=node.content,
            node_type=node.node_type.value
        )
    
    # Add edges with attributes
    for edge in edges:
        if edge.source_id in G.nodes and edge.target_id in G.nodes:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relationship_type=edge.relationship_type.value,
                description=edge.description
            )
    
    return G


def calculate_convergence_score(
    nodes: List[Node],
    edges: List[Edge],
    conclusion_id: str
) -> float:
    """
    Calculate convergence score: how well facts converge to the conclusion
    
    Algorithm:
    1. Build directed graph from nodes/edges
    2. Find all fact nodes
    3. Count paths from each fact to the conclusion
    4. Score based on:
       - Coverage: % of facts that have path to conclusion
       - Density: Average number of paths per fact
       - Directness: Shorter paths score higher
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        conclusion_id: ID of the conclusion node
        
    Returns:
        Convergence score between 0.0 and 1.0
    """
    G = build_networkx_graph(nodes, edges)
    
    # Get fact nodes
    fact_nodes = [n.id for n in nodes if n.node_type == NodeType.FACT]
    
    if not fact_nodes:
        return 0.0
    
    # Check if conclusion exists in graph
    if conclusion_id not in G.nodes:
        return 0.0
    
    # Calculate metrics
    facts_with_path = 0
    total_paths = 0
    path_lengths = []
    
    for fact_id in fact_nodes:
        if nx.has_path(G, fact_id, conclusion_id):
            facts_with_path += 1
            
            # Get all simple paths from fact to conclusion
            try:
                paths = list(nx.all_simple_paths(G, fact_id, conclusion_id, cutoff=10))
                total_paths += len(paths)
                
                # Record path lengths
                for path in paths:
                    path_lengths.append(len(path) - 1)  # -1 because we count edges
            except:
                # If too many paths, just count as having a path
                facts_with_path += 1
    
    # Coverage: what % of facts connect to conclusion
    coverage = facts_with_path / len(fact_nodes)
    
    # Density: how many paths on average (normalized)
    # More paths = more convergence
    avg_paths_per_fact = total_paths / len(fact_nodes) if fact_nodes else 0
    density_score = min(avg_paths_per_fact / 3.0, 1.0)  # Cap at 3 paths per fact
    
    # Directness: shorter paths are better
    if path_lengths:
        avg_path_length = sum(path_lengths) / len(path_lengths)
        # Penalize very long paths (>5)
        directness_score = max(0, 1.0 - (avg_path_length - 1) / 5.0)
    else:
        directness_score = 0.0
    
    # Weighted combination
    convergence = (
        coverage * 0.5 +          # Most important: do facts connect?
        density_score * 0.3 +     # Multiple paths show convergence
        directness_score * 0.2    # Shorter chains are clearer
    )
    
    return round(convergence, 3)


def calculate_graph_metrics(
    nodes: List[Node],
    edges: List[Edge],
    conclusion_id: str
) -> Dict[str, any]:
    """
    Calculate various graph metrics for the narrative
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        conclusion_id: ID of the conclusion node
        
    Returns:
        Dictionary of metrics
    """
    G = build_networkx_graph(nodes, edges)
    
    # Basic counts
    num_facts = sum(1 for n in nodes if n.node_type == NodeType.FACT)
    num_events = sum(1 for n in nodes if n.node_type == NodeType.EVENT)
    num_opinions = sum(1 for n in nodes if n.node_type == NodeType.OPINION)
    num_assumptions = sum(1 for n in nodes if n.node_type == NodeType.ASSUMPTION)
    
    # Path analysis
    fact_nodes = [n.id for n in nodes if n.node_type == NodeType.FACT]
    path_lengths = []
    
    for fact_id in fact_nodes:
        if conclusion_id in G.nodes and nx.has_path(G, fact_id, conclusion_id):
            try:
                # Get shortest path
                shortest = nx.shortest_path_length(G, fact_id, conclusion_id)
                path_lengths.append(shortest)
            except:
                pass
    
    avg_path_length = round(sum(path_lengths) / len(path_lengths), 2) if path_lengths else None
    max_depth = max(path_lengths) if path_lengths else None
    
    # Convergence score
    convergence = calculate_convergence_score(nodes, edges, conclusion_id)
    
    return {
        'num_facts': num_facts,
        'num_events': num_events,
        'num_opinions': num_opinions,
        'num_assumptions': num_assumptions,
        'avg_path_length': avg_path_length,
        'max_convergence_depth': max_depth,
        'convergence_score': convergence,
        'total_nodes': len(nodes),
        'total_edges': len(edges)
    }


def validate_graph(nodes: List[Node], edges: List[Edge]) -> Tuple[bool, List[str]]:
    """
    Validate that the graph is well-formed
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        
    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check for nodes
    if not nodes:
        errors.append("Graph has no nodes")
        return False, errors
    
    # Check for duplicate node IDs
    node_ids = [n.id for n in nodes]
    if len(node_ids) != len(set(node_ids)):
        errors.append("Duplicate node IDs found")
    
    # Check that edges reference valid nodes
    node_id_set = set(node_ids)
    for edge in edges:
        if edge.source_id not in node_id_set:
            errors.append(f"Edge references non-existent source node: {edge.source_id}")
        if edge.target_id not in node_id_set:
            errors.append(f"Edge references non-existent target node: {edge.target_id}")
    
    # Check for at least one conclusion
    conclusion_nodes = [n for n in nodes if n.node_type == NodeType.CONCLUSION]
    if not conclusion_nodes:
        errors.append("Graph has no conclusion node")
    
    # Check for self-loops
    for edge in edges:
        if edge.source_id == edge.target_id:
            errors.append(f"Self-loop detected on node: {edge.source_id}")
    
    return len(errors) == 0, errors


def find_disconnected_nodes(nodes: List[Node], edges: List[Edge]) -> List[str]:
    """
    Find nodes that are not connected to any other nodes
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        
    Returns:
        List of node IDs that are disconnected
    """
    if not edges:
        return [n.id for n in nodes]
    
    # Build set of connected nodes
    connected = set()
    for edge in edges:
        connected.add(edge.source_id)
        connected.add(edge.target_id)
    
    # Find disconnected
    disconnected = [n.id for n in nodes if n.id not in connected]
    
    return disconnected


def merge_duplicate_nodes(nodes: List[Node]) -> List[Node]:
    """
    Merge nodes with very similar content (basic deduplication)
    
    Args:
        nodes: List of nodes
        
    Returns:
        Deduplicated list of nodes
    """
    # Simple implementation: exact content match
    seen_content = {}
    unique_nodes = []
    
    for node in nodes:
        content_lower = node.content.lower().strip()
        
        if content_lower not in seen_content:
            seen_content[content_lower] = node
            unique_nodes.append(node)
    
    return unique_nodes


def create_narrative_graph(
    nodes: List[Node],
    edges: List[Edge],
    conclusion_id: str
) -> NarrativeGraph:
    """
    Create a NarrativeGraph with all metrics calculated
    
    Args:
        nodes: List of all nodes
        edges: List of all edges
        conclusion_id: ID of the conclusion node
        
    Returns:
        Complete NarrativeGraph object
    """
    # Calculate metrics
    metrics = calculate_graph_metrics(nodes, edges, conclusion_id)
    
    # Create graph
    graph = NarrativeGraph(
        nodes=nodes,
        edges=edges,
        conclusion_id=conclusion_id,
        convergence_score=metrics['convergence_score'],
        num_facts=metrics['num_facts'],
        num_conclusions=1,
        conclusion_ids=[conclusion_id],
        avg_path_length=metrics['avg_path_length'],
        max_convergence_depth=metrics['max_convergence_depth']
    )
    
    return graph


def create_narrative_item(
    ticker: str,
    position: Position,
    source_text: str,
    narrative_graph: NarrativeGraph,
    prediction=None,
    source_url: Optional[str] = None,
    author: Optional[str] = None
) -> NarrativeItem:
    """
    Create a complete NarrativeItem
    
    Args:
        ticker: Stock ticker symbol
        position: Investment position
        source_text: Original analysis text
        narrative_graph: Extracted narrative graph
        prediction: Optional price prediction
        source_url: Optional source URL
        author: Optional author name
        
    Returns:
        Complete NarrativeItem
    """
    return NarrativeItem(
        ticker=ticker,
        position=position,
        created_at=datetime.now(),
        source_text=source_text,
        narrative_graph=narrative_graph,
        prediction=prediction,
        actual_outcome=None,
        argument_quality_score=narrative_graph.convergence_score
    )


def get_node_by_id(nodes: List[Node], node_id: str) -> Optional[Node]:
    """
    Find a node by its ID
    
    Args:
        nodes: List of nodes
        node_id: Node ID to find
        
    Returns:
        Node object or None if not found
    """
    for node in nodes:
        if node.id == node_id:
            return node
    return None


def get_fact_nodes(nodes: List[Node]) -> List[Node]:
    """Get all fact nodes"""
    return [n for n in nodes if n.node_type == NodeType.FACT]


def get_conclusion_nodes(nodes: List[Node]) -> List[Node]:
    """Get all conclusion nodes"""
    return [n for n in nodes if n.node_type == NodeType.CONCLUSION]


def print_graph_summary(graph: NarrativeGraph) -> None:
    """
    Print a summary of the narrative graph
    
    Args:
        graph: NarrativeGraph to summarize
    """
    print(f"\n{'='*60}")
    print("NARRATIVE GRAPH SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nNodes: {len(graph.nodes)}")
    print(f"  - Facts: {graph.num_facts}")
    print(f"  - Events: {sum(1 for n in graph.nodes if n.node_type == NodeType.EVENT)}")
    print(f"  - Opinions: {sum(1 for n in graph.nodes if n.node_type == NodeType.OPINION)}")
    print(f"  - Conclusions: {graph.num_conclusions}")
    
    print(f"\nEdges: {len(graph.edges)}")
    print(f"  - Causal: {sum(1 for e in graph.edges if e.relationship_type == RelationshipType.CAUSAL)}")
    print(f"  - Supporting: {sum(1 for e in graph.edges if e.relationship_type == RelationshipType.SUPPORTING)}")
    print(f"  - Contradicting: {sum(1 for e in graph.edges if e.relationship_type == RelationshipType.CONTRADICTING)}")
    
    print(f"\nMetrics:")
    print(f"  - Convergence Score: {graph.convergence_score:.3f}")
    print(f"  - Avg Path Length: {graph.avg_path_length}")
    print(f"  - Max Depth: {graph.max_convergence_depth}")
    
    print(f"\nConclusion:")
    conclusion = graph.get_node(graph.conclusion_id)
    if conclusion:
        print(f'  "{conclusion.content}"')
    
    print(f"{'='*60}\n")

