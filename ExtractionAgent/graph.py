"""
LangGraph Workflow Orchestration

This module defines the LangGraph workflow that orchestrates the extraction
of causal narratives from financial analysis text.

Architecture:
- graph.py: Workflow orchestration with thin node wrappers (this file)
- process.py: Business logic for each extraction step
- chains.py: LLM prompts and interactions
- data_types.py: Data models and schemas

Node functions in this file are thin wrappers that:
1. Extract data from state
2. Call business logic in process.py
3. Format results for state update
"""

from typing import TypedDict, Annotated, List, Literal
from operator import add

from langgraph.graph import StateGraph, END

from data_types import NarrativeItem, NarrativeGraph, Node, Edge, Position, Prediction
from process import (
    process_extract_metadata,
    process_extract_conclusions,
    process_extract_nodes,
    process_extract_edges,
    process_build_narrative_graph,
    process_extract_prediction,
    process_create_narrative_item
)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ExtractionState(TypedDict):
    """
    State that flows through the LangGraph pipeline
    
    The state contains both input data and accumulated results as the
    extraction progresses through different stages.
    """
    # ===== INPUT =====
    source_text: str  # Original financial analysis text
    
    # ===== METADATA =====
    ticker: str  # Stock ticker symbol
    position: Position  # Investment stance (long/short/neutral)
    
    # ===== CONCLUSIONS =====
    conclusions: List[Node]  # All conclusions found in text
    current_conclusion_idx: int  # Index of conclusion being processed
    current_conclusion: Node  # Current conclusion being analyzed
    
    # ===== PER-CONCLUSION EXTRACTION =====
    raw_nodes: List[Node]  # Extracted nodes for current conclusion
    raw_edges: List[Edge]  # Extracted edges for current conclusion
    narrative_graph: NarrativeGraph  # Constructed graph for current conclusion
    prediction: Prediction  # Price prediction for current conclusion
    
    # ===== OUTPUTS =====
    narrative_items: Annotated[List[NarrativeItem], add]  # Final results
    
    # ===== ERROR HANDLING =====
    errors: Annotated[List[str], add]  # Accumulated errors
    messages: Annotated[List[str], add]  # Status messages


# ============================================================================
# GRAPH NODES (Processing Steps)
# ============================================================================

def extract_metadata_node(state: ExtractionState) -> dict:
    """
    Workflow Node 1: Extract ticker symbol and investment position
    
    Thin wrapper that delegates to process_extract_metadata()
    """
    ticker, position, error = process_extract_metadata(state['source_text'])
    
    if error:
        return {
            'ticker': ticker,
            'position': position,
            'errors': [error]
        }
    
    return {
        'ticker': ticker,
        'position': position,
        'messages': [f"Extracted metadata: {ticker} ({position.value})"]
    }


def extract_conclusions_node(state: ExtractionState) -> dict:
    """
    Workflow Node 2: Identify ALL conclusion statements
    
    Each conclusion will be processed separately to create individual NarrativeItems.
    Thin wrapper that delegates to process_extract_conclusions()
    """
    conclusions, error = process_extract_conclusions(state['source_text'])
    
    if error:
        return {
            'conclusions': conclusions,
            'current_conclusion_idx': 0,
            'errors': [error]
        }
    
    return {
        'conclusions': conclusions,
        'current_conclusion_idx': 0,
        'messages': [f"Found {len(conclusions)} conclusion(s)"]
    }


def set_current_conclusion_node(state: ExtractionState) -> dict:
    """
    Node 3: Set the current conclusion to process
    
    This node prepares the state for processing the next conclusion.
    """
    idx = state['current_conclusion_idx']
    conclusions = state['conclusions']
    
    if idx < len(conclusions):
        current = conclusions[idx]
        return {
            'current_conclusion': current,
            'messages': [f"Processing conclusion {idx+1}/{len(conclusions)}: \"{current.content}\""]
        }
    else:
        return {}


def extract_nodes_node(state: ExtractionState) -> dict:
    """
    Workflow Node 4: Extract nodes (facts, events, opinions, assumptions)
    
    Thin wrapper that delegates to process_extract_nodes()
    """
    current_conclusion = state['current_conclusion']
    all_nodes, error = process_extract_nodes(state['source_text'], current_conclusion)
    
    if error:
        return {
            'raw_nodes': all_nodes,
            'errors': [error]
        }
    
    return {
        'raw_nodes': all_nodes,
        'messages': [f"Extracted {len(all_nodes) - 1} supporting nodes"]
    }


def extract_edges_node(state: ExtractionState) -> dict:
    """
    Workflow Node 5: Extract causal and logical relationships between nodes
    
    Thin wrapper that delegates to process_extract_edges()
    """
    nodes = state['raw_nodes']
    conclusion = state['current_conclusion']
    
    edges, error = process_extract_edges(nodes, conclusion, state['source_text'])
    
    if error:
        return {
            'raw_edges': edges,
            'errors': [error]
        }
    
    return {
        'raw_edges': edges,
        'messages': [f"Extracted {len(edges)} relationships"]
    }


def build_narrative_graph_node(state: ExtractionState) -> dict:
    """
    Workflow Node 6: Build and validate the NarrativeGraph
    
    Thin wrapper that delegates to process_build_narrative_graph()
    """
    nodes = state['raw_nodes']
    edges = state['raw_edges']
    conclusion = state['current_conclusion']
    
    graph, errors = process_build_narrative_graph(nodes, edges, conclusion)
    
    if errors:
        return {
            'errors': errors,
            'messages': ["Graph validation failed"]
        }
    
    return {
        'narrative_graph': graph,
        'messages': [f"Built narrative graph (convergence: {graph.convergence_score:.3f})"]
    }


def extract_prediction_node(state: ExtractionState) -> dict:
    """
    Workflow Node 7: Extract price prediction from the analysis
    
    Thin wrapper that delegates to process_extract_prediction()
    """
    conclusion = state['current_conclusion']
    position = state['position']
    
    prediction, error = process_extract_prediction(state['source_text'], conclusion, position)
    
    if error:
        return {
            'prediction': prediction,
            'errors': [error]
        }
    
    return {
        'prediction': prediction,
        'messages': [f"Extracted prediction: {prediction.direction} ({prediction.confidence:.2f} confidence)"]
    }


def create_narrative_item_node(state: ExtractionState) -> dict:
    """
    Workflow Node 8: Create the final NarrativeItem
    
    Thin wrapper that delegates to process_create_narrative_item()
    """
    item, error = process_create_narrative_item(
        ticker=state['ticker'],
        position=state['position'],
        source_text=state['source_text'],
        narrative_graph=state['narrative_graph'],
        prediction=state['prediction']
    )
    
    if error:
        return {
            'current_conclusion_idx': state['current_conclusion_idx'] + 1,
            'errors': [error]
        }
    
    return {
        'narrative_items': [item],
        'current_conclusion_idx': state['current_conclusion_idx'] + 1,
        'messages': [f"Created NarrativeItem for conclusion {state['current_conclusion_idx'] + 1}"]
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue_processing(state: ExtractionState) -> Literal["continue", "end"]:
    """
    Conditional edge: Check if more conclusions to process
    
    Returns:
        "continue" if there are more conclusions to process
        "end" if all conclusions have been processed
    """
    idx = state.get('current_conclusion_idx', 0)
    conclusions = state.get('conclusions', [])
    
    if idx < len(conclusions):
        return "continue"
    else:
        return "end"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_extraction_graph() -> StateGraph:
    """
    Build the complete LangGraph workflow for narrative extraction
    
    Workflow:
    1. Extract metadata (ticker, position)
    2. Extract all conclusions
    3. For each conclusion:
       a. Set as current conclusion
       b. Extract nodes (facts, events, etc.)
       c. Extract edges (relationships)
       d. Build narrative graph with metrics
       e. Extract prediction
       f. Create NarrativeItem
    4. Return all NarrativeItems
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    workflow = StateGraph(ExtractionState)
    
    # Add all processing nodes
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("extract_conclusions", extract_conclusions_node)
    workflow.add_node("set_current_conclusion", set_current_conclusion_node)
    workflow.add_node("extract_nodes", extract_nodes_node)
    workflow.add_node("extract_edges", extract_edges_node)
    workflow.add_node("build_graph", build_narrative_graph_node)
    workflow.add_node("extract_prediction", extract_prediction_node)
    workflow.add_node("create_item", create_narrative_item_node)
    
    # Define the flow
    workflow.set_entry_point("extract_metadata")
    workflow.add_edge("extract_metadata", "extract_conclusions")
    
    # Conditional: check if we have conclusions to process
    workflow.add_conditional_edges(
        "extract_conclusions",
        should_continue_processing,
        {
            "continue": "set_current_conclusion",
            "end": END
        }
    )
    
    # Linear flow for processing each conclusion
    workflow.add_edge("set_current_conclusion", "extract_nodes")
    workflow.add_edge("extract_nodes", "extract_edges")
    workflow.add_edge("extract_edges", "build_graph")
    workflow.add_edge("build_graph", "extract_prediction")
    workflow.add_edge("extract_prediction", "create_item")
    
    # After creating item, check if more conclusions to process
    workflow.add_conditional_edges(
        "create_item",
        should_continue_processing,
        {
            "continue": "set_current_conclusion",
            "end": END
        }
    )
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def extract_narratives(source_text: str) -> List[NarrativeItem]:
    """
    Main entry point: Extract all narrative items from financial analysis text
    
    This function:
    1. Creates the LangGraph workflow
    2. Initializes the state with the source text
    3. Executes the workflow
    4. Returns the extracted NarrativeItem(s)
    
    Args:
        source_text: The financial analysis text to process
        
    Returns:
        List of NarrativeItem objects (one per conclusion found)
        
    Example:
        >>> text = "TSMC reported 20% revenue growth. The company is undervalued."
        >>> items = extract_narratives(text)
        >>> print(f"Found {len(items)} narrative(s)")
        >>> print(f"Convergence score: {items[0].narrative_graph.convergence_score}")
    """
    # Build the graph
    graph = build_extraction_graph()
    
    # Initialize state
    initial_state = ExtractionState(
        source_text=source_text,
        ticker='',
        position=Position.NEUTRAL,
        conclusions=[],
        current_conclusion_idx=0,
        current_conclusion=None,
        raw_nodes=[],
        raw_edges=[],
        narrative_graph=None,
        prediction=None,
        narrative_items=[],
        errors=[],
        messages=[]
    )
    
    # Execute the workflow
    print("\n" + "="*60)
    print("STARTING NARRATIVE EXTRACTION")
    print("="*60)
    
    try:
        final_state = graph.invoke(initial_state)
        
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Narratives extracted: {len(final_state.get('narrative_items', []))}")
        
        if final_state.get('errors'):
            print(f"Errors encountered: {len(final_state['errors'])}")
            for error in final_state['errors']:
                print(f"  - {error}")
        
        print("="*60 + "\n")
        
        return final_state.get('narrative_items', [])
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {str(e)}\n")
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_single_narrative(
    source_text: str,
    conclusion_text: str = None
) -> NarrativeItem:
    """
    Extract a single narrative for a specific conclusion
    
    Useful when you know there's only one conclusion or want to target
    a specific conclusion.
    
    Args:
        source_text: The financial analysis text
        conclusion_text: Optional specific conclusion to extract for
        
    Returns:
        Single NarrativeItem
    """
    items = extract_narratives(source_text)
    
    if not items:
        raise ValueError("No narratives could be extracted from the text")
    
    if conclusion_text:
        # Find item with matching conclusion
        for item in items:
            conclusion = item.narrative_graph.get_node(
                item.narrative_graph.conclusion_id
            )
            if conclusion and conclusion_text.lower() in conclusion.content.lower():
                return item
        raise ValueError(f"No narrative found for conclusion: {conclusion_text}")
    
    return items[0]


def batch_extract_narratives(texts: List[str]) -> List[List[NarrativeItem]]:
    """
    Extract narratives from multiple texts in batch
    
    Args:
        texts: List of financial analysis texts
        
    Returns:
        List of lists of NarrativeItems (one list per input text)
    """
    results = []
    
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}...")
        try:
            items = extract_narratives(text)
            results.append(items)
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            results.append([])
    
    return results

