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

from typing import TypedDict, Annotated, List, Literal, Optional
from operator import add
from pathlib import Path
import json
from datetime import datetime

from langgraph.graph import StateGraph, END

from .data_types import NarrativeItem, NarrativeGraph, Node, Edge, Position, Prediction
from .process import (
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
    
    print(f"📍 [NODE] set_current_conclusion: idx={idx}, total_conclusions={len(conclusions)}")
    
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
    print(f"📍 [NODE] extract_nodes: starting")
    current_conclusion = state['current_conclusion']
    all_nodes, error = process_extract_nodes(state['source_text'], current_conclusion)
    
    if error:
        print(f"❌ [ERROR] extract_nodes failed: {error}")
        return {
            'raw_nodes': all_nodes,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] extract_nodes: extracted {len(all_nodes)} nodes")
    return {
        'raw_nodes': all_nodes,
        'messages': [f"Extracted {len(all_nodes) - 1} supporting nodes"]
    }


def extract_edges_node(state: ExtractionState) -> dict:
    """
    Workflow Node 5: Extract causal and logical relationships between nodes
    
    Thin wrapper that delegates to process_extract_edges()
    """
    print(f"📍 [NODE] extract_edges: starting")
    nodes = state['raw_nodes']
    conclusion = state['current_conclusion']
    
    edges, error = process_extract_edges(nodes, conclusion, state['source_text'])
    
    if error:
        print(f"❌ [ERROR] extract_edges failed: {error}")
        return {
            'raw_edges': edges,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] extract_edges: extracted {len(edges)} edges")
    return {
        'raw_edges': edges,
        'messages': [f"Extracted {len(edges)} relationships"]
    }


def build_narrative_graph_node(state: ExtractionState) -> dict:
    """
    Workflow Node 6: Build and validate the NarrativeGraph
    
    Thin wrapper that delegates to process_build_narrative_graph()
    """
    print(f"📍 [NODE] build_narrative_graph: starting")
    nodes = state['raw_nodes']
    edges = state['raw_edges']
    conclusion = state['current_conclusion']
    
    graph, errors = process_build_narrative_graph(nodes, edges, conclusion)
    
    if errors:
        print(f"❌ [ERROR] build_narrative_graph failed: {errors}")
        return {
            'errors': errors,
            'messages': ["Graph validation failed"]
        }
    
    print(f"✅ [SUCCESS] build_narrative_graph: convergence={graph.convergence_score:.3f}")
    return {
        'narrative_graph': graph,
        'messages': [f"Built narrative graph (convergence: {graph.convergence_score:.3f})"]
    }


def extract_prediction_node(state: ExtractionState) -> dict:
    """
    Workflow Node 7: Extract price prediction from the analysis
    
    Thin wrapper that delegates to process_extract_prediction()
    """
    print(f"📍 [NODE] extract_prediction: starting")
    conclusion = state['current_conclusion']
    position = state['position']
    
    prediction, error = process_extract_prediction(state['source_text'], conclusion, position)
    
    result = {}
    if error:
        print(f"❌ [ERROR] extract_prediction failed: {error}")
        result = {
            'prediction': prediction,
            'errors': [error]
        }
    else:
        print(f"✅ [SUCCESS] extract_prediction: {prediction.direction} ({prediction.confidence:.2f})")
        result = {
            'prediction': prediction,
            'messages': [f"Extracted prediction: {prediction.direction} ({prediction.confidence:.2f} confidence)"]
        }
    
    print(f"🔄 [STATE UPDATE] extract_prediction returning: {list(result.keys())}")
    return result


def create_narrative_item_node(state: ExtractionState) -> dict:
    """
    Workflow Node 8: Create the final NarrativeItem
    
    Thin wrapper that delegates to process_create_narrative_item()
    """
    current_idx = state['current_conclusion_idx']
    print(f"📍 [NODE] create_narrative_item: current_idx={current_idx}")
    
    # Skip if graph failed to build
    if state.get('narrative_graph') is None:
        new_idx = current_idx + 1
        print(f"⚠️  [WARNING] Skipping conclusion {current_idx + 1}: narrative_graph is None. Incrementing idx {current_idx} -> {new_idx}")
        return {
            'current_conclusion_idx': new_idx,
            'errors': [f"Skipping conclusion {current_idx + 1}: narrative graph failed to build"]
        }
    
    item, error = process_create_narrative_item(
        ticker=state['ticker'],
        position=state['position'],
        source_text=state['source_text'],
        narrative_graph=state['narrative_graph'],
        prediction=state['prediction']
    )
    
    new_idx = current_idx + 1
    
    if error:
        print(f"❌ [ERROR] Failed to create item for conclusion {current_idx + 1}: {error}. Incrementing idx {current_idx} -> {new_idx}")
        return {
            'current_conclusion_idx': new_idx,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] Created item for conclusion {current_idx + 1}. Incrementing idx {current_idx} -> {new_idx}")
    return {
        'narrative_items': [item],
        'current_conclusion_idx': new_idx,
        'messages': [f"Created NarrativeItem for conclusion {current_idx + 1}"]
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
    
    decision = "continue" if idx < len(conclusions) else "end"
    print(f"🔀 [DECISION] should_continue_processing: idx={idx}, total_conclusions={len(conclusions)}, decision={decision}")
    
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

def extract_narratives(
    source_text: str, 
    save_json: bool = False,
    json_output_path: Optional[str] = None,
    separate_files: bool = False
) -> List[NarrativeItem]:
    """
    Main entry point: Extract all narrative items from financial analysis text
    
    This function:
    1. Creates the LangGraph workflow
    2. Initializes the state with the source text
    3. Executes the workflow
    4. Returns the extracted NarrativeItem(s)
    5. Optionally saves results to JSON
    
    Args:
        source_text: The financial analysis text to process
        save_json: If True, automatically save results to JSON file(s)
        json_output_path: Path for JSON output (default: output/narratives_TIMESTAMP.json)
        separate_files: If True, save each narrative to a separate JSON file
        
    Returns:
        List of NarrativeItem objects (one per conclusion found)
        
    Example:
        >>> text = "TSMC reported 20% revenue growth. The company is undervalued."
        >>> items = extract_narratives(text)
        >>> print(f"Found {len(items)} narrative(s)")
        >>> print(f"Convergence score: {items[0].narrative_graph.convergence_score}")
        
        >>> # With automatic JSON saving
        >>> items = extract_narratives(text, save_json=True, json_output_path="output/tsmc.json")
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
        # Set recursion_limit high enough for multiple conclusions
        # Each conclusion takes ~7 steps, plus 3 initial steps
        # Formula: 3 + (7 * expected_conclusions) + buffer
        # For safety, use 100 to handle up to ~14 conclusions
        final_state = graph.invoke(initial_state, {"recursion_limit": 100})
        
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
        
        narrative_items = final_state.get('narrative_items', [])
        
        # Auto-save to JSON if requested
        if save_json and narrative_items:
            save_narratives_to_json(
                narrative_items, 
                output_path=json_output_path,
                separate_files=separate_files
            )
        
        return narrative_items
        
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


# ============================================================================
# JSON EXPORT FUNCTIONS
# ============================================================================

def save_narratives_to_json(
    narrative_items: List[NarrativeItem],
    output_path: Optional[str] = None,
    separate_files: bool = False
) -> List[str]:
    """
    Save narrative items to JSON file(s)
    
    Args:
        narrative_items: List of NarrativeItem objects to save
        output_path: Path to output file or directory. If None, uses default
        separate_files: If True, save each narrative to a separate file
        
    Returns:
        List of file paths that were created
        
    Example:
        >>> items = extract_narratives(text)
        >>> save_narratives_to_json(items, "output/narratives.json")
        ['output/narratives.json']
        
        >>> save_narratives_to_json(items, "output/", separate_files=True)
        ['output/narrative_1.json', 'output/narrative_2.json', ...]
    """
    if not narrative_items:
        print("⚠️  No narrative items to save")
        return []
    
    # Create default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/narratives_{timestamp}.json"
    
    output_path = Path(output_path)
    created_files = []
    
    if separate_files:
        # Save each narrative to a separate file
        # If output_path is a file, use its parent directory
        if output_path.suffix:
            output_dir = output_path.parent
        else:
            output_dir = output_path
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(narrative_items, 1):
            # Create filename based on ticker and conclusion
            conclusion = item.narrative_graph.get_node(item.narrative_graph.conclusion_id)
            conclusion_snippet = conclusion.content[:50].replace(" ", "_") if conclusion else f"narrative_{i}"
            # Sanitize filename
            safe_name = "".join(c for c in conclusion_snippet if c.isalnum() or c in "._- ")
            
            file_path = output_dir / f"{item.ticker}_{i}_{safe_name}.json"
            
            # Use Pydantic's model_dump with custom serialization
            json_data = item.model_dump(mode='json')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            created_files.append(str(file_path))
            print(f"💾 Saved narrative {i} to: {file_path}")
    
    else:
        # Save all narratives to a single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert all items to dict using Pydantic's model_dump
        json_data = {
            "extraction_date": datetime.now().isoformat(),
            "num_narratives": len(narrative_items),
            "narratives": [item.model_dump(mode='json') for item in narrative_items]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        created_files.append(str(output_path))
        print(f"💾 Saved {len(narrative_items)} narrative(s) to: {output_path}")
    
    return created_files


def load_narratives_from_json(json_path: str) -> List[NarrativeItem]:
    """
    Load narrative items from JSON file
    
    Args:
        json_path: Path to JSON file created by save_narratives_to_json()
        
    Returns:
        List of NarrativeItem objects
        
    Example:
        >>> items = load_narratives_from_json("output/narratives.json")
        >>> print(f"Loaded {len(items)} narratives")
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both single-file and multi-file formats
    if "narratives" in data:
        # Single file with multiple narratives
        narrative_dicts = data["narratives"]
    elif isinstance(data, dict) and "ticker" in data:
        # Single narrative file
        narrative_dicts = [data]
    else:
        raise ValueError(f"Unrecognized JSON format in {json_path}")
    
    # Convert dicts back to NarrativeItem objects using Pydantic
    items = [NarrativeItem.model_validate(n) for n in narrative_dicts]
    
    print(f"📂 Loaded {len(items)} narrative(s) from: {json_path}")
    return items

