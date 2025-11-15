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

from .data_types import NarrativeItem, NarrativeGraph, Node, Edge, Position, Prediction, ExtractionConfig
from .process import (
    process_extract_metadata,
    process_extract_conclusion,
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
    config: ExtractionConfig  # Configuration with safety limits
    
    # ===== METADATA =====
    ticker: str  # Stock ticker symbol
    position: Position  # Investment stance (long/short/neutral)
    
    # ===== CONCLUSION =====
    conclusion: Optional[Node]  # The single main conclusion
    
    # ===== EXTRACTION =====
    raw_nodes: List[Node]  # Extracted nodes for the narrative
    raw_edges: List[Edge]  # Extracted edges for the narrative
    narrative_graph: Optional[NarrativeGraph]  # Constructed graph
    prediction: Optional[Prediction]  # Price prediction
    
    # ===== OUTPUT =====
    narrative_item: Optional[NarrativeItem]  # Final result
    
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
    print(f"📍 [NODE] extract_metadata: starting")
    ticker, position, error = process_extract_metadata(state['source_text'])
    
    if error:
        print(f"❌ [ERROR] extract_metadata failed: {error}")
        return {
            'ticker': ticker,
            'position': position,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] extract_metadata: {ticker} ({position.value})")
    return {
        'ticker': ticker,
        'position': position,
        'messages': [f"Extracted metadata: {ticker} ({position.value})"]
    }


def extract_conclusion_node(state: ExtractionState) -> dict:
    """
    Workflow Node 2: Identify the MAIN conclusion statement
    
    Extracts the single primary investment thesis/recommendation.
    Thin wrapper that delegates to process_extract_conclusion()
    """
    print(f"📍 [NODE] extract_conclusion: starting")
    ticker = state['ticker']
    position = state['position']
    
    conclusion, error = process_extract_conclusion(state['source_text'], ticker, position)
    
    if error:
        print(f"❌ [ERROR] extract_conclusion failed: {error}")
        return {
            'conclusion': None,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] extract_conclusion: \"{conclusion.content[:80]}...\"")
    return {
        'conclusion': conclusion,
        'messages': [f"Found main conclusion: \"{conclusion.content[:80]}...\""]
    }




def extract_nodes_node(state: ExtractionState) -> dict:
    """
    Workflow Node 3: Extract nodes (facts, events, opinions, assumptions)
    
    Thin wrapper that delegates to process_extract_nodes()
    """
    print(f"📍 [NODE] extract_nodes: starting")
    conclusion = state['conclusion']
    config = state.get('config', ExtractionConfig())
    
    if not conclusion:
        print(f"❌ [ERROR] No conclusion to extract nodes for")
        return {
            'raw_nodes': [],
            'errors': ["Cannot extract nodes without a conclusion"]
        }
    
    all_nodes, error = process_extract_nodes(state['source_text'], conclusion, config)
    
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
    Workflow Node 4: Extract causal and logical relationships between nodes
    
    Thin wrapper that delegates to process_extract_edges()
    """
    print(f"📍 [NODE] extract_edges: starting")
    nodes = state['raw_nodes']
    conclusion = state['conclusion']
    config = state.get('config', ExtractionConfig())
    
    edges, error = process_extract_edges(nodes, conclusion, state['source_text'], config)
    
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
    Workflow Node 5: Build and validate the NarrativeGraph
    
    Thin wrapper that delegates to process_build_narrative_graph()
    """
    print(f"📍 [NODE] build_narrative_graph: starting")
    nodes = state['raw_nodes']
    edges = state['raw_edges']
    conclusion = state['conclusion']
    config = state.get('config', ExtractionConfig())
    
    graph, errors = process_build_narrative_graph(nodes, edges, conclusion, config)
    
    if errors and graph is None:
        print(f"❌ [ERROR] build_narrative_graph failed: {errors}")
        return {
            'narrative_graph': None,
            'errors': errors,
            'messages': ["Graph validation failed"]
        }
    
    # Graph was created but may have warnings
    if errors:
        print(f"✅ [SUCCESS] build_narrative_graph: convergence={graph.convergence_score:.3f} (with warnings)")
        return {
            'narrative_graph': graph,
            'errors': errors,
            'messages': [f"Built narrative graph (convergence: {graph.convergence_score:.3f}) with warnings"]
        }
    
    print(f"✅ [SUCCESS] build_narrative_graph: convergence={graph.convergence_score:.3f}")
    return {
        'narrative_graph': graph,
        'messages': [f"Built narrative graph (convergence: {graph.convergence_score:.3f})"]
    }


def extract_prediction_node(state: ExtractionState) -> dict:
    """
    Workflow Node 6: Extract price prediction from the analysis
    
    Thin wrapper that delegates to process_extract_prediction()
    """
    print(f"📍 [NODE] extract_prediction: starting")
    conclusion = state['conclusion']
    position = state['position']
    
    prediction, error = process_extract_prediction(state['source_text'], conclusion, position)
    
    if error:
        print(f"❌ [ERROR] extract_prediction failed: {error}")
        return {
            'prediction': prediction,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] extract_prediction: {prediction.direction} ({prediction.confidence:.2f})")
    return {
        'prediction': prediction,
        'messages': [f"Extracted prediction: {prediction.direction} ({prediction.confidence:.2f} confidence)"]
    }


def create_narrative_item_node(state: ExtractionState) -> dict:
    """
    Workflow Node 7: Create the final NarrativeItem
    
    Thin wrapper that delegates to process_create_narrative_item()
    """
    print(f"📍 [NODE] create_narrative_item: starting")
    
    # Skip if graph failed to build
    if state.get('narrative_graph') is None:
        print(f"⚠️  [WARNING] Cannot create item: narrative_graph is None")
        return {
            'narrative_item': None,
            'errors': ["Cannot create narrative item: graph failed to build"]
        }
    
    item, error = process_create_narrative_item(
        ticker=state['ticker'],
        position=state['position'],
        source_text=state['source_text'],
        narrative_graph=state['narrative_graph'],
        prediction=state['prediction']
    )
    
    if error:
        print(f"❌ [ERROR] Failed to create item: {error}")
        return {
            'narrative_item': None,
            'errors': [error]
        }
    
    print(f"✅ [SUCCESS] Created narrative item")
    return {
        'narrative_item': item,
        'messages': ["Created NarrativeItem successfully"]
    }


# ============================================================================
# No conditional edges needed - linear workflow
# ============================================================================


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_extraction_graph() -> StateGraph:
    """
    Build the complete LangGraph workflow for narrative extraction
    
    Workflow (SINGLE GRAPH - NO LOOP):
    1. Extract metadata (ticker, position)
    2. Extract THE main conclusion
    3. Extract ALL nodes (facts, events, opinions) from entire text
    4. Extract ALL edges showing relationships
    5. Build ONE unified narrative graph
    6. Extract prediction
    7. Create NarrativeItem
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize graph
    workflow = StateGraph(ExtractionState)
    
    # Add all processing nodes (linear flow)
    workflow.add_node("extract_metadata", extract_metadata_node)
    workflow.add_node("extract_conclusion", extract_conclusion_node)
    workflow.add_node("extract_nodes", extract_nodes_node)
    workflow.add_node("extract_edges", extract_edges_node)
    workflow.add_node("build_graph", build_narrative_graph_node)
    workflow.add_node("extract_prediction", extract_prediction_node)
    workflow.add_node("create_item", create_narrative_item_node)
    
    # Define LINEAR flow - no loops
    workflow.set_entry_point("extract_metadata")
    workflow.add_edge("extract_metadata", "extract_conclusion")
    workflow.add_edge("extract_conclusion", "extract_nodes")
    workflow.add_edge("extract_nodes", "extract_edges")
    workflow.add_edge("extract_edges", "build_graph")
    workflow.add_edge("build_graph", "extract_prediction")
    workflow.add_edge("extract_prediction", "create_item")
    workflow.add_edge("create_item", END)
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def extract_narratives(
    source_text: str, 
    config: ExtractionConfig = None,
    save_json: bool = False,
    json_output_path: Optional[str] = None
) -> Optional[NarrativeItem]:
    """
    Main entry point: Extract a narrative from financial analysis text
    
    This function:
    1. Creates the LangGraph workflow
    2. Initializes the state with the source text and config
    3. Executes the workflow
    4. Returns the extracted NarrativeItem
    5. Optionally saves result to JSON
    
    Args:
        source_text: The financial analysis text to process
        config: Extraction configuration with safety limits (default: ExtractionConfig())
        save_json: If True, automatically save result to JSON file
        json_output_path: Path for JSON output (default: output/narrative_TIMESTAMP.json)
        
    Returns:
        Single NarrativeItem object, or None if extraction failed
        
    Example:
        >>> text = "TSMC reported 20% revenue growth. I recommend buying."
        >>> item = extract_narratives(text)
        >>> print(f"Convergence score: {item.narrative_graph.convergence_score}")
        
        >>> # With safety limits disabled
        >>> config = ExtractionConfig(enable_safety_limits=False)
        >>> item = extract_narratives(text, config=config)
        
        >>> # With automatic JSON saving
        >>> item = extract_narratives(text, save_json=True, json_output_path="output/tsmc.json")
    """
    if config is None:
        config = ExtractionConfig()
    
    # Build the graph
    graph = build_extraction_graph()
    
    # Initialize state
    initial_state = ExtractionState(
        source_text=source_text,
        config=config,
        ticker='',
        position=Position.NEUTRAL,
        conclusion=None,
        raw_nodes=[],
        raw_edges=[],
        narrative_graph=None,
        prediction=None,
        narrative_item=None,
        errors=[],
        messages=[]
    )
    
    # Execute the workflow
    print("\n" + "="*60)
    print("STARTING NARRATIVE EXTRACTION")
    print("="*60)
    
    try:
        # Linear workflow only needs 7 steps
        final_state = graph.invoke(initial_state, {"recursion_limit": 20})
        
        # Print summary
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        
        narrative_item = final_state.get('narrative_item')
        
        if narrative_item:
            print(f"✅ Successfully extracted narrative")
            print(f"   Ticker: {narrative_item.ticker}")
            print(f"   Position: {narrative_item.position.value}")
            print(f"   Convergence: {narrative_item.narrative_graph.convergence_score:.3f}")
            print(f"   Nodes: {len(narrative_item.narrative_graph.nodes)}")
            print(f"   Edges: {len(narrative_item.narrative_graph.edges)}")
        else:
            print(f"❌ Failed to extract narrative")
        
        if final_state.get('errors'):
            print(f"\n⚠️  Warnings/Errors: {len(final_state['errors'])}")
            for error in final_state['errors']:
                print(f"  - {error}")
        
        print("="*60 + "\n")
        
        # Auto-save to JSON if requested
        if save_json and narrative_item:
            save_narratives_to_json(
                [narrative_item],  # Wrap in list for compatibility
                output_path=json_output_path
            )
        
        return narrative_item
        
    except Exception as e:
        print(f"\n❌ Extraction failed: {str(e)}\n")
        raise


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def batch_extract_narratives(
    texts: List[str],
    config: ExtractionConfig = None
) -> List[Optional[NarrativeItem]]:
    """
    Extract narratives from multiple texts in batch
    
    Args:
        texts: List of financial analysis texts
        config: Extraction configuration (applied to all texts)
        
    Returns:
        List of NarrativeItems (one per input text, None if extraction failed)
    """
    if config is None:
        config = ExtractionConfig()
    
    results = []
    
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}/{len(texts)}...")
        try:
            item = extract_narratives(text, config=config)
            results.append(item)
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            results.append(None)
    
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

