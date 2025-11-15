"""
Causal Narratives Extraction Agent

This package extracts causal narrative structures from financial analysis texts,
measuring how well facts converge to support conclusions.
"""

# Load environment variables early so downstream modules (e.g., ChatOpenAI) have access
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    # dotenv is optional; ignore if unavailable
    pass

from .data_types import (
    Node,
    Edge,
    NarrativeGraph,
    NarrativeItem,
    NodeType,
    RelationshipType,
    Position,
    Prediction,
    ActualOutcome,
    ExtractionConfig
)

from .graph import (
    extract_narratives,
    batch_extract_narratives,
    save_narratives_to_json,
    load_narratives_from_json
)

from .chains import NarrativeChains, get_chains

from .process import (
    calculate_convergence_score,
    calculate_graph_metrics,
    validate_graph,
    print_graph_summary
)

__version__ = "0.1.0"

__all__ = [
    # Main functions
    "extract_narratives",
    "batch_extract_narratives",
    
    # JSON export/import
    "save_narratives_to_json",
    "load_narratives_from_json",
    
    # Data types
    "Node",
    "Edge",
    "NarrativeGraph",
    "NarrativeItem",
    "NodeType",
    "RelationshipType",
    "Position",
    "Prediction",
    "ActualOutcome",
    "ExtractionConfig",
    
    # Chains
    "NarrativeChains",
    "get_chains",
    
    # Utilities
    "calculate_convergence_score",
    "calculate_graph_metrics",
    "validate_graph",
    "print_graph_summary",
]

