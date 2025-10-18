"""
Causal Narratives Extraction Agent

This package extracts causal narrative structures from financial analysis texts,
measuring how well facts converge to support conclusions.
"""

from .data_types import (
    Node,
    Edge,
    NarrativeGraph,
    NarrativeItem,
    NodeType,
    RelationshipType,
    Position,
    Prediction,
    ActualOutcome
)

from .graph import (
    extract_narratives,
    extract_single_narrative,
    batch_extract_narratives
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
    "extract_single_narrative",
    "batch_extract_narratives",
    
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
    
    # Chains
    "NarrativeChains",
    "get_chains",
    
    # Utilities
    "calculate_convergence_score",
    "calculate_graph_metrics",
    "validate_graph",
    "print_graph_summary",
]

