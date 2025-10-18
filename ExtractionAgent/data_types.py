from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Tuple
from datetime import datetime
from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the causal narrative"""
    FACT = "fact"  # Concrete, verifiable facts (e.g., "Q3 revenue increased 20%")
    EVENT = "event"  # Specific events (e.g., "CEO resignation")
    OPINION = "opinion"  # Interpretations or judgments
    CONCLUSION = "conclusion"  # The final claim/thesis (e.g., "Stock is undervalued")
    ASSUMPTION = "assumption"  # Underlying assumptions in the argument


class RelationshipType(str, Enum):
    """Types of relationships between nodes"""
    CAUSAL = "causal"  # X causes Y
    SUPPORTING = "supporting"  # X supports/evidences Y
    CONTRADICTING = "contradicting"  # X contradicts Y
    CONDITIONAL = "conditional"  # X leads to Y under certain conditions


class Position(str, Enum):
    """Investment position/stance"""
    LONG = "long"  # Bullish/buy
    SHORT = "short"  # Bearish/sell
    NEUTRAL = "neutral"  # Hold/uncertain


class Node(BaseModel):
    """Represents a single concept, fact, or claim in the narrative"""
    id: str = Field(..., description="Unique identifier for the node")
    content: str = Field(..., description="The actual text content of this node")
    node_type: NodeType = Field(..., description="Type of node (fact, event, opinion, conclusion)")

class Edge(BaseModel):
    """Represents a causal or logical relationship between two nodes"""
    source_id: str = Field(..., description="ID of the source node")
    target_id: str = Field(..., description="ID of the target node")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    description: Optional[str] = Field(None, description="Optional textual description of the relationship")


class NarrativeGraph(BaseModel):
    """Graph structure representing the causal narrative"""
    nodes: List[Node] = Field(default_factory=list, description="All nodes in the narrative")
    edges: List[Edge] = Field(default_factory=list, description="All edges connecting nodes")
    conclusion_id: str = Field(..., description="ID of the single conclusion node in this graph")
    convergence_score: float = Field(..., ge=0.0, le=1.0, description="How well facts converge to the conclusion (0-1)")
    
    # Metrics for measuring argument quality
    num_facts: int = Field(default=0, description="Number of fact nodes")
    num_conclusions: int = Field(default=0, description="Number of conclusion nodes")
    conclusion_ids: List[str] = Field(default_factory=list, description="IDs of conclusion nodes")
    avg_path_length: Optional[float] = Field(None, description="Average path length from facts to conclusions")
    max_convergence_depth: Optional[int] = Field(None, description="Maximum chain depth to a conclusion")
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges pointing to this node"""
        return [edge for edge in self.edges if edge.target_id == node_id]
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges originating from this node"""
        return [edge for edge in self.edges if edge.source_id == node_id]


class Prediction(BaseModel):
    """Prediction about stock price movement"""
    direction: Literal["up", "down", "neutral"] = Field(..., description="Predicted price direction")
    magnitude: Optional[float] = Field(None, description="Expected % change (e.g., 0.05 for 5%)")
    time_horizon_days: int = Field(default=30, description="Prediction time horizon in days")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in prediction")


class ActualOutcome(BaseModel):
    """Actual observed stock price movement for verification"""
    price_at_analysis: float = Field(..., description="Stock price when analysis was created")
    price_at_verification: float = Field(..., description="Stock price at verification time")
    percent_change: float = Field(..., description="Actual % change in price")
    verification_date: datetime = Field(..., description="When the outcome was verified")
    prediction_correct: bool = Field(..., description="Whether the prediction was correct")


class NarrativeItem(BaseModel):
    """Complete narrative extraction from a financial analysis"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'TSMC', 'AAPL')")
    position: Position = Field(..., description="Investment stance (long/short/neutral)")
    created_at: datetime = Field(default_factory=datetime.now, description="When the analysis was created")
    source_text: str = Field(..., description="Original text of the financial analysis")    
    # Core extraction
    narrative_graph: NarrativeGraph = Field(..., description="Extracted causal narrative graph")
    
    # Prediction and verification
    prediction: Optional[Prediction] = Field(None, description="Predicted price movement")
    actual_outcome: Optional[ActualOutcome] = Field(None, description="Actual outcome for verification")
    
    # Quality metrics
    argument_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall argument quality")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ticker": "TSMC",
                "position": "long",
                "source_text": "TSMC reported 20% revenue growth...",
                "narrative_graph": {
                    "nodes": [],
                    "edges": [],
                    "convergence_score": 0.85
                }
            }
        }