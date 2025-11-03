"""
LLM Chains and Prompts for Narrative Extraction

This module contains all the prompts and LangChain chains used to extract
causal narratives from financial analysis text.
"""

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .data_types import Node, Edge, NodeType, RelationshipType, Position, Prediction


# ============================================================================
# SCHEMA DEFINITIONS FOR STRUCTURED OUTPUT
# ============================================================================

class MetadataOutput(BaseModel):
    """Schema for metadata extraction"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., TSMC, AAPL)")
    position: str = Field(..., description="Investment stance: long, short, or neutral")


class ConclusionList(BaseModel):
    """Schema for conclusion extraction"""
    conclusions: List[Node] = Field(..., description="List of all conclusion nodes found in text")


class NodeList(BaseModel):
    """Schema for node extraction"""
    nodes: List[Node] = Field(..., description="List of extracted nodes (facts, events, opinions, assumptions)")


class EdgeList(BaseModel):
    """Schema for edge extraction"""
    edges: List[Edge] = Field(..., description="List of causal/logical relationships between nodes")


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

EXTRACT_METADATA_PROMPT = """You are analyzing a financial analysis article about a stock.

Your task is to extract:
1. **Ticker Symbol**: The stock ticker mentioned (e.g., TSMC, AAPL, MSFT)
2. **Position**: The author's investment stance

Position definitions:
- "long": Bullish/recommending to buy (positive outlook)
- "short": Bearish/recommending to sell (negative outlook)  
- "neutral": Holding/uncertain (balanced or no clear recommendation)

Text to analyze:
{source_text}

Extract the ticker and position from this text."""


EXTRACT_CONCLUSIONS_PROMPT = """You are analyzing a financial analysis to identify CONCLUSION statements.

A CONCLUSION is the main claim, thesis, or recommendation the author is making. Examples:
- "TSMC stock is undervalued and presents a strong buying opportunity"
- "The company will likely underperform the market"
- "Investors should avoid this stock"
- "This is a good long-term hold"

NOT conclusions (these are facts or opinions):
- "Revenue increased 20%" (this is a fact)
- "The market is volatile" (too general)
- "Management is competent" (opinion, but not a final recommendation)

Important: 
- Each conclusion will be analyzed separately
- Extract ALL distinct conclusions in the text
- Use exact quotes or paraphrases from the text
- Assign each a unique ID (e.g., "conclusion_1", "conclusion_2")

Text to analyze:
{source_text}

Extract all conclusion statements as Node objects with:
- id: unique identifier (e.g., "conclusion_1")
- content: the conclusion statement text
- node_type: "conclusion" """


EXTRACT_NODES_PROMPT = """You are analyzing a financial text to extract nodes (facts, events, opinions, assumptions) that relate to a specific conclusion.

Current conclusion being analyzed: "{conclusion_content}"

Node type definitions:
- **FACT**: Concrete, verifiable facts (e.g., "Q3 revenue increased 20%", "P/E ratio is 15")
- **EVENT**: Specific events that happened or will happen (e.g., "CEO resigned", "Product launch scheduled")
- **OPINION**: Interpretations, judgments, or analysis (e.g., "Management is competent", "Valuation seems high")
- **ASSUMPTION**: Underlying assumptions in the argument (e.g., "Market will remain stable", "Growth will continue")

Instructions:
1. Extract ALL nodes from the text that support, relate to, or connect to the conclusion: "{conclusion_content}"
2. DO NOT extract the conclusion itself (it's already included)
3. Be specific and granular - extract individual facts rather than combining them
4. Include nodes that might contradict if present
5. Assign unique IDs (e.g., "fact_1", "event_1", "opinion_1", "assumption_1")

Text to analyze:
{source_text}

Extract nodes as Node objects with:
- id: unique identifier
- content: the node text content
- node_type: one of [fact, event, opinion, assumption]"""


EXTRACT_EDGES_PROMPT = """You are analyzing the causal and logical relationships between nodes in a financial argument.

Nodes in the analysis:
{nodes_description}

Target conclusion: "{conclusion_content}" (ID: {conclusion_id})

Relationship type definitions:
- **causal**: X directly causes Y (e.g., "inflation increase" → "interest rate hike")
- **supporting**: X provides evidence/support for Y (e.g., "revenue growth" → "company is strong")
- **contradicting**: X contradicts or weakens Y
- **conditional**: X leads to Y under certain conditions

Instructions:
1. Identify relationships that form causal chains toward the conclusion
2. Focus on direct relationships (not transitive)
3. Ensure most evidence nodes ultimately connect to the conclusion
4. Look for convergent patterns: multiple nodes → single conclusion

For each relationship, provide:
- source_id: the source node ID
- target_id: the target node ID  
- relationship_type: one of [causal, supporting, contradicting, conditional]
- description: brief explanation of the relationship (optional)

Extract all meaningful relationships as Edge objects."""


EXTRACT_PREDICTION_PROMPT = """You are analyzing a financial text to extract the author's stock price prediction.

Text: {source_text}

Conclusion: {conclusion_content}

Author's Position: {position}

Extract the prediction details:

1. **direction**: Expected price movement
   - "up" if predicting price increase
   - "down" if predicting price decrease  
   - "neutral" if no clear direction or expecting stability

2. **magnitude**: Expected percentage change (as decimal)
   - Examples: 0.10 for 10%, 0.25 for 25%, -0.15 for -15%
   - Set to null if not explicitly stated

3. **time_horizon_days**: Timeframe for the prediction
   - Default to 30 days if not specified
   - Common: 30 (1 month), 90 (1 quarter), 365 (1 year)

4. **confidence**: Prediction confidence level (0.0 to 1.0)
   - 0.9-1.0: Very confident ("definitely", "certainly")
   - 0.7-0.9: Confident ("likely", "should")
   - 0.5-0.7: Moderate ("could", "may")
   - 0.3-0.5: Low confidence ("uncertain", "possible")
   - Default to 0.5 if unclear

Extract as Prediction object."""


# ============================================================================
# LLM CHAINS
# ============================================================================

class NarrativeChains:
    """Collection of LLM chains for narrative extraction"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        """
        Initialize chains with specified model
        
        Args:
            model: OpenAI model name (default: gpt-4o)
            temperature: Sampling temperature (default: 0 for deterministic)
        """
        self.model = model
        self.temperature = temperature
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    
    def extract_metadata(self, source_text: str) -> MetadataOutput:
        """
        Extract ticker symbol and investment position from text
        
        Args:
            source_text: The financial analysis text
            
        Returns:
            MetadataOutput with ticker and position
        """
        # Truncate text for metadata extraction (first 1500 chars usually enough)
        text_preview = source_text[:1500]
        
        prompt = EXTRACT_METADATA_PROMPT.format(source_text=text_preview)
        
        chain = self.llm.with_structured_output(MetadataOutput)
        result = chain.invoke(prompt)
        
        return result
    
    def extract_conclusions(self, source_text: str) -> List[Node]:
        """
        Extract all conclusion statements from text
        
        Args:
            source_text: The financial analysis text
            
        Returns:
            List of Node objects with node_type=conclusion
        """
        prompt = EXTRACT_CONCLUSIONS_PROMPT.format(source_text=source_text)
        
        chain = self.llm.with_structured_output(ConclusionList)
        result = chain.invoke(prompt)
        
        # Ensure all nodes are marked as conclusions
        for node in result.conclusions:
            node.node_type = NodeType.CONCLUSION
        
        return result.conclusions
    
    def extract_nodes(self, source_text: str, conclusion: Node) -> List[Node]:
        """
        Extract nodes (facts, events, opinions, assumptions) related to a conclusion
        
        Args:
            source_text: The financial analysis text
            conclusion: The conclusion node to find supporting nodes for
            
        Returns:
            List of Node objects (excluding the conclusion)
        """
        prompt = EXTRACT_NODES_PROMPT.format(
            source_text=source_text,
            conclusion_content=conclusion.content
        )
        
        # Use mini model for node extraction (cheaper, still effective)
        chain = self.llm_mini.with_structured_output(NodeList)
        result = chain.invoke(prompt)
        
        return result.nodes
    
    def extract_edges(
        self, 
        nodes: List[Node], 
        conclusion: Node,
        source_text: str = None
    ) -> List[Edge]:
        """
        Extract causal/logical relationships between nodes
        
        Args:
            nodes: All nodes including the conclusion
            conclusion: The target conclusion node
            source_text: Optional source text for context
            
        Returns:
            List of Edge objects representing relationships
        """
        # Format nodes for prompt
        nodes_description = "\n".join([
            f"- {node.id}: \"{node.content}\" (type: {node.node_type.value})"
            for node in nodes
        ])
        
        prompt = EXTRACT_EDGES_PROMPT.format(
            nodes_description=nodes_description,
            conclusion_content=conclusion.content,
            conclusion_id=conclusion.id
        )
        
        chain = self.llm.with_structured_output(EdgeList)
        result = chain.invoke(prompt)
        
        return result.edges
    
    def extract_prediction(
        self, 
        source_text: str,
        conclusion: Node,
        position: Position
    ) -> Prediction:
        """
        Extract price prediction from text
        
        Args:
            source_text: The financial analysis text
            conclusion: The conclusion node
            position: The author's position (long/short/neutral)
            
        Returns:
            Prediction object
        """
        # Truncate text to last 1000 chars (predictions often at end)
        text_preview = source_text[-1000:]
        
        prompt = EXTRACT_PREDICTION_PROMPT.format(
            source_text=text_preview,
            conclusion_content=conclusion.content,
            position=position.value
        )
        
        chain = self.llm_mini.with_structured_output(Prediction)
        result = chain.invoke(prompt)
        
        return result


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

# Create default chains instance
chains = NarrativeChains()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_chains(model: str = "gpt-4o", temperature: float = 0) -> NarrativeChains:
    """
    Get a NarrativeChains instance with custom configuration
    
    Args:
        model: OpenAI model name
        temperature: Sampling temperature
        
    Returns:
        Configured NarrativeChains instance
    """
    return NarrativeChains(model=model, temperature=temperature)

