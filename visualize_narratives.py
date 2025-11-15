"""
Narrative Graph Visualization Tool

Creates interactive and static visualizations of extracted narrative graphs.

Usage:
    python visualize_narratives.py output/batch_narratives_test.json
    python visualize_narratives.py output/batch_narratives_test.json --format html
    python visualize_narratives.py output/batch_narratives_test.json --format png
    python visualize_narratives.py output/batch_narratives_test.json --narrative 0
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import networkx as nx
from datetime import datetime


def load_narratives(json_path: str) -> List[Dict]:
    """Load narratives from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['narratives']


def build_networkx_graph(narrative: Dict) -> nx.DiGraph:
    """Convert narrative to NetworkX directed graph."""
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node in narrative['narrative_graph']['nodes']:
        G.add_node(
            node['id'],
            label=node['content'][:60] + "..." if len(node['content']) > 60 else node['content'],
            full_content=node['content'],
            node_type=node['node_type'],
            asset=node.get('asset'),
            position=node.get('position')
        )
    
    # Add edges with attributes
    for edge in narrative['narrative_graph']['edges']:
        G.add_edge(
            edge['source_id'],
            edge['target_id'],
            relationship=edge['relationship_type'],
            description=edge['description']
        )
    
    return G


def visualize_with_pyvis(narrative: Dict, output_path: str):
    """
    Create interactive HTML visualization using pyvis.
    
    Requires: pip install pyvis
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("❌ pyvis not installed. Run: pip install pyvis")
        return
    
    # Create network
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#000000",
        notebook=False
    )
    
    # Configure physics
    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "stabilization": {
          "enabled": true,
          "iterations": 1000
        }
      },
      "nodes": {
        "font": {"size": 14}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "continuous"}
      }
    }
    """)
    
    # Color mapping
    color_map = {
        'fact': '#4A90E2',        # Blue
        'opinion': '#7ED321',     # Green
        'event': '#F5A623',       # Orange
        'assumption': '#BD10E0',  # Purple
        'conclusion': '#D0021B'   # Red
    }
    
    # Add nodes
    nodes = narrative['narrative_graph']['nodes']
    for node in nodes:
        node_type = node['node_type']
        color = color_map.get(node_type, '#999999')
        
        # Size based on type
        size = 30 if node_type == 'conclusion' else 20
        
        # Create hover info
        title = f"<b>{node_type.upper()}</b><br>{node['content']}"
        
        net.add_node(
            node['id'],
            label=node['id'],
            title=title,
            color=color,
            size=size,
            shape='dot'
        )
    
    # Add edges
    edges = narrative['narrative_graph']['edges']
    edge_color_map = {
        'supporting': '#7ED321',      # Green
        'contradicting': '#D0021B',   # Red
        'conditional': '#F5A623',     # Orange
        'causal': '#4A90E2'           # Blue
    }
    
    for edge in edges:
        rel_type = edge['relationship_type']
        color = edge_color_map.get(rel_type, '#999999')
        
        # Edge width based on type
        width = 3 if rel_type == 'causal' else 1.5 if rel_type == 'supporting' else 1
        
        net.add_edge(
            edge['source_id'],
            edge['target_id'],
            title=f"{rel_type}: {edge['description']}",
            color=color,
            width=width
        )
    
    # Add title
    ticker = narrative.get('ticker', 'Unknown')
    position = narrative.get('position', 'unknown')
    conv_score = narrative['narrative_graph']['convergence_score']
    
    net.heading = f"{ticker} - {position.upper()} (Convergence: {conv_score:.3f})"
    
    # Save
    net.save_graph(output_path)
    print(f"✅ Interactive HTML saved to: {output_path}")
    print(f"   Open in browser to explore!")


def visualize_with_matplotlib(narrative: Dict, output_path: str):
    """
    Create static PNG visualization using matplotlib.
    
    Requires: pip install matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("❌ matplotlib not installed. Run: pip install matplotlib")
        return
    
    G = build_networkx_graph(narrative)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color mapping
    color_map = {
        'fact': '#4A90E2',
        'opinion': '#7ED321',
        'event': '#F5A623',
        'assumption': '#BD10E0',
        'conclusion': '#D0021B'
    }
    
    # Separate nodes by type
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        node_type = G.nodes[node]['node_type']
        node_colors.append(color_map.get(node_type, '#999999'))
        node_sizes.append(3000 if node_type == 'conclusion' else 2000)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax
    )
    
    # Draw edges by type
    edge_color_map = {
        'supporting': '#7ED321',
        'contradicting': '#D0021B',
        'conditional': '#F5A623',
        'causal': '#4A90E2'
    }
    
    for edge_type, color in edge_color_map.items():
        edges_of_type = [(u, v) for u, v, d in G.edges(data=True) 
                         if d['relationship'] == edge_type]
        if edges_of_type:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges_of_type,
                edge_color=color,
                arrows=True,
                arrowsize=20,
                width=2,
                alpha=0.6,
                ax=ax
            )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels={n: n for n in G.nodes()},
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    # Title
    ticker = narrative.get('ticker', 'Unknown')
    position = narrative.get('position', 'unknown')
    conv_score = narrative['narrative_graph']['convergence_score']
    nodes_count = len(narrative['narrative_graph']['nodes'])
    edges_count = len(narrative['narrative_graph']['edges'])
    
    plt.title(
        f"{ticker} - {position.upper()}\n"
        f"Convergence: {conv_score:.3f} | Nodes: {nodes_count} | Edges: {edges_count}",
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    
    # Legend
    node_legend_elements = [
        mpatches.Patch(color=color_map['fact'], label='Facts'),
        mpatches.Patch(color=color_map['opinion'], label='Opinions'),
        mpatches.Patch(color=color_map['event'], label='Events'),
        mpatches.Patch(color=color_map['assumption'], label='Assumptions'),
        mpatches.Patch(color=color_map['conclusion'], label='Conclusion')
    ]
    
    edge_legend_elements = [
        mpatches.Patch(color=edge_color_map['supporting'], label='Supporting'),
        mpatches.Patch(color=edge_color_map['contradicting'], label='Contradicting'),
        mpatches.Patch(color=edge_color_map['conditional'], label='Conditional'),
        mpatches.Patch(color=edge_color_map['causal'], label='Causal')
    ]
    
    legend1 = ax.legend(
        handles=node_legend_elements,
        loc='upper left',
        title='Node Types',
        framealpha=0.9
    )
    ax.add_artist(legend1)
    
    ax.legend(
        handles=edge_legend_elements,
        loc='upper right',
        title='Edge Types',
        framealpha=0.9
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Static PNG saved to: {output_path}")
    plt.close()


def generate_summary_report(narratives: List[Dict], output_path: str):
    """Generate a text summary report."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NARRATIVE GRAPH ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, narrative in enumerate(narratives, 1):
            ticker = narrative.get('ticker', 'Unknown')
            position = narrative.get('position', 'unknown')
            graph = narrative['narrative_graph']
            
            f.write(f"\n{'=' * 80}\n")
            f.write(f"NARRATIVE {idx}: {ticker} - {position.upper()}\n")
            f.write(f"{'=' * 80}\n\n")
            
            # Basic stats
            f.write(f"Convergence Score: {graph['convergence_score']:.3f}\n")
            f.write(f"Total Nodes: {len(graph['nodes'])}\n")
            f.write(f"Total Edges: {len(graph['edges'])}\n")
            f.write(f"Avg Path Length: {graph['avg_path_length']:.2f}\n")
            f.write(f"Max Depth: {graph['max_convergence_depth']}\n\n")
            
            # Node breakdown
            node_types = {}
            for node in graph['nodes']:
                node_type = node['node_type']
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            f.write("Node Breakdown:\n")
            for node_type, count in sorted(node_types.items()):
                f.write(f"  - {node_type.capitalize()}: {count}\n")
            f.write("\n")
            
            # Edge breakdown
            edge_types = {}
            for edge in graph['edges']:
                edge_type = edge['relationship_type']
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            f.write("Edge Breakdown:\n")
            for edge_type, count in sorted(edge_types.items()):
                f.write(f"  - {edge_type.capitalize()}: {count}\n")
            f.write("\n")
            
            # Conclusion
            conclusion_node = next(
                (n for n in graph['nodes'] if n['node_type'] == 'conclusion'),
                None
            )
            if conclusion_node:
                f.write(f"Main Conclusion:\n")
                f.write(f"  \"{conclusion_node['content']}\"\n\n")
            
            # Prediction
            if narrative.get('prediction'):
                pred = narrative['prediction']
                f.write(f"Prediction:\n")
                f.write(f"  Direction: {pred['direction']}\n")
                if pred.get('magnitude'):
                    f.write(f"  Magnitude: {pred['magnitude']}\n")
                f.write(f"  Time Horizon: {pred['time_horizon_days']} days\n")
                f.write(f"  Confidence: {pred['confidence']:.2f}\n\n")
            
            # Top supporting evidence
            supporting_edges = [e for e in graph['edges'] 
                              if e['relationship_type'] == 'supporting' 
                              and e['target_id'] == graph['conclusion_id']]
            
            if supporting_edges:
                f.write(f"Top Supporting Evidence (showing 5):\n")
                for i, edge in enumerate(supporting_edges[:5], 1):
                    source_node = next(n for n in graph['nodes'] if n['id'] == edge['source_id'])
                    f.write(f"  {i}. [{source_node['node_type']}] {source_node['content'][:80]}...\n")
                f.write("\n")
            
            # Contradicting evidence
            contra_edges = [e for e in graph['edges'] 
                          if e['relationship_type'] == 'contradicting'
                          and e['target_id'] == graph['conclusion_id']]
            
            if contra_edges:
                f.write(f"Contradicting Evidence:\n")
                for i, edge in enumerate(contra_edges, 1):
                    source_node = next(n for n in graph['nodes'] if n['id'] == edge['source_id'])
                    f.write(f"  {i}. [{source_node['node_type']}] {source_node['content'][:80]}...\n")
                f.write("\n")
    
    print(f"✅ Summary report saved to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_narratives.py <json_file> [--format html|png|all] [--narrative N]")
        print("\nExamples:")
        print("  python visualize_narratives.py output/batch_narratives_test.json")
        print("  python visualize_narratives.py output/batch_narratives_test.json --format html")
        print("  python visualize_narratives.py output/batch_narratives_test.json --narrative 0")
        return
    
    json_path = sys.argv[1]
    
    # Parse arguments
    output_format = 'all'
    narrative_idx = None
    
    if '--format' in sys.argv:
        idx = sys.argv.index('--format')
        if idx + 1 < len(sys.argv):
            output_format = sys.argv[idx + 1]
    
    if '--narrative' in sys.argv:
        idx = sys.argv.index('--narrative')
        if idx + 1 < len(sys.argv):
            narrative_idx = int(sys.argv[idx + 1])
    
    # Load narratives
    print(f"\n📂 Loading narratives from: {json_path}")
    narratives = load_narratives(json_path)
    print(f"✅ Loaded {len(narratives)} narrative(s)")
    
    # Filter to specific narrative if requested
    if narrative_idx is not None:
        if narrative_idx < 0 or narrative_idx >= len(narratives):
            print(f"❌ Invalid narrative index. Must be 0-{len(narratives)-1}")
            return
        narratives = [narratives[narrative_idx]]
        print(f"📍 Processing narrative {narrative_idx} only")
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")
    
    for idx, narrative in enumerate(narratives):
        ticker = narrative.get('ticker', f'narrative_{idx}')
        position = narrative.get('position', 'unknown')
        base_name = f"{ticker}_{position}"
        
        print(f"\n--- {ticker} ({position}) ---")
        
        # HTML (interactive)
        if output_format in ['html', 'all']:
            html_path = output_dir / f"{base_name}_interactive.html"
            visualize_with_pyvis(narrative, str(html_path))
        
        # PNG (static)
        if output_format in ['png', 'all']:
            png_path = output_dir / f"{base_name}_static.png"
            visualize_with_matplotlib(narrative, str(png_path))
    
    # Generate summary report
    if output_format == 'all':
        report_path = output_dir / "analysis_report.txt"
        generate_summary_report(narratives, str(report_path))
    
    print(f"\n{'=' * 70}")
    print("✅ VISUALIZATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print()


if __name__ == "__main__":
    main()

