"""
Prerequisite Chain Visualization

Creates clear visualizations of topic-to-topic prerequisite chains
showing explicit learning paths and dependencies.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_tracing_system.data.schemas import KnowledgeGraph, KnowledgeConcept
from knowledge_tracing_system.data.prerequisite_builder import PrerequisiteChainBuilder


class PrerequisiteChainVisualizer:
    """
    Visualizes prerequisite chains with clear topic-to-topic dependencies.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.builder = PrerequisiteChainBuilder(knowledge_graph)

        # Build prerequisite graph (only prerequisite edges)
        self.prereq_graph = nx.DiGraph()
        for cid in self.kg.concepts:
            self.prereq_graph.add_node(cid)

        for edge in self.kg.edges:
            if edge.edge_type == 'prerequisite':
                self.prereq_graph.add_edge(
                    edge.source_concept_id,
                    edge.target_concept_id,
                    weight=edge.weight
                )

        # Colors
        self.mastery_cmap = LinearSegmentedColormap.from_list(
            'mastery', ['#ff6b6b', '#ffd93d', '#6bcb77']
        )

        self.chapter_colors = self._assign_chapter_colors()

    def _assign_chapter_colors(self) -> Dict[str, str]:
        """Assign colors to chapters."""
        chapters = set(c.chapter for c in self.kg.concepts.values())
        colors = [
            '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
            '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b',
            '#795548', '#673ab7', '#4caf50', '#ff9800', '#03a9f4'
        ]
        return {ch: colors[i % len(colors)] for i, ch in enumerate(sorted(chapters))}

    def visualize_concept_prerequisites(
        self,
        concept_id: str,
        student_mastery: Optional[Dict[str, float]] = None,
        max_depth: int = 4,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize all prerequisites for a specific concept as a tree.
        """
        fig, ax = plt.subplots(figsize=figsize)

        concept = self.kg.concepts.get(concept_id)
        if not concept:
            ax.text(0.5, 0.5, f"Concept {concept_id} not found", ha='center')
            return fig

        # Build prerequisite tree
        nodes_to_draw = set()
        edges_to_draw = []

        def collect_prereqs(cid: str, depth: int):
            if depth > max_depth or cid in nodes_to_draw:
                return
            nodes_to_draw.add(cid)

            prereqs = self.kg.get_prerequisites(cid)
            for prereq in prereqs:
                edges_to_draw.append((prereq, cid))
                collect_prereqs(prereq, depth + 1)

        collect_prereqs(concept_id, 0)

        if not nodes_to_draw:
            nodes_to_draw.add(concept_id)

        # Create subgraph
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(nodes_to_draw)
        subgraph.add_edges_from(edges_to_draw)

        # Hierarchical layout
        pos = self._hierarchical_layout(subgraph, concept_id)

        # Draw edges
        for u, v in subgraph.edges():
            ax.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#e74c3c',
                    lw=2,
                    connectionstyle='arc3,rad=0.1'
                )
            )

        # Draw nodes
        for node in subgraph.nodes():
            x, y = pos[node]
            node_concept = self.kg.concepts.get(node)

            # Color by mastery or chapter
            if student_mastery and node in student_mastery:
                color = self.mastery_cmap(student_mastery[node])
                mastery_text = f"\n{student_mastery[node]:.0%}"
            else:
                color = self.chapter_colors.get(node_concept.chapter, '#cccccc') if node_concept else '#cccccc'
                mastery_text = ""

            # Highlight target concept
            edgecolor = '#000000' if node == concept_id else '#666666'
            linewidth = 3 if node == concept_id else 1

            # Draw box
            name = node_concept.name if node_concept else node
            if len(name) > 25:
                name = name[:22] + '...'

            bbox = dict(
                boxstyle='round,pad=0.3',
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.9
            )

            chapter = node_concept.chapter[:12] if node_concept else ''
            text = f"{name}\n({chapter}){mastery_text}"

            ax.text(x, y, text, ha='center', va='center',
                   fontsize=8, fontweight='bold', bbox=bbox)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2,
                      marker='>', markersize=8, label='Prerequisite'),
        ]

        if student_mastery:
            legend_elements.extend([
                mpatches.Patch(color='#ff6b6b', label='Low Mastery (<33%)'),
                mpatches.Patch(color='#ffd93d', label='Medium Mastery (33-66%)'),
                mpatches.Patch(color='#6bcb77', label='High Mastery (>66%)'),
            ])

        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        ax.set_title(f'Prerequisites for: {concept.name}\n({len(nodes_to_draw)} concepts, {len(edges_to_draw)} dependencies)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')

        # Set limits with padding
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
            ax.set_ylim(min(ys) - 0.3, max(ys) + 0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def _hierarchical_layout(self, G: nx.DiGraph, root: str) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout with root at bottom."""
        # Calculate depth for each node (distance from root)
        depths = {root: 0}
        queue = [root]

        while queue:
            node = queue.pop(0)
            for pred in G.predecessors(node):
                if pred not in depths:
                    depths[pred] = depths[node] + 1
                    queue.append(pred)

        # Assign positions
        max_depth = max(depths.values()) if depths else 0

        # Group by depth
        depth_nodes = defaultdict(list)
        for node, depth in depths.items():
            depth_nodes[depth].append(node)

        pos = {}
        for depth, nodes in depth_nodes.items():
            y = depth  # Root at bottom (y=0)
            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1) * 2 - 1  # Center around 0
                pos[node] = (x, y)

        return pos

    def visualize_learning_path(
        self,
        target_concepts: List[str],
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (20, 14),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the optimal learning path to reach target concepts.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={'width_ratios': [2, 1]})

        # Get learning sequence
        sequence = self.builder.get_learning_sequence(target_concepts)

        if not sequence:
            ax1.text(0.5, 0.5, "No learning path found", ha='center')
            return fig

        # Build path graph
        path_graph = nx.DiGraph()
        for cid in sequence:
            path_graph.add_node(cid)

        # Add edges from prerequisites
        for cid in sequence:
            for prereq in self.kg.get_prerequisites(cid):
                if prereq in sequence:
                    path_graph.add_edge(prereq, cid)

        # Layered layout based on sequence order
        pos = {}
        layer_width = 4
        nodes_per_layer = max(1, len(sequence) // 6)

        for i, cid in enumerate(sequence):
            layer = i // nodes_per_layer
            pos_in_layer = i % nodes_per_layer
            x = (pos_in_layer + 1) / (nodes_per_layer + 1) * layer_width
            y = -layer * 1.5
            pos[cid] = (x, y)

        # Draw edges
        for u, v in path_graph.edges():
            ax1.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle='-|>',
                    color='#3498db',
                    lw=1.5,
                    connectionstyle='arc3,rad=0.1',
                    alpha=0.7
                )
            )

        # Draw nodes
        for i, cid in enumerate(sequence):
            x, y = pos[cid]
            concept = self.kg.concepts.get(cid)

            if student_mastery and cid in student_mastery:
                color = self.mastery_cmap(student_mastery[cid])
            else:
                color = '#cccccc'

            # Highlight targets
            if cid in target_concepts:
                edgecolor = '#e74c3c'
                linewidth = 3
            else:
                edgecolor = '#333333'
                linewidth = 1

            name = concept.name[:20] if concept else cid[:20]
            chapter = concept.chapter[:10] if concept else ''

            bbox = dict(
                boxstyle='round,pad=0.2',
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.9
            )

            ax1.text(x, y, f"{i+1}. {name}\n({chapter})",
                    ha='center', va='center', fontsize=7,
                    fontweight='bold', bbox=bbox)

        ax1.set_title(f'Learning Path ({len(sequence)} topics)',
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Right panel: Chapter breakdown
        chapter_order = defaultdict(list)
        for i, cid in enumerate(sequence):
            concept = self.kg.concepts.get(cid)
            if concept:
                chapter_order[concept.chapter].append((i+1, cid))

        # Draw chapter summary
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, len(chapter_order) + 1)

        y = len(chapter_order)
        for chapter, items in sorted(chapter_order.items(), key=lambda x: min(i for i, _ in x[1])):
            color = self.chapter_colors.get(chapter, '#cccccc')

            # Chapter header
            ax2.add_patch(FancyBboxPatch(
                (0.05, y - 0.4), 0.9, 0.8,
                boxstyle="round,pad=0.02",
                facecolor=color, alpha=0.3,
                edgecolor=color
            ))

            ax2.text(0.1, y, f"{chapter}", fontsize=9, fontweight='bold', va='center')

            # Topic count
            ax2.text(0.9, y, f"{len(items)} topics", fontsize=8, va='center', ha='right')

            y -= 1

        ax2.set_title('Chapters in Path', fontsize=11, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def visualize_full_prerequisite_graph(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (24, 18),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize the complete prerequisite graph with all chains.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Only include nodes with prerequisites or that are prerequisites
        active_nodes = set()
        for edge in self.kg.edges:
            if edge.edge_type == 'prerequisite':
                active_nodes.add(edge.source_concept_id)
                active_nodes.add(edge.target_concept_id)

        if not active_nodes:
            ax.text(0.5, 0.5, "No prerequisite relationships found", ha='center')
            return fig

        # Create subgraph
        subgraph = self.prereq_graph.subgraph(active_nodes).copy()

        # Use graphviz-style hierarchical layout
        try:
            pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
        except:
            # Fallback to spring layout with hierarchy hints
            pos = self._create_hierarchical_spring_layout(subgraph)

        # Normalize positions
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        x_range = max(xs) - min(xs) if max(xs) != min(xs) else 1
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 1

        pos = {n: ((p[0] - min(xs)) / x_range, (p[1] - min(ys)) / y_range)
               for n, p in pos.items()}

        # Group edges by weight for styling
        strong_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                       if d.get('weight', 1.0) >= 0.8]
        weak_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                     if d.get('weight', 1.0) < 0.8]

        # Draw weak edges first
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=weak_edges,
            edge_color='#bdc3c7', width=0.5, alpha=0.5,
            arrows=True, arrowsize=8,
            connectionstyle='arc3,rad=0.1', ax=ax
        )

        # Draw strong edges
        nx.draw_networkx_edges(
            subgraph, pos, edgelist=strong_edges,
            edge_color='#e74c3c', width=1.5, alpha=0.8,
            arrows=True, arrowsize=12,
            connectionstyle='arc3,rad=0.1', ax=ax
        )

        # Node colors
        node_colors = []
        node_sizes = []

        for node in subgraph.nodes():
            concept = self.kg.concepts.get(node)

            if student_mastery and node in student_mastery:
                node_colors.append(self.mastery_cmap(student_mastery[node]))
            elif concept:
                node_colors.append(self.chapter_colors.get(concept.chapter, '#cccccc'))
            else:
                node_colors.append('#cccccc')

            # Size by out-degree (how many concepts depend on this)
            out_deg = subgraph.out_degree(node)
            node_sizes.append(200 + out_deg * 50)

        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos, node_color=node_colors,
            node_size=node_sizes, alpha=0.9, ax=ax
        )

        # Labels for important nodes (high out-degree)
        important_nodes = [n for n in subgraph.nodes() if subgraph.out_degree(n) >= 2]
        labels = {}
        for node in important_nodes:
            concept = self.kg.concepts.get(node)
            labels[node] = concept.name[:15] if concept else node[:15]

        nx.draw_networkx_labels(
            subgraph, pos, labels, font_size=7,
            font_weight='bold', ax=ax
        )

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Strong Prerequisite'),
            plt.Line2D([0], [0], color='#bdc3c7', linewidth=1, label='Weak Prerequisite'),
        ]

        # Add chapter colors to legend
        for chapter, color in list(self.chapter_colors.items())[:6]:
            legend_elements.append(
                mpatches.Patch(color=color, label=chapter[:15], alpha=0.8)
            )

        ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=2)

        ax.set_title(f'Complete Prerequisite Graph\n({len(active_nodes)} concepts, {subgraph.number_of_edges()} prerequisites)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def _create_hierarchical_spring_layout(self, G: nx.DiGraph) -> Dict:
        """Create spring layout with hierarchical hints."""
        # Calculate topological levels
        levels = {}

        # Find roots (no incoming edges)
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]

        # BFS to assign levels
        for root in roots:
            queue = [(root, 0)]
            while queue:
                node, level = queue.pop(0)
                if node not in levels or levels[node] < level:
                    levels[node] = level
                for succ in G.successors(node):
                    queue.append((succ, level + 1))

        # Assign level 0 to unvisited
        for node in G.nodes():
            if node not in levels:
                levels[node] = 0

        # Create initial positions based on levels
        max_level = max(levels.values()) if levels else 0
        level_nodes = defaultdict(list)
        for node, level in levels.items():
            level_nodes[level].append(node)

        pos = {}
        for level, nodes in level_nodes.items():
            y = level / (max_level + 1) if max_level > 0 else 0.5
            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1)
                pos[node] = (x, y)

        # Refine with spring layout
        pos = nx.spring_layout(G, pos=pos, k=0.5, iterations=50)

        return pos

    def visualize_chapter_chains(
        self,
        chapter: str,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (18, 14),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize prerequisite chains within and into a specific chapter.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get chapter concepts
        chapter_concepts = [
            cid for cid, c in self.kg.concepts.items()
            if c.chapter == chapter
        ]

        if not chapter_concepts:
            ax.text(0.5, 0.5, f"No concepts in chapter: {chapter}", ha='center')
            return fig

        # Collect all related concepts (including external prerequisites)
        all_concepts = set(chapter_concepts)
        external_prereqs = set()

        for cid in chapter_concepts:
            for prereq in self.kg.get_prerequisites(cid):
                if prereq not in chapter_concepts:
                    external_prereqs.add(prereq)
                    all_concepts.add(prereq)

        # Build subgraph
        subgraph = nx.DiGraph()
        for cid in all_concepts:
            subgraph.add_node(cid)

        for cid in all_concepts:
            for prereq in self.kg.get_prerequisites(cid):
                if prereq in all_concepts:
                    subgraph.add_edge(prereq, cid)

        # Layout: external prereqs on left, chapter concepts on right
        pos = {}

        # Sort by difficulty
        chapter_by_diff = sorted(chapter_concepts,
                                key=lambda x: self.kg.concepts[x].difficulty_base.value
                                if x in self.kg.concepts else 3)

        external_by_diff = sorted(external_prereqs,
                                 key=lambda x: self.kg.concepts[x].difficulty_base.value
                                 if x in self.kg.concepts else 3)

        # Position external prereqs
        for i, cid in enumerate(external_by_diff):
            y = (i + 1) / (len(external_by_diff) + 1)
            pos[cid] = (0.1, y)

        # Position chapter concepts by difficulty level
        diff_groups = defaultdict(list)
        for cid in chapter_by_diff:
            concept = self.kg.concepts.get(cid)
            if concept:
                diff_groups[concept.difficulty_base.value].append(cid)

        x_offset = 0.4
        for diff_level in sorted(diff_groups.keys()):
            nodes = diff_groups[diff_level]
            for i, cid in enumerate(nodes):
                y = (i + 1) / (len(nodes) + 1)
                pos[cid] = (x_offset, y)
            x_offset += 0.2

        # Draw external to chapter edges
        for u, v in subgraph.edges():
            if u in external_prereqs:
                color = '#9b59b6'  # Purple for cross-chapter
                style = 'dashed'
            else:
                color = '#e74c3c'  # Red for intra-chapter
                style = 'solid'

            ax.annotate(
                '', xy=pos[v], xytext=pos[u],
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=color,
                    lw=1.5,
                    linestyle=style,
                    connectionstyle='arc3,rad=0.1'
                )
            )

        # Draw nodes
        for cid in all_concepts:
            x, y = pos[cid]
            concept = self.kg.concepts.get(cid)

            if cid in external_prereqs:
                color = self.chapter_colors.get(concept.chapter, '#cccccc') if concept else '#cccccc'
                alpha = 0.7
            else:
                if student_mastery and cid in student_mastery:
                    color = self.mastery_cmap(student_mastery[cid])
                else:
                    color = self.chapter_colors.get(chapter, '#3498db')
                alpha = 0.9

            name = concept.name[:20] if concept else cid[:20]
            ch = concept.chapter[:10] if concept else ''

            bbox = dict(
                boxstyle='round,pad=0.2',
                facecolor=color,
                edgecolor='#333333',
                alpha=alpha
            )

            ax.text(x, y, f"{name}\n({ch})", ha='center', va='center',
                   fontsize=8, fontweight='bold', bbox=bbox)

        # Add difficulty level labels
        for diff_level in sorted(diff_groups.keys()):
            x = 0.4 + (diff_level - min(diff_groups.keys())) * 0.2
            ax.text(x, 1.05, f"Difficulty {diff_level}",
                   ha='center', fontsize=9, fontweight='bold')

        if external_prereqs:
            ax.text(0.1, 1.05, "External\nPrerequisites",
                   ha='center', fontsize=9, fontweight='bold')

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Intra-chapter Prerequisite'),
            plt.Line2D([0], [0], color='#9b59b6', linewidth=2, linestyle='--', label='Cross-chapter Prerequisite'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        ax.set_title(f'Prerequisite Chains: {chapter}\n({len(chapter_concepts)} topics, {len(external_prereqs)} external prerequisites)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig


def run_prerequisite_visualization(sample_size: int = 10000):
    """Run prerequisite chain visualization."""
    from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader
    from knowledge_tracing_system.data.prerequisite_builder import build_enhanced_prerequisites
    from knowledge_tracing_system.run_simulation import SimulationRunner

    print("Loading data...")
    loader = ASSISTmentsDataLoader()
    loader.load_csv_files(sample_size=sample_size)
    loader.build_knowledge_concepts()
    loader.build_questions()
    loader.build_knowledge_graph()

    print("\nBuilding enhanced prerequisites...")
    enhanced_kg = build_enhanced_prerequisites(loader.knowledge_graph)

    # Get student mastery
    print("Loading student data...")
    runner = SimulationRunner(sample_size=sample_size)

    # Find student with good coverage
    student_concepts = defaultdict(set)
    for attempt in runner.data['question_attempts']:
        q = runner.data['questions'].get(attempt.question_id)
        if q:
            for cid in q.concept_ids:
                student_concepts[attempt.student_id].add(cid)

    best_student = max(student_concepts.items(), key=lambda x: len(x[1]))[0]
    print(f"Selected student: {best_student} ({len(student_concepts[best_student])} concepts)")

    knowledge_state = runner.simulate_knowledge_state(best_student)

    # Create visualizer with enhanced graph
    viz = PrerequisiteChainVisualizer(enhanced_kg)

    # Output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Full prerequisite graph
    print("\n1. Creating full prerequisite graph...")
    viz.visualize_full_prerequisite_graph(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "full_prerequisite_graph.png")
    )

    # 2. Find complex concept and show its prerequisites
    print("2. Creating concept prerequisite tree...")
    complex_concepts = [
        cid for cid, c in enhanced_kg.concepts.items()
        if c.difficulty_base.value >= 4 and len(enhanced_kg.get_prerequisites(cid)) > 0
    ]

    if complex_concepts:
        target = complex_concepts[0]
        viz.visualize_concept_prerequisites(
            target,
            student_mastery=knowledge_state.concept_mastery,
            save_path=str(output_dir / "concept_prerequisites.png")
        )

    # 3. Learning path to weak concepts
    print("3. Creating learning path...")
    weak_concepts = [cid for cid, m in knowledge_state.concept_mastery.items() if m < 0.5][:5]
    if weak_concepts:
        viz.visualize_learning_path(
            weak_concepts,
            student_mastery=knowledge_state.concept_mastery,
            save_path=str(output_dir / "learning_path_detailed.png")
        )

    # 4. Chapter chains
    print("4. Creating chapter chain visualizations...")
    chapters = list(set(c.chapter for c in enhanced_kg.concepts.values()))
    for i, chapter in enumerate(sorted(chapters)[:4]):
        viz.visualize_chapter_chains(
            chapter,
            student_mastery=knowledge_state.concept_mastery,
            save_path=str(output_dir / f"chapter_chain_{i+1}_{chapter[:15].replace(' ', '_')}.png")
        )

    print(f"\nAll visualizations saved to: {output_dir}")

    return viz, enhanced_kg, knowledge_state


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    run_prerequisite_visualization(sample_size=10000)
