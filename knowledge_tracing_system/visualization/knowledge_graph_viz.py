"""
Knowledge Graph Visualization with Student Mastery Overlay

Visualizes:
- Concept relationships (prerequisites, related topics)
- Student mastery levels per concept
- Learning path recommendations
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

# Try to import plotly for interactive visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_tracing_system.data.schemas import (
    KnowledgeGraph, KnowledgeConcept, KnowledgeState, StudentProfile
)


class KnowledgeGraphVisualizer:
    """
    Visualizes knowledge graphs with student mastery overlay.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize visualizer with knowledge graph.

        Args:
            knowledge_graph: The knowledge graph to visualize
        """
        self.kg = knowledge_graph
        self.G = self._build_networkx_graph()

        # Color schemes
        self.mastery_cmap = LinearSegmentedColormap.from_list(
            'mastery', ['#ff4444', '#ffaa00', '#44ff44']  # Red -> Yellow -> Green
        )

        self.edge_colors = {
            'prerequisite': '#e74c3c',  # Red - must learn first
            'related': '#3498db',        # Blue - related topics
            'similar_difficulty': '#9b59b6',  # Purple
            'same_topic': '#2ecc71'      # Green
        }

        # Chapter colors for grouping
        self.chapter_colors = {}

    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert KnowledgeGraph to NetworkX DiGraph."""
        G = nx.DiGraph()

        # Add nodes with attributes
        for concept_id, concept in self.kg.concepts.items():
            G.add_node(
                concept_id,
                name=concept.name,
                chapter=concept.chapter,
                topic=concept.topic,
                difficulty=concept.difficulty_base.value,
                bloom_level=concept.bloom_level
            )

        # Add edges with attributes
        for edge in self.kg.edges:
            G.add_edge(
                edge.source_concept_id,
                edge.target_concept_id,
                edge_type=edge.edge_type,
                weight=edge.weight
            )

        return G

    def _get_chapter_color(self, chapter: str) -> str:
        """Get consistent color for a chapter."""
        if chapter not in self.chapter_colors:
            # Generate distinct colors
            colors = [
                '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12',
                '#1abc9c', '#e91e63', '#00bcd4', '#ff5722', '#607d8b',
                '#795548', '#9c27b0', '#4caf50', '#ff9800', '#03a9f4'
            ]
            idx = len(self.chapter_colors) % len(colors)
            self.chapter_colors[chapter] = colors[idx]
        return self.chapter_colors[chapter]

    def visualize_full_graph(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
        show_labels: bool = True,
        layout: str = 'spring'
    ) -> plt.Figure:
        """
        Visualize the complete knowledge graph.

        Args:
            student_mastery: Dict of concept_id -> mastery (0-1)
            figsize: Figure size
            save_path: Path to save the figure
            show_labels: Whether to show concept names
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'hierarchical')

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'circular':
            pos = nx.circular_layout(self.G)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout()
        else:
            pos = nx.spring_layout(self.G, seed=42)

        # Prepare node colors based on mastery or chapter
        node_colors = []
        node_sizes = []

        for node in self.G.nodes():
            concept = self.kg.concepts.get(node)

            if student_mastery and node in student_mastery:
                # Color by mastery
                mastery = student_mastery[node]
                node_colors.append(self.mastery_cmap(mastery))
                node_sizes.append(500 + mastery * 1000)  # Larger = more mastered
            elif concept:
                # Color by chapter
                node_colors.append(self._get_chapter_color(concept.chapter))
                node_sizes.append(500 + concept.difficulty_base.value * 100)
            else:
                node_colors.append('#cccccc')
                node_sizes.append(500)

        # Draw edges by type
        for edge_type, color in self.edge_colors.items():
            edges = [(u, v) for u, v, d in self.G.edges(data=True)
                    if d.get('edge_type') == edge_type]
            if edges:
                style = 'solid' if edge_type == 'prerequisite' else 'dashed'
                width = 2.0 if edge_type == 'prerequisite' else 1.0
                nx.draw_networkx_edges(
                    self.G, pos, edgelist=edges,
                    edge_color=color, style=style, width=width,
                    alpha=0.7, arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1", ax=ax
                )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos, node_color=node_colors,
            node_size=node_sizes, alpha=0.9, ax=ax
        )

        # Draw labels
        if show_labels:
            labels = {}
            for node in self.G.nodes():
                concept = self.kg.concepts.get(node)
                if concept:
                    # Truncate long names
                    name = concept.name[:20] + '...' if len(concept.name) > 20 else concept.name
                    labels[node] = name
                else:
                    labels[node] = node

            nx.draw_networkx_labels(
                self.G, pos, labels, font_size=8,
                font_weight='bold', ax=ax
            )

        # Create legend
        legend_elements = []

        # Edge type legend
        for edge_type, color in self.edge_colors.items():
            style = '-' if edge_type == 'prerequisite' else '--'
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linestyle=style,
                          label=f'{edge_type.replace("_", " ").title()}')
            )

        # Mastery legend if applicable
        if student_mastery:
            legend_elements.append(mpatches.Patch(color='#ff4444', label='Low Mastery (0-33%)'))
            legend_elements.append(mpatches.Patch(color='#ffaa00', label='Medium Mastery (34-66%)'))
            legend_elements.append(mpatches.Patch(color='#44ff44', label='High Mastery (67-100%)'))

        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        # Title
        title = "Knowledge Graph"
        if student_mastery:
            avg_mastery = np.mean(list(student_mastery.values()))
            title += f" - Student Mastery (Avg: {avg_mastery:.1%})"
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved graph to {save_path}")

        return fig

    def _hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on prerequisites."""
        # Find root nodes (no prerequisites)
        roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]

        if not roots:
            # Fall back to spring layout
            return nx.spring_layout(self.G, seed=42)

        # BFS to assign levels
        levels = {}
        queue = [(r, 0) for r in roots]
        visited = set()

        while queue:
            node, level = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            levels[node] = max(levels.get(node, 0), level)

            for successor in self.G.successors(node):
                queue.append((successor, level + 1))

        # Assign unvisited nodes
        for node in self.G.nodes():
            if node not in levels:
                levels[node] = 0

        # Calculate positions
        max_level = max(levels.values()) if levels else 0
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)

        pos = {}
        for level, nodes in level_nodes.items():
            y = 1 - (level / (max_level + 1))
            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1)
                pos[node] = (x, y)

        return pos

    def visualize_student_progress(
        self,
        student_id: str,
        student_mastery: Dict[str, float],
        student_profile: Optional[StudentProfile] = None,
        figsize: Tuple[int, int] = (18, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive student progress visualization.

        Args:
            student_id: Student identifier
            student_mastery: Dict of concept_id -> mastery
            student_profile: Optional student profile for additional info
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)

        # Create grid layout
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Knowledge Graph (main)
        ax1 = fig.add_subplot(gs[0, :2])
        self._draw_graph_on_axis(ax1, student_mastery)
        ax1.set_title(f'Knowledge Graph - {student_id}', fontsize=12, fontweight='bold')

        # 2. Mastery by Chapter (bar chart)
        ax2 = fig.add_subplot(gs[0, 2])
        self._draw_chapter_mastery(ax2, student_mastery)

        # 3. Mastery Distribution (histogram)
        ax3 = fig.add_subplot(gs[1, 0])
        self._draw_mastery_distribution(ax3, student_mastery)

        # 4. Learning Path (recommended order)
        ax4 = fig.add_subplot(gs[1, 1])
        self._draw_learning_path(ax4, student_mastery)

        # 5. Statistics Summary
        ax5 = fig.add_subplot(gs[1, 2])
        self._draw_statistics(ax5, student_id, student_mastery, student_profile)

        plt.suptitle(f'Student Learning Progress Dashboard', fontsize=14, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved progress visualization to {save_path}")

        return fig

    def _draw_graph_on_axis(self, ax: plt.Axes, student_mastery: Dict[str, float]):
        """Draw knowledge graph on given axis."""
        pos = nx.spring_layout(self.G, k=1.5, iterations=50, seed=42)

        # Node colors by mastery
        node_colors = []
        for node in self.G.nodes():
            if node in student_mastery:
                node_colors.append(self.mastery_cmap(student_mastery[node]))
            else:
                node_colors.append('#cccccc')

        # Draw prerequisite edges
        prereq_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                       if d.get('edge_type') == 'prerequisite']
        nx.draw_networkx_edges(
            self.G, pos, edgelist=prereq_edges,
            edge_color='#e74c3c', arrows=True, arrowsize=10,
            alpha=0.6, ax=ax
        )

        # Draw related edges
        related_edges = [(u, v) for u, v, d in self.G.edges(data=True)
                        if d.get('edge_type') == 'related']
        nx.draw_networkx_edges(
            self.G, pos, edgelist=related_edges,
            edge_color='#3498db', style='dashed', arrows=False,
            alpha=0.3, ax=ax
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, pos, node_color=node_colors,
            node_size=300, alpha=0.9, ax=ax
        )

        ax.axis('off')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=self.mastery_cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='Mastery')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['0%', '50%', '100%'])

    def _draw_chapter_mastery(self, ax: plt.Axes, student_mastery: Dict[str, float]):
        """Draw mastery by chapter bar chart."""
        chapter_mastery = {}

        for concept_id, mastery in student_mastery.items():
            concept = self.kg.concepts.get(concept_id)
            if concept:
                chapter = concept.chapter
                if chapter not in chapter_mastery:
                    chapter_mastery[chapter] = []
                chapter_mastery[chapter].append(mastery)

        # Calculate averages
        chapters = []
        avg_mastery = []
        colors = []

        for chapter, masteries in sorted(chapter_mastery.items()):
            chapters.append(chapter[:15] + '...' if len(chapter) > 15 else chapter)
            avg = np.mean(masteries)
            avg_mastery.append(avg)
            colors.append(self.mastery_cmap(avg))

        # Horizontal bar chart
        y_pos = np.arange(len(chapters))
        ax.barh(y_pos, avg_mastery, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(chapters, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Average Mastery')
        ax.set_title('Mastery by Chapter', fontsize=10, fontweight='bold')

        # Add percentage labels
        for i, v in enumerate(avg_mastery):
            ax.text(v + 0.02, i, f'{v:.0%}', va='center', fontsize=8)

    def _draw_mastery_distribution(self, ax: plt.Axes, student_mastery: Dict[str, float]):
        """Draw mastery distribution histogram."""
        mastery_values = list(student_mastery.values())

        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        colors = [self.mastery_cmap(b + 0.1) for b in bins[:-1]]

        n, bins_out, patches = ax.hist(mastery_values, bins=bins, edgecolor='white', alpha=0.8)

        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Mastery Level')
        ax.set_ylabel('Number of Concepts')
        ax.set_title('Mastery Distribution', fontsize=10, fontweight='bold')

        # Add labels
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_xticklabels(labels, fontsize=8)

    def _draw_learning_path(self, ax: plt.Axes, student_mastery: Dict[str, float]):
        """Draw recommended learning path."""
        # Find concepts needing work (mastery < 0.6)
        needs_work = [(cid, m) for cid, m in student_mastery.items() if m < 0.6]
        needs_work.sort(key=lambda x: x[1])  # Sort by mastery

        # Get top 10 concepts to focus on
        focus_concepts = needs_work[:10]

        if not focus_concepts:
            ax.text(0.5, 0.5, 'All concepts mastered!',
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.axis('off')
            return

        # Create ordered list with prerequisites considered
        ordered = []
        for cid, mastery in focus_concepts:
            prereqs = self.kg.get_prerequisites(cid)
            # Add unmastered prerequisites first
            for prereq in prereqs:
                if prereq in student_mastery and student_mastery[prereq] < 0.6:
                    if prereq not in [x[0] for x in ordered]:
                        ordered.append((prereq, student_mastery.get(prereq, 0)))
            if cid not in [x[0] for x in ordered]:
                ordered.append((cid, mastery))

        # Display as list
        ax.axis('off')
        ax.set_title('Recommended Learning Path', fontsize=10, fontweight='bold')

        y_start = 0.95
        for i, (cid, mastery) in enumerate(ordered[:8]):
            concept = self.kg.concepts.get(cid)
            name = concept.name[:25] if concept else cid[:25]

            # Arrow indicator
            color = self.mastery_cmap(mastery)
            ax.add_patch(plt.Circle((0.05, y_start - i * 0.1), 0.02, color=color))
            ax.text(0.1, y_start - i * 0.1, f'{i+1}. {name}',
                   va='center', fontsize=9)
            ax.text(0.9, y_start - i * 0.1, f'{mastery:.0%}',
                   va='center', ha='right', fontsize=9, color=color)

    def _draw_statistics(
        self,
        ax: plt.Axes,
        student_id: str,
        student_mastery: Dict[str, float],
        student_profile: Optional[StudentProfile]
    ):
        """Draw statistics summary."""
        ax.axis('off')
        ax.set_title('Statistics Summary', fontsize=10, fontweight='bold')

        mastery_values = list(student_mastery.values())

        stats = [
            ('Student ID', student_id),
            ('', ''),
            ('Total Concepts', str(len(self.kg.concepts))),
            ('Practiced Concepts', str(len(student_mastery))),
            ('Coverage', f'{len(student_mastery)/len(self.kg.concepts):.0%}'),
            ('', ''),
            ('Average Mastery', f'{np.mean(mastery_values):.1%}'),
            ('Median Mastery', f'{np.median(mastery_values):.1%}'),
            ('Min Mastery', f'{min(mastery_values):.1%}'),
            ('Max Mastery', f'{max(mastery_values):.1%}'),
            ('', ''),
            ('Mastered (>80%)', str(sum(1 for m in mastery_values if m > 0.8))),
            ('Learning (40-80%)', str(sum(1 for m in mastery_values if 0.4 <= m <= 0.8))),
            ('Needs Work (<40%)', str(sum(1 for m in mastery_values if m < 0.4))),
        ]

        if student_profile:
            stats.extend([
                ('', ''),
                ('Learning Pace', student_profile.learning_pace),
                ('Overall Mastery', f'{student_profile.overall_mastery:.1%}'),
            ])

        y_start = 0.95
        for i, (label, value) in enumerate(stats):
            if label:
                ax.text(0.05, y_start - i * 0.055, f'{label}:', fontsize=9, fontweight='bold')
                ax.text(0.6, y_start - i * 0.055, value, fontsize=9)

    def create_interactive_graph(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create interactive Plotly visualization.

        Args:
            student_mastery: Dict of concept_id -> mastery
            save_path: Path to save HTML file

        Returns:
            Plotly Figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None

        pos = nx.spring_layout(self.G, k=2, iterations=50, seed=42)

        # Create edge traces
        edge_traces = []

        for edge_type, color in self.edge_colors.items():
            edge_x = []
            edge_y = []

            for u, v, d in self.G.edges(data=True):
                if d.get('edge_type') == edge_type:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:
                edge_traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=1 if edge_type != 'prerequisite' else 2, color=color),
                    hoverinfo='none',
                    name=edge_type.replace('_', ' ').title()
                ))

        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []

        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            concept = self.kg.concepts.get(node)
            if student_mastery and node in student_mastery:
                mastery = student_mastery[node]
                node_colors.append(mastery)
                node_sizes.append(15 + mastery * 20)
            else:
                node_colors.append(0.5)
                node_sizes.append(15)

            # Hover text
            if concept:
                text = f"<b>{concept.name}</b><br>"
                text += f"Chapter: {concept.chapter}<br>"
                text += f"Difficulty: {concept.difficulty_base.name}<br>"
                if student_mastery and node in student_mastery:
                    text += f"Mastery: {student_mastery[node]:.1%}"
                node_text.append(text)
            else:
                node_text.append(node)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[self.kg.concepts.get(n, type('', (), {'name': n})()).name[:15]
                  for n in self.G.nodes()],
            textposition='top center',
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title='Mastery',
                    tickvals=[0, 0.5, 1],
                    ticktext=['0%', '50%', '100%']
                ),
                line=dict(width=1, color='white')
            ),
            name='Concepts'
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        title = "Interactive Knowledge Graph"
        if student_mastery:
            avg = np.mean(list(student_mastery.values()))
            title += f" - Average Mastery: {avg:.1%}"

        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1200,
            height=800
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive graph to {save_path}")

        return fig

    def export_graph_data(self, save_path: str, student_mastery: Optional[Dict[str, float]] = None):
        """Export graph data to JSON for external visualization tools."""
        nodes = []
        for concept_id, concept in self.kg.concepts.items():
            node_data = {
                'id': concept_id,
                'name': concept.name,
                'chapter': concept.chapter,
                'topic': concept.topic,
                'difficulty': concept.difficulty_base.value,
                'bloom_level': concept.bloom_level
            }
            if student_mastery and concept_id in student_mastery:
                node_data['mastery'] = student_mastery[concept_id]
            nodes.append(node_data)

        edges = []
        for edge in self.kg.edges:
            edges.append({
                'source': edge.source_concept_id,
                'target': edge.target_concept_id,
                'type': edge.edge_type,
                'weight': edge.weight
            })

        data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_concepts': len(nodes),
                'total_edges': len(edges),
                'chapters': list(set(n['chapter'] for n in nodes))
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Exported graph data to {save_path}")


def visualize_from_simulation(sample_size: int = 10000):
    """
    Load data and create visualizations.

    Args:
        sample_size: Number of records to load
    """
    from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader
    from knowledge_tracing_system.run_simulation import SimulationRunner

    print("Loading data...")
    runner = SimulationRunner(sample_size=sample_size)

    # Get a student with good amount of data
    student_ids = list(runner.data['student_profiles'].keys())

    # Find student with most attempts
    student_attempts = {}
    for attempt in runner.data['question_attempts']:
        sid = attempt.student_id
        student_attempts[sid] = student_attempts.get(sid, 0) + 1

    best_student = max(student_attempts.items(), key=lambda x: x[1])[0]
    print(f"Selected student: {best_student} ({student_attempts[best_student]} attempts)")

    # Get student's knowledge state
    knowledge_state = runner.simulate_knowledge_state(best_student)
    student_profile = runner.data['student_profiles'].get(best_student)

    # Create visualizer
    viz = KnowledgeGraphVisualizer(runner.data['knowledge_graph'])

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Full knowledge graph
    print("\nCreating full knowledge graph visualization...")
    fig1 = viz.visualize_full_graph(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "knowledge_graph.png")
    )

    # 2. Student progress dashboard
    print("Creating student progress dashboard...")
    fig2 = viz.visualize_student_progress(
        student_id=best_student,
        student_mastery=knowledge_state.concept_mastery,
        student_profile=student_profile,
        save_path=str(output_dir / "student_progress.png")
    )

    # 3. Interactive graph
    print("Creating interactive graph...")
    fig3 = viz.create_interactive_graph(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "interactive_graph.html")
    )

    # 4. Export data
    print("Exporting graph data...")
    viz.export_graph_data(
        str(output_dir / "graph_data.json"),
        student_mastery=knowledge_state.concept_mastery
    )

    print(f"\nAll visualizations saved to: {output_dir}")

    # Show figures
    plt.show()

    return viz, runner


if __name__ == "__main__":
    visualize_from_simulation(sample_size=10000)
