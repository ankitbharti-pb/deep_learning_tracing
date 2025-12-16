"""
Chapter-Focused Knowledge Graph Visualization

Shows:
- Intra-chapter dependencies (within same chapter)
- Inter-chapter dependencies (across chapters)
- Topic-level prerequisite chains
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import json

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
    KnowledgeGraph, KnowledgeConcept, DifficultyLevel
)


class ChapterDependencyVisualizer:
    """
    Visualizes chapter and topic dependencies with clear
    inter and intra-chapter relationships.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.G = self._build_graph()

        # Organize concepts by chapter
        self.chapters: Dict[str, List[str]] = defaultdict(list)
        for concept_id, concept in self.kg.concepts.items():
            self.chapters[concept.chapter].append(concept_id)

        # Chapter colors
        self.chapter_colors = self._assign_chapter_colors()

        # Mastery colormap
        self.mastery_cmap = LinearSegmentedColormap.from_list(
            'mastery', ['#ff6b6b', '#ffd93d', '#6bcb77']
        )

    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from knowledge graph."""
        G = nx.DiGraph()

        for concept_id, concept in self.kg.concepts.items():
            G.add_node(
                concept_id,
                name=concept.name,
                chapter=concept.chapter,
                topic=concept.topic,
                difficulty=concept.difficulty_base.value,
                prerequisites=concept.prerequisites
            )

        for edge in self.kg.edges:
            G.add_edge(
                edge.source_concept_id,
                edge.target_concept_id,
                edge_type=edge.edge_type,
                weight=edge.weight
            )

        return G

    def _assign_chapter_colors(self) -> Dict[str, str]:
        """Assign distinct colors to each chapter."""
        colors = [
            '#e74c3c',  # Red
            '#3498db',  # Blue
            '#2ecc71',  # Green
            '#9b59b6',  # Purple
            '#f39c12',  # Orange
            '#1abc9c',  # Teal
            '#e91e63',  # Pink
            '#00bcd4',  # Cyan
            '#ff5722',  # Deep Orange
            '#607d8b',  # Blue Grey
            '#795548',  # Brown
            '#673ab7',  # Deep Purple
            '#4caf50',  # Light Green
            '#ff9800',  # Amber
            '#03a9f4',  # Light Blue
        ]

        chapter_colors = {}
        for i, chapter in enumerate(sorted(self.chapters.keys())):
            chapter_colors[chapter] = colors[i % len(colors)]

        return chapter_colors

    def _get_inter_intra_edges(self) -> Tuple[List, List]:
        """Separate edges into inter-chapter and intra-chapter."""
        inter_edges = []
        intra_edges = []

        for u, v, data in self.G.edges(data=True):
            if data.get('edge_type') != 'prerequisite':
                continue

            u_chapter = self.G.nodes[u].get('chapter', '')
            v_chapter = self.G.nodes[v].get('chapter', '')

            if u_chapter == v_chapter:
                intra_edges.append((u, v, data))
            else:
                inter_edges.append((u, v, data))

        return inter_edges, intra_edges

    def visualize_chapter_overview(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (20, 16),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create chapter-level overview showing inter-chapter dependencies.

        Args:
            student_mastery: Optional mastery scores per concept
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create chapter-level graph
        chapter_graph = nx.DiGraph()
        chapter_stats = {}

        for chapter, concepts in self.chapters.items():
            # Calculate chapter statistics
            if student_mastery:
                masteries = [student_mastery.get(c, 0) for c in concepts]
                avg_mastery = np.mean(masteries) if masteries else 0
            else:
                avg_mastery = 0.5

            chapter_stats[chapter] = {
                'num_concepts': len(concepts),
                'avg_mastery': avg_mastery,
                'avg_difficulty': np.mean([
                    self.kg.concepts[c].difficulty_base.value
                    for c in concepts if c in self.kg.concepts
                ])
            }
            chapter_graph.add_node(chapter, **chapter_stats[chapter])

        # Add inter-chapter edges
        inter_edges, _ = self._get_inter_intra_edges()
        chapter_edge_counts = defaultdict(int)

        for u, v, _ in inter_edges:
            u_chapter = self.G.nodes[u].get('chapter', '')
            v_chapter = self.G.nodes[v].get('chapter', '')
            if u_chapter and v_chapter and u_chapter != v_chapter:
                chapter_edge_counts[(u_chapter, v_chapter)] += 1

        for (src, tgt), count in chapter_edge_counts.items():
            chapter_graph.add_edge(src, tgt, weight=count)

        # Layout
        pos = nx.spring_layout(chapter_graph, k=3, iterations=100, seed=42)

        # Draw chapter nodes
        node_colors = []
        node_sizes = []

        for chapter in chapter_graph.nodes():
            stats = chapter_stats[chapter]
            node_colors.append(self.mastery_cmap(stats['avg_mastery']))
            node_sizes.append(500 + stats['num_concepts'] * 50)

        # Draw edges with varying thickness
        edge_weights = [chapter_graph[u][v]['weight'] for u, v in chapter_graph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1

        for (u, v), weight in zip(chapter_graph.edges(), edge_weights):
            nx.draw_networkx_edges(
                chapter_graph, pos,
                edgelist=[(u, v)],
                width=1 + (weight / max_weight) * 4,
                alpha=0.6,
                edge_color='#e74c3c',
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1",
                ax=ax
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            chapter_graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

        # Labels with stats
        labels = {}
        for chapter in chapter_graph.nodes():
            stats = chapter_stats[chapter]
            label = f"{chapter}\n({stats['num_concepts']} topics)"
            if student_mastery:
                label += f"\n{stats['avg_mastery']:.0%} mastery"
            labels[chapter] = label

        nx.draw_networkx_labels(
            chapter_graph, pos, labels,
            font_size=9, font_weight='bold',
            ax=ax
        )

        # Add edge labels (number of dependencies)
        edge_labels = {(u, v): str(d['weight'])
                      for u, v, d in chapter_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            chapter_graph, pos, edge_labels,
            font_size=8, font_color='red',
            ax=ax
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color='#ff6b6b', label='Low Mastery (0-33%)'),
            mpatches.Patch(color='#ffd93d', label='Medium Mastery (34-66%)'),
            mpatches.Patch(color='#6bcb77', label='High Mastery (67-100%)'),
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2,
                      label='Inter-Chapter Prerequisite')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

        ax.set_title('Chapter-Level Dependencies\n(Node size = number of topics, Edge thickness = number of dependencies)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved chapter overview to {save_path}")

        return fig

    def visualize_chapter_detail(
        self,
        chapter: str,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Detailed visualization of a single chapter showing intra-chapter dependencies.

        Args:
            chapter: Chapter name to visualize
            student_mastery: Optional mastery scores
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        if chapter not in self.chapters:
            raise ValueError(f"Chapter '{chapter}' not found. Available: {list(self.chapters.keys())}")

        fig, ax = plt.subplots(figsize=figsize)

        # Get concepts in this chapter
        chapter_concepts = set(self.chapters[chapter])

        # Create subgraph
        subgraph = self.G.subgraph(chapter_concepts).copy()

        # Add external dependencies (from other chapters)
        external_deps = defaultdict(list)  # concept -> list of external prereqs

        for concept_id in chapter_concepts:
            concept = self.kg.concepts.get(concept_id)
            if concept:
                for prereq in concept.prerequisites:
                    if prereq not in chapter_concepts:
                        external_deps[concept_id].append(prereq)

        # Hierarchical layout based on difficulty
        pos = self._hierarchical_chapter_layout(subgraph, chapter_concepts)

        # Draw intra-chapter edges (prerequisites)
        intra_prereq_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                              if d.get('edge_type') == 'prerequisite']

        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=intra_prereq_edges,
            edge_color='#e74c3c',
            width=2,
            arrows=True,
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )

        # Draw related edges
        related_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                        if d.get('edge_type') == 'related']

        nx.draw_networkx_edges(
            subgraph, pos,
            edgelist=related_edges,
            edge_color='#3498db',
            width=1,
            style='dashed',
            alpha=0.5,
            ax=ax
        )

        # Node colors and sizes
        node_colors = []
        node_sizes = []

        for node in subgraph.nodes():
            concept = self.kg.concepts.get(node)

            if student_mastery and node in student_mastery:
                mastery = student_mastery[node]
                node_colors.append(self.mastery_cmap(mastery))
            else:
                node_colors.append('#cccccc')

            # Size by difficulty
            diff = concept.difficulty_base.value if concept else 3
            node_sizes.append(400 + diff * 150)

        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax
        )

        # Labels
        labels = {}
        for node in subgraph.nodes():
            concept = self.kg.concepts.get(node)
            if concept:
                name = concept.name
                if len(name) > 20:
                    name = name[:18] + '...'
                labels[node] = name
            else:
                labels[node] = node

        nx.draw_networkx_labels(
            subgraph, pos, labels,
            font_size=9, font_weight='bold',
            ax=ax
        )

        # Show external dependencies
        if external_deps:
            ext_text = "External Prerequisites:\n"
            for concept_id, prereqs in list(external_deps.items())[:5]:
                concept = self.kg.concepts.get(concept_id)
                concept_name = concept.name[:15] if concept else concept_id[:15]
                for prereq_id in prereqs[:2]:
                    prereq = self.kg.concepts.get(prereq_id)
                    prereq_name = prereq.name[:15] if prereq else prereq_id[:15]
                    prereq_chapter = prereq.chapter[:10] if prereq else "Unknown"
                    ext_text += f"  {concept_name} <- {prereq_name} ({prereq_chapter})\n"

            ax.text(0.02, 0.02, ext_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Prerequisite'),
            plt.Line2D([0], [0], color='#3498db', linewidth=1, linestyle='--', label='Related'),
            mpatches.Patch(color='#ff6b6b', label='Low Mastery'),
            mpatches.Patch(color='#ffd93d', label='Medium Mastery'),
            mpatches.Patch(color='#6bcb77', label='High Mastery'),
            mpatches.Patch(color='#cccccc', label='Not Practiced'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # Difficulty scale
        ax.text(0.02, 0.98, 'Difficulty: Node size increases with difficulty\nTop = Easier, Bottom = Harder',
               transform=ax.transAxes, fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        ax.set_title(f'Chapter: {chapter}\n({len(chapter_concepts)} topics, {len(intra_prereq_edges)} prerequisites)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved chapter detail to {save_path}")

        return fig

    def _hierarchical_chapter_layout(
        self,
        subgraph: nx.DiGraph,
        concepts: Set[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on difficulty levels."""
        # Group by difficulty
        difficulty_groups = defaultdict(list)

        for node in subgraph.nodes():
            concept = self.kg.concepts.get(node)
            if concept:
                diff = concept.difficulty_base.value
            else:
                diff = 3
            difficulty_groups[diff].append(node)

        pos = {}
        num_levels = len(difficulty_groups)

        for level_idx, (diff, nodes) in enumerate(sorted(difficulty_groups.items())):
            y = 1 - (level_idx / max(num_levels, 1))

            for i, node in enumerate(nodes):
                x = (i + 1) / (len(nodes) + 1)
                pos[node] = (x, y)

        return pos

    def visualize_all_chapters(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (24, 20),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive visualization showing all chapters with
        both inter and intra dependencies.

        Args:
            student_mastery: Optional mastery scores
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        num_chapters = len(self.chapters)
        cols = min(4, num_chapters)
        rows = (num_chapters + cols - 1) // cols + 1  # +1 for overview

        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(rows, cols, hspace=0.4, wspace=0.3)

        # Chapter overview in first row (spanning all columns)
        ax_overview = fig.add_subplot(gs[0, :])
        self._draw_chapter_overview_on_axis(ax_overview, student_mastery)

        # Individual chapters
        for idx, (chapter, concepts) in enumerate(sorted(self.chapters.items())):
            row = (idx // cols) + 1
            col = idx % cols

            if row < rows:
                ax = fig.add_subplot(gs[row, col])
                self._draw_chapter_on_axis(ax, chapter, concepts, student_mastery)

        plt.suptitle('Knowledge Graph: Inter and Intra-Chapter Dependencies',
                    fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved all chapters visualization to {save_path}")

        return fig

    def _draw_chapter_overview_on_axis(
        self,
        ax: plt.Axes,
        student_mastery: Optional[Dict[str, float]]
    ):
        """Draw chapter overview on axis."""
        # Create chapter graph
        chapter_graph = nx.DiGraph()

        for chapter, concepts in self.chapters.items():
            if student_mastery:
                masteries = [student_mastery.get(c, 0) for c in concepts]
                avg_mastery = np.mean(masteries) if masteries else 0
            else:
                avg_mastery = 0.5

            chapter_graph.add_node(chapter, num_concepts=len(concepts), avg_mastery=avg_mastery)

        # Inter-chapter edges
        inter_edges, _ = self._get_inter_intra_edges()
        chapter_edges = defaultdict(int)

        for u, v, _ in inter_edges:
            u_ch = self.G.nodes[u].get('chapter', '')
            v_ch = self.G.nodes[v].get('chapter', '')
            if u_ch and v_ch and u_ch != v_ch:
                chapter_edges[(u_ch, v_ch)] += 1

        for (src, tgt), count in chapter_edges.items():
            chapter_graph.add_edge(src, tgt, weight=count)

        pos = nx.spring_layout(chapter_graph, k=2, seed=42)

        # Draw
        node_colors = [self.mastery_cmap(chapter_graph.nodes[n]['avg_mastery'])
                      for n in chapter_graph.nodes()]
        node_sizes = [300 + chapter_graph.nodes[n]['num_concepts'] * 30
                     for n in chapter_graph.nodes()]

        nx.draw_networkx_edges(chapter_graph, pos, ax=ax, edge_color='#e74c3c',
                              alpha=0.6, arrows=True, arrowsize=15)
        nx.draw_networkx_nodes(chapter_graph, pos, ax=ax, node_color=node_colors,
                              node_size=node_sizes, alpha=0.9)

        labels = {ch: f"{ch[:12]}..." if len(ch) > 12 else ch for ch in chapter_graph.nodes()}
        nx.draw_networkx_labels(chapter_graph, pos, labels, ax=ax, font_size=8)

        ax.set_title('Inter-Chapter Dependencies Overview', fontsize=11, fontweight='bold')
        ax.axis('off')

    def _draw_chapter_on_axis(
        self,
        ax: plt.Axes,
        chapter: str,
        concepts: List[str],
        student_mastery: Optional[Dict[str, float]]
    ):
        """Draw single chapter on axis."""
        subgraph = self.G.subgraph(concepts).copy()

        if len(subgraph.nodes()) == 0:
            ax.text(0.5, 0.5, 'No topics', ha='center', va='center')
            ax.set_title(chapter[:20], fontsize=10)
            ax.axis('off')
            return

        pos = nx.spring_layout(subgraph, seed=42)

        # Edges
        prereq_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                       if d.get('edge_type') == 'prerequisite']
        related_edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                        if d.get('edge_type') == 'related']

        if prereq_edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=prereq_edges,
                                  edge_color='#e74c3c', width=1.5, arrows=True,
                                  arrowsize=10, ax=ax)
        if related_edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=related_edges,
                                  edge_color='#3498db', width=0.5, style='dashed',
                                  alpha=0.5, ax=ax)

        # Nodes
        node_colors = []
        for node in subgraph.nodes():
            if student_mastery and node in student_mastery:
                node_colors.append(self.mastery_cmap(student_mastery[node]))
            else:
                node_colors.append('#cccccc')

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                              node_size=200, alpha=0.9, ax=ax)

        # Stats
        num_prereqs = len(prereq_edges)
        num_related = len(related_edges)

        if student_mastery:
            masteries = [student_mastery.get(c, 0) for c in concepts]
            avg_m = np.mean(masteries) if masteries else 0
            title = f"{chapter[:18]}\n{len(concepts)} topics | {avg_m:.0%} mastery"
        else:
            title = f"{chapter[:18]}\n{len(concepts)} topics"

        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    def visualize_dependency_matrix(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a dependency matrix showing inter-chapter relationships.

        Args:
            student_mastery: Optional mastery scores
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        chapters = sorted(self.chapters.keys())
        n = len(chapters)

        # Create dependency matrix
        dep_matrix = np.zeros((n, n))

        inter_edges, intra_edges = self._get_inter_intra_edges()

        # Count inter-chapter dependencies
        for u, v, _ in inter_edges:
            u_ch = self.G.nodes[u].get('chapter', '')
            v_ch = self.G.nodes[v].get('chapter', '')

            if u_ch in chapters and v_ch in chapters:
                i = chapters.index(u_ch)
                j = chapters.index(v_ch)
                dep_matrix[i, j] += 1

        # Count intra-chapter dependencies (diagonal)
        for u, v, _ in intra_edges:
            u_ch = self.G.nodes[u].get('chapter', '')
            if u_ch in chapters:
                i = chapters.index(u_ch)
                dep_matrix[i, i] += 1

        # Plot dependency matrix
        im = ax1.imshow(dep_matrix, cmap='YlOrRd', aspect='auto')

        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))

        short_chapters = [ch[:12] + '...' if len(ch) > 12 else ch for ch in chapters]
        ax1.set_xticklabels(short_chapters, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(short_chapters, fontsize=8)

        ax1.set_xlabel('Target Chapter (depends on)', fontsize=10)
        ax1.set_ylabel('Source Chapter (prerequisite)', fontsize=10)
        ax1.set_title('Dependency Matrix\n(Diagonal = Intra-chapter, Off-diagonal = Inter-chapter)',
                     fontsize=11, fontweight='bold')

        # Add text annotations
        for i in range(n):
            for j in range(n):
                if dep_matrix[i, j] > 0:
                    ax1.text(j, i, int(dep_matrix[i, j]), ha='center', va='center',
                            fontsize=8, color='white' if dep_matrix[i, j] > dep_matrix.max()/2 else 'black')

        plt.colorbar(im, ax=ax1, label='Number of Dependencies')

        # Plot chapter statistics
        chapter_stats = []
        for chapter in chapters:
            concepts = self.chapters[chapter]

            # Count dependencies
            intra = sum(1 for u, v, _ in intra_edges
                       if self.G.nodes[u].get('chapter') == chapter)
            inter_out = sum(1 for u, v, _ in inter_edges
                          if self.G.nodes[u].get('chapter') == chapter)
            inter_in = sum(1 for u, v, _ in inter_edges
                         if self.G.nodes[v].get('chapter') == chapter)

            if student_mastery:
                avg_mastery = np.mean([student_mastery.get(c, 0) for c in concepts])
            else:
                avg_mastery = 0.5

            chapter_stats.append({
                'chapter': chapter,
                'num_topics': len(concepts),
                'intra_deps': intra,
                'inter_out': inter_out,
                'inter_in': inter_in,
                'avg_mastery': avg_mastery
            })

        # Bar chart of dependencies
        x = np.arange(n)
        width = 0.25

        intra_deps = [s['intra_deps'] for s in chapter_stats]
        inter_out = [s['inter_out'] for s in chapter_stats]
        inter_in = [s['inter_in'] for s in chapter_stats]

        ax2.bar(x - width, intra_deps, width, label='Intra-chapter', color='#2ecc71')
        ax2.bar(x, inter_out, width, label='Inter-chapter (outgoing)', color='#e74c3c')
        ax2.bar(x + width, inter_in, width, label='Inter-chapter (incoming)', color='#3498db')

        ax2.set_xticks(x)
        ax2.set_xticklabels(short_chapters, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Number of Dependencies')
        ax2.set_title('Dependencies by Chapter', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved dependency matrix to {save_path}")

        return fig

    def visualize_learning_path(
        self,
        student_mastery: Dict[str, float],
        target_concepts: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (18, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize recommended learning path based on prerequisites.

        Args:
            student_mastery: Current mastery scores
            target_concepts: Concepts to target (if None, finds weakest)
            figsize: Figure size
            save_path: Path to save

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Find concepts needing work
        if target_concepts is None:
            weak_concepts = [(cid, m) for cid, m in student_mastery.items() if m < 0.6]
            weak_concepts.sort(key=lambda x: x[1])
            target_concepts = [c[0] for c in weak_concepts[:10]]

        # Build learning path considering prerequisites
        learning_path = []
        visited = set()

        def get_prereqs_recursive(concept_id: str, depth: int = 0) -> List[Tuple[str, int]]:
            if concept_id in visited or depth > 5:
                return []

            visited.add(concept_id)
            prereqs = []

            concept = self.kg.concepts.get(concept_id)
            if concept:
                for prereq_id in concept.prerequisites:
                    prereq_mastery = student_mastery.get(prereq_id, 0)
                    if prereq_mastery < 0.7:  # Only include unmastered prereqs
                        prereqs.extend(get_prereqs_recursive(prereq_id, depth + 1))
                        prereqs.append((prereq_id, depth + 1))

            return prereqs

        for target in target_concepts:
            prereqs = get_prereqs_recursive(target)
            learning_path.extend(prereqs)
            if target not in visited:
                learning_path.append((target, 0))
                visited.add(target)

        # Remove duplicates while preserving order
        seen = set()
        unique_path = []
        for item in learning_path:
            if item[0] not in seen:
                seen.add(item[0])
                unique_path.append(item)

        # Sort by depth (prerequisites first)
        unique_path.sort(key=lambda x: -x[1])

        # Visualize as flow diagram
        ax1.set_xlim(-0.5, 1.5)
        ax1.set_ylim(-0.5, max(len(unique_path), 1) + 0.5)

        for i, (concept_id, depth) in enumerate(unique_path[:15]):  # Limit to 15
            concept = self.kg.concepts.get(concept_id)
            name = concept.name[:25] if concept else concept_id[:25]
            chapter = concept.chapter[:15] if concept else 'Unknown'
            mastery = student_mastery.get(concept_id, 0)

            # Draw box
            color = self.mastery_cmap(mastery)
            y = len(unique_path) - i - 1

            rect = FancyBboxPatch((0, y - 0.3), 1, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor='black',
                                 alpha=0.8)
            ax1.add_patch(rect)

            # Text
            ax1.text(0.5, y, f"{i+1}. {name}\n({chapter}) - {mastery:.0%}",
                    ha='center', va='center', fontsize=8, fontweight='bold')

            # Arrow to next
            if i < len(unique_path) - 1 and i < 14:
                ax1.annotate('', xy=(0.5, y - 0.35), xytext=(0.5, y - 0.65),
                           arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

        ax1.set_title('Recommended Learning Path\n(Prerequisites First)',
                     fontsize=11, fontweight='bold')
        ax1.axis('off')

        # Chapter breakdown of learning path
        path_chapters = defaultdict(list)
        for concept_id, _ in unique_path:
            concept = self.kg.concepts.get(concept_id)
            if concept:
                path_chapters[concept.chapter].append(concept_id)

        chapters = list(path_chapters.keys())
        counts = [len(path_chapters[ch]) for ch in chapters]
        colors = [self.chapter_colors.get(ch, '#cccccc') for ch in chapters]

        ax2.barh(range(len(chapters)), counts, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(chapters)))
        ax2.set_yticklabels([ch[:20] for ch in chapters], fontsize=9)
        ax2.set_xlabel('Number of Topics to Learn')
        ax2.set_title('Learning Path by Chapter', fontsize=11, fontweight='bold')

        # Add count labels
        for i, count in enumerate(counts):
            ax2.text(count + 0.1, i, str(count), va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved learning path to {save_path}")

        return fig

    def create_interactive_chapter_graph(
        self,
        student_mastery: Optional[Dict[str, float]] = None,
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """Create interactive Plotly visualization with chapter grouping."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available")
            return None

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            subplot_titles=('Knowledge Graph by Chapter', 'Chapter Statistics'),
            specs=[[{"type": "scatter"}, {"type": "bar"}]]
        )

        # Layout with chapter grouping
        pos = {}
        chapter_list = sorted(self.chapters.keys())

        for ch_idx, chapter in enumerate(chapter_list):
            concepts = self.chapters[chapter]
            angle = 2 * np.pi * ch_idx / len(chapter_list)

            # Place chapter concepts in a cluster
            center_x = 2 * np.cos(angle)
            center_y = 2 * np.sin(angle)

            for i, concept_id in enumerate(concepts):
                inner_angle = 2 * np.pi * i / len(concepts)
                radius = 0.3 + 0.1 * (i % 3)

                x = center_x + radius * np.cos(inner_angle)
                y = center_y + radius * np.sin(inner_angle)
                pos[concept_id] = (x, y)

        # Draw edges
        inter_edges, intra_edges = self._get_inter_intra_edges()

        # Intra-chapter edges
        for u, v, _ in intra_edges:
            if u in pos and v in pos:
                fig.add_trace(go.Scatter(
                    x=[pos[u][0], pos[v][0], None],
                    y=[pos[u][1], pos[v][1], None],
                    mode='lines',
                    line=dict(color='#3498db', width=1),
                    hoverinfo='none',
                    showlegend=False
                ), row=1, col=1)

        # Inter-chapter edges
        for u, v, _ in inter_edges:
            if u in pos and v in pos:
                fig.add_trace(go.Scatter(
                    x=[pos[u][0], pos[v][0], None],
                    y=[pos[u][1], pos[v][1], None],
                    mode='lines',
                    line=dict(color='#e74c3c', width=2),
                    hoverinfo='none',
                    showlegend=False
                ), row=1, col=1)

        # Draw nodes by chapter
        for chapter in chapter_list:
            concepts = self.chapters[chapter]

            node_x = [pos[c][0] for c in concepts if c in pos]
            node_y = [pos[c][1] for c in concepts if c in pos]

            if student_mastery:
                colors = [student_mastery.get(c, 0) for c in concepts if c in pos]
            else:
                colors = [0.5] * len(node_x)

            hover_text = []
            for c in concepts:
                if c in pos:
                    concept = self.kg.concepts.get(c)
                    if concept:
                        text = f"<b>{concept.name}</b><br>"
                        text += f"Chapter: {concept.chapter}<br>"
                        text += f"Difficulty: {concept.difficulty_base.name}<br>"
                        if student_mastery and c in student_mastery:
                            text += f"Mastery: {student_mastery[c]:.1%}"
                        hover_text.append(text)

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors,
                    colorscale='RdYlGn',
                    cmin=0,
                    cmax=1,
                    line=dict(width=1, color='white')
                ),
                text=hover_text,
                hoverinfo='text',
                name=chapter[:20]
            ), row=1, col=1)

        # Bar chart of chapter statistics
        chapter_stats = []
        for chapter in chapter_list:
            concepts = self.chapters[chapter]
            intra = sum(1 for u, v, _ in intra_edges
                       if self.G.nodes[u].get('chapter') == chapter)
            inter = sum(1 for u, v, _ in inter_edges
                       if self.G.nodes[u].get('chapter') == chapter or
                          self.G.nodes[v].get('chapter') == chapter)

            chapter_stats.append({
                'chapter': chapter,
                'topics': len(concepts),
                'intra': intra,
                'inter': inter
            })

        fig.add_trace(go.Bar(
            y=[s['chapter'][:15] for s in chapter_stats],
            x=[s['topics'] for s in chapter_stats],
            orientation='h',
            name='Topics',
            marker_color='#3498db'
        ), row=1, col=2)

        fig.update_layout(
            title='Interactive Knowledge Graph with Chapter Dependencies',
            showlegend=True,
            height=700,
            width=1400
        )

        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive chapter graph to {save_path}")

        return fig


def run_chapter_visualization(sample_size: int = 10000):
    """Run chapter-focused visualization."""
    from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader
    from knowledge_tracing_system.run_simulation import SimulationRunner

    print("Loading data...")
    runner = SimulationRunner(sample_size=sample_size)

    # Find student with most coverage
    student_attempts = defaultdict(set)
    for attempt in runner.data['question_attempts']:
        question = runner.data['questions'].get(attempt.question_id)
        if question:
            for cid in question.concept_ids:
                student_attempts[attempt.student_id].add(cid)

    best_student = max(student_attempts.items(), key=lambda x: len(x[1]))[0]
    print(f"Selected student: {best_student} ({len(student_attempts[best_student])} concepts practiced)")

    # Get knowledge state
    knowledge_state = runner.simulate_knowledge_state(best_student)

    # Create visualizer
    viz = ChapterDependencyVisualizer(runner.data['knowledge_graph'])

    # Output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # 1. Chapter overview
    print("\n1. Creating chapter overview...")
    viz.visualize_chapter_overview(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "chapter_overview.png")
    )

    # 2. All chapters view
    print("2. Creating all chapters visualization...")
    viz.visualize_all_chapters(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "all_chapters.png")
    )

    # 3. Dependency matrix
    print("3. Creating dependency matrix...")
    viz.visualize_dependency_matrix(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "dependency_matrix.png")
    )

    # 4. Learning path
    print("4. Creating learning path...")
    viz.visualize_learning_path(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "learning_path.png")
    )

    # 5. Individual chapter details (first 3)
    print("5. Creating individual chapter details...")
    for i, chapter in enumerate(sorted(viz.chapters.keys())[:3]):
        viz.visualize_chapter_detail(
            chapter=chapter,
            student_mastery=knowledge_state.concept_mastery,
            save_path=str(output_dir / f"chapter_{i+1}_{chapter[:20].replace(' ', '_')}.png")
        )

    # 6. Interactive
    print("6. Creating interactive visualization...")
    viz.create_interactive_chapter_graph(
        student_mastery=knowledge_state.concept_mastery,
        save_path=str(output_dir / "interactive_chapters.html")
    )

    print(f"\nAll visualizations saved to: {output_dir}")

    return viz, runner


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    run_chapter_visualization(sample_size=10000)
