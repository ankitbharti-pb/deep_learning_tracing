"""
Quick Prerequisite Chain Visualization Script
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader
from knowledge_tracing_system.data.prerequisite_builder import build_enhanced_prerequisites

def main():
    print("Loading data...")
    loader = ASSISTmentsDataLoader()
    loader.load_csv_files(sample_size=5000)
    loader.build_knowledge_concepts()
    loader.build_knowledge_graph()

    print("Building enhanced prerequisites...")
    kg = build_enhanced_prerequisites(loader.knowledge_graph)

    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Get prerequisite edges only
    prereq_edges = [(e.source_concept_id, e.target_concept_id, e.weight)
                    for e in kg.edges if e.edge_type == 'prerequisite']

    print(f"Total prerequisite edges: {len(prereq_edges)}")

    # Build graph
    G = nx.DiGraph()
    for cid in kg.concepts:
        G.add_node(cid)
    for src, tgt, w in prereq_edges:
        G.add_edge(src, tgt, weight=w)

    # Group by chapter
    chapters = defaultdict(list)
    for cid, c in kg.concepts.items():
        chapters[c.chapter].append(cid)

    chapter_colors = {
        'Algebra': '#e74c3c',
        'Equations': '#3498db',
        'Geometry': '#2ecc71',
        'Fractions': '#9b59b6',
        'Number Systems': '#f39c12',
        'Decimals': '#1abc9c',
        'Percentages': '#e91e63',
        'General Mathematics': '#607d8b',
        'Probability & Statistics': '#00bcd4',
        'Ratios & Proportions': '#ff5722',
        'Graphing': '#795548',
        'Exponents & Powers': '#673ab7',
    }

    # 1. Create full prerequisite graph
    print("\n1. Creating full prerequisite graph...")
    fig, ax = plt.subplots(figsize=(20, 16))

    # Only nodes with edges
    active_nodes = set()
    for src, tgt, _ in prereq_edges:
        active_nodes.add(src)
        active_nodes.add(tgt)

    subG = G.subgraph(active_nodes)
    pos = nx.spring_layout(subG, k=2, iterations=50, seed=42)

    # Draw edges
    nx.draw_networkx_edges(subG, pos, edge_color='#e74c3c', alpha=0.6,
                          arrows=True, arrowsize=10, ax=ax)

    # Draw nodes colored by chapter
    for chapter, cids in chapters.items():
        chapter_nodes = [c for c in cids if c in active_nodes]
        if chapter_nodes:
            color = chapter_colors.get(chapter, '#cccccc')
            nx.draw_networkx_nodes(subG, pos, nodelist=chapter_nodes,
                                  node_color=color, node_size=300, alpha=0.8, ax=ax)

    # Labels for high-degree nodes
    degrees = dict(subG.degree())
    high_degree = [n for n, d in degrees.items() if d >= 4]
    labels = {n: kg.concepts[n].name[:12] for n in high_degree if n in kg.concepts}
    nx.draw_networkx_labels(subG, pos, labels, font_size=7, ax=ax)

    ax.set_title(f'Full Prerequisite Graph\n({len(active_nodes)} concepts, {len(prereq_edges)} prerequisites)',
                fontsize=14, fontweight='bold')
    ax.axis('off')

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=c, label=ch[:15])
                     for ch, c in list(chapter_colors.items())[:8]]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'full_prerequisite_graph.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'full_prerequisite_graph.png'}")
    plt.close()

    # 2. Create individual chapter graphs
    for chapter in ['Algebra', 'Equations', 'Geometry', 'Fractions']:
        if chapter not in chapters:
            continue

        print(f"\n2. Creating {chapter} prerequisite chain...")
        fig, ax = plt.subplots(figsize=(16, 12))

        chapter_concepts = set(chapters[chapter])

        # Get external prerequisites
        external = set()
        for cid in chapter_concepts:
            for prereq in kg.get_prerequisites(cid):
                if prereq not in chapter_concepts:
                    external.add(prereq)

        all_nodes = chapter_concepts | external

        # Build subgraph
        subG = nx.DiGraph()
        for n in all_nodes:
            subG.add_node(n)

        for src, tgt, w in prereq_edges:
            if src in all_nodes and tgt in all_nodes:
                subG.add_edge(src, tgt)

        # Layout
        pos = {}
        # External on left
        for i, n in enumerate(sorted(external)):
            pos[n] = (0, (i + 1) / (len(external) + 1))

        # Chapter concepts by difficulty
        by_diff = defaultdict(list)
        for cid in chapter_concepts:
            c = kg.concepts.get(cid)
            if c:
                by_diff[c.difficulty_base.value].append(cid)

        x_offset = 0.3
        for diff in sorted(by_diff.keys()):
            nodes = by_diff[diff]
            for i, n in enumerate(nodes):
                pos[n] = (x_offset, (i + 1) / (len(nodes) + 1))
            x_offset += 0.25

        # Draw edges
        intra_edges = [(u, v) for u, v in subG.edges() if u in chapter_concepts]
        inter_edges = [(u, v) for u, v in subG.edges() if u in external]

        if intra_edges:
            nx.draw_networkx_edges(subG, pos, edgelist=intra_edges,
                                  edge_color='#e74c3c', width=2, arrows=True,
                                  arrowsize=12, ax=ax)
        if inter_edges:
            nx.draw_networkx_edges(subG, pos, edgelist=inter_edges,
                                  edge_color='#9b59b6', width=1.5, style='dashed',
                                  arrows=True, arrowsize=10, ax=ax)

        # Draw nodes
        chapter_color = chapter_colors.get(chapter, '#3498db')

        if chapter_concepts:
            nx.draw_networkx_nodes(subG, pos, nodelist=list(chapter_concepts),
                                  node_color=chapter_color, node_size=400,
                                  alpha=0.9, ax=ax)
        if external:
            # Color external by their chapter
            for ext_node in external:
                ext_c = kg.concepts.get(ext_node)
                if ext_c:
                    ext_color = chapter_colors.get(ext_c.chapter, '#cccccc')
                else:
                    ext_color = '#cccccc'
                nx.draw_networkx_nodes(subG, pos, nodelist=[ext_node],
                                      node_color=ext_color, node_size=300,
                                      alpha=0.7, ax=ax)

        # Labels
        labels = {}
        for n in all_nodes:
            c = kg.concepts.get(n)
            if c:
                labels[n] = c.name[:15]

        nx.draw_networkx_labels(subG, pos, labels, font_size=7, ax=ax)

        # Add difficulty labels
        ax.text(0, 1.05, 'External\\nPrereqs', ha='center', fontsize=9, fontweight='bold')
        x_pos = 0.3
        for diff in sorted(by_diff.keys()):
            ax.text(x_pos, 1.05, f'Diff {diff}', ha='center', fontsize=9, fontweight='bold')
            x_pos += 0.25

        ax.set_title(f'{chapter} Prerequisite Chain\\n({len(chapter_concepts)} topics, {len(external)} external prerequisites)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.15, 1.1)
        ax.set_ylim(-0.1, 1.15)

        # Legend
        legend_patches = [
            mpatches.Patch(color='#e74c3c', label='Intra-chapter'),
            mpatches.Patch(color='#9b59b6', label='Cross-chapter'),
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=9)

        plt.tight_layout()
        fname = f'{chapter.lower().replace(" ", "_")}_chain.png'
        plt.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir / fname}")
        plt.close()

    # 3. Create dependency summary table
    print("\n3. Creating summary...")

    summary = []
    for chapter, cids in sorted(chapters.items()):
        intra_count = sum(1 for s, t, _ in prereq_edges
                        if s in cids and t in cids)
        inter_in = sum(1 for s, t, _ in prereq_edges
                      if s not in cids and t in cids)
        inter_out = sum(1 for s, t, _ in prereq_edges
                       if s in cids and t not in cids)
        summary.append({
            'chapter': chapter,
            'topics': len(cids),
            'intra': intra_count,
            'inter_in': inter_in,
            'inter_out': inter_out
        })

    print("\nChapter Dependency Summary:")
    print("-" * 70)
    print(f"{'Chapter':<25} {'Topics':>8} {'Intra':>8} {'Inter-In':>10} {'Inter-Out':>10}")
    print("-" * 70)
    for s in summary:
        print(f"{s['chapter'][:24]:<25} {s['topics']:>8} {s['intra']:>8} {s['inter_in']:>10} {s['inter_out']:>10}")
    print("-" * 70)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
