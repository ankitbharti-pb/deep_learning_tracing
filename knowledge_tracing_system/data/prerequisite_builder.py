"""
Prerequisite Chain Builder

Creates explicit topic-to-topic prerequisite chains based on
educational domain knowledge for mathematics concepts.
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from knowledge_tracing_system.data.schemas import (
    KnowledgeGraph, KnowledgeConcept, KnowledgeGraphEdge, DifficultyLevel
)


class PrerequisiteChainBuilder:
    """
    Builds explicit prerequisite chains based on educational domain knowledge.
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.concept_by_name: Dict[str, str] = {}  # name -> concept_id
        self.concept_by_keyword: Dict[str, List[str]] = defaultdict(list)

        # Build lookup tables
        for cid, concept in self.kg.concepts.items():
            name_lower = concept.name.lower()
            self.concept_by_name[name_lower] = cid

            # Extract keywords
            keywords = self._extract_keywords(concept.name)
            for kw in keywords:
                self.concept_by_keyword[kw].append(cid)

    def _extract_keywords(self, name: str) -> List[str]:
        """Extract meaningful keywords from concept name."""
        # Remove common words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'of', 'to', 'in', 'for', 'with', 'as'}
        words = re.findall(r'\b[a-zA-Z]+\b', name.lower())
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _find_concept_by_pattern(self, patterns: List[str]) -> List[str]:
        """Find concepts matching any of the patterns."""
        matches = []
        for cid, concept in self.kg.concepts.items():
            name_lower = concept.name.lower()
            for pattern in patterns:
                if pattern.lower() in name_lower:
                    matches.append(cid)
                    break
        return matches

    def build_educational_prerequisites(self) -> KnowledgeGraph:
        """
        Build comprehensive prerequisite chains based on math education sequence.
        """
        # Clear existing prerequisite edges
        self.kg.edges = [e for e in self.kg.edges if e.edge_type != 'prerequisite']

        # Define prerequisite rules
        prerequisite_rules = self._get_prerequisite_rules()

        # Apply rules
        for rule in prerequisite_rules:
            prereq_patterns = rule['prereq']
            target_patterns = rule['target']
            weight = rule.get('weight', 1.0)

            prereq_concepts = self._find_concept_by_pattern(prereq_patterns)
            target_concepts = self._find_concept_by_pattern(target_patterns)

            for prereq_id in prereq_concepts:
                for target_id in target_concepts:
                    if prereq_id != target_id:
                        self._add_prerequisite(prereq_id, target_id, weight)

        # Build difficulty-based prerequisites within chapters
        self._build_difficulty_prerequisites()

        # Build cross-chapter foundational prerequisites
        self._build_foundational_prerequisites()

        print(f"Built {len([e for e in self.kg.edges if e.edge_type == 'prerequisite'])} prerequisite edges")

        return self.kg

    def _get_prerequisite_rules(self) -> List[Dict]:
        """
        Define educational prerequisite rules.
        Format: {'prereq': [patterns], 'target': [patterns], 'weight': float}
        """
        return [
            # Number fundamentals -> Operations
            {'prereq': ['number line', 'ordering'], 'target': ['addition', 'subtraction'], 'weight': 1.0},
            {'prereq': ['addition', 'subtraction'], 'target': ['multiplication', 'division'], 'weight': 1.0},

            # Fractions chain
            {'prereq': ['ordering fraction', 'equivalent fraction'], 'target': ['add fraction', 'subtract fraction'], 'weight': 1.0},
            {'prereq': ['add fraction', 'subtract fraction'], 'target': ['multiply fraction', 'divide fraction'], 'weight': 1.0},
            {'prereq': ['fraction'], 'target': ['decimal', 'percent'], 'weight': 0.8},

            # Decimals chain
            {'prereq': ['place value', 'decimal'], 'target': ['decimal addition', 'decimal subtraction'], 'weight': 1.0},
            {'prereq': ['decimal'], 'target': ['percent'], 'weight': 0.9},

            # Percentages
            {'prereq': ['fraction', 'decimal'], 'target': ['percent'], 'weight': 0.8},
            {'prereq': ['percent'], 'target': ['discount', 'tax', 'interest'], 'weight': 1.0},

            # Ratios and Proportions
            {'prereq': ['fraction', 'equivalent'], 'target': ['ratio', 'proportion'], 'weight': 0.9},
            {'prereq': ['ratio'], 'target': ['proportion', 'scale'], 'weight': 1.0},
            {'prereq': ['proportion'], 'target': ['similar', 'scale factor'], 'weight': 1.0},

            # Algebra foundations
            {'prereq': ['order of operations', 'expression'], 'target': ['variable', 'algebraic'], 'weight': 1.0},
            {'prereq': ['variable'], 'target': ['equation', 'expression'], 'weight': 1.0},
            {'prereq': ['simplify', 'expression'], 'target': ['solve', 'equation'], 'weight': 1.0},
            {'prereq': ['equation'], 'target': ['linear equation', 'system'], 'weight': 1.0},

            # Linear equations chain
            {'prereq': ['slope', 'intercept'], 'target': ['linear equation', 'graph line'], 'weight': 1.0},
            {'prereq': ['linear equation'], 'target': ['system of equation'], 'weight': 1.0},
            {'prereq': ['write linear', 'slope'], 'target': ['finding slope'], 'weight': 0.8},

            # Polynomials
            {'prereq': ['variable', 'exponent'], 'target': ['polynomial', 'monomial'], 'weight': 1.0},
            {'prereq': ['polynomial'], 'target': ['factor', 'polynomial factor'], 'weight': 1.0},
            {'prereq': ['greatest common factor'], 'target': ['factor', 'simplif'], 'weight': 0.9},

            # Exponents chain
            {'prereq': ['multiplication'], 'target': ['exponent', 'power'], 'weight': 0.9},
            {'prereq': ['exponent'], 'target': ['scientific notation', 'square root'], 'weight': 1.0},

            # Geometry foundations
            {'prereq': ['angle', 'line'], 'target': ['triangle', 'polygon'], 'weight': 1.0},
            {'prereq': ['triangle'], 'target': ['pythagorean', 'similar triangle', 'congruent'], 'weight': 1.0},

            # Area and perimeter
            {'prereq': ['perimeter'], 'target': ['area'], 'weight': 0.8},
            {'prereq': ['area rectangle', 'area square'], 'target': ['area triangle', 'area circle'], 'weight': 1.0},
            {'prereq': ['area'], 'target': ['surface area', 'volume'], 'weight': 1.0},

            # Circle geometry
            {'prereq': ['radius', 'diameter'], 'target': ['circumference', 'area circle'], 'weight': 1.0},
            {'prereq': ['circle'], 'target': ['arc', 'sector', 'circle graph'], 'weight': 0.9},

            # Statistics chain
            {'prereq': ['mean', 'median', 'mode'], 'target': ['range', 'standard deviation'], 'weight': 1.0},
            {'prereq': ['data', 'table'], 'target': ['graph', 'chart', 'plot'], 'weight': 0.8},
            {'prereq': ['histogram', 'bar graph'], 'target': ['box and whisker', 'stem and leaf'], 'weight': 0.9},

            # Probability
            {'prereq': ['fraction', 'ratio'], 'target': ['probability'], 'weight': 0.8},
            {'prereq': ['probability'], 'target': ['compound probability', 'expected'], 'weight': 1.0},

            # Coordinate geometry
            {'prereq': ['number line', 'integer'], 'target': ['coordinate', 'ordered pair'], 'weight': 1.0},
            {'prereq': ['coordinate', 'ordered pair'], 'target': ['graph', 'plot', 'scatter'], 'weight': 1.0},

            # Word problems require operations
            {'prereq': ['addition', 'subtraction', 'multiplication', 'division'], 'target': ['word problem'], 'weight': 0.7},

            # Absolute value
            {'prereq': ['number line', 'integer'], 'target': ['absolute value'], 'weight': 1.0},

            # Inequalities
            {'prereq': ['equation', 'solve'], 'target': ['inequality'], 'weight': 1.0},
        ]

    def _add_prerequisite(self, prereq_id: str, target_id: str, weight: float = 1.0):
        """Add a prerequisite edge if not already exists."""
        # Check if edge already exists
        for edge in self.kg.edges:
            if (edge.source_concept_id == prereq_id and
                edge.target_concept_id == target_id and
                edge.edge_type == 'prerequisite'):
                return

        edge = KnowledgeGraphEdge(
            source_concept_id=prereq_id,
            target_concept_id=target_id,
            edge_type='prerequisite',
            weight=weight
        )
        self.kg.edges.append(edge)

        # Update concept's prerequisites list
        concept = self.kg.concepts.get(target_id)
        if concept and prereq_id not in concept.prerequisites:
            concept.prerequisites.append(prereq_id)

    def _build_difficulty_prerequisites(self):
        """Build prerequisites based on difficulty within each chapter."""
        # Group by chapter
        chapter_concepts = defaultdict(list)
        for cid, concept in self.kg.concepts.items():
            chapter_concepts[concept.chapter].append((cid, concept))

        for chapter, concepts in chapter_concepts.items():
            # Sort by difficulty
            concepts.sort(key=lambda x: x[1].difficulty_base.value)

            # Create chains within difficulty levels
            for i in range(len(concepts) - 1):
                curr_id, curr = concepts[i]
                next_id, next_concept = concepts[i + 1]

                diff_gap = next_concept.difficulty_base.value - curr.difficulty_base.value

                # Only connect if difficulty increases by 1
                if diff_gap == 1:
                    self._add_prerequisite(curr_id, next_id, 0.6)

    def _build_foundational_prerequisites(self):
        """Build cross-chapter foundational prerequisites."""
        # Define foundational chapters
        foundational_order = [
            'Number Systems',
            'Fractions',
            'Decimals',
            'Percentages',
            'Ratios & Proportions',
            'Algebra',
            'Equations',
            'Geometry',
            'Probability & Statistics'
        ]

        # Get beginner concepts from each chapter
        for i in range(len(foundational_order) - 1):
            curr_chapter = foundational_order[i]
            next_chapter = foundational_order[i + 1]

            # Find beginner concepts in current chapter
            curr_beginners = [
                cid for cid, c in self.kg.concepts.items()
                if c.chapter == curr_chapter and c.difficulty_base.value <= 2
            ]

            # Find beginner concepts in next chapter
            next_beginners = [
                cid for cid, c in self.kg.concepts.items()
                if c.chapter == next_chapter and c.difficulty_base.value <= 2
            ]

            # Create cross-chapter links (limited to avoid explosion)
            for curr_id in curr_beginners[:3]:
                for next_id in next_beginners[:2]:
                    self._add_prerequisite(curr_id, next_id, 0.5)

    def get_prerequisite_chains(self, concept_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Get all prerequisite chains leading to a concept.

        Returns list of chains, where each chain is a list of concept IDs
        from foundational to target.
        """
        chains = []

        def dfs(current_id: str, current_chain: List[str], depth: int):
            if depth > max_depth:
                return

            prereqs = self.kg.get_prerequisites(current_id)

            if not prereqs:
                # Found a root - this is a complete chain
                chains.append(list(reversed(current_chain)))
                return

            for prereq_id in prereqs:
                if prereq_id not in current_chain:  # Avoid cycles
                    dfs(prereq_id, current_chain + [prereq_id], depth + 1)

        dfs(concept_id, [concept_id], 0)

        return chains

    def get_learning_sequence(self, target_concepts: List[str]) -> List[str]:
        """
        Get optimal learning sequence for target concepts,
        respecting all prerequisites.
        """
        # Collect all required concepts
        required = set(target_concepts)
        to_process = list(target_concepts)

        while to_process:
            current = to_process.pop(0)
            prereqs = self.kg.get_prerequisites(current)
            for prereq in prereqs:
                if prereq not in required:
                    required.add(prereq)
                    to_process.append(prereq)

        # Topological sort
        in_degree = {c: 0 for c in required}
        for c in required:
            for prereq in self.kg.get_prerequisites(c):
                if prereq in required:
                    in_degree[c] += 1

        # Start with concepts that have no prerequisites
        queue = [c for c in required if in_degree[c] == 0]
        sequence = []

        while queue:
            # Sort by difficulty for consistent ordering
            queue.sort(key=lambda x: self.kg.concepts[x].difficulty_base.value
                      if x in self.kg.concepts else 3)
            current = queue.pop(0)
            sequence.append(current)

            # Update in-degrees
            for c in required:
                if current in self.kg.get_prerequisites(c):
                    in_degree[c] -= 1
                    if in_degree[c] == 0:
                        queue.append(c)

        return sequence

    def print_prerequisite_tree(self, concept_id: str, indent: int = 0, visited: Set[str] = None):
        """Print prerequisite tree for debugging."""
        if visited is None:
            visited = set()

        if concept_id in visited:
            return
        visited.add(concept_id)

        concept = self.kg.concepts.get(concept_id)
        name = concept.name if concept else concept_id
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{name}")

        prereqs = self.kg.get_prerequisites(concept_id)
        for prereq in prereqs:
            self.print_prerequisite_tree(prereq, indent + 1, visited)


def build_enhanced_prerequisites(knowledge_graph: KnowledgeGraph) -> KnowledgeGraph:
    """Build enhanced prerequisite chains for a knowledge graph."""
    builder = PrerequisiteChainBuilder(knowledge_graph)
    return builder.build_educational_prerequisites()


if __name__ == "__main__":
    from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader

    print("Loading data...")
    loader = ASSISTmentsDataLoader()
    loader.load_csv_files(sample_size=10000)
    loader.build_knowledge_concepts()
    loader.build_knowledge_graph()

    print("\nOriginal prerequisite edges:", len([e for e in loader.knowledge_graph.edges if e.edge_type == 'prerequisite']))

    print("\nBuilding enhanced prerequisites...")
    builder = PrerequisiteChainBuilder(loader.knowledge_graph)
    enhanced_kg = builder.build_educational_prerequisites()

    print("\nEnhanced prerequisite edges:", len([e for e in enhanced_kg.edges if e.edge_type == 'prerequisite']))

    # Print sample chains
    print("\n" + "="*60)
    print("Sample Prerequisite Trees:")
    print("="*60)

    # Find some complex concepts
    complex_concepts = [
        cid for cid, c in enhanced_kg.concepts.items()
        if c.difficulty_base.value >= 4
    ][:3]

    for cid in complex_concepts:
        concept = enhanced_kg.concepts[cid]
        print(f"\n{concept.name} ({concept.chapter}):")
        builder.print_prerequisite_tree(cid)
