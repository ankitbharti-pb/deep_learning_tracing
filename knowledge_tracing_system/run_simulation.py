"""
Simulation Runner for Knowledge Tracing System

Loads ASSISTments 2009 data and runs sample simulations
to demonstrate the knowledge tracing capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime
from typing import Dict, List
import random

from knowledge_tracing_system.data.data_loader import ASSISTmentsDataLoader
from knowledge_tracing_system.data.schemas import (
    KnowledgeState, StudentProfile, QuestionAttempt, ContentInteraction
)


class SimulationRunner:
    """
    Runs simulations using loaded ASSISTments data.
    """

    def __init__(self, sample_size: int = 10000):
        """
        Initialize simulation with data.

        Args:
            sample_size: Number of records to load (use None for full dataset)
        """
        self.loader = ASSISTmentsDataLoader()
        self.data = self.loader.load_all(
            sample_size=sample_size,
            content_per_student=10,
            chatbot_per_student=5
        )

    def get_student_learning_sequence(self, student_id: str) -> Dict:
        """
        Get a student's complete learning sequence for simulation.

        Args:
            student_id: The student ID to retrieve

        Returns:
            Dictionary with profile, attempts, content, and chatbot interactions
        """
        return self.loader.get_student_data(student_id)

    def simulate_knowledge_state(self, student_id: str) -> KnowledgeState:
        """
        Simulate current knowledge state for a student based on their history.

        This is a simplified simulation - the actual model would use
        deep learning to compute these values.

        Args:
            student_id: The student ID

        Returns:
            KnowledgeState object with mastery estimates
        """
        student_data = self.get_student_learning_sequence(student_id)
        profile = student_data['profile']

        if profile is None:
            raise ValueError(f"Student {student_id} not found")

        # Calculate concept mastery from attempts
        concept_attempts: Dict[str, List[bool]] = {}
        for attempt in student_data['question_attempts']:
            question = self.data['questions'].get(attempt.question_id)
            if question:
                for concept_id in question.concept_ids:
                    if concept_id not in concept_attempts:
                        concept_attempts[concept_id] = []
                    concept_attempts[concept_id].append(attempt.is_correct)

        # Simple mastery calculation (exponential moving average)
        concept_mastery = {}
        for concept_id, attempts in concept_attempts.items():
            if attempts:
                # Recent attempts weighted more heavily
                weights = [0.5 ** (len(attempts) - i - 1) for i in range(len(attempts))]
                weighted_sum = sum(w * a for w, a in zip(weights, attempts))
                mastery = weighted_sum / sum(weights)
                concept_mastery[concept_id] = mastery

        # Identify improving/declining concepts
        improving = []
        declining = []
        stable = []

        for concept_id, attempts in concept_attempts.items():
            if len(attempts) >= 3:
                recent_avg = sum(attempts[-3:]) / 3
                older_avg = sum(attempts[:-3]) / max(len(attempts) - 3, 1) if len(attempts) > 3 else 0.5
                if recent_avg > older_avg + 0.1:
                    improving.append(concept_id)
                elif recent_avg < older_avg - 0.1:
                    declining.append(concept_id)
                else:
                    stable.append(concept_id)

        # Determine ready for assessment vs needs review
        ready = [cid for cid, m in concept_mastery.items() if m >= 0.8]
        needs_review = [cid for cid, m in concept_mastery.items() if m < 0.5]

        # Next recommended concepts
        all_concepts = set(self.data['concepts'].keys())
        practiced = set(concept_mastery.keys())
        unpracticed = list(all_concepts - practiced)

        # Recommend concepts with mastered prerequisites
        recommended = []
        for concept_id in unpracticed[:5]:
            concept = self.data['concepts'].get(concept_id)
            if concept:
                prereqs = concept.prerequisites
                if all(concept_mastery.get(p, 0) >= 0.6 for p in prereqs):
                    recommended.append(concept_id)

        return KnowledgeState(
            student_id=student_id,
            timestamp=datetime.now(),
            concept_mastery=concept_mastery,
            concepts_improving=improving,
            concepts_declining=declining,
            concepts_stable=stable,
            ready_for_assessment=ready,
            needs_review=needs_review,
            next_recommended_concepts=recommended[:3],
            mastery_confidence={cid: (max(0, m - 0.1), min(1, m + 0.1))
                              for cid, m in concept_mastery.items()}
        )

    def run_sample_simulation(self) -> None:
        """Run a sample simulation and print results."""
        print("\n" + "=" * 70)
        print("KNOWLEDGE TRACING SIMULATION DEMO")
        print("=" * 70)

        # Get a sample student
        student_ids = list(self.data['student_profiles'].keys())
        sample_student = random.choice(student_ids)

        print(f"\n1. STUDENT PROFILE")
        print("-" * 40)
        profile = self.data['student_profiles'][sample_student]
        print(f"   Student ID: {profile.student_id}")
        print(f"   Overall Mastery: {profile.overall_mastery:.2%}")
        print(f"   Learning Pace: {profile.learning_pace}")
        print(f"   Total Learning Time: {profile.total_learning_time_hours:.1f} hours")

        # Show preferred mediums
        print(f"\n   Learning Medium Preferences:")
        for medium, weight in sorted(profile.preferred_medium.items(),
                                    key=lambda x: x[1], reverse=True):
            print(f"     - {medium.value}: {weight:.1%}")

        # Simulate knowledge state
        print(f"\n2. SIMULATED KNOWLEDGE STATE")
        print("-" * 40)
        knowledge_state = self.simulate_knowledge_state(sample_student)

        # Show top 5 mastered concepts
        sorted_mastery = sorted(knowledge_state.concept_mastery.items(),
                               key=lambda x: x[1], reverse=True)

        print(f"   Top 5 Mastered Concepts:")
        for concept_id, mastery in sorted_mastery[:5]:
            concept = self.data['concepts'].get(concept_id)
            name = concept.name if concept else concept_id
            print(f"     - {name}: {mastery:.2%}")

        print(f"\n   Bottom 5 Concepts (Need Review):")
        for concept_id, mastery in sorted_mastery[-5:]:
            concept = self.data['concepts'].get(concept_id)
            name = concept.name if concept else concept_id
            print(f"     - {name}: {mastery:.2%}")

        print(f"\n   Improving: {len(knowledge_state.concepts_improving)} concepts")
        print(f"   Declining: {len(knowledge_state.concepts_declining)} concepts")
        print(f"   Ready for Assessment: {len(knowledge_state.ready_for_assessment)} concepts")
        print(f"   Needs Review: {len(knowledge_state.needs_review)} concepts")

        # Show learning history summary
        print(f"\n3. LEARNING HISTORY")
        print("-" * 40)
        student_data = self.get_student_learning_sequence(sample_student)

        print(f"   Question Attempts: {len(student_data['question_attempts'])}")
        correct = sum(1 for a in student_data['question_attempts'] if a.is_correct)
        total = len(student_data['question_attempts'])
        print(f"   Accuracy: {correct}/{total} ({correct/total:.1%})" if total > 0 else "   No attempts")

        print(f"   Content Interactions: {len(student_data['content_interactions'])}")
        print(f"   Chatbot Queries: {len(student_data['chatbot_interactions'])}")
        print(f"   Learning Sessions: {len(student_data['sessions'])}")

        # Show sample interactions
        if student_data['question_attempts']:
            print(f"\n   Recent Question Attempts:")
            for attempt in student_data['question_attempts'][-3:]:
                question = self.data['questions'].get(attempt.question_id)
                topic = question.topic if question else "Unknown"
                result = "Correct" if attempt.is_correct else "Wrong"
                print(f"     - {topic}: {result} ({attempt.time_spent_seconds}s)")

        if student_data['chatbot_interactions']:
            print(f"\n   Recent Chatbot Interactions:")
            for chat in student_data['chatbot_interactions'][-3:]:
                print(f"     - [{chat.query_type}] {chat.query[:50]}...")
                print(f"       Sentiment: {chat.sentiment}")

        # Show knowledge graph for student's concepts
        print(f"\n4. CONCEPT RELATIONSHIPS")
        print("-" * 40)
        practiced_concepts = list(knowledge_state.concept_mastery.keys())[:3]
        for concept_id in practiced_concepts:
            concept = self.data['concepts'].get(concept_id)
            if concept:
                prereqs = self.data['knowledge_graph'].get_prerequisites(concept_id)
                related = self.data['knowledge_graph'].get_related(concept_id)
                print(f"   {concept.name}:")
                print(f"     Prerequisites: {len(prereqs)}")
                print(f"     Related Concepts: {len(related)}")

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)

    def get_statistics(self) -> Dict:
        """Get overall statistics about the loaded data."""
        return {
            'num_concepts': len(self.data['concepts']),
            'num_questions': len(self.data['questions']),
            'num_students': len(self.data['student_profiles']),
            'num_question_attempts': len(self.data['question_attempts']),
            'num_content_interactions': len(self.data['content_interactions']),
            'num_chatbot_interactions': len(self.data['chatbot_interactions']),
            'num_sessions': len(self.data['learning_sessions']),
            'num_graph_edges': len(self.data['knowledge_graph'].edges)
        }


def main():
    """Main entry point for running simulations."""
    print("Initializing Knowledge Tracing Simulation...")
    print("Using ASSISTments 2009 dataset with synthetic content/chatbot data\n")

    # Use smaller sample for quick demo
    runner = SimulationRunner(sample_size=10000)

    # Print data statistics
    stats = runner.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")

    # Run sample simulation
    runner.run_sample_simulation()

    return runner


if __name__ == "__main__":
    runner = main()
