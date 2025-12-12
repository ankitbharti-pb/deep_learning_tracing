"""
Personalized Learning Engine

Combines knowledge state with learning preferences to provide:
1. Optimal content recommendations
2. Adaptive difficulty calibration
3. Learning path optimization
4. Medium selection (video/reading/gamified)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class LearningMedium(Enum):
    VIDEO = "video"
    READING = "reading"
    GAMIFIED = "gamified"
    INTERACTIVE = "interactive"
    PRACTICE = "practice"


@dataclass
class LearnerProfile:
    """Comprehensive learner profile built from interactions"""
    student_id: str

    # Learning style preferences (normalized probabilities)
    medium_preferences: Dict[LearningMedium, float]

    # Temporal patterns
    optimal_session_duration_minutes: int
    peak_learning_hours: List[int]  # Hours of day (0-23)
    avg_time_between_sessions_hours: float

    # Cognitive characteristics
    learning_speed: float  # 0.5-2.0 (1.0 = average)
    retention_rate: float  # 0-1 (how well knowledge is retained)
    attention_span_factor: float  # 0.5-1.5

    # Performance patterns
    best_performing_difficulty: int  # 1-5
    struggle_threshold: float  # Accuracy below which student struggles

    # Engagement patterns
    prefers_hints: bool
    revisit_frequency: float  # How often they revisit content
    gamification_response: float  # 0-1 (how well they respond to gamification)


class LearningPreferenceAnalyzer(nn.Module):
    """
    Neural network to analyze and predict learning preferences from interaction history.
    Updates student profile based on observed behaviors.
    """

    def __init__(
        self,
        num_medium_types: int = 5,
        interaction_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_medium_types = num_medium_types

        # Interaction encoder (processes sequence of interactions)
        self.interaction_encoder = nn.LSTM(
            input_size=interaction_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Medium preference predictor
        self.medium_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_medium_types),
            nn.Softmax(dim=-1)
        )

        # Learning speed estimator
        self.speed_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output 0-1, scaled to 0.5-2.0
        )

        # Retention rate estimator (based on forgetting patterns)
        self.retention_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optimal difficulty predictor
        self.difficulty_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # 5 difficulty levels
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        interaction_sequence: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze interaction sequence to predict learning preferences

        Args:
            interaction_sequence: [batch, seq_len, interaction_dim]
            sequence_lengths: Actual lengths of sequences

        Returns:
            Dictionary of predicted preferences
        """
        # Encode interactions
        packed = nn.utils.rnn.pack_padded_sequence(
            interaction_sequence, sequence_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.interaction_encoder(packed)

        # Combine bidirectional hidden states
        hidden_combined = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        # Predict preferences
        medium_prefs = self.medium_predictor(hidden_combined)
        learning_speed = self.speed_estimator(hidden_combined) * 1.5 + 0.5  # Scale to 0.5-2.0
        retention_rate = self.retention_estimator(hidden_combined)
        difficulty_prefs = self.difficulty_predictor(hidden_combined)

        return {
            'medium_preferences': medium_prefs,
            'learning_speed': learning_speed.squeeze(-1),
            'retention_rate': retention_rate.squeeze(-1),
            'difficulty_preferences': difficulty_prefs
        }


class ZoneOfProximalDevelopment:
    """
    Implements Vygotsky's Zone of Proximal Development (ZPD) for difficulty calibration.
    Finds the optimal difficulty where learning is challenging but achievable.
    """

    def __init__(
        self,
        target_success_rate: float = 0.7,  # Optimal challenge level
        min_success_rate: float = 0.5,
        max_success_rate: float = 0.9
    ):
        self.target_success_rate = target_success_rate
        self.min_success_rate = min_success_rate
        self.max_success_rate = max_success_rate

    def calculate_zpd_difficulty(
        self,
        current_mastery: float,
        recent_accuracy: float,
        learning_speed: float
    ) -> Tuple[float, float]:
        """
        Calculate optimal difficulty range for student

        Args:
            current_mastery: Current concept mastery (0-1)
            recent_accuracy: Recent performance accuracy (0-1)
            learning_speed: Student's learning speed factor

        Returns:
            (min_difficulty, max_difficulty) as floats from 0-1
        """
        # Base difficulty from mastery
        base_difficulty = current_mastery

        # Adjust based on recent performance
        if recent_accuracy > self.max_success_rate:
            # Too easy, increase difficulty
            adjustment = (recent_accuracy - self.target_success_rate) * 0.3
        elif recent_accuracy < self.min_success_rate:
            # Too hard, decrease difficulty
            adjustment = (recent_accuracy - self.target_success_rate) * 0.3
        else:
            adjustment = 0

        # Factor in learning speed
        speed_factor = (learning_speed - 1.0) * 0.1

        optimal_difficulty = base_difficulty + adjustment + speed_factor
        optimal_difficulty = np.clip(optimal_difficulty, 0.1, 0.95)

        # Calculate range around optimal
        range_width = 0.15 / learning_speed  # Faster learners get wider range

        return (
            max(0.05, optimal_difficulty - range_width),
            min(0.98, optimal_difficulty + range_width)
        )


class ContentRecommendationEngine(nn.Module):
    """
    Neural content recommendation engine that considers:
    - Current knowledge state
    - Learning preferences
    - Available content
    - Learning objectives
    """

    def __init__(
        self,
        knowledge_dim: int = 128,
        content_dim: int = 128,
        hidden_dim: int = 256,
        num_medium_types: int = 5
    ):
        super().__init__()
        self.knowledge_dim = knowledge_dim
        self.content_dim = content_dim

        # Knowledge state encoder
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(knowledge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Content encoder
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Medium preference integration
        self.medium_embedding = nn.Embedding(num_medium_types, hidden_dim)

        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Expected learning gain predictor
        self.gain_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        knowledge_state: torch.Tensor,
        content_features: torch.Tensor,
        medium_types: torch.Tensor,
        medium_preferences: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Score content items for recommendation

        Args:
            knowledge_state: Student's knowledge state [batch, knowledge_dim]
            content_features: Features of candidate content [batch, num_content, content_dim]
            medium_types: Medium type of each content [batch, num_content]
            medium_preferences: Student's medium preferences [batch, num_medium_types]

        Returns:
            Scores and expected gains for each content item
        """
        batch_size, num_content, _ = content_features.shape

        # Encode knowledge state
        knowledge_encoded = self.knowledge_encoder(knowledge_state)  # [batch, hidden]
        knowledge_expanded = knowledge_encoded.unsqueeze(1).expand(-1, num_content, -1)

        # Encode content
        content_encoded = self.content_encoder(content_features)  # [batch, num_content, hidden]

        # Get medium embeddings and weight by preferences
        medium_emb = self.medium_embedding(medium_types)  # [batch, num_content, hidden]

        # Weight medium embeddings by student preferences
        medium_weights = torch.gather(
            medium_preferences.unsqueeze(1).expand(-1, num_content, -1),
            dim=2,
            index=medium_types.unsqueeze(-1)
        )
        weighted_medium = medium_emb * medium_weights

        # Combine all features
        combined = torch.cat([
            knowledge_expanded,
            content_encoded,
            weighted_medium
        ], dim=-1)

        # Score each content
        scores = self.scorer(combined).squeeze(-1)  # [batch, num_content]

        # Predict expected learning gain
        gain_input = torch.cat([knowledge_expanded, content_encoded], dim=-1)
        expected_gains = self.gain_predictor(gain_input).squeeze(-1)

        # Final ranking combines score and gain
        final_scores = scores + 0.3 * expected_gains

        return {
            'scores': final_scores,
            'expected_gains': expected_gains,
            'base_scores': scores
        }


class LearningPathOptimizer:
    """
    Optimizes learning path based on:
    - Prerequisite relationships
    - Current knowledge state
    - Learning objectives
    - Time constraints
    """

    def __init__(self, knowledge_graph: Dict[str, List[str]]):
        """
        Args:
            knowledge_graph: Adjacency list of concept prerequisites
        """
        self.knowledge_graph = knowledge_graph
        self.concepts = list(knowledge_graph.keys())

    def topological_sort(self, target_concepts: List[str]) -> List[str]:
        """Get concepts in valid learning order"""
        # Get all prerequisites recursively
        to_learn = set()
        stack = list(target_concepts)

        while stack:
            concept = stack.pop()
            if concept not in to_learn:
                to_learn.add(concept)
                prerequisites = self.knowledge_graph.get(concept, [])
                stack.extend(prerequisites)

        # Topological sort
        visited = set()
        order = []

        def dfs(concept):
            if concept in visited:
                return
            visited.add(concept)
            for prereq in self.knowledge_graph.get(concept, []):
                if prereq in to_learn:
                    dfs(prereq)
            order.append(concept)

        for concept in to_learn:
            dfs(concept)

        return order

    def optimize_path(
        self,
        target_concepts: List[str],
        current_mastery: Dict[str, float],
        available_time_hours: float,
        learning_speed: float,
        avg_concept_time_hours: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Create optimized learning path

        Args:
            target_concepts: Concepts student wants to learn
            current_mastery: Current mastery levels
            available_time_hours: Time budget
            learning_speed: Student's learning speed
            avg_concept_time_hours: Average time per concept

        Returns:
            Ordered list of concepts with recommendations
        """
        # Get valid learning order
        ordered_concepts = self.topological_sort(target_concepts)

        # Filter out mastered concepts (>0.8 mastery)
        concepts_to_learn = [
            c for c in ordered_concepts
            if current_mastery.get(c, 0) < 0.8
        ]

        # Calculate adjusted time per concept
        adjusted_time = avg_concept_time_hours / learning_speed

        # Build path within time budget
        path = []
        total_time = 0

        for concept in concepts_to_learn:
            mastery = current_mastery.get(concept, 0)

            # Estimate time needed based on current mastery
            if mastery < 0.3:
                time_needed = adjusted_time * 1.2  # More time for new concepts
            elif mastery < 0.6:
                time_needed = adjusted_time * 0.8  # Review
            else:
                time_needed = adjusted_time * 0.5  # Quick brush-up

            if total_time + time_needed <= available_time_hours:
                path.append({
                    'concept': concept,
                    'current_mastery': mastery,
                    'estimated_time_hours': time_needed,
                    'priority': 'high' if concept in target_concepts else 'prerequisite',
                    'recommended_activities': self._get_activities(mastery)
                })
                total_time += time_needed

        return path

    def _get_activities(self, mastery: float) -> List[str]:
        """Recommend activities based on mastery level"""
        if mastery < 0.3:
            return ['introduction_video', 'reading_material', 'basic_practice']
        elif mastery < 0.6:
            return ['practice_questions', 'interactive_examples', 'review_video']
        elif mastery < 0.8:
            return ['advanced_practice', 'application_problems', 'peer_discussion']
        else:
            return ['challenge_problems', 'teaching_others', 'real_world_application']


class AdaptiveLearningSystem:
    """
    Main system that combines all components for adaptive personalized learning
    """

    def __init__(
        self,
        knowledge_state_model: nn.Module,
        preference_analyzer: LearningPreferenceAnalyzer,
        recommendation_engine: ContentRecommendationEngine,
        knowledge_graph: Dict[str, List[str]]
    ):
        self.knowledge_state_model = knowledge_state_model
        self.preference_analyzer = preference_analyzer
        self.recommendation_engine = recommendation_engine
        self.path_optimizer = LearningPathOptimizer(knowledge_graph)
        self.zpd_calculator = ZoneOfProximalDevelopment()

    def get_personalized_recommendation(
        self,
        student_id: str,
        interaction_history: torch.Tensor,
        knowledge_state: torch.Tensor,
        target_concepts: List[str],
        available_content: Dict[str, torch.Tensor],
        available_time_hours: float = 2.0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive personalized learning recommendation

        Returns:
            Full recommendation including content, path, and settings
        """
        # Analyze learning preferences
        preferences = self.preference_analyzer(
            interaction_history,
            torch.tensor([interaction_history.shape[1]])
        )

        # Get learning speed
        learning_speed = preferences['learning_speed'].item()
        retention_rate = preferences['retention_rate'].item()

        # Calculate recent accuracy (from last 20 interactions)
        recent_correct = interaction_history[0, -20:, 0].mean().item()  # Assuming first dim is correctness

        # Get ZPD difficulty range
        avg_mastery = knowledge_state.mean().item()
        min_diff, max_diff = self.zpd_calculator.calculate_zpd_difficulty(
            avg_mastery, recent_correct, learning_speed
        )

        # Optimize learning path
        # Convert knowledge state tensor to dict
        mastery_dict = {f"concept_{i}": knowledge_state[0, i].item()
                       for i in range(knowledge_state.shape[1])}

        learning_path = self.path_optimizer.optimize_path(
            target_concepts=target_concepts,
            current_mastery=mastery_dict,
            available_time_hours=available_time_hours,
            learning_speed=learning_speed
        )

        # Get content recommendations for first concept in path
        if learning_path:
            first_concept = learning_path[0]['concept']
            # Get content features for this concept
            content_features = available_content.get('features', torch.randn(1, 10, 128))
            medium_types = available_content.get('mediums', torch.randint(0, 5, (1, 10)))

            content_scores = self.recommendation_engine(
                knowledge_state,
                content_features,
                medium_types,
                preferences['medium_preferences']
            )

            # Get top recommendations
            top_indices = torch.topk(content_scores['scores'], k=3, dim=-1).indices[0]
            top_contents = top_indices.tolist()
        else:
            top_contents = []

        return {
            'student_id': student_id,
            'timestamp': datetime.now().isoformat(),

            # Learning profile
            'learning_profile': {
                'learning_speed': learning_speed,
                'retention_rate': retention_rate,
                'medium_preferences': {
                    m.value: preferences['medium_preferences'][0, i].item()
                    for i, m in enumerate(LearningMedium)
                },
                'optimal_difficulty_range': (min_diff, max_diff)
            },

            # Learning path
            'learning_path': learning_path,

            # Content recommendations
            'recommended_content': {
                'content_ids': top_contents,
                'expected_gains': content_scores['expected_gains'][0, top_indices].tolist() if top_contents else []
            },

            # Session settings
            'session_settings': {
                'recommended_duration_minutes': int(30 / learning_speed * 1.5),
                'break_interval_minutes': int(25 / learning_speed),
                'questions_per_set': int(10 * learning_speed)
            },

            # Alerts
            'alerts': self._generate_alerts(retention_rate, recent_correct, mastery_dict)
        }

    def _generate_alerts(
        self,
        retention_rate: float,
        recent_accuracy: float,
        mastery_dict: Dict[str, float]
    ) -> List[Dict[str, str]]:
        """Generate alerts for student or instructor"""
        alerts = []

        if retention_rate < 0.4:
            alerts.append({
                'type': 'retention',
                'severity': 'warning',
                'message': 'Student may benefit from more frequent review sessions'
            })

        if recent_accuracy < 0.5:
            alerts.append({
                'type': 'difficulty',
                'severity': 'warning',
                'message': 'Recent performance suggests content may be too challenging'
            })

        declining_concepts = [c for c, m in mastery_dict.items() if m < 0.3]
        if len(declining_concepts) > 3:
            alerts.append({
                'type': 'knowledge_gap',
                'severity': 'info',
                'message': f'Multiple concepts need attention: {len(declining_concepts)} concepts below 30% mastery'
            })

        return alerts


# Example configuration and usage
def create_adaptive_learning_system(
    num_concepts: int,
    knowledge_graph: Dict[str, List[str]]
) -> AdaptiveLearningSystem:
    """Factory function to create the full adaptive learning system"""

    # Import the knowledge state model
    from models.knowledge_state import HybridKnowledgeStateEngine

    # Initialize components
    knowledge_model = HybridKnowledgeStateEngine(
        num_concepts=num_concepts,
        num_questions=num_concepts * 10,  # Assume 10 questions per concept
        embedding_dim=128
    )

    preference_analyzer = LearningPreferenceAnalyzer(
        num_medium_types=5,
        interaction_dim=64,
        hidden_dim=128
    )

    recommendation_engine = ContentRecommendationEngine(
        knowledge_dim=128,
        content_dim=128,
        hidden_dim=256
    )

    return AdaptiveLearningSystem(
        knowledge_state_model=knowledge_model,
        preference_analyzer=preference_analyzer,
        recommendation_engine=recommendation_engine,
        knowledge_graph=knowledge_graph
    )


if __name__ == "__main__":
    # Demo usage
    knowledge_graph = {
        "algebra_basics": [],
        "linear_equations": ["algebra_basics"],
        "quadratic_equations": ["linear_equations"],
        "polynomials": ["algebra_basics"],
        "factoring": ["polynomials"],
        "functions": ["linear_equations"],
        "graphing": ["functions", "linear_equations"]
    }

    system = create_adaptive_learning_system(
        num_concepts=7,
        knowledge_graph=knowledge_graph
    )

    # Simulate interaction history
    interaction_history = torch.randn(1, 50, 64)  # 50 interactions
    knowledge_state = torch.rand(1, 7)  # Random mastery levels

    recommendation = system.get_personalized_recommendation(
        student_id="student_123",
        interaction_history=interaction_history,
        knowledge_state=knowledge_state,
        target_concepts=["quadratic_equations", "graphing"],
        available_content={
            'features': torch.randn(1, 20, 128),
            'mediums': torch.randint(0, 5, (1, 20))
        },
        available_time_hours=3.0
    )

    print("Personalized Recommendation:")
    print(f"Learning Speed: {recommendation['learning_profile']['learning_speed']:.2f}")
    print(f"Optimal Difficulty: {recommendation['learning_profile']['optimal_difficulty_range']}")
    print(f"\nLearning Path:")
    for item in recommendation['learning_path']:
        print(f"  - {item['concept']} (mastery: {item['current_mastery']:.2f})")
