"""
Knowledge Tracing System - Main Orchestration Service

A comprehensive personalized learning system that combines:
- Deep Learning Knowledge Tracing (DLKT)
- Multi-modal interaction analysis
- LLM-based chatbot integration
- Personalized content recommendation

Based on: "Deep Learning Based Knowledge Tracing: A Review of the Literature"
by Siyi Zhu and Wenjie Chen, ICBDIE 2025
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for the Knowledge Tracing System"""
    # Model dimensions
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_gnn_layers: int = 3

    # Knowledge structure
    num_concepts: int = 100
    num_questions: int = 1000
    memory_size: int = 50

    # Learning preferences
    num_medium_types: int = 5
    max_seq_len: int = 200

    # Training
    learning_rate: float = 0.001
    dropout: float = 0.2
    batch_size: int = 32

    # Thresholds
    mastery_threshold: float = 0.8
    forgetting_threshold_hours: float = 168  # 1 week


class KnowledgeTracingSystem:
    """
    Main orchestration class for the personalized learning system.

    Integrates:
    1. Hybrid Knowledge State Engine (GNN + Attention + Memory)
    2. Learning Preference Analyzer
    3. Content Recommendation Engine
    4. LLM Chatbot Analyzer
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Initializing Knowledge Tracing System on {self.device}")

        # Initialize components
        self._init_knowledge_model()
        self._init_preference_analyzer()
        self._init_recommendation_engine()
        self._init_chatbot_analyzer()

        # Knowledge graph (would be loaded from database in production)
        self.knowledge_graph = {}

        # Student state cache
        self.student_states: Dict[str, Dict[str, Any]] = {}

    def _init_knowledge_model(self):
        """Initialize the hybrid knowledge state model"""
        from models.knowledge_state import HybridKnowledgeStateEngine

        self.knowledge_model = HybridKnowledgeStateEngine(
            num_concepts=self.config.num_concepts,
            num_questions=self.config.num_questions,
            embedding_dim=self.config.embedding_dim,
            memory_size=self.config.memory_size,
            num_attention_heads=self.config.num_attention_heads,
            num_gnn_layers=self.config.num_gnn_layers,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        ).to(self.device)

        logger.info("Knowledge model initialized")

    def _init_preference_analyzer(self):
        """Initialize learning preference analyzer"""
        from recommendation.personalized_learning import LearningPreferenceAnalyzer

        self.preference_analyzer = LearningPreferenceAnalyzer(
            num_medium_types=self.config.num_medium_types,
            interaction_dim=self.config.embedding_dim // 2,
            hidden_dim=self.config.hidden_dim,
            num_layers=2
        ).to(self.device)

        logger.info("Preference analyzer initialized")

    def _init_recommendation_engine(self):
        """Initialize content recommendation engine"""
        from recommendation.personalized_learning import ContentRecommendationEngine

        self.recommendation_engine = ContentRecommendationEngine(
            knowledge_dim=self.config.embedding_dim,
            content_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            num_medium_types=self.config.num_medium_types
        ).to(self.device)

        logger.info("Recommendation engine initialized")

    def _init_chatbot_analyzer(self):
        """Initialize LLM chatbot analyzer"""
        from llm.chatbot_analyzer import ChatbotQueryAnalyzer, LLMKnowledgeEnhancer

        # Placeholder concept vocabulary
        concept_vocab = [f"concept_{i}" for i in range(self.config.num_concepts)]

        self.query_analyzer = ChatbotQueryAnalyzer(concept_vocab)
        self.llm_enhancer = LLMKnowledgeEnhancer(
            concept_dim=self.config.embedding_dim,
            query_dim=768,  # Standard BERT/LLM embedding size
            hidden_dim=self.config.hidden_dim,
            num_concepts=self.config.num_concepts
        ).to(self.device)

        logger.info("Chatbot analyzer initialized")

    def process_question_attempt(
        self,
        student_id: str,
        question_id: int,
        concept_ids: List[int],
        is_correct: bool,
        time_spent_seconds: int,
        difficulty: float,
        hints_used: int = 0
    ) -> Dict[str, Any]:
        """
        Process a question attempt and update knowledge state

        Args:
            student_id: Unique student identifier
            question_id: ID of the attempted question
            concept_ids: Concepts tested by the question
            is_correct: Whether the answer was correct
            time_spent_seconds: Time spent on the question
            difficulty: Question difficulty (0-1)
            hints_used: Number of hints used

        Returns:
            Updated knowledge state and recommendations
        """
        logger.info(f"Processing question attempt for student {student_id}")

        # Get or initialize student state
        state = self._get_student_state(student_id)

        # Add interaction to history
        interaction = {
            'type': 'question_attempt',
            'question_id': question_id,
            'concept_ids': concept_ids,
            'is_correct': is_correct,
            'time_spent': time_spent_seconds,
            'difficulty': difficulty,
            'hints_used': hints_used,
            'timestamp': datetime.now().isoformat()
        }
        state['interactions'].append(interaction)

        # Prepare tensors for model
        with torch.no_grad():
            # Build sequence from recent interactions
            recent = [i for i in state['interactions'][-self.config.max_seq_len:]
                     if i['type'] == 'question_attempt']

            if len(recent) < 2:
                # Not enough history, use simplified update
                return self._simplified_update(student_id, state, interaction)

            # Convert to tensors
            question_ids = torch.tensor([[i['question_id'] for i in recent]], device=self.device)
            concepts = torch.tensor([[i['concept_ids'][0] for i in recent]], device=self.device)
            responses = torch.tensor([[1 if i['is_correct'] else 0 for i in recent]], device=self.device)
            difficulties = torch.tensor([[[i['difficulty']] for i in recent]], device=self.device)
            time_gaps = torch.tensor([[[0.0] for _ in recent]], device=self.device)  # Simplified

            # Get adjacency matrix (identity for now)
            adj_matrix = torch.eye(self.config.num_concepts, device=self.device)
            edge_types = torch.zeros(self.config.num_concepts, self.config.num_concepts,
                                    dtype=torch.long, device=self.device)

            # Forward pass
            output = self.knowledge_model(
                question_ids=question_ids,
                concept_ids=concepts,
                responses=responses,
                difficulties=difficulties,
                time_gaps=time_gaps,
                adjacency_matrix=adj_matrix,
                edge_types=edge_types,
                target_question=torch.tensor([[question_id]], device=self.device),
                target_concept=torch.tensor([[concept_ids[0]]], device=self.device),
                value_memory=state.get('memory')
            )

            # Update state
            state['memory'] = output['updated_memory']
            state['knowledge_state'] = output['knowledge_state'].cpu()

            # Get mastery levels
            mastery_vector = self.knowledge_model.get_knowledge_state_vector(output['updated_memory'])
            state['concept_mastery'] = {
                f"concept_{i}": mastery_vector[0, i].item()
                for i in range(self.config.num_concepts)
            }

        # Save state
        self.student_states[student_id] = state

        # Generate response
        return {
            'student_id': student_id,
            'prediction_correct': output['prediction'].item() > 0.5,
            'predicted_probability': output['prediction'].item(),
            'updated_mastery': {
                f"concept_{cid}": state['concept_mastery'].get(f"concept_{cid}", 0.5)
                for cid in concept_ids
            },
            'attention_weights': output['attention_weights'][0].tolist() if output['attention_weights'] is not None else [],
            'recommendations': self._get_next_recommendations(student_id, state)
        }

    def process_content_interaction(
        self,
        student_id: str,
        content_id: str,
        concept_ids: List[int],
        medium: str,
        duration_seconds: int,
        completion_percentage: float,
        engagement_score: float
    ) -> Dict[str, Any]:
        """
        Process content viewing interaction (video, reading, etc.)

        Returns:
            Updated preferences and recommendations
        """
        logger.info(f"Processing content interaction for student {student_id}")

        state = self._get_student_state(student_id)

        # Add interaction
        interaction = {
            'type': 'content_view',
            'content_id': content_id,
            'concept_ids': concept_ids,
            'medium': medium,
            'duration_seconds': duration_seconds,
            'completion_percentage': completion_percentage,
            'engagement_score': engagement_score,
            'timestamp': datetime.now().isoformat()
        }
        state['interactions'].append(interaction)

        # Update medium preferences
        medium_counts = state.get('medium_counts', {})
        medium_scores = state.get('medium_scores', {})

        medium_counts[medium] = medium_counts.get(medium, 0) + 1
        medium_scores[medium] = medium_scores.get(medium, 0) + engagement_score * completion_percentage / 100

        state['medium_counts'] = medium_counts
        state['medium_scores'] = medium_scores

        # Calculate preferences
        total_count = sum(medium_counts.values())
        preferences = {
            m: (medium_scores.get(m, 0) / medium_counts.get(m, 1)) * (medium_counts.get(m, 0) / total_count)
            for m in medium_counts
        }

        # Normalize
        pref_sum = sum(preferences.values()) or 1
        state['medium_preferences'] = {m: p / pref_sum for m, p in preferences.items()}

        self.student_states[student_id] = state

        return {
            'student_id': student_id,
            'updated_preferences': state['medium_preferences'],
            'recommended_medium': max(state['medium_preferences'], key=state['medium_preferences'].get),
            'engagement_level': engagement_score,
            'learning_impact': self._estimate_learning_impact(completion_percentage, engagement_score)
        }

    def process_chatbot_query(
        self,
        student_id: str,
        query: str,
        query_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Process a chatbot query and extract knowledge signals

        Args:
            student_id: Student identifier
            query: Raw query text
            query_embedding: Optional pre-computed LLM embedding

        Returns:
            Query analysis and knowledge state updates
        """
        logger.info(f"Processing chatbot query for student {student_id}")

        state = self._get_student_state(student_id)

        # Analyze query text
        analysis = self.query_analyzer.analyze_query(query)

        # Add to interactions
        interaction = {
            'type': 'chatbot_query',
            'query': query,
            'query_type': analysis.query_type.value,
            'sentiment': analysis.sentiment.value,
            'concepts': analysis.concepts_mentioned,
            'understanding_estimate': analysis.understanding_level,
            'timestamp': datetime.now().isoformat()
        }
        state['interactions'].append(interaction)

        # Neural analysis if embedding provided
        neural_signals = None
        if query_embedding is not None:
            with torch.no_grad():
                concept_embeddings = self.knowledge_model.gnn_module.concept_embeddings.weight
                knowledge_state = state.get('knowledge_state', torch.zeros(1, self.config.embedding_dim))

                neural_signals = self.llm_enhancer(
                    query_embedding.unsqueeze(0).to(self.device) if query_embedding.dim() == 1 else query_embedding.to(self.device),
                    concept_embeddings,
                    knowledge_state.to(self.device)
                )

        self.student_states[student_id] = state

        response = {
            'student_id': student_id,
            'query_analysis': {
                'type': analysis.query_type.value,
                'sentiment': analysis.sentiment.value,
                'concepts_mentioned': analysis.concepts_mentioned,
                'complexity': analysis.complexity_level,
                'understanding_estimate': analysis.understanding_level
            },
            'misconceptions_detected': analysis.misconceptions,
            'knowledge_gaps': analysis.knowledge_gaps,
            'suggested_response_type': analysis.suggested_response_type,
            'follow_up_recommended': analysis.follow_up_recommended
        }

        if neural_signals:
            response['neural_understanding'] = neural_signals['understanding_level'].item()

        return response

    def get_student_dashboard(self, student_id: str) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for a student

        Returns:
            Complete learning analytics and recommendations
        """
        state = self._get_student_state(student_id)

        # Calculate various metrics
        interactions = state['interactions']
        question_attempts = [i for i in interactions if i['type'] == 'question_attempt']
        content_views = [i for i in interactions if i['type'] == 'content_view']
        chatbot_queries = [i for i in interactions if i['type'] == 'chatbot_query']

        # Performance metrics
        if question_attempts:
            recent_attempts = question_attempts[-20:]
            accuracy = sum(1 for a in recent_attempts if a['is_correct']) / len(recent_attempts)
            avg_time = sum(a['time_spent'] for a in recent_attempts) / len(recent_attempts)
        else:
            accuracy = 0.5
            avg_time = 0

        # Learning curve (accuracy over time)
        learning_curve = []
        window_size = 10
        for i in range(0, len(question_attempts), window_size):
            window = question_attempts[i:i+window_size]
            if window:
                learning_curve.append({
                    'period': i // window_size,
                    'accuracy': sum(1 for a in window if a['is_correct']) / len(window),
                    'count': len(window)
                })

        # Concept mastery
        mastery = state.get('concept_mastery', {})
        mastered_concepts = [c for c, m in mastery.items() if m >= self.config.mastery_threshold]
        struggling_concepts = [c for c, m in mastery.items() if m < 0.4]

        return {
            'student_id': student_id,
            'summary': {
                'total_questions_attempted': len(question_attempts),
                'total_content_viewed': len(content_views),
                'total_chatbot_queries': len(chatbot_queries),
                'overall_accuracy': accuracy,
                'average_response_time_seconds': avg_time
            },
            'concept_mastery': mastery,
            'mastered_concepts': mastered_concepts,
            'struggling_concepts': struggling_concepts,
            'learning_curve': learning_curve,
            'medium_preferences': state.get('medium_preferences', {}),
            'learning_style': self._infer_learning_style(state),
            'recommendations': self._get_next_recommendations(student_id, state),
            'alerts': self._generate_alerts(state, accuracy)
        }

    def _get_student_state(self, student_id: str) -> Dict[str, Any]:
        """Get or initialize student state"""
        if student_id not in self.student_states:
            self.student_states[student_id] = {
                'student_id': student_id,
                'created_at': datetime.now().isoformat(),
                'interactions': [],
                'memory': None,
                'knowledge_state': None,
                'concept_mastery': {},
                'medium_preferences': {},
                'medium_counts': {},
                'medium_scores': {}
            }
        return self.student_states[student_id]

    def _simplified_update(
        self,
        student_id: str,
        state: Dict[str, Any],
        interaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplified update for cold start"""
        # Simple ELO-like update
        for cid in interaction['concept_ids']:
            key = f"concept_{cid}"
            current = state['concept_mastery'].get(key, 0.5)
            if interaction['is_correct']:
                new_mastery = current + 0.1 * (1 - current)
            else:
                new_mastery = current - 0.1 * current
            state['concept_mastery'][key] = max(0, min(1, new_mastery))

        return {
            'student_id': student_id,
            'prediction_correct': None,
            'predicted_probability': 0.5,
            'updated_mastery': {
                f"concept_{cid}": state['concept_mastery'].get(f"concept_{cid}", 0.5)
                for cid in interaction['concept_ids']
            },
            'recommendations': self._get_next_recommendations(student_id, state),
            'note': 'Simplified update due to limited history'
        }

    def _get_next_recommendations(
        self,
        student_id: str,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        mastery = state.get('concept_mastery', {})
        preferences = state.get('medium_preferences', {})

        # Find concepts to focus on
        struggling = sorted(
            [(c, m) for c, m in mastery.items() if m < 0.6],
            key=lambda x: x[1]
        )[:5]

        # Find concepts ready for advancement
        ready_for_challenge = sorted(
            [(c, m) for c, m in mastery.items() if 0.6 <= m < 0.9],
            key=lambda x: -x[1]
        )[:3]

        # Recommend medium
        recommended_medium = max(preferences, key=preferences.get) if preferences else 'reading'

        return {
            'priority_concepts': [c for c, _ in struggling],
            'challenge_concepts': [c for c, _ in ready_for_challenge],
            'recommended_medium': recommended_medium,
            'suggested_activities': self._suggest_activities(struggling, preferences)
        }

    def _suggest_activities(
        self,
        struggling_concepts: List[tuple],
        preferences: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate suggested learning activities"""
        activities = []

        for concept, mastery in struggling_concepts[:3]:
            if mastery < 0.3:
                activity_type = 'introduction'
            elif mastery < 0.5:
                activity_type = 'practice'
            else:
                activity_type = 'reinforcement'

            preferred_medium = max(preferences, key=preferences.get) if preferences else 'reading'

            activities.append({
                'concept': concept,
                'activity_type': activity_type,
                'medium': preferred_medium,
                'estimated_time_minutes': 15 if activity_type == 'introduction' else 10
            })

        return activities

    def _infer_learning_style(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Infer learning style from behavior"""
        interactions = state['interactions']

        # Analyze patterns
        question_times = [i['time_spent'] for i in interactions
                        if i['type'] == 'question_attempt' and 'time_spent' in i]
        avg_time = sum(question_times) / len(question_times) if question_times else 60

        # Infer pace
        if avg_time < 30:
            pace = 'fast'
        elif avg_time > 90:
            pace = 'careful'
        else:
            pace = 'moderate'

        # Engagement pattern
        content_views = [i for i in interactions if i['type'] == 'content_view']
        if content_views:
            avg_completion = sum(i.get('completion_percentage', 50) for i in content_views) / len(content_views)
            engagement = 'high' if avg_completion > 80 else 'moderate' if avg_completion > 50 else 'low'
        else:
            engagement = 'unknown'

        return {
            'learning_pace': pace,
            'engagement_level': engagement,
            'preferred_medium': max(state.get('medium_preferences', {'reading': 1}),
                                   key=state.get('medium_preferences', {'reading': 1}).get),
            'hint_usage': 'frequent' if any(i.get('hints_used', 0) > 0 for i in interactions) else 'rare'
        }

    def _estimate_learning_impact(
        self,
        completion: float,
        engagement: float
    ) -> float:
        """Estimate learning impact of content interaction"""
        # Simple model: impact = completion * engagement factor
        return (completion / 100) * (0.5 + 0.5 * engagement)

    def _generate_alerts(
        self,
        state: Dict[str, Any],
        accuracy: float
    ) -> List[Dict[str, str]]:
        """Generate alerts for student or instructor"""
        alerts = []

        if accuracy < 0.4:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'message': 'Student accuracy has dropped below 40%. Consider reviewing fundamentals.'
            })

        # Check for frustration signals
        recent_queries = [i for i in state['interactions'][-10:]
                        if i['type'] == 'chatbot_query']
        frustrated = sum(1 for q in recent_queries if q.get('sentiment') == 'frustrated')
        if frustrated >= 2:
            alerts.append({
                'type': 'engagement',
                'severity': 'medium',
                'message': 'Student showing signs of frustration. Consider simplifying content.'
            })

        return alerts

    def save_state(self, filepath: str):
        """Save system state to file"""
        # Convert tensors to lists for JSON serialization
        serializable_states = {}
        for student_id, state in self.student_states.items():
            s = dict(state)
            s['memory'] = None  # Don't save tensor
            s['knowledge_state'] = None
            serializable_states[student_id] = s

        with open(filepath, 'w') as f:
            json.dump(serializable_states, f, indent=2)
        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str):
        """Load system state from file"""
        with open(filepath, 'r') as f:
            self.student_states = json.load(f)
        logger.info(f"State loaded from {filepath}")


def main():
    """Demo usage of the Knowledge Tracing System"""
    # Initialize system
    config = SystemConfig(
        num_concepts=50,
        num_questions=500,
        embedding_dim=64,  # Smaller for demo
        hidden_dim=128
    )

    system = KnowledgeTracingSystem(config)

    # Simulate student interactions
    student_id = "student_001"

    # Question attempts
    print("\n=== Processing Question Attempts ===")
    for i in range(10):
        result = system.process_question_attempt(
            student_id=student_id,
            question_id=i,
            concept_ids=[i % 10],
            is_correct=(i % 3 != 0),  # Some wrong answers
            time_spent_seconds=30 + (i * 5),
            difficulty=0.3 + (i * 0.05),
            hints_used=1 if i % 4 == 0 else 0
        )
        print(f"Question {i}: Correct={i % 3 != 0}, Mastery={result['updated_mastery']}")

    # Content interaction
    print("\n=== Processing Content Interaction ===")
    result = system.process_content_interaction(
        student_id=student_id,
        content_id="video_001",
        concept_ids=[3, 4],
        medium="video",
        duration_seconds=600,
        completion_percentage=85,
        engagement_score=0.8
    )
    print(f"Updated preferences: {result['updated_preferences']}")

    # Chatbot query
    print("\n=== Processing Chatbot Query ===")
    result = system.process_chatbot_query(
        student_id=student_id,
        query="I don't understand how to solve quadratic equations. Can you explain?"
    )
    print(f"Query analysis: {result['query_analysis']}")
    print(f"Suggested response type: {result['suggested_response_type']}")

    # Get dashboard
    print("\n=== Student Dashboard ===")
    dashboard = system.get_student_dashboard(student_id)
    print(f"Overall accuracy: {dashboard['summary']['overall_accuracy']:.2%}")
    print(f"Mastered concepts: {dashboard['mastered_concepts']}")
    print(f"Struggling concepts: {dashboard['struggling_concepts']}")
    print(f"Learning style: {dashboard['learning_style']}")
    print(f"Recommendations: {dashboard['recommendations']}")


if __name__ == "__main__":
    main()
