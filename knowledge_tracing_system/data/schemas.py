"""
Data Schemas for Multi-Interaction Knowledge Tracing System

Captures all student interactions:
- Practice questions (difficulty, chapter, topic, subject)
- LLM chatbot queries
- Learning medium preferences (reading, videos, gamified)
- Engagement patterns
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import json


class LearningMedium(Enum):
    VIDEO = "video"
    READING = "reading"
    GAMIFIED = "gamified"
    INTERACTIVE = "interactive"
    AUDIO = "audio"
    PRACTICE = "practice"
    DISCUSSION = "discussion"


class DifficultyLevel(Enum):
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


class InteractionType(Enum):
    QUESTION_ATTEMPT = "question_attempt"
    CONTENT_VIEW = "content_view"
    CHATBOT_QUERY = "chatbot_query"
    HINT_REQUEST = "hint_request"
    SKIP = "skip"
    BOOKMARK = "bookmark"
    NOTE_TAKING = "note_taking"
    REVISIT = "revisit"
    ASSESSMENT = "assessment"


@dataclass
class KnowledgeConcept:
    """Represents a knowledge concept/topic"""
    concept_id: str
    name: str
    subject: str
    chapter: str
    topic: str
    difficulty_base: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    bloom_level: str = "understanding"  # remember, understand, apply, analyze, evaluate, create
    estimated_learning_time_minutes: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "name": self.name,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "difficulty_base": self.difficulty_base.value,
            "prerequisites": self.prerequisites,
            "related_concepts": self.related_concepts,
            "bloom_level": self.bloom_level,
            "estimated_learning_time_minutes": self.estimated_learning_time_minutes
        }


@dataclass
class Question:
    """Represents a practice question"""
    question_id: str
    concept_ids: List[str]  # Can test multiple concepts
    subject: str
    chapter: str
    topic: str
    difficulty: DifficultyLevel
    question_type: str  # mcq, fill_blank, short_answer, coding, etc.
    content: str
    options: Optional[List[str]] = None
    correct_answer: Any = None
    hints: List[str] = field(default_factory=list)
    explanation: str = ""
    discrimination_index: float = 0.5  # IRT parameter
    guessing_parameter: float = 0.25  # IRT parameter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "concept_ids": self.concept_ids,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "difficulty": self.difficulty.value,
            "question_type": self.question_type,
            "discrimination_index": self.discrimination_index,
            "guessing_parameter": self.guessing_parameter
        }


@dataclass
class QuestionAttempt:
    """Records a student's attempt at a question"""
    attempt_id: str
    student_id: str
    question_id: str
    timestamp: datetime
    response: Any
    is_correct: bool
    time_spent_seconds: int
    hints_used: int = 0
    attempts_count: int = 1  # Number of attempts on this question
    confidence_level: Optional[int] = None  # 1-5 self-reported
    device_type: str = "desktop"
    session_id: str = ""

    # Derived features for KT
    time_since_last_attempt_hours: float = 0.0
    time_since_last_concept_attempt_hours: float = 0.0
    streak_correct: int = 0
    streak_incorrect: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "student_id": self.student_id,
            "question_id": self.question_id,
            "timestamp": self.timestamp.isoformat(),
            "is_correct": self.is_correct,
            "time_spent_seconds": self.time_spent_seconds,
            "hints_used": self.hints_used,
            "attempts_count": self.attempts_count,
            "confidence_level": self.confidence_level,
            "device_type": self.device_type,
            "time_since_last_attempt_hours": self.time_since_last_attempt_hours,
            "streak_correct": self.streak_correct,
            "streak_incorrect": self.streak_incorrect
        }


@dataclass
class ContentInteraction:
    """Records interaction with learning content"""
    interaction_id: str
    student_id: str
    content_id: str
    concept_ids: List[str]
    medium: LearningMedium
    timestamp: datetime
    duration_seconds: int
    completion_percentage: float  # 0-100
    engagement_actions: Dict[str, int] = field(default_factory=dict)
    # e.g., {"pauses": 3, "rewinds": 2, "speed_changes": 1, "highlights": 5}

    # Engagement metrics
    scroll_depth: float = 0.0  # For reading
    video_completion_points: List[float] = field(default_factory=list)  # Timestamps watched
    quiz_scores_during: List[float] = field(default_factory=list)  # Embedded quizzes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "student_id": self.student_id,
            "content_id": self.content_id,
            "concept_ids": self.concept_ids,
            "medium": self.medium.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "completion_percentage": self.completion_percentage,
            "engagement_actions": self.engagement_actions
        }


@dataclass
class ChatbotInteraction:
    """Records interaction with LLM-based chatbot"""
    interaction_id: str
    student_id: str
    session_id: str
    timestamp: datetime
    query: str
    response: str
    concept_ids_detected: List[str]  # Concepts mentioned in query

    # Query classification
    query_type: str = "explanation"  # explanation, clarification, example, practice, off_topic
    complexity_level: int = 1  # 1-5 based on query sophistication
    sentiment: str = "neutral"  # frustrated, confused, curious, confident
    follow_up_count: int = 0  # How many follow-ups in conversation

    # LLM-extracted signals
    misconceptions_detected: List[str] = field(default_factory=list)
    knowledge_gaps_indicated: List[str] = field(default_factory=list)
    understanding_indicators: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "student_id": self.student_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "query": self.query,
            "concept_ids_detected": self.concept_ids_detected,
            "query_type": self.query_type,
            "complexity_level": self.complexity_level,
            "sentiment": self.sentiment,
            "misconceptions_detected": self.misconceptions_detected,
            "knowledge_gaps_indicated": self.knowledge_gaps_indicated
        }


@dataclass
class LearningSession:
    """Aggregated session data"""
    session_id: str
    student_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    device_type: str = "desktop"
    platform: str = "web"

    # Aggregated metrics
    total_questions_attempted: int = 0
    questions_correct: int = 0
    content_pieces_viewed: int = 0
    chatbot_queries: int = 0
    total_time_seconds: int = 0
    concepts_touched: List[str] = field(default_factory=list)

    # Session quality metrics
    focus_score: float = 0.0  # Based on tab switches, idle time
    engagement_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "student_id": self.student_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "device_type": self.device_type,
            "total_questions_attempted": self.total_questions_attempted,
            "questions_correct": self.questions_correct,
            "content_pieces_viewed": self.content_pieces_viewed,
            "total_time_seconds": self.total_time_seconds,
            "concepts_touched": self.concepts_touched
        }


@dataclass
class StudentProfile:
    """Comprehensive student learning profile"""
    student_id: str
    created_at: datetime

    # Learning style preferences (learned over time)
    preferred_medium: Dict[LearningMedium, float] = field(default_factory=dict)
    # e.g., {VIDEO: 0.4, READING: 0.3, GAMIFIED: 0.3}

    optimal_session_length_minutes: int = 30
    best_learning_times: List[int] = field(default_factory=list)  # Hours of day
    learning_pace: str = "medium"  # slow, medium, fast

    # Cognitive profile
    working_memory_estimate: float = 0.5  # 0-1
    attention_span_estimate: float = 0.5  # 0-1
    preferred_difficulty_range: tuple = (2, 4)  # Min, max difficulty

    # Historical performance
    overall_mastery: float = 0.0  # 0-1
    concept_mastery: Dict[str, float] = field(default_factory=dict)
    subject_strengths: Dict[str, float] = field(default_factory=dict)
    subject_weaknesses: Dict[str, float] = field(default_factory=dict)

    # Behavioral patterns
    avg_session_length_minutes: float = 0.0
    total_learning_time_hours: float = 0.0
    consistency_score: float = 0.0  # Regular study pattern
    completion_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "created_at": self.created_at.isoformat(),
            "preferred_medium": {k.value: v for k, v in self.preferred_medium.items()},
            "optimal_session_length_minutes": self.optimal_session_length_minutes,
            "learning_pace": self.learning_pace,
            "overall_mastery": self.overall_mastery,
            "concept_mastery": self.concept_mastery,
            "subject_strengths": self.subject_strengths,
            "subject_weaknesses": self.subject_weaknesses
        }


@dataclass
class KnowledgeState:
    """Current knowledge state for a student (output of KT model)"""
    student_id: str
    timestamp: datetime

    # Per-concept mastery probabilities
    concept_mastery: Dict[str, float] = field(default_factory=dict)

    # Temporal dynamics
    concepts_improving: List[str] = field(default_factory=list)
    concepts_declining: List[str] = field(default_factory=list)  # Forgetting
    concepts_stable: List[str] = field(default_factory=list)

    # Readiness indicators
    ready_for_assessment: List[str] = field(default_factory=list)
    needs_review: List[str] = field(default_factory=list)
    next_recommended_concepts: List[str] = field(default_factory=list)

    # Confidence intervals
    mastery_confidence: Dict[str, tuple] = field(default_factory=dict)
    # e.g., {"concept_1": (0.7, 0.9)} for 80% CI

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "timestamp": self.timestamp.isoformat(),
            "concept_mastery": self.concept_mastery,
            "concepts_improving": self.concepts_improving,
            "concepts_declining": self.concepts_declining,
            "ready_for_assessment": self.ready_for_assessment,
            "needs_review": self.needs_review,
            "next_recommended_concepts": self.next_recommended_concepts
        }


@dataclass
class ContentRecommendation:
    """Personalized content recommendation"""
    student_id: str
    timestamp: datetime
    concept_id: str

    recommended_content_id: str
    medium: LearningMedium
    difficulty: DifficultyLevel
    estimated_duration_minutes: int

    # Reasoning
    recommendation_reason: str  # "knowledge_gap", "reinforcement", "advancement"
    confidence_score: float  # How confident the system is
    expected_mastery_gain: float

    # Alternative options
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "timestamp": self.timestamp.isoformat(),
            "concept_id": self.concept_id,
            "recommended_content_id": self.recommended_content_id,
            "medium": self.medium.value,
            "difficulty": self.difficulty.value,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "recommendation_reason": self.recommendation_reason,
            "confidence_score": self.confidence_score,
            "expected_mastery_gain": self.expected_mastery_gain
        }


# Knowledge Graph Schema
@dataclass
class KnowledgeGraphEdge:
    """Edge in knowledge graph"""
    source_concept_id: str
    target_concept_id: str
    edge_type: str  # "prerequisite", "related", "similar_difficulty", "same_topic"
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_concept_id,
            "target": self.target_concept_id,
            "type": self.edge_type,
            "weight": self.weight
        }


@dataclass
class KnowledgeGraph:
    """Full knowledge graph structure"""
    concepts: Dict[str, KnowledgeConcept] = field(default_factory=dict)
    edges: List[KnowledgeGraphEdge] = field(default_factory=list)

    def add_concept(self, concept: KnowledgeConcept):
        self.concepts[concept.concept_id] = concept

    def add_edge(self, edge: KnowledgeGraphEdge):
        self.edges.append(edge)

    def get_prerequisites(self, concept_id: str) -> List[str]:
        return [e.source_concept_id for e in self.edges
                if e.target_concept_id == concept_id and e.edge_type == "prerequisite"]

    def get_related(self, concept_id: str) -> List[str]:
        related = []
        for e in self.edges:
            if e.edge_type == "related":
                if e.source_concept_id == concept_id:
                    related.append(e.target_concept_id)
                elif e.target_concept_id == concept_id:
                    related.append(e.source_concept_id)
        return related

    def to_adjacency_dict(self) -> Dict[str, List[str]]:
        adj = {cid: [] for cid in self.concepts}
        for edge in self.edges:
            adj[edge.source_concept_id].append(edge.target_concept_id)
        return adj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
            "edges": [e.to_dict() for e in self.edges]
        }
