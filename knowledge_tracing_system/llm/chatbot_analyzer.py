"""
LLM-based Chatbot Interaction Analyzer

Inspired by LKT (Language model-based Knowledge Tracing) from the paper.
Uses LLM to:
1. Analyze student queries for knowledge signals
2. Detect misconceptions
3. Identify knowledge gaps
4. Enhance cold-start predictions
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re


class QueryType(Enum):
    EXPLANATION = "explanation"  # "What is X?"
    CLARIFICATION = "clarification"  # "I don't understand Y"
    EXAMPLE = "example"  # "Can you give an example of Z?"
    PRACTICE = "practice"  # "Can I try a problem?"
    COMPARISON = "comparison"  # "What's the difference between A and B?"
    APPLICATION = "application"  # "How do I use this for...?"
    DEBUGGING = "debugging"  # "Why is my answer wrong?"
    METACOGNITIVE = "metacognitive"  # "Am I understanding this correctly?"
    OFF_TOPIC = "off_topic"


class StudentSentiment(Enum):
    CONFIDENT = "confident"
    CURIOUS = "curious"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"


@dataclass
class QueryAnalysis:
    """Result of analyzing a student query"""
    query_type: QueryType
    sentiment: StudentSentiment
    concepts_mentioned: List[str]
    complexity_level: int  # 1-5
    misconceptions: List[str]
    knowledge_gaps: List[str]
    understanding_level: float  # 0-1
    follow_up_recommended: bool
    suggested_response_type: str


class ChatbotQueryAnalyzer:
    """
    Analyzes chatbot queries to extract knowledge tracing signals.
    Can be used with any LLM backend (OpenAI, Anthropic, local models).
    """

    def __init__(self, concept_vocabulary: List[str]):
        """
        Args:
            concept_vocabulary: List of known concepts for matching
        """
        self.concept_vocabulary = set(c.lower() for c in concept_vocabulary)
        self.concept_patterns = self._build_concept_patterns(concept_vocabulary)

    def _build_concept_patterns(self, concepts: List[str]) -> Dict[str, re.Pattern]:
        """Build regex patterns for concept detection"""
        patterns = {}
        for concept in concepts:
            # Create pattern that matches the concept with word boundaries
            pattern = re.compile(
                r'\b' + re.escape(concept.lower()) + r'\b',
                re.IGNORECASE
            )
            patterns[concept] = pattern
        return patterns

    def extract_concepts(self, text: str) -> List[str]:
        """Extract mentioned concepts from text"""
        found_concepts = []
        text_lower = text.lower()

        for concept, pattern in self.concept_patterns.items():
            if pattern.search(text_lower):
                found_concepts.append(concept)

        return found_concepts

    def classify_query_type(self, query: str) -> QueryType:
        """Classify the type of student query"""
        query_lower = query.lower()

        # Pattern matching for query types
        if any(p in query_lower for p in ["what is", "what are", "define", "explain"]):
            return QueryType.EXPLANATION
        elif any(p in query_lower for p in ["don't understand", "confused", "unclear", "lost"]):
            return QueryType.CLARIFICATION
        elif any(p in query_lower for p in ["example", "show me", "demonstrate"]):
            return QueryType.EXAMPLE
        elif any(p in query_lower for p in ["practice", "try", "quiz", "test me"]):
            return QueryType.PRACTICE
        elif any(p in query_lower for p in ["difference between", "compare", "versus", "vs"]):
            return QueryType.COMPARISON
        elif any(p in query_lower for p in ["how do i use", "apply", "real world", "practical"]):
            return QueryType.APPLICATION
        elif any(p in query_lower for p in ["wrong", "mistake", "error", "why didn't"]):
            return QueryType.DEBUGGING
        elif any(p in query_lower for p in ["am i right", "is this correct", "understanding"]):
            return QueryType.METACOGNITIVE
        else:
            return QueryType.EXPLANATION  # Default

    def detect_sentiment(self, query: str) -> StudentSentiment:
        """Detect student sentiment from query"""
        query_lower = query.lower()

        frustrated_signals = ["frustrated", "annoying", "why won't", "still don't", "keep getting wrong"]
        confused_signals = ["confused", "lost", "don't understand", "makes no sense", "???", "huh"]
        curious_signals = ["interesting", "curious", "wonder", "cool", "neat"]
        confident_signals = ["i think", "i know", "pretty sure", "got it", "understand"]

        if any(s in query_lower for s in frustrated_signals):
            return StudentSentiment.FRUSTRATED
        elif any(s in query_lower for s in confused_signals):
            return StudentSentiment.CONFUSED
        elif any(s in query_lower for s in curious_signals):
            return StudentSentiment.CURIOUS
        elif any(s in query_lower for s in confident_signals):
            return StudentSentiment.CONFIDENT
        else:
            return StudentSentiment.NEUTRAL

    def estimate_complexity(self, query: str) -> int:
        """Estimate query complexity (1-5)"""
        # Simple heuristics - would use LLM for better accuracy
        word_count = len(query.split())
        has_technical_terms = len(self.extract_concepts(query)) > 0
        has_multiple_parts = any(c in query for c in [',', 'and', 'also', 'additionally'])

        complexity = 1

        if word_count > 10:
            complexity += 1
        if word_count > 25:
            complexity += 1
        if has_technical_terms:
            complexity += 1
        if has_multiple_parts:
            complexity += 1

        return min(5, complexity)

    def analyze_query(self, query: str, conversation_history: List[Dict] = None) -> QueryAnalysis:
        """
        Comprehensive query analysis

        Args:
            query: The student's query text
            conversation_history: Previous messages in conversation

        Returns:
            QueryAnalysis with all extracted signals
        """
        query_type = self.classify_query_type(query)
        sentiment = self.detect_sentiment(query)
        concepts = self.extract_concepts(query)
        complexity = self.estimate_complexity(query)

        # Detect potential misconceptions (simplified - use LLM for real detection)
        misconceptions = self._detect_misconceptions(query, concepts)

        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(query, query_type, concepts)

        # Estimate understanding level
        understanding = self._estimate_understanding(query_type, sentiment, complexity)

        # Determine if follow-up is needed
        follow_up = sentiment in [StudentSentiment.CONFUSED, StudentSentiment.FRUSTRATED]

        # Suggest response type
        response_type = self._suggest_response_type(query_type, sentiment, understanding)

        return QueryAnalysis(
            query_type=query_type,
            sentiment=sentiment,
            concepts_mentioned=concepts,
            complexity_level=complexity,
            misconceptions=misconceptions,
            knowledge_gaps=knowledge_gaps,
            understanding_level=understanding,
            follow_up_recommended=follow_up,
            suggested_response_type=response_type
        )

    def _detect_misconceptions(self, query: str, concepts: List[str]) -> List[str]:
        """Detect potential misconceptions in the query"""
        misconceptions = []
        query_lower = query.lower()

        # Common misconception patterns (would be expanded with real data)
        misconception_patterns = {
            "algebra": [
                ("equals means calculate", r"equals.*answer"),
                ("variables are unknown numbers", r"variable.*number"),
            ],
            "fractions": [
                ("adding numerators and denominators", r"add.*top.*bottom"),
            ],
            "physics": [
                ("heavier falls faster", r"heavier.*fall.*faster"),
            ]
        }

        for concept in concepts:
            if concept.lower() in misconception_patterns:
                for misconception_name, pattern in misconception_patterns[concept.lower()]:
                    if re.search(pattern, query_lower):
                        misconceptions.append(misconception_name)

        return misconceptions

    def _identify_knowledge_gaps(
        self,
        query: str,
        query_type: QueryType,
        concepts: List[str]
    ) -> List[str]:
        """Identify potential knowledge gaps"""
        gaps = []

        # Basic queries about fundamental concepts suggest gaps
        if query_type == QueryType.EXPLANATION and concepts:
            gaps.extend([f"basics_of_{c}" for c in concepts])

        # Confusion suggests gaps in prerequisites
        if query_type == QueryType.CLARIFICATION:
            gaps.extend([f"prerequisites_for_{c}" for c in concepts])

        return gaps

    def _estimate_understanding(
        self,
        query_type: QueryType,
        sentiment: StudentSentiment,
        complexity: int
    ) -> float:
        """Estimate student's understanding level from query characteristics"""
        base_understanding = 0.5

        # Query type adjustments
        type_adjustments = {
            QueryType.EXPLANATION: -0.1,  # Asking basic questions
            QueryType.CLARIFICATION: -0.2,  # Confused
            QueryType.EXAMPLE: 0.0,
            QueryType.PRACTICE: 0.1,  # Ready to practice
            QueryType.COMPARISON: 0.15,  # Deeper thinking
            QueryType.APPLICATION: 0.2,  # Applying knowledge
            QueryType.METACOGNITIVE: 0.1,  # Self-aware
            QueryType.DEBUGGING: 0.05,
            QueryType.OFF_TOPIC: 0.0
        }

        # Sentiment adjustments
        sentiment_adjustments = {
            StudentSentiment.CONFIDENT: 0.15,
            StudentSentiment.CURIOUS: 0.1,
            StudentSentiment.NEUTRAL: 0.0,
            StudentSentiment.CONFUSED: -0.2,
            StudentSentiment.FRUSTRATED: -0.25
        }

        # Complexity adjustment (higher complexity queries suggest deeper engagement)
        complexity_adjustment = (complexity - 3) * 0.05

        understanding = (
            base_understanding +
            type_adjustments.get(query_type, 0) +
            sentiment_adjustments.get(sentiment, 0) +
            complexity_adjustment
        )

        return max(0.0, min(1.0, understanding))

    def _suggest_response_type(
        self,
        query_type: QueryType,
        sentiment: StudentSentiment,
        understanding: float
    ) -> str:
        """Suggest appropriate response type"""
        if sentiment == StudentSentiment.FRUSTRATED:
            return "empathetic_scaffolded"
        elif sentiment == StudentSentiment.CONFUSED:
            return "simplified_step_by_step"
        elif query_type == QueryType.EXAMPLE:
            return "concrete_example"
        elif query_type == QueryType.PRACTICE:
            return "practice_problem"
        elif query_type == QueryType.COMPARISON:
            return "comparative_analysis"
        elif understanding < 0.3:
            return "foundational_explanation"
        elif understanding > 0.7:
            return "advanced_extension"
        else:
            return "standard_explanation"


class LLMKnowledgeEnhancer(nn.Module):
    """
    Neural component that integrates LLM-extracted signals with knowledge tracing.
    Inspired by LKT model which achieved AUC 0.8513.
    """

    def __init__(
        self,
        concept_dim: int = 128,
        query_dim: int = 768,  # Typical LLM embedding size
        hidden_dim: int = 256,
        num_concepts: int = 100
    ):
        super().__init__()
        self.concept_dim = concept_dim
        self.query_dim = query_dim

        # Query embedding projection
        self.query_projector = nn.Linear(query_dim, hidden_dim)

        # Concept-query attention
        self.query_concept_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Knowledge state updater based on query signals
        self.state_updater = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, concept_dim)
        )

        # Misconception detector
        self.misconception_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_concepts),
            nn.Sigmoid()  # Probability of misconception per concept
        )

        # Understanding level predictor
        self.understanding_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        query_embedding: torch.Tensor,
        concept_embeddings: torch.Tensor,
        current_knowledge_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Enhance knowledge state with query signals

        Args:
            query_embedding: LLM embedding of student query [batch, query_dim]
            concept_embeddings: Embeddings of all concepts [num_concepts, concept_dim]
            current_knowledge_state: Current knowledge state [batch, state_dim]

        Returns:
            Enhanced knowledge signals
        """
        batch_size = query_embedding.shape[0]

        # Project query
        query_proj = self.query_projector(query_embedding)  # [batch, hidden]

        # Expand concept embeddings for batch
        concepts_expanded = concept_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Attend to relevant concepts
        query_for_attn = query_proj.unsqueeze(1)  # [batch, 1, hidden]
        attended_concepts, attn_weights = self.query_concept_attention(
            query_for_attn, concepts_expanded, concepts_expanded
        )
        attended_concepts = attended_concepts.squeeze(1)  # [batch, hidden]

        # Combine query and attended concepts
        combined = torch.cat([query_proj, attended_concepts], dim=-1)

        # Update knowledge state
        state_update = self.state_updater(combined)

        # Predict misconceptions
        misconception_probs = self.misconception_predictor(query_proj)

        # Predict understanding level
        understanding = self.understanding_predictor(query_proj)

        return {
            'state_update': state_update,
            'misconception_probs': misconception_probs,
            'understanding_level': understanding.squeeze(-1),
            'concept_attention': attn_weights.squeeze(1),
            'attended_concepts': attended_concepts
        }


class ChatbotKnowledgeTracer:
    """
    Main class for integrating chatbot interactions with knowledge tracing
    """

    def __init__(
        self,
        concept_vocabulary: List[str],
        llm_enhancer: LLMKnowledgeEnhancer
    ):
        self.query_analyzer = ChatbotQueryAnalyzer(concept_vocabulary)
        self.llm_enhancer = llm_enhancer
        self.concept_vocabulary = concept_vocabulary

    def process_interaction(
        self,
        student_id: str,
        query: str,
        query_embedding: torch.Tensor,
        current_knowledge_state: torch.Tensor,
        concept_embeddings: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Process a chatbot interaction and update knowledge state

        Args:
            student_id: Student identifier
            query: Raw query text
            query_embedding: LLM embedding of query
            current_knowledge_state: Current KT state
            concept_embeddings: Concept embedding matrix

        Returns:
            Analysis results and state updates
        """
        # Analyze query text
        text_analysis = self.query_analyzer.analyze_query(query)

        # Neural enhancement
        neural_signals = self.llm_enhancer(
            query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding,
            concept_embeddings,
            current_knowledge_state
        )

        # Combine text and neural analysis
        combined_analysis = {
            'student_id': student_id,
            'query': query,

            # Text-based analysis
            'query_type': text_analysis.query_type.value,
            'sentiment': text_analysis.sentiment.value,
            'concepts_mentioned': text_analysis.concepts_mentioned,
            'complexity_level': text_analysis.complexity_level,
            'text_based_misconceptions': text_analysis.misconceptions,
            'text_based_gaps': text_analysis.knowledge_gaps,
            'text_understanding_estimate': text_analysis.understanding_level,

            # Neural analysis
            'neural_understanding': neural_signals['understanding_level'].item(),
            'misconception_probabilities': {
                self.concept_vocabulary[i]: neural_signals['misconception_probs'][0, i].item()
                for i in range(min(len(self.concept_vocabulary), neural_signals['misconception_probs'].shape[1]))
                if neural_signals['misconception_probs'][0, i].item() > 0.3
            },
            'concept_relevance': {
                self.concept_vocabulary[i]: neural_signals['concept_attention'][0, i].item()
                for i in range(min(len(self.concept_vocabulary), neural_signals['concept_attention'].shape[1]))
            },

            # State update
            'knowledge_state_update': neural_signals['state_update'],

            # Recommendations
            'follow_up_recommended': text_analysis.follow_up_recommended,
            'suggested_response_type': text_analysis.suggested_response_type
        }

        return combined_analysis

    def get_session_summary(
        self,
        interactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Summarize a chat session for knowledge tracing

        Args:
            interactions: List of processed interactions

        Returns:
            Session-level summary for KT model
        """
        if not interactions:
            return {}

        # Aggregate metrics
        total_interactions = len(interactions)
        query_types = [i['query_type'] for i in interactions]
        sentiments = [i['sentiment'] for i in interactions]
        all_concepts = []
        for i in interactions:
            all_concepts.extend(i['concepts_mentioned'])

        # Calculate averages
        avg_understanding = sum(i['neural_understanding'] for i in interactions) / total_interactions
        avg_complexity = sum(i['complexity_level'] for i in interactions) / total_interactions

        # Identify struggling areas
        all_misconceptions = {}
        for i in interactions:
            for concept, prob in i.get('misconception_probabilities', {}).items():
                if concept not in all_misconceptions:
                    all_misconceptions[concept] = []
                all_misconceptions[concept].append(prob)

        persistent_misconceptions = {
            c: sum(probs) / len(probs)
            for c, probs in all_misconceptions.items()
            if len(probs) >= 2 and sum(probs) / len(probs) > 0.4
        }

        return {
            'total_interactions': total_interactions,
            'query_type_distribution': {qt: query_types.count(qt) / total_interactions for qt in set(query_types)},
            'sentiment_distribution': {s: sentiments.count(s) / total_interactions for s in set(sentiments)},
            'concepts_discussed': list(set(all_concepts)),
            'concept_frequency': {c: all_concepts.count(c) for c in set(all_concepts)},
            'average_understanding': avg_understanding,
            'average_query_complexity': avg_complexity,
            'persistent_misconceptions': persistent_misconceptions,
            'frustration_episodes': sentiments.count('frustrated'),
            'confusion_episodes': sentiments.count('confused'),
            'learning_signals': {
                'asking_for_examples': query_types.count('example'),
                'asking_for_practice': query_types.count('practice'),
                'metacognitive_queries': query_types.count('metacognitive')
            }
        }


# Prompt templates for LLM-based analysis (for use with external LLM APIs)
LLM_ANALYSIS_PROMPTS = {
    'concept_extraction': """
    Analyze the following student query and extract any mathematical/educational concepts mentioned.
    Query: {query}

    Return a JSON object with:
    - concepts: list of concepts mentioned
    - main_topic: the primary topic being discussed
    - prerequisite_concepts: concepts the student should know before this
    """,

    'misconception_detection': """
    Analyze the following student query for potential misconceptions.
    Query: {query}
    Topic: {topic}

    Return a JSON object with:
    - misconceptions: list of identified misconceptions
    - severity: how serious each misconception is (1-5)
    - correction_suggestion: how to address each misconception
    """,

    'understanding_assessment': """
    Based on the following student query, assess their understanding level.
    Query: {query}
    Previous context: {context}

    Return a JSON object with:
    - understanding_level: 0-1 score
    - evidence: what indicates this level
    - gaps: identified knowledge gaps
    - strengths: demonstrated understanding
    """,

    'response_generation': """
    Generate an appropriate response for this student query.
    Query: {query}
    Student profile:
    - Understanding level: {understanding}
    - Sentiment: {sentiment}
    - Learning style preference: {learning_style}

    The response should:
    1. Match the student's level
    2. Address any confusion
    3. Build on what they know
    4. Be encouraging but accurate
    """
}
