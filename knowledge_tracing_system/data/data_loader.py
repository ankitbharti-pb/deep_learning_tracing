"""
ASSISTments 2009 Data Loader and Dummy Data Generator

Converts ASSISTments dataset to the knowledge tracing schema format
and generates synthetic data for content/chatbot interactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import uuid
import random
from pathlib import Path

from .schemas import (
    KnowledgeConcept, Question, QuestionAttempt, ContentInteraction,
    ChatbotInteraction, LearningSession, StudentProfile, KnowledgeState,
    ContentRecommendation, KnowledgeGraph, KnowledgeGraphEdge,
    LearningMedium, DifficultyLevel, InteractionType
)


class ASSISTmentsDataLoader:
    """
    Loads ASSISTments 2009 data and converts to schema format.
    Also generates synthetic data for content/chatbot interactions.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "assismentdata"
        self.data_dir = Path(data_dir)

        # Loaded data
        self.raw_df: Optional[pd.DataFrame] = None
        self.concepts: Dict[str, KnowledgeConcept] = {}
        self.questions: Dict[str, Question] = {}
        self.question_attempts: List[QuestionAttempt] = []
        self.content_interactions: List[ContentInteraction] = []
        self.chatbot_interactions: List[ChatbotInteraction] = []
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.student_profiles: Dict[str, StudentProfile] = {}
        self.knowledge_graph: KnowledgeGraph = KnowledgeGraph()

        # Mapping tables
        self.skill_to_concept: Dict[int, str] = {}
        self.problem_to_question: Dict[int, str] = {}
        self.user_to_student: Dict[int, str] = {}

        # Random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

    def load_csv_files(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and combine all CSV files from data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        dfs = []
        for csv_file in csv_files:
            print(f"Loading {csv_file.name}...")
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(csv_file, low_memory=False, encoding=encoding)
                    print(f"  Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode {csv_file.name} with any supported encoding")
            dfs.append(df)

        self.raw_df = pd.concat(dfs, ignore_index=True)

        # Sample if requested (for faster testing)
        if sample_size and len(self.raw_df) > sample_size:
            self.raw_df = self.raw_df.sample(n=sample_size, random_state=42)

        print(f"Loaded {len(self.raw_df):,} records from {len(csv_files)} file(s)")
        return self.raw_df

    def _map_difficulty(self, avg_correct_rate: float) -> DifficultyLevel:
        """Map average correct rate to difficulty level."""
        if avg_correct_rate >= 0.85:
            return DifficultyLevel.BEGINNER
        elif avg_correct_rate >= 0.70:
            return DifficultyLevel.EASY
        elif avg_correct_rate >= 0.50:
            return DifficultyLevel.MEDIUM
        elif avg_correct_rate >= 0.30:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT

    def build_knowledge_concepts(self) -> Dict[str, KnowledgeConcept]:
        """Extract unique skills and create KnowledgeConcept objects."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_csv_files() first.")

        # Get unique skills with their statistics
        skill_stats = self.raw_df.groupby(['skill_id', 'skill_name']).agg({
            'correct': 'mean',
            'problem_id': 'nunique',
            'user_id': 'nunique'
        }).reset_index()

        skill_stats.columns = ['skill_id', 'skill_name', 'avg_correct', 'num_problems', 'num_students']

        for _, row in skill_stats.iterrows():
            skill_id = int(row['skill_id']) if pd.notna(row['skill_id']) else 0
            skill_name = str(row['skill_name']) if pd.notna(row['skill_name']) else "Unknown"

            concept_id = f"concept_{skill_id}"
            self.skill_to_concept[skill_id] = concept_id

            # Determine difficulty based on average correct rate
            difficulty = self._map_difficulty(row['avg_correct'])

            # Determine Bloom's level based on skill name patterns
            bloom_level = self._infer_bloom_level(skill_name)

            concept = KnowledgeConcept(
                concept_id=concept_id,
                name=skill_name,
                subject="Mathematics",
                chapter=self._infer_chapter(skill_name),
                topic=skill_name,
                difficulty_base=difficulty,
                prerequisites=[],  # Will be populated later
                related_concepts=[],
                bloom_level=bloom_level,
                estimated_learning_time_minutes=self._estimate_learning_time(difficulty)
            )

            self.concepts[concept_id] = concept
            self.knowledge_graph.add_concept(concept)

        print(f"Created {len(self.concepts)} knowledge concepts")
        return self.concepts

    def _infer_bloom_level(self, skill_name: str) -> str:
        """Infer Bloom's taxonomy level from skill name."""
        skill_lower = skill_name.lower()

        if any(word in skill_lower for word in ['identify', 'recall', 'define', 'list']):
            return "remember"
        elif any(word in skill_lower for word in ['explain', 'describe', 'interpret', 'classify']):
            return "understand"
        elif any(word in skill_lower for word in ['solve', 'calculate', 'apply', 'use']):
            return "apply"
        elif any(word in skill_lower for word in ['analyze', 'compare', 'contrast', 'examine']):
            return "analyze"
        elif any(word in skill_lower for word in ['evaluate', 'judge', 'assess', 'critique']):
            return "evaluate"
        elif any(word in skill_lower for word in ['create', 'design', 'construct', 'develop']):
            return "create"
        else:
            return "apply"  # Default for math problems

    def _infer_chapter(self, skill_name: str) -> str:
        """Infer chapter from skill name."""
        skill_lower = skill_name.lower()

        chapter_mapping = {
            'equation': 'Equations',
            'graph': 'Graphing',
            'fraction': 'Fractions',
            'decimal': 'Decimals',
            'percent': 'Percentages',
            'geometry': 'Geometry',
            'area': 'Geometry',
            'perimeter': 'Geometry',
            'volume': 'Geometry',
            'angle': 'Geometry',
            'probability': 'Probability & Statistics',
            'statistic': 'Probability & Statistics',
            'box and whisker': 'Probability & Statistics',
            'mean': 'Probability & Statistics',
            'median': 'Probability & Statistics',
            'ratio': 'Ratios & Proportions',
            'proportion': 'Ratios & Proportions',
            'algebra': 'Algebra',
            'variable': 'Algebra',
            'expression': 'Algebra',
            'polynomial': 'Algebra',
            'factor': 'Algebra',
            'number': 'Number Systems',
            'integer': 'Number Systems',
            'exponent': 'Exponents & Powers',
            'power': 'Exponents & Powers',
        }

        for keyword, chapter in chapter_mapping.items():
            if keyword in skill_lower:
                return chapter

        return "General Mathematics"

    def _estimate_learning_time(self, difficulty: DifficultyLevel) -> int:
        """Estimate learning time based on difficulty."""
        time_mapping = {
            DifficultyLevel.BEGINNER: 15,
            DifficultyLevel.EASY: 20,
            DifficultyLevel.MEDIUM: 30,
            DifficultyLevel.HARD: 45,
            DifficultyLevel.EXPERT: 60
        }
        return time_mapping.get(difficulty, 30)

    def build_questions(self) -> Dict[str, Question]:
        """Create Question objects from problem data."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_csv_files() first.")

        # Get unique problems with their statistics
        problem_stats = self.raw_df.groupby(['problem_id', 'skill_id']).agg({
            'correct': 'mean',
            'attempt_count': 'mean',
            'hint_count': 'mean',
            'ms_first_response': 'mean'
        }).reset_index()

        for _, row in problem_stats.iterrows():
            problem_id = int(row['problem_id'])
            skill_id = int(row['skill_id']) if pd.notna(row['skill_id']) else 0

            question_id = f"question_{problem_id}"
            self.problem_to_question[problem_id] = question_id

            concept_id = self.skill_to_concept.get(skill_id, f"concept_{skill_id}")
            difficulty = self._map_difficulty(row['correct'])

            # Calculate discrimination index based on attempt variance
            discrimination = min(0.9, max(0.1, 1 - row['correct']))

            # Calculate guessing parameter
            guessing = 0.25 if difficulty in [DifficultyLevel.BEGINNER, DifficultyLevel.EASY] else 0.2

            question = Question(
                question_id=question_id,
                concept_ids=[concept_id],
                subject="Mathematics",
                chapter=self.concepts.get(concept_id, KnowledgeConcept(
                    concept_id=concept_id, name="Unknown", subject="Math",
                    chapter="General", topic="Unknown", difficulty_base=DifficultyLevel.MEDIUM
                )).chapter,
                topic=self.concepts.get(concept_id, KnowledgeConcept(
                    concept_id=concept_id, name="Unknown", subject="Math",
                    chapter="General", topic="Unknown", difficulty_base=DifficultyLevel.MEDIUM
                )).topic,
                difficulty=difficulty,
                question_type="algebra",  # From ASSISTments answer_type
                content=f"Problem {problem_id}",  # Actual content not in dataset
                options=None,
                correct_answer=None,
                hints=[f"Hint {i+1}" for i in range(int(row['hint_count']) or 1)],
                explanation="",
                discrimination_index=discrimination,
                guessing_parameter=guessing
            )

            self.questions[question_id] = question

        print(f"Created {len(self.questions)} questions")
        return self.questions

    def build_question_attempts(self) -> List[QuestionAttempt]:
        """Convert raw data to QuestionAttempt objects."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_csv_files() first.")

        # Sort by user and order_id to maintain temporal sequence
        df_sorted = self.raw_df.sort_values(['user_id', 'order_id'])

        # Create base timestamp (simulating data collection period)
        base_time = datetime(2009, 9, 1, 8, 0, 0)  # Start of school year

        # Track user's last attempt time and concept attempt time
        user_last_attempt: Dict[int, datetime] = {}
        user_concept_last_attempt: Dict[Tuple[int, int], datetime] = {}
        user_streaks: Dict[int, Dict[str, int]] = {}  # correct/incorrect streaks

        for idx, row in df_sorted.iterrows():
            user_id = int(row['user_id'])
            problem_id = int(row['problem_id'])
            skill_id = int(row['skill_id']) if pd.notna(row['skill_id']) else 0

            # Map to schema IDs
            student_id = f"student_{user_id}"
            self.user_to_student[user_id] = student_id
            question_id = self.problem_to_question.get(problem_id, f"question_{problem_id}")

            # Calculate timestamp
            time_offset = timedelta(
                days=random.randint(0, 180),  # Spread across semester
                hours=random.randint(8, 20),  # School hours
                minutes=random.randint(0, 59)
            )
            timestamp = base_time + time_offset

            # Calculate time since last attempt
            time_since_last = 0.0
            if user_id in user_last_attempt:
                delta = timestamp - user_last_attempt[user_id]
                time_since_last = delta.total_seconds() / 3600  # hours

            time_since_concept = 0.0
            if (user_id, skill_id) in user_concept_last_attempt:
                delta = timestamp - user_concept_last_attempt[(user_id, skill_id)]
                time_since_concept = delta.total_seconds() / 3600

            # Track streaks
            if user_id not in user_streaks:
                user_streaks[user_id] = {'correct': 0, 'incorrect': 0}

            is_correct = bool(row['correct'])
            if is_correct:
                user_streaks[user_id]['correct'] += 1
                user_streaks[user_id]['incorrect'] = 0
            else:
                user_streaks[user_id]['incorrect'] += 1
                user_streaks[user_id]['correct'] = 0

            attempt = QuestionAttempt(
                attempt_id=f"attempt_{row['order_id']}",
                student_id=student_id,
                question_id=question_id,
                timestamp=timestamp,
                response=row.get('answer_text', ''),
                is_correct=is_correct,
                time_spent_seconds=int(row['ms_first_response'] / 1000) if pd.notna(row['ms_first_response']) else 30,
                hints_used=int(row['hint_count']) if pd.notna(row['hint_count']) else 0,
                attempts_count=int(row['attempt_count']) if pd.notna(row['attempt_count']) else 1,
                confidence_level=None,
                device_type="desktop",
                session_id=f"session_{row['assignment_id']}",
                time_since_last_attempt_hours=time_since_last,
                time_since_last_concept_attempt_hours=time_since_concept,
                streak_correct=user_streaks[user_id]['correct'],
                streak_incorrect=user_streaks[user_id]['incorrect']
            )

            self.question_attempts.append(attempt)

            # Update tracking
            user_last_attempt[user_id] = timestamp
            user_concept_last_attempt[(user_id, skill_id)] = timestamp

        print(f"Created {len(self.question_attempts)} question attempts")
        return self.question_attempts

    def build_student_profiles(self) -> Dict[str, StudentProfile]:
        """Create StudentProfile objects from user data."""
        if self.raw_df is None:
            raise ValueError("Data not loaded. Call load_csv_files() first.")

        # Aggregate user statistics
        user_stats = self.raw_df.groupby('user_id').agg({
            'correct': ['mean', 'sum', 'count'],
            'skill_id': 'nunique',
            'ms_first_response': 'mean',
            'hint_count': 'mean',
            'attempt_count': 'mean'
        }).reset_index()

        user_stats.columns = ['user_id', 'avg_correct', 'total_correct', 'total_attempts',
                             'skills_practiced', 'avg_response_time', 'avg_hints', 'avg_attempts']

        for _, row in user_stats.iterrows():
            user_id = int(row['user_id'])
            student_id = f"student_{user_id}"

            # Infer learning pace from response time
            avg_time = row['avg_response_time'] / 1000 if pd.notna(row['avg_response_time']) else 30
            if avg_time < 15:
                pace = "fast"
            elif avg_time < 45:
                pace = "medium"
            else:
                pace = "slow"

            # Calculate concept mastery from skill performance
            skill_perf = self.raw_df[self.raw_df['user_id'] == user_id].groupby('skill_id')['correct'].mean()
            concept_mastery = {
                self.skill_to_concept.get(int(sid), f"concept_{int(sid)}"): float(perf)
                for sid, perf in skill_perf.items() if pd.notna(sid)
            }

            # Infer preferred medium (synthetic - based on hint usage pattern)
            preferred_medium = {}
            hint_ratio = row['avg_hints'] / max(row['avg_attempts'], 1) if pd.notna(row['avg_hints']) else 0
            if hint_ratio > 0.5:
                preferred_medium = {
                    LearningMedium.VIDEO: 0.4,
                    LearningMedium.INTERACTIVE: 0.3,
                    LearningMedium.READING: 0.2,
                    LearningMedium.PRACTICE: 0.1
                }
            else:
                preferred_medium = {
                    LearningMedium.PRACTICE: 0.4,
                    LearningMedium.READING: 0.3,
                    LearningMedium.VIDEO: 0.2,
                    LearningMedium.INTERACTIVE: 0.1
                }

            profile = StudentProfile(
                student_id=student_id,
                created_at=datetime(2009, 9, 1),
                preferred_medium=preferred_medium,
                optimal_session_length_minutes=30,
                best_learning_times=[9, 10, 14, 15, 16],  # School hours
                learning_pace=pace,
                working_memory_estimate=min(1.0, row['avg_correct'] + 0.2),
                attention_span_estimate=min(1.0, 1 - (row['avg_hints'] / 10 if pd.notna(row['avg_hints']) else 0.3)),
                preferred_difficulty_range=(2, 4),
                overall_mastery=float(row['avg_correct']),
                concept_mastery=concept_mastery,
                subject_strengths={"Mathematics": float(row['avg_correct'])},
                subject_weaknesses={},
                avg_session_length_minutes=float(row['total_attempts']) * 2,  # Estimate
                total_learning_time_hours=float(row['total_attempts']) * 0.03,
                consistency_score=min(1.0, row['total_attempts'] / 100),
                completion_rate=float(row['avg_correct'])
            )

            self.student_profiles[student_id] = profile

        print(f"Created {len(self.student_profiles)} student profiles")
        return self.student_profiles

    def build_knowledge_graph(self) -> KnowledgeGraph:
        """Build knowledge graph with concept relationships."""
        # Create edges based on co-occurrence and performance patterns
        concept_ids = list(self.concepts.keys())

        # Group concepts by chapter for related edges
        chapter_concepts: Dict[str, List[str]] = {}
        for cid, concept in self.concepts.items():
            chapter = concept.chapter
            if chapter not in chapter_concepts:
                chapter_concepts[chapter] = []
            chapter_concepts[chapter].append(cid)

        # Add related edges within chapters
        for chapter, cids in chapter_concepts.items():
            for i, cid1 in enumerate(cids):
                for cid2 in cids[i+1:]:
                    edge = KnowledgeGraphEdge(
                        source_concept_id=cid1,
                        target_concept_id=cid2,
                        edge_type="related",
                        weight=0.7
                    )
                    self.knowledge_graph.add_edge(edge)

        # Add prerequisite edges based on difficulty ordering
        sorted_concepts = sorted(
            self.concepts.items(),
            key=lambda x: x[1].difficulty_base.value
        )

        for i in range(len(sorted_concepts) - 1):
            easier_cid = sorted_concepts[i][0]
            harder_cid = sorted_concepts[i + 1][0]

            # Only add prerequisite if same chapter and difficulty difference is 1
            easier = self.concepts[easier_cid]
            harder = self.concepts[harder_cid]

            if easier.chapter == harder.chapter:
                diff = harder.difficulty_base.value - easier.difficulty_base.value
                if 0 < diff <= 2:
                    edge = KnowledgeGraphEdge(
                        source_concept_id=easier_cid,
                        target_concept_id=harder_cid,
                        edge_type="prerequisite",
                        weight=1.0 / diff
                    )
                    self.knowledge_graph.add_edge(edge)
                    harder.prerequisites.append(easier_cid)

        print(f"Built knowledge graph with {len(self.knowledge_graph.edges)} edges")
        return self.knowledge_graph

    def generate_content_interactions(self, num_per_student: int = 10) -> List[ContentInteraction]:
        """Generate synthetic content interaction data."""
        mediums = list(LearningMedium)
        content_types = {
            LearningMedium.VIDEO: {"avg_duration": 600, "completion_range": (0.3, 1.0)},
            LearningMedium.READING: {"avg_duration": 300, "completion_range": (0.4, 1.0)},
            LearningMedium.GAMIFIED: {"avg_duration": 450, "completion_range": (0.5, 1.0)},
            LearningMedium.INTERACTIVE: {"avg_duration": 400, "completion_range": (0.4, 1.0)},
            LearningMedium.AUDIO: {"avg_duration": 500, "completion_range": (0.3, 0.9)},
            LearningMedium.PRACTICE: {"avg_duration": 200, "completion_range": (0.6, 1.0)},
            LearningMedium.DISCUSSION: {"avg_duration": 350, "completion_range": (0.2, 0.8)},
        }

        concept_ids = list(self.concepts.keys())
        base_time = datetime(2009, 9, 1, 8, 0, 0)

        for student_id, profile in self.student_profiles.items():
            # Weight mediums by preference
            medium_weights = [
                profile.preferred_medium.get(m, 0.1) for m in mediums
            ]
            total_weight = sum(medium_weights)
            medium_probs = [w / total_weight for w in medium_weights]

            for i in range(num_per_student):
                medium = np.random.choice(mediums, p=medium_probs)
                config = content_types[medium]

                # Pick concept based on mastery (prefer weaker areas)
                mastery_weights = [
                    1 - profile.concept_mastery.get(cid, 0.5)
                    for cid in concept_ids
                ]
                total_m = sum(mastery_weights)
                concept_probs = [w / total_m for w in mastery_weights]
                concept_id = np.random.choice(concept_ids, p=concept_probs)

                timestamp = base_time + timedelta(
                    days=random.randint(0, 180),
                    hours=random.randint(8, 20),
                    minutes=random.randint(0, 59)
                )

                duration = int(np.random.normal(config["avg_duration"], config["avg_duration"] * 0.3))
                duration = max(60, duration)

                completion = random.uniform(*config["completion_range"])

                # Generate engagement actions
                engagement = {}
                if medium == LearningMedium.VIDEO:
                    engagement = {
                        "pauses": random.randint(0, 5),
                        "rewinds": random.randint(0, 3),
                        "speed_changes": random.randint(0, 2)
                    }
                elif medium == LearningMedium.READING:
                    engagement = {
                        "highlights": random.randint(0, 10),
                        "notes": random.randint(0, 3),
                        "bookmarks": random.randint(0, 2)
                    }
                elif medium == LearningMedium.INTERACTIVE:
                    engagement = {
                        "interactions": random.randint(5, 20),
                        "hints_requested": random.randint(0, 5)
                    }

                interaction = ContentInteraction(
                    interaction_id=f"content_{student_id}_{i}",
                    student_id=student_id,
                    content_id=f"content_{concept_id}_{medium.value}",
                    concept_ids=[concept_id],
                    medium=medium,
                    timestamp=timestamp,
                    duration_seconds=duration,
                    completion_percentage=completion * 100,
                    engagement_actions=engagement,
                    scroll_depth=completion if medium == LearningMedium.READING else 0.0,
                    video_completion_points=[completion] if medium == LearningMedium.VIDEO else [],
                    quiz_scores_during=[random.uniform(0.5, 1.0)] if random.random() > 0.5 else []
                )

                self.content_interactions.append(interaction)

        print(f"Generated {len(self.content_interactions)} content interactions")
        return self.content_interactions

    def generate_chatbot_interactions(self, num_per_student: int = 5) -> List[ChatbotInteraction]:
        """Generate synthetic chatbot interaction data."""
        query_types = ["explanation", "clarification", "example", "practice", "definition", "application"]
        sentiments = ["confident", "curious", "confused", "frustrated", "neutral"]

        # Sample queries by type
        query_templates = {
            "explanation": [
                "Can you explain how {concept} works?",
                "I don't understand {concept}. Can you help?",
                "What is the main idea behind {concept}?",
            ],
            "clarification": [
                "I'm confused about {concept}. What's the difference between this and {related}?",
                "Can you clarify {concept}?",
                "I thought {concept} was different. Can you explain again?",
            ],
            "example": [
                "Can you give me an example of {concept}?",
                "Show me how to solve a {concept} problem.",
                "What's a real-world example of {concept}?",
            ],
            "practice": [
                "Can I try a practice problem on {concept}?",
                "Give me a problem to solve for {concept}.",
                "I want to practice {concept}.",
            ],
            "definition": [
                "What is {concept}?",
                "Define {concept}.",
                "What does {concept} mean?",
            ],
            "application": [
                "When would I use {concept} in real life?",
                "How is {concept} applied?",
                "What are practical uses of {concept}?",
            ]
        }

        misconception_templates = [
            "thinks {concept} is the same as {related}",
            "confuses the order of operations in {concept}",
            "misunderstands the definition of {concept}",
        ]

        base_time = datetime(2009, 9, 1, 8, 0, 0)
        concept_ids = list(self.concepts.keys())

        for student_id, profile in self.student_profiles.items():
            session_id = f"chat_session_{student_id}"

            for i in range(num_per_student):
                # Pick concept (prefer weaker areas for chatbot)
                mastery_weights = [
                    1 - profile.concept_mastery.get(cid, 0.5)
                    for cid in concept_ids
                ]
                total_m = sum(mastery_weights)
                concept_probs = [w / total_m for w in mastery_weights]
                concept_id = np.random.choice(concept_ids, p=concept_probs)
                concept = self.concepts[concept_id]

                # Determine query type and sentiment based on mastery
                mastery = profile.concept_mastery.get(concept_id, 0.5)
                if mastery < 0.3:
                    query_type = random.choice(["explanation", "definition", "example"])
                    sentiment = random.choice(["confused", "frustrated", "curious"])
                elif mastery < 0.6:
                    query_type = random.choice(["clarification", "example", "practice"])
                    sentiment = random.choice(["curious", "neutral"])
                else:
                    query_type = random.choice(["practice", "application"])
                    sentiment = random.choice(["confident", "curious"])

                # Generate query
                template = random.choice(query_templates[query_type])
                related_concept = random.choice(concept_ids)
                query = template.format(
                    concept=concept.name,
                    related=self.concepts.get(related_concept, concept).name
                )

                timestamp = base_time + timedelta(
                    days=random.randint(0, 180),
                    hours=random.randint(8, 20),
                    minutes=random.randint(0, 59)
                )

                # Generate misconceptions for struggling students
                misconceptions = []
                if mastery < 0.4 and random.random() > 0.5:
                    misc_template = random.choice(misconception_templates)
                    misconceptions.append(misc_template.format(
                        concept=concept.name,
                        related=self.concepts.get(related_concept, concept).name
                    ))

                # Knowledge gaps
                knowledge_gaps = []
                if mastery < 0.5:
                    prereqs = concept.prerequisites[:2] if concept.prerequisites else []
                    knowledge_gaps.extend(prereqs)

                interaction = ChatbotInteraction(
                    interaction_id=f"chat_{student_id}_{i}",
                    student_id=student_id,
                    session_id=session_id,
                    timestamp=timestamp,
                    query=query,
                    response=f"[AI Response about {concept.name}]",
                    concept_ids_detected=[concept_id],
                    query_type=query_type,
                    complexity_level=random.randint(1, 3) if mastery > 0.5 else random.randint(1, 2),
                    sentiment=sentiment,
                    follow_up_count=random.randint(0, 3),
                    misconceptions_detected=misconceptions,
                    knowledge_gaps_indicated=knowledge_gaps,
                    understanding_indicators={concept_id: mastery}
                )

                self.chatbot_interactions.append(interaction)

        print(f"Generated {len(self.chatbot_interactions)} chatbot interactions")
        return self.chatbot_interactions

    def build_learning_sessions(self) -> Dict[str, LearningSession]:
        """Build learning sessions from question attempts."""
        # Group attempts by student and assignment
        session_data: Dict[str, Dict] = {}

        for attempt in self.question_attempts:
            session_id = attempt.session_id
            student_id = attempt.student_id

            if session_id not in session_data:
                session_data[session_id] = {
                    'student_id': student_id,
                    'start_time': attempt.timestamp,
                    'end_time': attempt.timestamp,
                    'questions_attempted': 0,
                    'questions_correct': 0,
                    'total_time': 0,
                    'concepts': set()
                }

            data = session_data[session_id]
            data['end_time'] = max(data['end_time'], attempt.timestamp)
            data['questions_attempted'] += 1
            data['questions_correct'] += 1 if attempt.is_correct else 0
            data['total_time'] += attempt.time_spent_seconds

            # Get concept from question
            question = self.questions.get(attempt.question_id)
            if question:
                data['concepts'].update(question.concept_ids)

        for session_id, data in session_data.items():
            session = LearningSession(
                session_id=session_id,
                student_id=data['student_id'],
                start_time=data['start_time'],
                end_time=data['end_time'],
                device_type="desktop",
                platform="web",
                total_questions_attempted=data['questions_attempted'],
                questions_correct=data['questions_correct'],
                content_pieces_viewed=0,
                chatbot_queries=0,
                total_time_seconds=data['total_time'],
                concepts_touched=list(data['concepts']),
                focus_score=random.uniform(0.6, 1.0),
                engagement_score=random.uniform(0.5, 1.0)
            )

            self.learning_sessions[session_id] = session

        print(f"Built {len(self.learning_sessions)} learning sessions")
        return self.learning_sessions

    def load_all(self, sample_size: Optional[int] = None,
                 content_per_student: int = 10,
                 chatbot_per_student: int = 5) -> Dict:
        """
        Load all data and generate synthetic interactions.

        Args:
            sample_size: Optional limit on number of raw records to process
            content_per_student: Number of synthetic content interactions per student
            chatbot_per_student: Number of synthetic chatbot interactions per student

        Returns:
            Dictionary with all loaded/generated data
        """
        print("=" * 60)
        print("Loading ASSISTments 2009 Data")
        print("=" * 60)

        # Load raw data
        self.load_csv_files(sample_size)

        # Build schema objects
        self.build_knowledge_concepts()
        self.build_questions()
        self.build_question_attempts()
        self.build_student_profiles()
        self.build_knowledge_graph()

        # Generate synthetic data
        self.generate_content_interactions(content_per_student)
        self.generate_chatbot_interactions(chatbot_per_student)
        self.build_learning_sessions()

        print("=" * 60)
        print("Data Loading Complete!")
        print(f"  - Concepts: {len(self.concepts)}")
        print(f"  - Questions: {len(self.questions)}")
        print(f"  - Question Attempts: {len(self.question_attempts)}")
        print(f"  - Students: {len(self.student_profiles)}")
        print(f"  - Content Interactions: {len(self.content_interactions)}")
        print(f"  - Chatbot Interactions: {len(self.chatbot_interactions)}")
        print(f"  - Learning Sessions: {len(self.learning_sessions)}")
        print(f"  - Knowledge Graph Edges: {len(self.knowledge_graph.edges)}")
        print("=" * 60)

        return {
            'concepts': self.concepts,
            'questions': self.questions,
            'question_attempts': self.question_attempts,
            'student_profiles': self.student_profiles,
            'content_interactions': self.content_interactions,
            'chatbot_interactions': self.chatbot_interactions,
            'learning_sessions': self.learning_sessions,
            'knowledge_graph': self.knowledge_graph
        }

    def get_student_data(self, student_id: str) -> Dict:
        """Get all data for a specific student."""
        return {
            'profile': self.student_profiles.get(student_id),
            'question_attempts': [a for a in self.question_attempts if a.student_id == student_id],
            'content_interactions': [c for c in self.content_interactions if c.student_id == student_id],
            'chatbot_interactions': [c for c in self.chatbot_interactions if c.student_id == student_id],
            'sessions': {k: v for k, v in self.learning_sessions.items() if v.student_id == student_id}
        }

    def get_concept_data(self, concept_id: str) -> Dict:
        """Get all data for a specific concept."""
        concept = self.concepts.get(concept_id)
        questions = [q for q in self.questions.values() if concept_id in q.concept_ids]
        attempts = [a for a in self.question_attempts
                   if self.questions.get(a.question_id) and concept_id in self.questions[a.question_id].concept_ids]

        return {
            'concept': concept,
            'questions': questions,
            'attempts': attempts,
            'prerequisites': self.knowledge_graph.get_prerequisites(concept_id),
            'related': self.knowledge_graph.get_related(concept_id)
        }

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert all data to pandas DataFrames for analysis."""
        return {
            'concepts': pd.DataFrame([c.to_dict() for c in self.concepts.values()]),
            'questions': pd.DataFrame([q.to_dict() for q in self.questions.values()]),
            'question_attempts': pd.DataFrame([a.to_dict() for a in self.question_attempts]),
            'student_profiles': pd.DataFrame([p.to_dict() for p in self.student_profiles.values()]),
            'content_interactions': pd.DataFrame([c.to_dict() for c in self.content_interactions]),
            'chatbot_interactions': pd.DataFrame([c.to_dict() for c in self.chatbot_interactions]),
            'learning_sessions': pd.DataFrame([s.to_dict() for s in self.learning_sessions.values()]),
            'knowledge_graph_edges': pd.DataFrame([e.to_dict() for e in self.knowledge_graph.edges])
        }


def load_sample_data(sample_size: int = 10000) -> Dict:
    """
    Quick function to load a sample of data for testing.

    Args:
        sample_size: Number of raw records to sample

    Returns:
        Dictionary with all data
    """
    loader = ASSISTmentsDataLoader()
    return loader.load_all(sample_size=sample_size)


def load_full_data() -> Dict:
    """Load the complete dataset."""
    loader = ASSISTmentsDataLoader()
    return loader.load_all()


if __name__ == "__main__":
    # Test the data loader
    print("Testing ASSISTments Data Loader")
    print("-" * 40)

    # Load sample data
    data = load_sample_data(sample_size=5000)

    # Print sample records
    print("\nSample Knowledge Concept:")
    concept = list(data['concepts'].values())[0]
    print(f"  {concept.concept_id}: {concept.name} ({concept.difficulty_base.name})")

    print("\nSample Question:")
    question = list(data['questions'].values())[0]
    print(f"  {question.question_id}: {question.topic} - Difficulty: {question.difficulty.name}")

    print("\nSample Student Profile:")
    profile = list(data['student_profiles'].values())[0]
    print(f"  {profile.student_id}: Mastery={profile.overall_mastery:.2f}, Pace={profile.learning_pace}")

    print("\nSample Question Attempt:")
    attempt = data['question_attempts'][0]
    print(f"  {attempt.attempt_id}: Correct={attempt.is_correct}, Time={attempt.time_spent_seconds}s")

    print("\nSample Content Interaction:")
    content = data['content_interactions'][0]
    print(f"  {content.interaction_id}: {content.medium.value}, Duration={content.duration_seconds}s")

    print("\nSample Chatbot Interaction:")
    chat = data['chatbot_interactions'][0]
    print(f"  {chat.interaction_id}: Type={chat.query_type}, Sentiment={chat.sentiment}")
