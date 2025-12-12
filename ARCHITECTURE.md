# Personalized Knowledge Tracing System Architecture

## Overview

This system implements a **hybrid deep learning knowledge tracing** approach that combines the best techniques from the literature to provide personalized learning experiences. The design is based on the comprehensive review paper "Deep Learning Based Knowledge Tracing: A Review of the Literature" (Zhu & Chen, ICBDIE 2025).

## Model Selection Rationale

Based on the paper's analysis of 30+ DLKT models, we selected a **hybrid approach** combining the top performers from each category:

### Performance Comparison (from paper)

| Category | Selected Model | AUC Score | Why We Chose It |
|----------|---------------|-----------|-----------------|
| **GNN-based** | MVGKT | **0.9470** | Best overall performance, excellent for modeling knowledge concept relationships |
| **Memory Network** | DKVMN | **0.9190** | Excellent personalization through student-specific memory slots |
| **LLM-based** | LKT | **0.8513** | Best for semantic understanding and cold-start handling |
| **Attention-based** | AKT | **0.8346** | Combines attention with educational measurement theory |
| **RNN-based** | QIKT | **0.8416** | Best for capturing temporal dynamics in learning |

### Why Hybrid?

Each model type has unique strengths:

1. **GNN (MVGKT-style)**: Captures relationships between knowledge concepts
2. **Memory Networks (DKVMN-style)**: Enables student-specific personalization
3. **Attention (AKT-style)**: Handles forgetting and difficulty adaptation
4. **LLM Integration (LKT-style)**: Analyzes natural language queries

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     PERSONALIZED KNOWLEDGE TRACING SYSTEM                       │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                        DATA COLLECTION LAYER                              │ │
│  ├────────────────┬────────────────┬────────────────┬───────────────────────┤ │
│  │   Question     │   Content      │   Chatbot      │   Session             │ │
│  │   Attempts     │   Interactions │   Queries      │   Analytics           │ │
│  │   - Response   │   - Videos     │   - Queries    │   - Time patterns     │ │
│  │   - Time       │   - Reading    │   - Context    │   - Device info       │ │
│  │   - Hints      │   - Games      │   - Sentiment  │   - Engagement        │ │
│  └────────────────┴────────────────┴────────────────┴───────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                    HYBRID KNOWLEDGE STATE ENGINE                          │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────────┐   │ │
│  │  │  GNN Module     │  │ Attention       │  │  Memory Network        │   │ │
│  │  │  (MVGKT-style)  │  │ (AKT-style)     │  │  (DKVMN-style)         │   │ │
│  │  │                 │  │                 │  │                        │   │ │
│  │  │  • Knowledge    │  │  • Temporal     │  │  • Student-specific    │   │ │
│  │  │    graph conv   │  │    attention    │  │    memory slots        │   │ │
│  │  │  • Concept      │  │  • Forgetting   │  │  • Dynamic read/write  │   │ │
│  │  │    relations    │  │    modeling     │  │  • Forgetting curve    │   │ │
│  │  └─────────┬───────┘  └────────┬────────┘  └───────────┬────────────┘   │ │
│  │            │                   │                       │                 │ │
│  │            └───────────────────┼───────────────────────┘                 │ │
│  │                                ▼                                         │ │
│  │               ┌─────────────────────────────────┐                        │ │
│  │               │   FUSION LAYER                  │                        │ │
│  │               │   P(mastery | concept, student) │                        │ │
│  │               └─────────────────────────────────┘                        │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                    LLM INTEGRATION (LKT-style)                            │ │
│  │  • Semantic analysis of student queries                                   │ │
│  │  • Misconception detection from chat patterns                             │ │
│  │  • Knowledge gap identification                                           │ │
│  │  • Cold-start enhancement                                                 │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐ │
│  │                    PERSONALIZATION ENGINE                                 │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐ │ │
│  │  │ Learning Style   │  │ Content          │  │ Path                   │ │ │
│  │  │ Analyzer         │  │ Recommender      │  │ Optimizer              │ │ │
│  │  │                  │  │                  │  │                        │ │ │
│  │  │ • Medium prefs   │  │ • ZPD-based      │  │ • Prerequisite-aware   │ │ │
│  │  │ • Pace detection │  │   difficulty     │  │ • Time-constrained     │ │ │
│  │  │ • Engagement     │  │ • Personalized   │  │ • Mastery-based        │ │ │
│  │  └──────────────────┘  └──────────────────┘  └────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Knowledge Graph Module (GNN-based)

**Inspired by: MVGKT (AUC: 0.9470)**

```python
class KnowledgeGraphModule:
    """
    Models relationships between knowledge concepts using graph convolutions.

    Features:
    - Spatio-temporal graph neural network
    - Edge type awareness (prerequisite, related, similar)
    - Topic difficulty analysis
    """
```

**Key Benefits:**
- Captures complex relationships between concepts
- Models prerequisite dependencies
- Enables knowledge transfer between related topics

### 2. Temporal Attention Module (Attention-based)

**Inspired by: AKT (AUC: 0.8346) + RKT + FoLiBiKT**

```python
class TemporalAttentionModule:
    """
    Attention mechanism with forgetting awareness.

    Features:
    - Multi-head self-attention over learning history
    - Difficulty-aware weighting
    - Forgetting decay parameters
    """
```

**Key Benefits:**
- Weights historical interactions by relevance
- Models knowledge forgetting over time
- Adapts to question difficulty

### 3. Dynamic Memory Module (Memory Network-based)

**Inspired by: DKVMN (AUC: 0.9190)**

```python
class DynamicMemoryModule:
    """
    Student-specific memory for personalization.

    Features:
    - Key-value memory structure
    - Dynamic read/write operations
    - Individual learning pattern storage
    """
```

**Key Benefits:**
- Maintains student-specific knowledge state
- Enables personalization at scale
- Captures individual learning patterns

### 4. LLM Chatbot Analyzer

**Inspired by: LKT (AUC: 0.8513)**

```python
class ChatbotQueryAnalyzer:
    """
    Analyzes student queries for knowledge signals.

    Features:
    - Query type classification
    - Sentiment detection
    - Misconception identification
    - Knowledge gap detection
    """
```

**Key Benefits:**
- Extracts learning signals from natural language
- Handles cold-start problem
- Provides interpretable insights

## Data Flow

### 1. Question Attempt Flow

```
Student answers question
         │
         ▼
┌─────────────────────────┐
│ Extract features:       │
│ - Question embedding    │
│ - Concept IDs           │
│ - Response (correct/    │
│   incorrect)            │
│ - Time spent            │
│ - Difficulty            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ GNN Module:             │
│ - Get concept relations │
│ - Propagate through     │
│   knowledge graph       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Attention Module:       │
│ - Attend to history     │
│ - Apply forgetting      │
│ - Weight by difficulty  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Memory Module:          │
│ - Read current state    │
│ - Update with response  │
│ - Store new state       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Output:                 │
│ - Updated mastery       │
│ - Prediction            │
│ - Recommendations       │
└─────────────────────────┘
```

### 2. Content Interaction Flow

```
Student watches video / reads material
         │
         ▼
┌─────────────────────────┐
│ Track engagement:       │
│ - Duration              │
│ - Completion %          │
│ - Pauses/rewinds        │
│ - In-video quiz scores  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Update preferences:     │
│ - Medium effectiveness  │
│ - Optimal duration      │
│ - Engagement patterns   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Estimate learning:      │
│ - Implicit mastery gain │
│ - Concept reinforcement │
└─────────────────────────┘
```

### 3. Chatbot Interaction Flow

```
Student asks question
         │
         ▼
┌─────────────────────────┐
│ Text Analysis:          │
│ - Query type            │
│ - Sentiment             │
│ - Concepts mentioned    │
│ - Complexity level      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Neural Analysis:        │
│ - LLM embedding         │
│ - Misconception prob    │
│ - Understanding level   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Knowledge Update:       │
│ - Identify gaps         │
│ - Flag misconceptions   │
│ - Update state          │
└─────────────────────────┘
```

## Personalization Features

### Learning Medium Preferences

The system tracks effectiveness of different content types:

| Medium | Tracked Metrics | Personalization |
|--------|-----------------|-----------------|
| **Video** | Watch time, completion, rewinds | Recommend video length |
| **Reading** | Scroll depth, highlights, time | Recommend reading level |
| **Gamified** | Scores, attempts, engagement | Adjust game difficulty |
| **Interactive** | Click patterns, exploration | Customize interactions |
| **Practice** | Accuracy, time, hint usage | Calibrate difficulty |

### Zone of Proximal Development (ZPD)

```python
class ZoneOfProximalDevelopment:
    """
    Implements Vygotsky's ZPD for optimal challenge level.

    Target: 70% success rate (challenging but achievable)
    """

    def calculate_zpd_difficulty(self, mastery, recent_accuracy, speed):
        # Find the sweet spot where learning is maximized
        pass
```

### Learning Path Optimization

```python
class LearningPathOptimizer:
    """
    Creates personalized learning sequences.

    Considers:
    - Prerequisite relationships
    - Current mastery levels
    - Time constraints
    - Learning speed
    """
```

## Usage Examples

### Basic Usage

```python
from knowledge_tracing_system import KnowledgeTracingSystem, SystemConfig

# Initialize system
config = SystemConfig(
    num_concepts=100,
    num_questions=1000,
    embedding_dim=128
)
system = KnowledgeTracingSystem(config)

# Process question attempt
result = system.process_question_attempt(
    student_id="student_001",
    question_id=42,
    concept_ids=[5, 8],
    is_correct=True,
    time_spent_seconds=45,
    difficulty=0.6
)

print(f"Predicted next: {result['predicted_probability']:.2%}")
print(f"Updated mastery: {result['updated_mastery']}")
print(f"Recommendations: {result['recommendations']}")
```

### Get Student Dashboard

```python
# Get comprehensive learning analytics
dashboard = system.get_student_dashboard("student_001")

print(f"Overall accuracy: {dashboard['summary']['overall_accuracy']:.2%}")
print(f"Learning style: {dashboard['learning_style']}")
print(f"Struggling with: {dashboard['struggling_concepts']}")
print(f"Next steps: {dashboard['recommendations']}")
```

### Process Chatbot Query

```python
# Analyze a student's question
result = system.process_chatbot_query(
    student_id="student_001",
    query="I keep getting confused about when to use integration vs differentiation"
)

print(f"Query type: {result['query_analysis']['type']}")
print(f"Sentiment: {result['query_analysis']['sentiment']}")
print(f"Detected concepts: {result['query_analysis']['concepts_mentioned']}")
print(f"Misconceptions: {result['misconceptions_detected']}")
```

## Model Training

### Data Requirements

Following the datasets from the paper:

| Dataset | Records | Students | Questions | Use Case |
|---------|---------|----------|-----------|----------|
| ASSISTments2009 | 340K | 4,217 | 26K | Math tutoring |
| ASSISTments2012 | 2.54M | 27K | 45K | Predictive analytics |
| Ednet | 130M | 780K | Large | Deep learning |
| Custom | - | - | - | Your domain |

### Training Loop

```python
# Pseudo-code for training
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        output = model(
            question_ids=batch['questions'],
            concept_ids=batch['concepts'],
            responses=batch['responses'],
            ...
        )

        # Loss: Binary cross-entropy for prediction
        loss = F.binary_cross_entropy(
            output['prediction'],
            batch['next_correct'].float()
        )

        # Backward pass
        loss.backward()
        optimizer.step()
```

### Evaluation Metrics

Primary metric from paper: **AUC (Area Under ROC Curve)**

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_pred)
```

## Challenges and Solutions

Based on the paper's identified challenges:

### 1. Data Sparsity
- **Problem**: Many students have limited interaction history
- **Solution**: LLM integration for cold-start, memory networks for quick adaptation

### 2. Model Interpretability
- **Problem**: Deep models are black boxes
- **Solution**: Attention weights, explicit mastery scores, concept-level explanations

### 3. Forgetting Modeling
- **Problem**: Knowledge decays over time
- **Solution**: Temporal attention with forgetting factors (RKT-style)

### 4. Multi-Concept Questions
- **Problem**: Questions test multiple concepts
- **Solution**: GNN propagation, multi-concept embeddings

## Future Enhancements

Based on paper's future directions:

1. **Multimodal Integration**: Incorporate video engagement, text annotations, voice patterns
2. **Knowledge Graphs**: Richer semantic relationships using external knowledge
3. **Transfer Learning**: Share knowledge across subjects/platforms
4. **Real-time Adaptation**: Faster model updates during sessions

## Directory Structure

```
knowledge_tracing_system/
├── __init__.py
├── main.py                 # Main orchestration
├── models/
│   ├── __init__.py
│   └── knowledge_state.py  # Hybrid KT model
├── data/
│   ├── __init__.py
│   └── schemas.py          # Data structures
├── recommendation/
│   ├── __init__.py
│   └── personalized_learning.py  # Recommendation engine
└── llm/
    ├── __init__.py
    └── chatbot_analyzer.py # LLM integration
```

## References

1. Zhu, S., & Chen, W. (2025). Deep Learning Based Knowledge Tracing: A Review of the Literature. ICBDIE 2025.
2. Piech, C., et al. (2015). Deep Knowledge Tracing. NeurIPS.
3. Zhang, J., et al. (2017). Dynamic Key-Value Memory Networks for Knowledge Tracing. WWW.
4. Ghosh, A., et al. (2020). Context-aware Attentive Knowledge Tracing (AKT). KDD.
5. Xia, Z., et al. (2023). Multi-Variate Knowledge Tracking (MVGKT). IEEE TLT.
6. Lee, U., et al. (2024). Language Model Can Do Knowledge Tracing (LKT). arXiv.
