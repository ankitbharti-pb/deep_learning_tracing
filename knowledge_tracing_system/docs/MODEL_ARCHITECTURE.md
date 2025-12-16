# Deep Learning Knowledge Tracing Models

## Overview

This document provides detailed technical descriptions of the four core deep learning models used in our hybrid knowledge tracing system. Each model addresses specific challenges in educational data mining and student modeling.

| Model | Based On | Primary Function | Key Strength |
|-------|----------|------------------|--------------|
| GNN Module | MVGKT | Concept relationships | Multi-concept questions |
| Memory Networks | DKVMN | Student personalization | Individual learning patterns |
| Attention Module | AKT | Temporal dynamics | Forgetting & difficulty |
| LLM Integration | LKT | Semantic understanding | Cold-start & NLP queries |

---

## 1. Graph Neural Network (GNN) Module

### 1.1 Background: MVGKT

**MVGKT** (Multi-View Graph Knowledge Tracing) achieves state-of-the-art performance with **AUC: 0.9470** on benchmark datasets. It models knowledge concepts as a graph where relationships (prerequisites, similarities) are explicitly captured.

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GNN Module Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Student interaction sequence + Knowledge Graph           │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Concept    │    │  Adjacency   │    │  Interaction │       │
│  │  Embeddings  │    │   Matrix     │    │   Sequence   │       │
│  │   (N × d)    │    │   (N × N)    │    │   (T × 2)    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └─────────┬─────────┘                   │                │
│                   ▼                             │                │
│         ┌─────────────────┐                     │                │
│         │  Graph Conv     │                     │                │
│         │  Layer 1        │                     │                │
│         │  (GCN/GAT)      │                     │                │
│         └────────┬────────┘                     │                │
│                  ▼                              │                │
│         ┌─────────────────┐                     │                │
│         │  Graph Conv     │                     │                │
│         │  Layer 2        │                     │                │
│         └────────┬────────┘                     │                │
│                  ▼                              │                │
│         ┌─────────────────┐                     │                │
│         │  Concept-aware  │◄────────────────────┘                │
│         │  Aggregation    │                                      │
│         └────────┬────────┘                                      │
│                  ▼                                               │
│         ┌─────────────────┐                                      │
│         │  Knowledge      │                                      │
│         │  State Output   │                                      │
│         └─────────────────┘                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Mathematical Formulation

#### Graph Convolution Operation

For each concept node $c_i$, the GNN aggregates information from neighboring concepts:

$$h_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i d_j}} W^{(l)} h_j^{(l)} \right)$$

Where:
- $h_i^{(l)}$ = hidden state of concept $i$ at layer $l$
- $\mathcal{N}(i)$ = neighbors of concept $i$ (prerequisites + related)
- $d_i, d_j$ = degrees of nodes $i$ and $j$
- $W^{(l)}$ = learnable weight matrix
- $\sigma$ = activation function (ReLU)

#### Multi-View Aggregation

MVGKT uses multiple views of the knowledge graph:

1. **Prerequisite View**: Directed edges for learning dependencies
2. **Similarity View**: Undirected edges for related concepts
3. **Difficulty View**: Edges weighted by difficulty proximity

$$H_{final} = \text{Concat}(H_{prereq}, H_{similar}, H_{diff}) \cdot W_{agg}$$

#### Knowledge State Propagation

When a student answers question $q$ testing concepts $\{c_1, c_2, ..., c_k\}$:

$$\Delta h_{c_i} = \alpha \cdot r \cdot \sum_{j \in \mathcal{N}(c_i)} w_{ij} \cdot h_{c_j}$$

Where:
- $r$ = response correctness (1 or 0)
- $\alpha$ = learning rate parameter
- $w_{ij}$ = edge weight between concepts

### 1.4 Key Components

#### 1.4.1 Concept Embedding Layer
```python
class ConceptEmbedding(nn.Module):
    def __init__(self, num_concepts, embed_dim):
        self.concept_embed = nn.Embedding(num_concepts, embed_dim)
        self.difficulty_embed = nn.Embedding(5, embed_dim // 4)  # 5 difficulty levels

    def forward(self, concept_ids, difficulties):
        c_emb = self.concept_embed(concept_ids)
        d_emb = self.difficulty_embed(difficulties)
        return torch.cat([c_emb, d_emb], dim=-1)
```

#### 1.4.2 Graph Attention Layer
```python
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4):
        self.attention = nn.MultiheadAttention(in_features, num_heads)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        # Compute attention weights based on graph structure
        attn_mask = (adj_matrix == 0)  # Mask non-neighbors
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask)
        return self.linear(attn_output)
```

### 1.5 Goals & Objectives

| Goal | Description | How GNN Achieves It |
|------|-------------|---------------------|
| **Capture Prerequisites** | Model learning dependencies | Directed edges in graph |
| **Handle Multi-Concept Questions** | Questions testing multiple skills | Aggregate across concept nodes |
| **Transfer Learning** | Knowledge in one area helps another | Message passing between related nodes |
| **Explicit Interpretability** | Understand why predictions are made | Edge weights show relationships |

### 1.6 Training Process

1. **Graph Construction**: Build adjacency matrix from prerequisite data
2. **Forward Pass**: Propagate information through GNN layers
3. **Prediction**: Predict probability of correct response
4. **Loss**: Binary cross-entropy between prediction and actual response
5. **Backpropagation**: Update both GNN weights and concept embeddings

---

## 2. Memory Networks (DKVMN) Module

### 2.1 Background: DKVMN

**DKVMN** (Dynamic Key-Value Memory Networks) achieves **AUC: 0.9190** by maintaining student-specific memory that evolves with each interaction. It separates "what to store" (keys) from "how much is known" (values).

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 DKVMN Module Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Memory Matrix                         │    │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐    │    │
│  │  │  Key 1  │  Key 2  │  Key 3  │   ...   │  Key N  │    │    │
│  │  │(concept)│(concept)│(concept)│         │(concept)│    │    │
│  │  ├─────────┼─────────┼─────────┼─────────┼─────────┤    │    │
│  │  │ Value 1 │ Value 2 │ Value 3 │   ...   │ Value N │    │    │
│  │  │(mastery)│(mastery)│(mastery)│         │(mastery)│    │    │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Input: Question q    ┌─────────────┐                           │
│         ─────────────►│   Read      │──────► Knowledge State     │
│                       │  Operation  │        (for prediction)    │
│                       └─────────────┘                           │
│                              │                                   │
│  After Response r     ┌─────────────┐                           │
│         ─────────────►│   Write     │──────► Updated Memory      │
│                       │  Operation  │                            │
│                       └─────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Mathematical Formulation

#### Memory Structure

- **Key Matrix** $M_k \in \mathbb{R}^{N \times d_k}$: Static concept representations
- **Value Matrix** $M_v \in \mathbb{R}^{N \times d_v}$: Dynamic mastery states (per student)

#### Read Operation

When student encounters question $q$ with embedding $e_q$:

1. **Attention Weights**:
$$w_i = \text{softmax}\left(\frac{e_q \cdot M_k[i]}{\sqrt{d_k}}\right)$$

2. **Read Vector** (current knowledge state):
$$r_t = \sum_{i=1}^{N} w_i \cdot M_v[i]$$

3. **Prediction**:
$$p(correct) = \sigma(W_p \cdot [r_t; e_q] + b_p)$$

#### Write Operation

After observing response $r_t$ (correct/incorrect):

1. **Erase Vector** (what to forget):
$$e_t = \sigma(W_e \cdot [e_q; r_t] + b_e)$$

2. **Add Vector** (what to learn):
$$a_t = \tanh(W_a \cdot [e_q; r_t] + b_a)$$

3. **Memory Update**:
$$M_v[i] \leftarrow M_v[i] \cdot (1 - w_i \cdot e_t) + w_i \cdot a_t$$

### 2.4 Key Components

#### 2.4.1 Memory Initialization
```python
class DKVMNMemory(nn.Module):
    def __init__(self, num_concepts, key_dim, value_dim):
        # Static key matrix (shared across students)
        self.key_matrix = nn.Parameter(torch.randn(num_concepts, key_dim))

        # Value matrix initialized per student
        self.value_dim = value_dim

    def init_student_memory(self, batch_size):
        # Initialize value matrix for new students
        return torch.zeros(batch_size, self.num_concepts, self.value_dim)
```

#### 2.4.2 Read Head
```python
class ReadHead(nn.Module):
    def __init__(self, key_dim, value_dim):
        self.query_proj = nn.Linear(key_dim, key_dim)

    def forward(self, query, key_matrix, value_matrix):
        # Compute attention
        query = self.query_proj(query)
        attention = F.softmax(torch.matmul(query, key_matrix.T) / sqrt(key_dim), dim=-1)

        # Read from memory
        read_vector = torch.matmul(attention, value_matrix)
        return read_vector, attention
```

#### 2.4.3 Write Head
```python
class WriteHead(nn.Module):
    def __init__(self, input_dim, value_dim):
        self.erase_layer = nn.Linear(input_dim, value_dim)
        self.add_layer = nn.Linear(input_dim, value_dim)

    def forward(self, input_vector, attention, value_matrix):
        # Compute erase and add vectors
        erase = torch.sigmoid(self.erase_layer(input_vector))
        add = torch.tanh(self.add_layer(input_vector))

        # Update memory
        erase_term = value_matrix * (1 - attention.unsqueeze(-1) * erase.unsqueeze(1))
        add_term = attention.unsqueeze(-1) * add.unsqueeze(1)

        return erase_term + add_term
```

### 2.5 Goals & Objectives

| Goal | Description | How DKVMN Achieves It |
|------|-------------|----------------------|
| **Student Personalization** | Each student has unique state | Separate value matrix per student |
| **Knowledge Persistence** | Remember past interactions | Memory persists across time |
| **Selective Updates** | Only update relevant concepts | Attention-based writing |
| **Interpretable States** | Understand what student knows | Value vectors = mastery levels |

### 2.6 Advantages Over Traditional Methods

1. **Dynamic Updates**: Memory evolves with each interaction
2. **Soft Attention**: Partial updates to related concepts
3. **Scalability**: Fixed memory size regardless of history length
4. **Cold Start**: Can initialize from prior knowledge

---

## 3. Attention Module (AKT-style)

### 3.1 Background: AKT

**AKT** (Attentive Knowledge Tracing) achieves **AUC: 0.8346** by using self-attention to model temporal dependencies and incorporating explicit forgetting mechanisms. It extends the Transformer architecture for educational data.

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  AKT Module Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Sequence: [(q₁,r₁,t₁), (q₂,r₂,t₂), ..., (qₜ,rₜ,tₜ)]     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Embedding Layer                             │    │
│  │  Question Embed + Response Embed + Position Embed        │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Multi-Head Self-Attention                      │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │    │
│  │  │ Head 1  │  │ Head 2  │  │ Head 3  │  │ Head 4  │     │    │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘     │    │
│  │       └──────┬─────┴──────┬─────┴──────┬─────┘          │    │
│  │              ▼            ▼            ▼                 │    │
│  │         ┌─────────────────────────────────┐              │    │
│  │         │      Concat + Linear            │              │    │
│  │         └─────────────────────────────────┘              │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Forgetting Layer                            │    │
│  │         Exponential decay based on time gap              │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Difficulty-Aware Layer                         │    │
│  │     Modulate attention by question difficulty            │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Feed-Forward Network                        │    │
│  │           + Residual Connection + LayerNorm              │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│                    Prediction Layer                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Mathematical Formulation

#### Input Embedding

For interaction at time $t$ with question $q_t$, response $r_t$:

$$x_t = E_q[q_t] + E_r[r_t] + E_{pos}[t]$$

Where:
- $E_q$ = question embedding matrix
- $E_r$ = response embedding (correct/incorrect)
- $E_{pos}$ = positional encoding

#### Self-Attention with Forgetting

Standard attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

AKT adds **temporal decay** to attention weights:

$$\alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k}) \cdot \gamma^{\Delta t_{ij}}}{\sum_k \exp(q_i \cdot k_k / \sqrt{d_k}) \cdot \gamma^{\Delta t_{ik}}}$$

Where:
- $\Delta t_{ij} = t_i - t_j$ = time gap between interactions
- $\gamma \in (0, 1)$ = forgetting factor (learned or fixed)

#### Difficulty-Aware Attention

AKT modulates attention based on question difficulty:

$$\alpha'_{ij} = \alpha_{ij} \cdot \sigma(W_d \cdot [d_i; d_j])$$

Where $d_i, d_j$ are difficulty embeddings of questions $i$ and $j$.

#### Rasch Model Integration

AKT incorporates Item Response Theory (IRT):

$$p(r_t = 1) = \sigma(\theta_t - \beta_{q_t})$$

Where:
- $\theta_t$ = student ability (from attention output)
- $\beta_{q_t}$ = question difficulty parameter

### 3.4 Key Components

#### 3.4.1 Temporal Position Encoding
```python
class TemporalPositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, timestamps=None):
        if timestamps is not None:
            # Use actual timestamps for position
            return x + self.pe[timestamps]
        return x + self.pe[:x.size(1)]
```

#### 3.4.2 Forgetting-Aware Attention
```python
class ForgettingAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.forget_factor = nn.Parameter(torch.tensor(0.9))  # Learnable

    def forward(self, query, key, value, time_gaps):
        # Compute standard attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_model)

        # Apply temporal decay
        decay = self.forget_factor ** time_gaps
        attn_weights = attn_weights * decay

        # Softmax and apply
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights
```

#### 3.4.3 Difficulty Integration
```python
class DifficultyAwareLayer(nn.Module):
    def __init__(self, d_model, num_difficulties=5):
        self.diff_embed = nn.Embedding(num_difficulties, d_model)
        self.diff_attention = nn.Linear(d_model * 2, 1)

    def forward(self, hidden_states, difficulties):
        diff_emb = self.diff_embed(difficulties)

        # Modulate hidden states by difficulty
        combined = torch.cat([hidden_states, diff_emb], dim=-1)
        diff_weight = torch.sigmoid(self.diff_attention(combined))

        return hidden_states * diff_weight
```

### 3.5 Goals & Objectives

| Goal | Description | How AKT Achieves It |
|------|-------------|---------------------|
| **Model Forgetting** | Students forget over time | Exponential decay in attention |
| **Difficulty Adaptation** | Harder questions = different learning | Difficulty-aware attention |
| **Long-Range Dependencies** | Learn from distant past | Self-attention mechanism |
| **Sequence Modeling** | Order of learning matters | Positional encoding |

### 3.6 Forgetting Curves

AKT can model different forgetting patterns:

1. **Exponential Decay**: $m(t) = m_0 \cdot e^{-\lambda t}$
2. **Power Law**: $m(t) = m_0 \cdot t^{-\alpha}$
3. **Learned Decay**: Network learns optimal decay per concept

---

## 4. LLM Integration (LKT-style) Module

### 4.1 Background: LKT

**LKT** (Large Language Model Knowledge Tracing) achieves **AUC: 0.8513** by leveraging pre-trained language models to understand semantic content of questions and student queries. It excels at cold-start scenarios and natural language understanding.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  LKT Module Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                Pre-trained LLM                           │    │
│  │            (BERT / GPT / LLaMA / etc.)                   │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │  Transformer Layers (frozen or fine-tuned)      │    │    │
│  │  │  [Layer 1] → [Layer 2] → ... → [Layer N]        │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  Input Types:                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐   │
│  │ Question Text    │  │ Student Query    │  │ Concept Desc │   │
│  │ "Solve 2x + 3"   │  │ "I don't get it" │  │ "Algebra..." │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬───────┘   │
│           │                     │                    │           │
│           ▼                     ▼                    ▼           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Semantic Feature Extraction                 │    │
│  │         [CLS] token or Mean Pooling                      │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Projection Layers                           │    │
│  │    LLM embedding space → KT embedding space              │    │
│  └────────────────────────┬────────────────────────────────┘    │
│                           ▼                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ Question      │  │ Misconception │  │ Knowledge     │        │
│  │ Difficulty    │  │ Detection     │  │ Gap Analysis  │        │
│  │ Prediction    │  │               │  │               │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Mathematical Formulation

#### Text Encoding

Given question text $T = [t_1, t_2, ..., t_n]$:

$$H = \text{LLM}(T) \in \mathbb{R}^{n \times d_{llm}}$$

$$e_{text} = \text{Pool}(H) \in \mathbb{R}^{d_{llm}}$$

Pooling strategies:
- **CLS Token**: $e_{text} = H[0]$ (BERT-style)
- **Mean Pooling**: $e_{text} = \frac{1}{n}\sum_i H[i]$
- **Attention Pooling**: $e_{text} = \sum_i \alpha_i H[i]$

#### Projection to KT Space

$$e_{kt} = W_{proj} \cdot e_{text} + b_{proj}$$

Where $W_{proj} \in \mathbb{R}^{d_{kt} \times d_{llm}}$ maps LLM embeddings to KT model dimension.

#### Semantic Similarity for Cold Start

For new question $q_{new}$ without interaction history:

$$\text{sim}(q_{new}, c_i) = \frac{e_{q_{new}} \cdot e_{c_i}}{||e_{q_{new}}|| \cdot ||e_{c_i}||}$$

Initial difficulty estimate:
$$\hat{d}_{q_{new}} = \sum_i \text{sim}(q_{new}, c_i) \cdot d_{c_i}$$

#### Misconception Detection

Given student query $Q$, detect misconceptions:

$$P(misconception_j | Q) = \sigma(W_m \cdot [e_Q; e_{concept}] + b_m)$$

### 4.4 Key Components

#### 4.4.1 LLM Encoder
```python
class LLMEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, texts):
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors='pt')

        # Get embeddings
        outputs = self.model(**inputs)

        # Use [CLS] token
        return outputs.last_hidden_state[:, 0, :]
```

#### 4.4.2 Query Analyzer
```python
class QueryAnalyzer(nn.Module):
    def __init__(self, llm_dim, num_query_types=8, num_sentiments=5):
        self.llm_encoder = LLMEncoder()

        # Query type classifier
        self.query_classifier = nn.Sequential(
            nn.Linear(llm_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_query_types)
        )

        # Sentiment classifier
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(llm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_sentiments)
        )

        # Misconception detector
        self.misconception_detector = nn.Sequential(
            nn.Linear(llm_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, query_text, concept_embeddings):
        query_emb = self.llm_encoder(query_text)

        # Classify query type
        query_type = self.query_classifier(query_emb)

        # Detect sentiment
        sentiment = self.sentiment_classifier(query_emb)

        # Check for misconceptions
        combined = torch.cat([query_emb.unsqueeze(1).expand(-1, len(concept_embeddings), -1),
                             concept_embeddings], dim=-1)
        misconceptions = self.misconception_detector(combined)

        return {
            'query_type': query_type,
            'sentiment': sentiment,
            'misconceptions': misconceptions,
            'embedding': query_emb
        }
```

#### 4.4.3 Cold Start Handler
```python
class ColdStartHandler(nn.Module):
    def __init__(self, llm_dim, kt_dim, num_concepts):
        self.projection = nn.Linear(llm_dim, kt_dim)
        self.concept_embeddings = nn.Embedding(num_concepts, kt_dim)

    def estimate_initial_state(self, question_texts, concept_descriptions):
        # Get LLM embeddings for questions
        q_emb = self.llm_encoder(question_texts)
        q_proj = self.projection(q_emb)

        # Get concept embeddings
        c_emb = self.llm_encoder(concept_descriptions)
        c_proj = self.projection(c_emb)

        # Compute similarity-based initial estimates
        similarity = F.cosine_similarity(q_proj.unsqueeze(1),
                                         c_proj.unsqueeze(0), dim=-1)

        # Return weighted concept associations
        return F.softmax(similarity, dim=-1)
```

### 4.5 Query Types & Sentiments

#### Query Types Detected
| Type | Example | Indicates |
|------|---------|-----------|
| `explanation` | "Can you explain...?" | Needs conceptual understanding |
| `clarification` | "I'm confused about..." | Partial understanding |
| `example` | "Show me an example" | Needs concrete instances |
| `practice` | "Give me a problem" | Ready to apply |
| `definition` | "What is...?" | Basic knowledge gap |
| `application` | "When would I use...?" | Seeking real-world connection |
| `comparison` | "What's the difference...?" | Organizing knowledge |
| `verification` | "Is this correct...?" | Checking understanding |

#### Sentiments Detected
| Sentiment | Indicators | Action |
|-----------|------------|--------|
| `confident` | "I think I understand" | Provide challenge |
| `curious` | "I wonder if..." | Encourage exploration |
| `confused` | "I don't get it" | Simplify explanation |
| `frustrated` | "This is impossible" | Provide encouragement + hints |
| `neutral` | Direct questions | Standard response |

### 4.6 Goals & Objectives

| Goal | Description | How LKT Achieves It |
|------|-------------|---------------------|
| **Cold Start** | Handle new questions/students | Semantic similarity |
| **NLU** | Understand student queries | Pre-trained language models |
| **Misconception Detection** | Identify wrong mental models | Query analysis |
| **Rich Representations** | Capture semantic nuances | LLM embeddings |
| **Transfer Learning** | Leverage general knowledge | Pre-trained weights |

---

## 5. Hybrid Integration

### 5.1 Combining All Modules

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid KT Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Student Interaction (question, response, query, time)    │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │     GNN     │  │    DKVMN    │  │     AKT     │             │
│  │   Module    │  │   Module    │  │   Module    │             │
│  │             │  │             │  │             │             │
│  │ Concept     │  │ Student     │  │ Temporal    │             │
│  │ Relations   │  │ Memory      │  │ Attention   │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│                  ┌─────────────┐                                │
│                  │   Fusion    │◄─────┐                         │
│                  │   Layer     │      │                         │
│                  └──────┬──────┘      │                         │
│                         │             │                         │
│                         ▼             │                         │
│                  ┌─────────────┐  ┌───┴───────┐                 │
│                  │ Prediction  │  │    LLM    │                 │
│                  │   Head      │  │  Module   │                 │
│                  └──────┬──────┘  │           │                 │
│                         │         │ Cold Start│                 │
│                         ▼         │ + Queries │                 │
│                  P(correct)       └───────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Fusion Strategy

```python
class HybridFusion(nn.Module):
    def __init__(self, gnn_dim, dkvmn_dim, akt_dim, llm_dim, output_dim):
        self.gnn_proj = nn.Linear(gnn_dim, output_dim)
        self.dkvmn_proj = nn.Linear(dkvmn_dim, output_dim)
        self.akt_proj = nn.Linear(akt_dim, output_dim)
        self.llm_proj = nn.Linear(llm_dim, output_dim)

        # Learnable weights for each module
        self.module_weights = nn.Parameter(torch.ones(4) / 4)

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, gnn_out, dkvmn_out, akt_out, llm_out=None):
        # Project to common space
        g = self.gnn_proj(gnn_out)
        d = self.dkvmn_proj(dkvmn_out)
        a = self.akt_proj(akt_out)

        # Normalize weights
        weights = F.softmax(self.module_weights, dim=0)

        # Weighted combination
        if llm_out is not None:
            l = self.llm_proj(llm_out)
            fused = weights[0]*g + weights[1]*d + weights[2]*a + weights[3]*l
        else:
            fused = weights[0]*g + weights[1]*d + weights[2]*a

        return self.fusion(fused)
```

### 5.3 Training Objective

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{gnn} + \lambda_2 \mathcal{L}_{mem} + \lambda_3 \mathcal{L}_{reg}$$

Where:
- $\mathcal{L}_{pred}$ = Binary cross-entropy for response prediction
- $\mathcal{L}_{gnn}$ = Graph structure preservation loss
- $\mathcal{L}_{mem}$ = Memory consistency regularization
- $\mathcal{L}_{reg}$ = L2 regularization

---

## 6. Performance Comparison

| Model | AUC | Strengths | Weaknesses |
|-------|-----|-----------|------------|
| **MVGKT (GNN)** | 0.9470 | Multi-concept, interpretable | Requires graph construction |
| **DKVMN** | 0.9190 | Personalization, memory | Fixed memory size |
| **AKT** | 0.8346 | Forgetting, attention | Computationally expensive |
| **LKT** | 0.8513 | Cold start, NLU | Requires text data |
| **Hybrid** | ~0.95+ | Best of all | Complexity |

---

## 7. References & Papers

### 7.1 Core Model Papers

#### GNN-Based Models

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **GIKT: A Graph-based Interaction Model for Knowledge Tracing** | 2020 | [arXiv:2009.05991](https://arxiv.org/abs/2009.05991) | First GNN for KT, models question-skill interactions |
| **Graph-based Knowledge Tracing: Modeling Student Proficiency Using GNN** | 2021 | [IEEE](https://ieeexplore.ieee.org/document/9415160) | Graph convolution for concept dependencies |
| **MVGKT: Multi-View Graph Knowledge Tracing** | 2023 | [Paper](https://dl.acm.org/doi/10.1145/3534678.3539399) | State-of-the-art, multiple graph views |
| **SKT: Structure-aware Knowledge Tracing** | 2022 | [arXiv:2201.00453](https://arxiv.org/abs/2201.00453) | Hierarchical concept structure |

#### Memory Network Models

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **DKVMN: Dynamic Key-Value Memory Networks for Knowledge Tracing** | 2017 | [arXiv:1611.08108](https://arxiv.org/abs/1611.08108) | Original DKVMN paper - **Must Read** |
| **SKVMN: Sequential Key-Value Memory Networks** | 2019 | [IJCAI](https://www.ijcai.org/proceedings/2019/0214.pdf) | Adds sequential dependencies to DKVMN |
| **Deep-IRT: Deep Item Response Theory** | 2019 | [arXiv:1904.05033](https://arxiv.org/abs/1904.05033) | Combines DKVMN with IRT |
| **LPKT: Learning Process-consistent Knowledge Tracing** | 2021 | [KDD](https://dl.acm.org/doi/10.1145/3447548.3467237) | Models learning/forgetting processes |

#### Attention-Based Models

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **SAKT: Self-Attentive Knowledge Tracing** | 2019 | [arXiv:1907.06837](https://arxiv.org/abs/1907.06837) | First transformer for KT |
| **AKT: Context-Aware Attentive Knowledge Tracing** | 2020 | [arXiv:2007.12324](https://arxiv.org/abs/2007.12324) | Adds difficulty & forgetting - **Must Read** |
| **SAINT: Separated Self-AttentIve Neural Knowledge Tracing** | 2020 | [arXiv:2010.12042](https://arxiv.org/abs/2010.12042) | Separate encoders for exercises/responses |
| **SAINT+: Integrating Temporal Features** | 2021 | [LAK](https://dl.acm.org/doi/10.1145/3448139.3448188) | Adds elapsed time and lag time |
| **simpleKT: A Simple But Tough-to-Beat Baseline** | 2022 | [arXiv:2203.02099](https://arxiv.org/abs/2203.02099) | Simplified attention, strong baseline |
| **sparseKT: Sparse Attention Knowledge Tracing** | 2023 | [arXiv:2303.02185](https://arxiv.org/abs/2303.02185) | Efficient sparse attention |

#### LLM-Based Models

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **LKT: Large Language Model Enhanced Knowledge Tracing** | 2024 | [arXiv:2402.01789](https://arxiv.org/abs/2402.01789) | LLM embeddings for cold start |
| **CLKT: Contrastive Learning for Knowledge Tracing** | 2022 | [WWW](https://dl.acm.org/doi/10.1145/3485447.3512105) | Contrastive pre-training |
| **GPT4KT: GPT-4 for Knowledge Tracing** | 2024 | [arXiv:2401.15582](https://arxiv.org/abs/2401.15582) | Zero-shot KT with GPT-4 |
| **QDKT: Question-aware Deep Knowledge Tracing** | 2023 | [arXiv:2305.10165](https://arxiv.org/abs/2305.10165) | Question text understanding |

#### Foundational Papers

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **DKT: Deep Knowledge Tracing** | 2015 | [arXiv:1506.05908](https://arxiv.org/abs/1506.05908) | **Foundational paper** - First deep learning for KT |
| **BKT: Bayesian Knowledge Tracing** | 1994 | [Paper](https://link.springer.com/article/10.1007/BF01099821) | Classical probabilistic approach |
| **IRT: Item Response Theory** | 1968 | [Book](https://www.springer.com/gp/book/9780387946610) | Psychometric foundation |
| **KT Survey: A Survey of Knowledge Tracing** | 2022 | [arXiv:2201.06953](https://arxiv.org/abs/2201.06953) | Comprehensive survey - **Must Read** |

### 7.2 Forgetting Models

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **DKT-Forget: Augmenting DKT with Forgetting** | 2019 | [L@S](https://dl.acm.org/doi/10.1145/3330430.3333635) | Explicit forgetting in DKT |
| **HawkesKT: Temporal Dynamics via Hawkes Process** | 2021 | [WSDM](https://dl.acm.org/doi/10.1145/3437963.3441802) | Continuous-time forgetting |
| **FoLiBiKT: Forgetting-aware Long-interval Bi-directional KT** | 2023 | [Paper](https://dl.acm.org/doi/10.1145/3580305.3599414) | Bidirectional with forgetting |
| **RKT: Relation-aware Knowledge Tracing** | 2020 | [arXiv:2008.00928](https://arxiv.org/abs/2008.00928) | Forgetting + concept relations |

### 7.3 Interpretability & Explainability

| Paper | Year | Link | Description |
|-------|------|------|-------------|
| **Interpretable KT: Towards Interpretable Deep KT** | 2019 | [arXiv:1906.04482](https://arxiv.org/abs/1906.04482) | Attention visualization |
| **QIKT: Question-centric Interpretable KT** | 2023 | [arXiv:2302.06885](https://arxiv.org/abs/2302.06885) | Question-level interpretation |
| **Explainable KT: A Survey** | 2023 | [arXiv:2305.04304](https://arxiv.org/abs/2305.04304) | Survey on explainability |

### 7.4 Datasets

| Dataset | Records | Link | Description |
|---------|---------|------|-------------|
| **ASSISTments 2009** | 340K | [Harvard Dataverse](https://sites.google.com/site/assistmaborern/datasets) | Math tutoring - **Used in this project** |
| **ASSISTments 2012** | 2.7M | [Harvard Dataverse](https://sites.google.com/site/assistmaborern/datasets) | Larger version |
| **ASSISTments 2015** | 700K | [Harvard Dataverse](https://sites.google.com/site/assistmaborern/datasets) | 100 skills |
| **EdNet** | 130M | [GitHub](https://github.com/riiid/ednet) | Largest KT dataset |
| **Junyi Academy** | 25M | [PSLC](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1275) | Chinese math platform |
| **Statics2011** | 190K | [PSLC](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507) | Engineering statics |
| **Algebra 2005** | 800K | [PSLC](https://pslcdatashop.web.cmu.edu/KDDCup/) | KDD Cup dataset |

### 7.5 Recommended Reading Order

**For beginners:**
1. [DKT (2015)](https://arxiv.org/abs/1506.05908) - Understand the foundation
2. [DKVMN (2017)](https://arxiv.org/abs/1611.08108) - Memory networks approach
3. [SAKT (2019)](https://arxiv.org/abs/1907.06837) - Attention mechanism
4. [KT Survey (2022)](https://arxiv.org/abs/2201.06953) - Comprehensive overview

**For this project's models:**
1. [AKT (2020)](https://arxiv.org/abs/2007.12324) - Attention + Forgetting
2. [GIKT (2020)](https://arxiv.org/abs/2009.05991) - Graph-based approach
3. [DKVMN (2017)](https://arxiv.org/abs/1611.08108) - Memory networks
4. [LKT (2024)](https://arxiv.org/abs/2402.01789) - LLM integration

### 7.6 Code Repositories

| Repository | Link | Models |
|------------|------|--------|
| **pyKT** | [GitHub](https://github.com/pykt-team/pykt-toolkit) | 20+ KT models, benchmarks |
| **EduKTM** | [GitHub](https://github.com/bigdata-ustc/EduKTM) | Educational KT models |
| **KnowledgeTracing** | [GitHub](https://github.com/jennyzhang0215/KnowledgeTracing) | DKT, DKVMN implementations |
| **AKT Official** | [GitHub](https://github.com/arghosh/AKT) | Official AKT code |
| **SAINT Official** | [GitHub](https://github.com/Shivanandmn/SAINT_pytorch) | PyTorch SAINT |

### 7.7 Tutorials & Resources

| Resource | Link | Description |
|----------|------|-------------|
| **KT Tutorial (EDM 2021)** | [Slides](https://www.educationaldatamining.org/EDM2021/proceedings/2021.EDM-tutorials.1.pdf) | Comprehensive tutorial |
| **Deep Learning for Education** | [Course](https://www.coursera.org/learn/deep-learning-in-education) | Coursera course |
| **pyKT Documentation** | [Docs](https://pykt-team.github.io/pykt-toolkit/) | Implementation guide |
| **Awesome KT** | [GitHub](https://github.com/pykt-team/awesome-kt) | Curated paper list |

---

## 8. Implementation Notes

### Hardware Requirements
- **GNN**: GPU recommended for graph operations
- **DKVMN**: Moderate memory for student value matrices
- **AKT**: High memory for attention matrices
- **LLM**: GPU required for inference, or use API

### Recommended Hyperparameters
```yaml
gnn:
  num_layers: 2
  hidden_dim: 128
  dropout: 0.3

dkvmn:
  key_dim: 64
  value_dim: 64
  num_slots: 50

akt:
  num_heads: 4
  num_layers: 2
  d_model: 256
  forget_factor: 0.9

llm:
  model: "bert-base-uncased"
  freeze: true
  projection_dim: 256
```
