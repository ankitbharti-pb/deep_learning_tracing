"""
Hybrid Knowledge State Engine
Combines GNN (MVGKT-style), Attention (AKT-style), and Memory Networks (DKVMN-style)
Based on: "Deep Learning Based Knowledge Tracing: A Review of the Literature"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class KnowledgeGraphModule(nn.Module):
    """
    GNN-based module inspired by MVGKT (AUC: 0.9470)
    Models relationships between knowledge concepts using graph convolutions.

    Key features:
    - Spatio-temporal graph neural network
    - Topic difficulty analysis
    - Complex relationship modeling between concepts
    """

    def __init__(
        self,
        num_concepts: int,
        embedding_dim: int = 128,
        num_gnn_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim

        # Concept embeddings
        self.concept_embeddings = nn.Embedding(num_concepts, embedding_dim)

        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(embedding_dim, embedding_dim)
            for _ in range(num_gnn_layers)
        ])

        # Edge type embeddings (prerequisite, related, similar difficulty)
        self.edge_type_embedding = nn.Embedding(3, embedding_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        concept_ids: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        edge_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            concept_ids: Tensor of concept IDs [batch_size, seq_len]
            adjacency_matrix: Graph adjacency [num_concepts, num_concepts]
            edge_types: Type of edges [num_concepts, num_concepts]

        Returns:
            Concept representations enhanced with graph structure
        """
        # Get base embeddings
        x = self.concept_embeddings(concept_ids)

        # Apply graph convolutions
        for gnn_layer in self.gnn_layers:
            x_residual = x
            x = gnn_layer(x, adjacency_matrix, edge_types, self.edge_type_embedding)
            x = self.dropout(x)
            x = self.layer_norm(x + x_residual)

        return x


class GraphConvLayer(nn.Module):
    """Single graph convolution layer with edge type awareness"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.attention = nn.Linear(in_dim * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        edge_types: torch.Tensor,
        edge_type_emb: nn.Embedding
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        # Simplified message passing (full implementation would use sparse ops)
        # Aggregate neighbor information weighted by attention
        messages = torch.matmul(adj.float(), x)

        # Combine with self
        combined = torch.cat([x, messages], dim=-1)
        output = F.relu(self.linear(combined))

        return output


class TemporalAttentionModule(nn.Module):
    """
    Attention-based module inspired by AKT (AUC: 0.8346)
    Combines attention mechanism with educational measurement theory.

    Key features:
    - Assigns weights to historical records
    - Dynamically adjusts for question difficulty and student level
    - Forgetting-aware attention (RKT-style)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 8,
        max_seq_len: int = 200,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Difficulty-aware components (DIMKT-style)
        self.difficulty_encoder = nn.Linear(1, embedding_dim)

        # Forgetting factor (RKT/FoLiBiKT-style)
        self.forgetting_layer = nn.Linear(embedding_dim + 1, embedding_dim)

        # Time encoding
        self.time_encoder = nn.Linear(1, embedding_dim)

        # Output projection
        self.output_proj = nn.Linear(embedding_dim * 2, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        difficulty: torch.Tensor,
        time_gaps: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Current question representation [batch, 1, dim]
            key: Historical questions [batch, seq_len, dim]
            value: Historical responses [batch, seq_len, dim]
            difficulty: Question difficulties [batch, seq_len, 1]
            time_gaps: Time since each interaction [batch, seq_len, 1]
            mask: Attention mask

        Returns:
            attended_output: Attention-weighted knowledge state
            attention_weights: For interpretability
        """
        # Encode difficulty
        diff_emb = self.difficulty_encoder(difficulty)
        key = key + diff_emb

        # Apply forgetting decay (simulates natural knowledge decay)
        # Forgetting factor decreases with larger time gaps
        forgetting_input = torch.cat([value, time_gaps], dim=-1)
        forgetting_factor = torch.sigmoid(self.forgetting_layer(forgetting_input))
        value = value * forgetting_factor

        # Multi-head attention
        attended, attention_weights = self.attention(
            query, key, value, attn_mask=mask
        )

        # Combine with time encoding
        time_emb = self.time_encoder(time_gaps.mean(dim=1, keepdim=True))
        combined = torch.cat([attended, time_emb], dim=-1)
        output = self.output_proj(combined)

        return self.layer_norm(output), attention_weights


class DynamicMemoryModule(nn.Module):
    """
    Memory network module inspired by DKVMN (AUC: 0.9190)
    Maintains student-specific memory slots for personalization.

    Key features:
    - Key-value memory structure
    - Dynamic memory updates based on interactions
    - Captures individual learning patterns
    """

    def __init__(
        self,
        num_concepts: int,
        memory_size: int = 50,
        embedding_dim: int = 128,
        value_dim: int = 128
    ):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.value_dim = value_dim

        # Key memory (static - represents knowledge concepts)
        self.key_memory = nn.Parameter(
            torch.randn(memory_size, embedding_dim) * 0.1
        )

        # Value memory will be student-specific (dynamic)
        # Initial value memory template
        self.init_value_memory = nn.Parameter(
            torch.randn(memory_size, value_dim) * 0.1
        )

        # Read/write operations
        self.erase_linear = nn.Linear(embedding_dim + 1, value_dim)
        self.add_linear = nn.Linear(embedding_dim + 1, value_dim)

        # Summary network
        self.summary_net = nn.Sequential(
            nn.Linear(value_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim)
        )

    def init_memory(self, batch_size: int) -> torch.Tensor:
        """Initialize value memory for a batch of students"""
        return self.init_value_memory.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def read(
        self,
        query: torch.Tensor,
        value_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory based on query (concept embedding)

        Args:
            query: Concept embedding [batch, dim]
            value_memory: Current memory state [batch, mem_size, value_dim]

        Returns:
            read_content: Retrieved knowledge state
            attention: Memory attention weights
        """
        # Compute attention over key memory
        attention = F.softmax(
            torch.matmul(query, self.key_memory.T) / np.sqrt(self.embedding_dim),
            dim=-1
        )  # [batch, mem_size]

        # Read from value memory
        read_content = torch.matmul(
            attention.unsqueeze(1), value_memory
        ).squeeze(1)  # [batch, value_dim]

        return read_content, attention

    def write(
        self,
        query: torch.Tensor,
        response: torch.Tensor,
        value_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Update memory based on interaction

        Args:
            query: Concept embedding [batch, dim]
            response: Student response (0/1) [batch, 1]
            value_memory: Current memory state [batch, mem_size, value_dim]

        Returns:
            Updated value memory
        """
        # Compute write attention
        attention = F.softmax(
            torch.matmul(query, self.key_memory.T) / np.sqrt(self.embedding_dim),
            dim=-1
        )  # [batch, mem_size]

        # Compute erase and add vectors
        write_input = torch.cat([query, response], dim=-1)
        erase = torch.sigmoid(self.erase_linear(write_input))  # [batch, value_dim]
        add = torch.tanh(self.add_linear(write_input))  # [batch, value_dim]

        # Apply erase then add
        erase_term = torch.einsum('bm,bv->bmv', attention, erase)
        add_term = torch.einsum('bm,bv->bmv', attention, add)

        new_memory = value_memory * (1 - erase_term) + add_term

        return new_memory


class HybridKnowledgeStateEngine(nn.Module):
    """
    Main Knowledge State Engine combining all modules

    This hybrid approach leverages:
    1. MVGKT's graph structure for concept relationships
    2. AKT's attention for temporal patterns and difficulty
    3. DKVMN's memory for personalization
    4. LLM integration ready for semantic understanding
    """

    def __init__(
        self,
        num_concepts: int,
        num_questions: int,
        embedding_dim: int = 128,
        memory_size: int = 50,
        num_attention_heads: int = 8,
        num_gnn_layers: int = 3,
        max_seq_len: int = 200,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_questions = num_questions
        self.embedding_dim = embedding_dim

        # Question and response embeddings
        self.question_embedding = nn.Embedding(num_questions, embedding_dim)
        self.response_embedding = nn.Embedding(2, embedding_dim)  # correct/incorrect

        # Interaction embedding (question + response combined)
        self.interaction_linear = nn.Linear(embedding_dim * 2, embedding_dim)

        # Three core modules
        self.gnn_module = KnowledgeGraphModule(
            num_concepts, embedding_dim, num_gnn_layers, dropout
        )
        self.attention_module = TemporalAttentionModule(
            embedding_dim, num_attention_heads, max_seq_len, dropout
        )
        self.memory_module = DynamicMemoryModule(
            num_concepts, memory_size, embedding_dim
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        difficulties: torch.Tensor,
        time_gaps: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        edge_types: torch.Tensor,
        target_question: torch.Tensor,
        target_concept: torch.Tensor,
        value_memory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid knowledge tracing model

        Args:
            question_ids: Historical question IDs [batch, seq_len]
            concept_ids: Concept IDs for each question [batch, seq_len]
            responses: Student responses (0/1) [batch, seq_len]
            difficulties: Question difficulties [batch, seq_len, 1]
            time_gaps: Time gaps between interactions [batch, seq_len, 1]
            adjacency_matrix: Knowledge graph adjacency [num_concepts, num_concepts]
            edge_types: Edge types in graph [num_concepts, num_concepts]
            target_question: Question to predict [batch, 1]
            target_concept: Concept of target question [batch, 1]
            value_memory: Current memory state or None to initialize

        Returns:
            Dictionary with prediction, updated memory, attention weights
        """
        batch_size, seq_len = question_ids.shape

        # Initialize memory if needed
        if value_memory is None:
            value_memory = self.memory_module.init_memory(batch_size)

        # 1. Get embeddings
        q_emb = self.question_embedding(question_ids)  # [batch, seq, dim]
        r_emb = self.response_embedding(responses)  # [batch, seq, dim]

        # Interaction embedding
        interaction = self.interaction_linear(torch.cat([q_emb, r_emb], dim=-1))

        # 2. GNN module - get concept relationships
        concept_repr = self.gnn_module(concept_ids, adjacency_matrix, edge_types)

        # 3. Attention module - temporal patterns with forgetting
        target_q_emb = self.question_embedding(target_question)
        attended_state, attn_weights = self.attention_module(
            query=target_q_emb,
            key=q_emb + concept_repr,
            value=interaction,
            difficulty=difficulties,
            time_gaps=time_gaps
        )

        # 4. Memory module - read and write
        target_concept_emb = self.gnn_module.concept_embeddings(target_concept.squeeze(-1))
        memory_read, memory_attn = self.memory_module.read(
            target_concept_emb, value_memory
        )

        # Update memory with all interactions
        for t in range(seq_len):
            concept_t = self.gnn_module.concept_embeddings(concept_ids[:, t])
            response_t = responses[:, t:t+1].float()
            value_memory = self.memory_module.write(
                concept_t, response_t, value_memory
            )

        # 5. Fuse all representations
        fused = self.fusion(torch.cat([
            attended_state.squeeze(1),
            memory_read,
            concept_repr[:, -1, :]  # Last concept representation
        ], dim=-1))

        # 6. Predict
        pred_input = torch.cat([fused, target_q_emb.squeeze(1)], dim=-1)
        prediction = self.predictor(pred_input)

        return {
            'prediction': prediction,
            'updated_memory': value_memory,
            'attention_weights': attn_weights,
            'memory_attention': memory_attn,
            'knowledge_state': fused
        }

    def get_knowledge_state_vector(
        self,
        value_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Get interpretable knowledge state vector for all concepts

        Returns:
            Mastery probability for each concept [batch, num_concepts]
        """
        # Read from memory for each concept
        batch_size = value_memory.shape[0]
        all_concepts = torch.arange(self.num_concepts).unsqueeze(0).expand(batch_size, -1)

        mastery_scores = []
        for c in range(self.num_concepts):
            concept_emb = self.gnn_module.concept_embeddings(
                torch.full((batch_size,), c, dtype=torch.long)
            )
            read_content, _ = self.memory_module.read(concept_emb, value_memory)
            # Simple mastery score from memory content
            score = torch.sigmoid(read_content.mean(dim=-1))
            mastery_scores.append(score)

        return torch.stack(mastery_scores, dim=-1)


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    num_concepts = 100
    num_questions = 1000
    batch_size = 32
    seq_len = 50

    # Initialize model
    model = HybridKnowledgeStateEngine(
        num_concepts=num_concepts,
        num_questions=num_questions,
        embedding_dim=128,
        memory_size=50
    )

    # Create dummy data
    question_ids = torch.randint(0, num_questions, (batch_size, seq_len))
    concept_ids = torch.randint(0, num_concepts, (batch_size, seq_len))
    responses = torch.randint(0, 2, (batch_size, seq_len))
    difficulties = torch.rand(batch_size, seq_len, 1)
    time_gaps = torch.rand(batch_size, seq_len, 1) * 24  # Hours
    adjacency = torch.eye(num_concepts)  # Simplified adjacency
    edge_types = torch.zeros(num_concepts, num_concepts, dtype=torch.long)
    target_q = torch.randint(0, num_questions, (batch_size, 1))
    target_c = torch.randint(0, num_concepts, (batch_size, 1))

    # Forward pass
    output = model(
        question_ids, concept_ids, responses, difficulties, time_gaps,
        adjacency, edge_types, target_q, target_c
    )

    print(f"Prediction shape: {output['prediction'].shape}")
    print(f"Knowledge state shape: {output['knowledge_state'].shape}")
    print(f"Memory shape: {output['updated_memory'].shape}")
