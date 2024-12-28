# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==============================
# 1. Positional Encoding Module
# ==============================

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in 
    "Attention is All You Need" by Vaswani et al.
    Injects information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): The dimensionality of the embeddings.
            max_len (int): The maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) with sinusoidal values
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Register as buffer to avoid updating during training

    def forward(self, x):
        """
        Adds positional encoding to the input embeddings.

        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Positionally encoded embeddings.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

# ==============================
# 2. Specialized Multi-Head Attention
# ==============================

class SpecializedMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_division):
        super(SpecializedMultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_division = head_division
        self.metafunctions = list(head_division.keys())

        assert sum(head_division.values()) == num_heads, "Sum of head divisions must equal num_heads."

        # Compute head_dim
        head_dim = hidden_dim // num_heads

        # Create linear layers with correct output dimensions per metafunction
        self.query_linears = nn.ModuleDict()
        self.key_linears = nn.ModuleDict()
        self.value_linears = nn.ModuleDict()

        for func in self.metafunctions:
            num_func_heads = self.head_division[func]
            output_dim = num_func_heads * head_dim  # Only produce enough units for the metafunction's heads

            self.query_linears[func] = nn.Linear(hidden_dim, output_dim)
            self.key_linears[func] = nn.Linear(hidden_dim, output_dim)
            self.value_linears[func] = nn.Linear(hidden_dim, output_dim)

        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        batch_size, query_len, _ = query.size()
        key_len = key.size(1)
        head_dim = self.hidden_dim // self.num_heads

        attention_outputs = []

        for func in self.metafunctions:
            num_func_heads = self.head_division[func]

            # Linear projections produce output_dim = num_func_heads * head_dim
            Q = self.query_linears[func](query)  # (batch_size, query_len, num_func_heads * head_dim)
            K = self.key_linears[func](key)      # (batch_size, key_len, num_func_heads * head_dim)
            V = self.value_linears[func](value)  # (batch_size, key_len, num_func_heads * head_dim)

            # Reshape for multi-head attention per metafunction
            Q = Q.view(batch_size, query_len, num_func_heads, head_dim).transpose(1, 2)
            K = K.view(batch_size, key_len, num_func_heads, head_dim).transpose(1, 2)
            V = V.view(batch_size, key_len, num_func_heads, head_dim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_func_heads, query_len, head_dim)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, query_len, num_func_heads * head_dim)

            attention_outputs.append(attn_output)

        # Concatenate along the last dimension to get (batch_size, query_len, hidden_dim)
        concatenated = torch.cat(attention_outputs, dim=-1)
        output = self.out_linear(concatenated)

        return output

# ==============================
# 3. Hierarchical Attention Module
# ==============================

class HierarchicalAttention(nn.Module):
    """
    Implements a hierarchical attention mechanism focusing on ideational, interpersonal,
    and textual metafunctions of language.
    """
    def __init__(self, hidden_dim, num_heads, head_division):
        """
        Args:
            hidden_dim (int): The dimensionality of the input embeddings.
            num_heads (int): Total number of attention heads.
            head_division (dict): Number of heads assigned to each metafunction.
        """
        super(HierarchicalAttention, self).__init__()
        self.specialized_attn = SpecializedMultiHeadAttention(hidden_dim, num_heads, head_division)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        """
        Performs hierarchical attention by applying specialized attention mechanisms.

        Args:
            query (Tensor): Query embeddings of shape (batch_size, query_len, hidden_dim).
            key (Tensor): Key embeddings of shape (batch_size, key_len, hidden_dim).
            value (Tensor): Value embeddings of shape (batch_size, value_len, hidden_dim).

        Returns:
            Tensor: Hierarchical attention output of shape (batch_size, query_len, hidden_dim).
        """
        attn_output = self.specialized_attn(query, key, value)  # (batch_size, query_len, hidden_dim)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)  # Residual connection with LayerNorm
        return attn_output

# ==============================
# 4. Meta-Cognitive Attention Module
# ==============================

class MetaCognitiveAttention(nn.Module):
    """
    Enhances the HierarchicalAttention with meta-cognitive capabilities by introducing
    auxiliary loss terms to encourage self-modeling.
    """
    def __init__(self, hidden_dim, num_heads, head_division):
        """
        Args:
            hidden_dim (int): The dimensionality of the input embeddings.
            num_heads (int): Total number of attention heads.
            head_division (dict): Number of heads assigned to each metafunction.
        """
        super(MetaCognitiveAttention, self).__init__()
        self.hierarchical_attention = HierarchicalAttention(hidden_dim, num_heads, head_division)
        self.auxiliary_predictor = nn.Linear(hidden_dim, num_heads)  # Predict attention distributions

    def forward(self, query, key, value, ideal_distributions=None):
        """
        Performs hierarchical attention and computes auxiliary loss if ideal distributions are provided.

        Args:
            query (Tensor): Query embeddings of shape (batch_size, query_len, hidden_dim).
            key (Tensor): Key embeddings of shape (batch_size, key_len, hidden_dim).
            value (Tensor): Value embeddings of shape (batch_size, value_len, hidden_dim).
            ideal_distributions (Tensor, optional): Ideal attention distributions for auxiliary loss.

        Returns:
            Tuple[Tensor, Tensor or None]: 
                - Attention output of shape (batch_size, query_len, hidden_dim).
                - Auxiliary loss if ideal_distributions is provided; otherwise, None.
        """
        attn_output = self.hierarchical_attention(query, key, value)  # (batch_size, query_len, hidden_dim)
        
        # Predict attention distributions (for self-modeling purposes)
        predicted_distributions = self.auxiliary_predictor(attn_output)  # (batch_size, query_len, num_heads)
        
        # Compute auxiliary loss if ideal distributions are provided
        aux_loss = None
        if ideal_distributions is not None:
            aux_loss = F.mse_loss(predicted_distributions, ideal_distributions)
        
        return attn_output, aux_loss

# ==============================
# 5. Transformer Encoder Layer
# ==============================

class TransformerEncoderLayer(nn.Module):
    """
    Implements a single Transformer encoder layer with hierarchical and meta-cognitive attention.
    """
    def __init__(self, hidden_dim, num_heads, head_division):
        """
        Args:
            hidden_dim (int): The dimensionality of the input embeddings.
            num_heads (int): Total number of attention heads.
            head_division (dict): Number of heads assigned to each metafunction.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.meta_cognitive_attn = MetaCognitiveAttention(hidden_dim, num_heads, head_division)
        
        # Position-wise Feed-Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer Normalization and Dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, ideal_distributions=None):
        """
        Performs the forward pass of the TransformerEncoderLayer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, hidden_dim).
            ideal_distributions (Tensor, optional): Ideal attention distributions for auxiliary loss.

        Returns:
            Tuple[Tensor, Tensor or None]: 
                - Output tensor of shape (batch_size, sequence_length, hidden_dim).
                - Auxiliary loss if ideal_distributions is provided; otherwise, None.
        """
        # Self-Attention with Meta-Cognitive Enhancements
        attn_output, aux_loss = self.meta_cognitive_attn(x, x, x, ideal_distributions)
        x = attn_output  # (batch_size, sequence_length, hidden_dim)

        # Feed-Forward Network
        ff_output = self.feed_forward(x)  # (batch_size, sequence_length, hidden_dim)
        ff_output = self.dropout(ff_output)
        ff_output = self.layer_norm(ff_output + x)  # Residual connection with LayerNorm

        return ff_output, aux_loss

# ==============================
# 6. Language Model
# ==============================

class LanguageModel(nn.Module):
    """
    Defines the complete Language Model using an embedding layer, sinusoidal 
    positional encoding, multiple Transformer encoder layers with hierarchical and 
    meta-cognitive attention, and a final linear layer to produce logits over the vocabulary.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=6, 
                 dropout=0.2, num_heads=12, head_division=None, max_seq_len=5000):
        """
        Args:
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimensionality of the token embeddings.
            hidden_dim (int): The dimensionality of the hidden representations.
            num_layers (int, optional): Number of Transformer encoder layers. Default is 6.
            dropout (float, optional): Dropout probability for regularization. Default is 0.2.
            num_heads (int, optional): Number of attention heads in multi-head attention. Default is 12.
            head_division (dict, optional): Number of heads for each metafunction. 
                                            Default allocates equally.
            max_seq_len (int, optional): Maximum sequence length for positional encoding. Default is 5000.
        """
        super(LanguageModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer converts token indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Linear layer to project embeddings to the hidden dimension used in the Transformer
        self.embedding_linear = nn.Linear(embedding_dim, hidden_dim)

        # Sinusoidal Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)

        # Define head division for metafunctions if not provided
        if head_division is None:
            # Default: Equally divide heads among metafunctions
            head_division = {
                'ideational': num_heads // 3,
                'interpersonal': num_heads // 3,
                'textual': num_heads - 2 * (num_heads // 3)  # Assign remaining heads to textual
            }
        else:
            assert sum(head_division.values()) == num_heads, "Sum of head divisions must equal num_heads."

        # Stack of Transformer encoder layers with hierarchical and meta-cognitive attention
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, head_division) for _ in range(num_layers)
        ])

        # Final linear layer to project hidden states to vocabulary size for prediction
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ideal_distributions=None):
        """
        Performs the forward pass of the LanguageModel.

        Args:
            x (Tensor): Input tensor of token indices with shape (batch_size, sequence_length).
            ideal_distributions (List[Tensor], optional): List of ideal attention distributions 
                                                          for each encoder layer.

        Returns:
            Tuple[Tensor, Tensor or None]: 
                - Logits over the vocabulary for the next token prediction,
                  with shape (batch_size, vocab_size).
                - Total auxiliary loss accumulated from all encoder layers if ideal_distributions is provided; otherwise, None.
        """
        # Convert token indices to embeddings
        embedded = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)

        # Project embeddings to the hidden dimension
        embedded = self.embedding_linear(embedded)  # (batch_size, sequence_length, hidden_dim)

        # Apply sinusoidal positional encoding
        embedded = self.positional_encoding(embedded)  # (batch_size, sequence_length, hidden_dim)

        # Apply dropout for regularization
        embedded = self.dropout(embedded)

        aux_losses = []

        # Pass through each Transformer encoder layer sequentially
        for idx, layer in enumerate(self.transformer_encoder):
            if ideal_distributions is not None and idx < len(ideal_distributions):
                layer_output, aux_loss = layer(embedded, ideal_distributions[idx])
                aux_losses.append(aux_loss)
            else:
                layer_output, _ = layer(embedded)
            embedded = layer_output  # (batch_size, sequence_length, hidden_dim)

        # Extract the output from the last time step for prediction
        last_output = embedded[:, -1, :]  # (batch_size, hidden_dim)

        # Project the last hidden state to logits over the vocabulary
        output = self.fc(last_output)  # (batch_size, vocab_size)

        # Sum all auxiliary losses
        total_aux_loss = torch.stack(aux_losses).sum() if aux_losses else None

        return output, total_aux_loss

# ==============================
# 7. Example Usage (Optional)
# ==============================

if __name__ == "__main__":
    # Example configuration
    vocab_size = 10000
    embedding_dim = 512
    hidden_dim = 768
    num_layers = 6
    num_heads = 12
    head_division = {
        'ideational': 4,
        'interpersonal': 4,
        'textual': 4
    }
    max_seq_len = 512

    # Initialize the model
    model = LanguageModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_division=head_division,
        max_seq_len=max_seq_len
    )

    # Dummy input (batch_size=2, sequence_length=10)
    dummy_input = torch.randint(0, vocab_size, (2, 10))

    # Forward pass without ideal distributions
    logits, aux_loss = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (2, vocab_size)
    print("Auxiliary Loss:", aux_loss)    # Expected: None

    # Forward pass with ideal distributions (random example)
    ideal_distributions = [torch.rand(2, 10, num_heads) for _ in range(num_layers)]
    logits, aux_loss = model(dummy_input, ideal_distributions)
    print("Logits shape:", logits.shape)  # Expected: (2, vocab_size)
    print("Auxiliary Loss:", aux_loss.item())  # Scalar value
