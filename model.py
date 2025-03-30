import torch 
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initializes the input embeddings for the Transformer model. 
        Each input token is mapped to a dense vector of size d_model. 
        We scale the embeddings by sqrt(d_model) as per the paper to stabilize training.

        Input Parameters:
            d_model (int): Size of each embedding vector.
            vocab_size (int): Number of unique tokens in the vocabulary.
        """
        super().__init__()
        self.d_model = d_model  # Hidden size of the model
        self.vocab_size = vocab_size  # Number of unique tokens in vocabulary

        # nn.Embedding creates a lookup table that maps each token index to a d_model-dimensional vector
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        """
        Takes a batch of token indices and returns their dense vector representations. 
        The embeddings are scaled to prevent vanishing gradients during training.

        Input:
            x (Tensor): Shape (batch_size, seq_len) containing input token indices.

        Returns:
            Tensor: Shape (batch_size, seq_len, d_model), representing input tokens as dense vectors.
        """

        # Embedding lookup -> shape becomes (batch_size, seq_len, d_model)
        # Scaling embeddings by sqrt(d_model) as suggested in the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Adds positional encoding to input embeddings to incorporate order information. 
        Since transformers don’t have a sense of sequence, this step helps in understanding 
        the position of each token.

        Input Parameters:
            d_model (int): Size of the embedding vector. Must match input embeddings.
            seq_len (int): Maximum length of input sequences.
            dropout (float): Dropout probability to prevent overfitting.
        """
         
        super().__init__()
        self.d_model = d_model  # Hidden size of the model
        self.seq_len = seq_len  # Maximum length of input sequences
        self.dropout = nn.Dropout(dropout)  # Regularization to prevent overfitting

        # Creating a matrix of zeros with shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Creating a position vector of shape (seq_len, 1)
        # position = [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # div_term controls the rate of change of the sin and cos functions
        # Using exponential decay to ensure values remain within a valid range
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Applying sine to even indices and cosine to odd indices as per the paper
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Reshape pe to (1, seq_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)

        # Registering as a buffer to make it part of the model state without gradient updates
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Adds positional encodings to input embeddings to include order information.

        Input:
            x (Tensor): Shape (batch_size, seq_len, d_model), input embeddings.

        Returns:
            Tensor: Shape (batch_size, seq_len, d_model), embeddings with positional encodings.
        """
        # Adding positional encoding (pe) to the input tensor x
        # Using requires_grad(False) to avoid updating positional encodings during training
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)  # Applying dropout


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        """
        Implements Layer Normalization to stabilize training by normalizing inputs. 
        Unlike batch normalization, it normalizes across the feature dimension rather 
        than the batch dimension, making it suitable for sequential data.

        Input Parameters:
            eps (float): A small constant to prevent division by zero during normalization.
        """
        
        super().__init__()
        self.eps = eps  # Small constant to avoid division by zero

        # Learnable parameters for scaling and shifting normalized values
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Shift parameter initialized to 0

    def forward(self, x):
        """
        Performs layer normalization by normalizing the input tensor values. 
        The mean and standard deviation are computed across the last dimension.

        Input:
            x (Tensor): Shape (batch_size, seq_len, d_model), input tensor to be normalized.

        Returns:
            Tensor: Shape (batch_size, seq_len, d_model), normalized tensor.
        """
        
        # Calculating mean and standard deviation along the last dimension
        mean = x.mean(dim=-1, keepdim=True)  # Mean along feature dimension
        std = x.std(dim=-1, keepdim=True)    # Standard deviation along feature dimension

        # Normalizing input and applying learnable parameters
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements the Position-wise FeedForward Network described in the Transformer model. 
    This block is applied independently to each position of the input sequence.

    Input Parameters:
        d_model (int): Dimensionality of input and output (model size).
        d_ff (int): Dimensionality of the hidden layer (usually larger than d_model).
        dropout (float): Dropout probability to prevent overfitting.

    The block performs:
        1. Linear Transformation (d_model -> d_ff)
        2. Activation Function (ReLU)
        3. Dropout (to prevent overfitting)
        4. Linear Transformation (d_ff -> d_model)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()

        # First linear transformation: w1 * x + b1
        self.linear_1 = nn.Linear(d_model, d_ff)  # Input to hidden dimension

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Second linear transformation: w2 * x + b2
        self.linear_2 = nn.Linear(d_ff, d_model)  # Hidden to output dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForwardBlock.

        Input:
            x (Tensor): Shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Shape (batch_size, seq_len, d_model), transformed output
        """

        # Step 1: Linear transformation to hidden dimension
        # Step 2: Apply ReLU activation
        # Step 3: Apply Dropout
        # Step 4: Linear transformation back to input dimension
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



class MultiHeadAttentionBlock(nn.Module):
    """
    Implements Multi-Head Attention mechanism as described in paper.

    The Multi-Head Attention layer allows the model to jointly attend 
    to information from different representation subspaces.

    Input Parameters:
        d_model (int): Dimensionality of input and output (model size).
        h (int): Number of attention heads.
        dropout (float): Dropout probability for regularization.

    """
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model  # Hidden size of the model
        self.h = h              # Number of attention heads

        # Ensuring d_model is divisible by h
        # This ensures each head gets an equal share of dimensions
        assert d_model % h == 0, "d_model must be divisible by h"

        # Size of vectors per head (d_k = d_model / h)
        # d_k controls the dimension of query, key, value for each head
        self.d_k = d_model // h

        # Linear layers to project inputs into Query, Key, and Value matrices
        # W_q, W_k, W_v are the learnable parameters for linear transformations
        self.w_q = nn.Linear(d_model, d_model)  # W_q: Maps input to queries
        self.w_k = nn.Linear(d_model, d_model)  # W_k: Maps input to keys
        self.w_v = nn.Linear(d_model, d_model)  # W_v: Maps input to values

        # Linear layer to project concatenated outputs from all heads
        self.w_o = nn.Linear(d_model, d_model)  # W_o: Maps concatenated output back

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Calculates scaled dot-product attention as described in the paper:
        
        Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V

        Parameters:
            query (Tensor): Query matrix of shape (batch_size, h, seq_len, d_k)
            key (Tensor): Key matrix of shape (batch_size, h, seq_len, d_k)
            value (Tensor): Value matrix of shape (batch_size, h, seq_len, d_k)
            mask (Tensor): Mask tensor to ignore certain positions
            dropout (nn.Dropout): Dropout layer for regularization

        Returns:
            output (Tensor): Output of attention mechanism
            attention_scores (Tensor): Attention weights (softmax outputs)
        """

        d_k = query.shape[-1]  # d_k from input shape (batch_size, h, seq_len, d_k)

        # Step 1: Calculate raw attention scores using scaled dot-product
        # (Q * K^T) / sqrt(d_k) to avoid exploding gradients
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  #@use to matrix mutiplication (shortcut for torch.matmul)

        # Step 2: Apply mask if provided to avoid attending to padding positions
        if mask is not None:
            # Mask out unwanted positions by setting them to a very low value (-1e9)
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Step 3: Apply softmax to get normalized attention weights
        attention_scores = attention_scores.softmax(dim=-1)

        # Step 4: Apply dropout to attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Step 5: Multiply attention weights by the value matrix
        output = attention_scores @ value

        # Return output and attention scores
        return output, attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for Multi-Head Attention Block.

        Parameters:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_model)
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_model)
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_model)
            mask (Tensor): Mask tensor to ignore certain positions

        Returns:
            Tensor: Shape (batch_size, seq_len, d_model)
        """

        # Step 1: Linear projections to obtain Q, K, V matrices
        # Shape: (batch_size, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Step 2: Reshape into multiple heads
        # Original shape: (batch_size, seq_len, d_model)
        # New shape: (batch_size, seq_len, h, d_k)
        # Transpose to: (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)   #view functionis same as reshape function for pytorch
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Step 3: Apply scaled dot-product attention
        # Outputs (batch_size, h, seq_len, d_k) and attention scores
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Step 4: Concatenate multiple heads back into a single tensor
        # Transpose to: (batch_size, seq_len, h, d_k)
        # Reshape to original shape: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)  #contiguous() is a method that makes sure tensors are stored in a continuous memory block. This is needed when tensors are reshaped or transposed. It rearranges the memory layout so PyTorch can efficiently process the tensor.

        # Step 5: Final linear projection
        # Shape: (batch_size, seq_len, d_model)
        return self.w_o(x)



class ResidualConnection(nn.Module):
    """
    Implements the Residual Connection with Layer Normalization 

    The residual connection helps in avoiding vanishing gradients in deep networks 
    by allowing gradients to flow through the network unchanged. The Layer Normalization 
    normalizes inputs, improving convergence.

    Input Parameters:
        dropout (float): Dropout probability for regularization.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        
        # Dropout for regularization to avoid overfitting
        self.dropout = nn.Dropout(dropout)

        # Layer Normalization ensures stable training by normalizing inputs
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        """
        Forward pass for Residual Connection.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            sublayer (Callable): The sublayer function to apply (e.g., multi-head attention or feed-forward)

        Returns:
            Tensor: Output of the form (batch_size, seq_len, d_model)
        """

        # Step 1: Apply Layer Normalization to the input
        # Step 2: Pass normalized input to the sublayer (e.g., Attention or Feed-Forward)
        # Step 3: Apply Dropout to the output of the sublayer for regularization
        # Step 4: Add the input 'x' (residual connection) to sublayer output
        # This operation preserves gradients during backpropagation and avoids vanishing gradient issues
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    Implementation of a single Encoder Block

    Each Encoder Block consists of:
    - Multi-Head Self-Attention Mechanism
    - Feed-Forward Neural Network
    - Two Residual Connections with Layer Normalization

    Input Parameters:
        self_attention_block (MultiHeadAttentionBlock): Multi-Head Attention mechanism.
        feed_forward_block (FeedForwardBlock): Feed-forward neural network.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        # Multi-Head Self-Attention mechanism
        self.self_attention_block = self_attention_block

        # Feed-Forward Neural Network
        self.feed_forward_block = feed_forward_block

        # ModuleList to handle residual connections with Layer Normalization
        # Creating two residual connections, one for attention and one for feed-forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout),  # For Self-Attention
            ResidualConnection(dropout)   # For Feed-Forward
        ])

    def forward(self, x, src_mask):
        """
        Forward pass for a single Encoder Block.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            src_mask (Tensor): Mask tensor to ignore certain positions

        Returns:
            Tensor: Encoded output of shape (batch_size, seq_len, d_model)
        """

        # Step 1: Multi-Head Self-Attention with Residual Connection
        # Using lambda to pass query, key, and value as x
        # Shape remains (batch_size, seq_len, d_model)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # Step 2: Feed-Forward Block with Residual Connection
        # Directly passing feed-forward block as sublayer
        # Shape remains (batch_size, seq_len, d_model)
        x = self.residual_connections[1](x, self.feed_forward_block)

        # Step 3: Return the output of the encoder block
        return x


class Encoder(nn.Module):
    """
    Implements the Nx Encoders for the Transformer model.


    Input Parameters:
        layers (nn.ModuleList): A list of encoder layers, each implementing
                                multi-head attention and feed-forward operations.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Storing encoder layers in nn.ModuleList
        # nn.ModuleList ensures proper registration of parameters
        self.layers = layers

        # Final Layer Normalization to stabilize output
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Forward pass for the Encoder.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (Tensor): Mask tensor to ignore certain positions

        Returns:
            Tensor: Encoded output of shape (batch_size, seq_len, d_model)
        """

        # Step 1: Sequentially pass input through each encoder layer
        for layer in self.layers:
            x = layer(x, mask)  # Each layer expects (input, mask)

        # Step 2: Apply final Layer Normalization
        # Normalizes the final output to stabilize training
        return self.norm(x)


# class DecoderBlock(nn.Module):

#     def __init__(self,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock,dropout:float):
#         super().__init__()
#         self.self_attention_block=self_attention_block 
#         self.cross_attention_block=cross_attention_block
#         self.feed_forward_block=feed_forward_block 
#         self.residual_connections=nn.Module([ResidualConnection(dropout) for _ in range(3)])

    
#     def forward(self,x,encoder_output,src_mask,tgt_mask):
#         x=self.residual_connections[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
#         x=self.residual_connections[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
#         x=self.residual_connections[2](x,self.feed_forward_block)

#         return x 
    

# class Decoder(nn.Module):
#     def __init__(self,layers:nn.ModuleList)->None:
#         super().__init__()
#         self.layers=layers
#         self.norm=LayerNormalization() 
    
#     def forward(self,x,encoder_output,src_mask,tgt_mask):
#         for layer in self.layers:
#             x=layer(x,encoder_output,src_mask,tgt_mask)
        
#         return self.norm(x)
    
class DecoderBlock(nn.Module):
    """
    Implements a single Decoder Block in the Transformer model.

    Each Decoder Block consists of:
    - Masked Multi-Head Self-Attention
    - Cross-Attention (Multi-Head Attention over Encoder output)
    - Feed-Forward Neural Network
    - Three Residual Connections with Layer Normalization

    Input Parameters:
        self_attention_block (MultiHeadAttentionBlock): Masked self-attention mechanism.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention mechanism.
        feed_forward_block (FeedForwardBlock): Feed-forward neural network.
        dropout (float): Dropout probability for regularization.
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,cross_attention_block: MultiHeadAttentionBlock,feed_forward_block: FeedForwardBlock,dropout: float) -> None:
        super().__init__()

        # Masked Self-Attention Mechanism
        self.self_attention_block = self_attention_block

        # Cross-Attention Mechanism (Encoder-Decoder Attention)
        self.cross_attention_block = cross_attention_block

        # Feed-Forward Neural Network
        self.feed_forward_block = feed_forward_block

        # Residual Connections
        # Creating three residual connections for attention, cross-attention, and feed-forward
        # self.residual_connections = nn.ModuleList([
        #     ResidualConnection(dropout),  # For Self-Attention
        #     ResidualConnection(dropout),  # For Cross-Attention
        #     ResidualConnection(dropout)   # For Feed-Forward
        # ])
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for a single Decoder Block.

        Parameters:
            x (Tensor): Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (Tensor): Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask (Tensor): Mask for the encoder input
            tgt_mask (Tensor): Mask for the target input (to preserve auto-regression)

        Returns:
            Tensor: Decoded output of shape (batch_size, tgt_seq_len, d_model)
        """

        # Step 1: Masked Multi-Head Self-Attention with Residual Connection
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Step 2: Cross-Attention with Residual Connection
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))

        # Step 3: Feed-Forward Network with Residual Connection
        x = self.residual_connections[2](x, self.feed_forward_block)

        # Step 4: Return the output of the decoder block
        return x


class Decoder(nn.Module):
    """
    Implements the Transformer Decoder consisting of multiple Decoder Blocks.

    Input Parameters:
        layers (nn.ModuleList): List of DecoderBlocks.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        # Store decoder layers
        self.layers = layers

        # Final Layer Normalization
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the Transformer Decoder.

        Parameters:
            x (Tensor): Target input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output (Tensor): Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask (Tensor): Mask for the encoder input
            tgt_mask (Tensor): Mask for the target input (to preserve auto-regression)

        Returns:
            Tensor: Decoded output of shape (batch_size, tgt_seq_len, d_model)
        """

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        # Final normalization
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    The ProjectionLayer maps the model output to the target vocabulary space.
    
    Parameters:
        d_model (int): The dimension of the model output (embedding size).
        vocab_size (int): The size of the target vocabulary.

    Forward Pass:
        Input Shape: (batch_size, seq_len, d_model)
        Output Shape: (batch_size, seq_len, vocab_size)
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # Linear layer to project to vocab size y=x.W^T+b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the projection and returns log-probabilities.

        Parameters:
            x (Tensor): Model output of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Log probabilities of shape (batch_size, seq_len, vocab_size)
        """
        # Linear projection + Log Softmax for numerical stability
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    """
    The Transformer class implements the full Transformer model architecture
    for sequence-to-sequence tasks, integrating the encoder, decoder, embeddings,
    positional encodings, and projection layer.

    Parameters:
        encoder (Encoder): The encoder stack that processes source sequences.
        decoder (Decoder): The decoder stack that generates target sequences.
        src_embed (InputEmbeddings): Embedding layer for the source sequence.
        tgt_embed (InputEmbeddings): Embedding layer for the target sequence.
        src_pos (PositionalEncoding): Positional encoding for the source sequence.
        tgt_pos (PositionalEncoding): Positional encoding for the target sequence.
        projection_layer (ProjectionLayer): Final projection layer to map decoder output to vocabulary.

    Forward Pass:
        - Encodes the input sequence using the encoder.
        - Decodes using the decoder with encoder output and target sequence.
        - Projects the decoder output to vocabulary probabilities.

    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder                  # Encoder stack
        self.decoder = decoder                  # Decoder stack
        self.src_embed = src_embed              # Source sequence embedding
        self.tgt_embed = tgt_embed              # Target sequence embedding
        self.src_pos = src_pos                  # Source positional encoding
        self.tgt_pos = tgt_pos                  # Target positional encoding
        self.projection_layer = projection_layer # Projection to vocab

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input source sequence.

        Parameters:
            src (Tensor): Source input tensor of shape (batch_size, src_seq_len)
            src_mask (Tensor): Mask for source input to ignore padding

        Returns:
            Tensor: Encoded source of shape (batch_size, src_seq_len, d_model)
        """
        # Step 1: Apply input embeddings
        src = self.src_embed(src)
        
        # Step 2: Apply positional encoding
        src = self.src_pos(src)
        
        # Step 3: Pass through the encoder stack
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decodes the target sequence using encoder outputs.

        Parameters:
            encoder_output (Tensor): Encoded source tensor of shape (batch_size, src_seq_len, d_model)
            src_mask (Tensor): Mask for the source input
            tgt (Tensor): Target input tensor of shape (batch_size, tgt_seq_len)
            tgt_mask (Tensor): Mask for the target input to ignore padding

        Returns:
            Tensor: Decoded output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Step 1: Apply target embeddings
        tgt = self.tgt_embed(tgt)
        
        # Step 2: Apply positional encoding
        tgt = self.tgt_pos(tgt)
        
        # Step 3: Pass through the decoder stack
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the model output to the target vocabulary space.

        Parameters:
            x (Tensor): Decoder output of shape (batch_size, tgt_seq_len, d_model)

        Returns:
            Tensor: Predicted log-probabilities of shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Linear projection and softmax to get probability distribution
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,  d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Builds a complete Transformer model by assembling its components.

    Parameters:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of the source sequence.
        tgt_seq_len (int): Maximum length of the target sequence.
        d_model (int): Dimensionality of input/hidden layers (default: 512).
        N (int): Number of layers in the encoder and decoder (default: 6).
        h (int): Number of attention heads (default: 8).
        dropout (float): Dropout rate for regularization (default: 0.1).
        d_ff (int): Dimensionality of feed-forward layer (default: 2048).

    Returns:
        Transformer: A configured Transformer model instance.
    """

    # Step 1: Create Embedding Layers
    # InputEmbeddings: Transforms input indices into dense vectors
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Step 2: Create Positional Encoding Layers
    # PositionalEncoding: Adds positional information to embeddings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Step 3: Create Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        # MultiHeadAttentionBlock: Self-attention mechanism for encoder
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

        # FeedForwardBlock: Position-wise feed-forward network
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        # EncoderBlock: Encapsulates self-attention and feed-forward network with residual connections
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        
        encoder_blocks.append(encoder_block)

    # Step 4: Create Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        # MultiHeadAttentionBlock for self-attention and cross-attention in the decoder
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)

        # FeedForwardBlock for the decoder
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)

        # DecoderBlock: Encapsulates self-attention, cross-attention, and feed-forward network
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                     decoder_cross_attention_block, feed_forward_block, dropout)
        
        decoder_blocks.append(decoder_block)

    # Step 5: Create Encoder and Decoder
    # Encoder: Stacks N encoder blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Decoder: Stacks N decoder blocks
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Step 6: Create the Projection Layer
    # ProjectionLayer: Maps model output to vocabulary size with log probabilities
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Step 7: Assemble the Transformer
    # Transformer: Brings together encoder, decoder, embedding, positional encoding, and projection layer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Step 8: Parameter Initialization
    # Xavier Initialization: Ensures better gradient flow in deep networks
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


if __name__=='__main__':
    print("Inside name main:")
    # Example parameters
    vocab_size = 10000  # Example vocabulary size
    d_model = 512  # Embedding dimension

    # Instantiate the embedding layer
    embedding_layer = InputEmbeddings(d_model, vocab_size)

    # Sample input tensor (batch of token indices)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # Example batch of token indices

    # Forward pass
    embedded_x = embedding_layer(x)
    print(embedded_x.shape)  # Expected shape: (batch_size, sequence_length, d_model)

    print(embedded_x)

    print("Heelo world")
