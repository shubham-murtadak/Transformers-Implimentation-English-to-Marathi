import torch 
import math
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self,d_model: int, vocab_size:int):
        super().__init__()
        self.d_model=d_model 
        self.vocab_size=vocab_size 
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)

    
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
