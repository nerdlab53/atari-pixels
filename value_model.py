"""
Defines the Value Model architecture.

This model is a simple MLP that learns to predict the expected future reward
(the "value") from a given latent state representation of the game.
"""
import torch
import torch.nn as nn

class ValueModel(nn.Module):
    """
    An MLP that predicts the value of a given latent state.
    
    The model takes a sequence of discrete latent codes, embeds them,
    flattens the result, and passes it through linear layers to output
    a single scalar value.
    """
    def __init__(self, latent_seq_len, latent_vocab_size, embedding_dim=64, hidden_size=512):
        super().__init__()
        self.latent_seq_len = latent_seq_len
        self.latent_vocab_size = latent_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        # --- Embedding Layer ---
        # Converts the discrete latent codes into dense vectors.
        self.latent_embedding = nn.Embedding(latent_vocab_size, embedding_dim)

        # --- MLP Head ---
        # Processes the flattened embeddings to predict the state's value.
        self.mlp = nn.Sequential(
            nn.Linear(latent_seq_len * embedding_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size // 2, 1) # Output a single scalar value
        )

    def forward(self, latent_codes):
        """
        Forward pass of the Value Model.

        Args:
            latent_codes (torch.Tensor): A batch of latent code sequences.
                                        Shape: (batch_size, latent_seq_len)

        Returns:
            torch.Tensor: The predicted value for each state in the batch.
                          Shape: (batch_size, 1)
        """
        # 1. Embed the latent codes
        # -> (batch_size, latent_seq_len, embedding_dim)
        embeds = self.latent_embedding(latent_codes)

        # 2. Flatten the embeddings for the MLP
        # -> (batch_size, latent_seq_len * embedding_dim)
        flattened_embeds = embeds.view(embeds.size(0), -1)

        # 3. Predict the value
        predicted_value = self.mlp(flattened_embeds)
        
        return predicted_value 