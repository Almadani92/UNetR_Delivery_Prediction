import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TransformerClassifier(nn.Module):
    def __init__(self, num_channels, signal_length, d_model=128, num_heads=8, num_encoder_layers=6, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(num_channels * signal_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=d_model * 4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classification head
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        batch_size, num_patches, num_channels, signal_length = x.size()

        # Reshape the tensor to combine patches and channels
        x = x.view(batch_size * num_patches, num_channels * signal_length)

        # Embedding
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Shape: (seq_length, batch_size * num_patches, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the mean over the sequence dimension (seq_length) for each patch
        x = x.view(-1, num_patches, x.size(0), x.size(1))
        x = x.mean(dim=0)  # Shape: (num_patches, batch_size, d_model)

        # Classification head
        x = self.fc(x.mean(dim=0))  # Take the mean over the patch dimension

        # Squeeze to get rid of the unnecessary dimension
        x = x.squeeze(1)

        return x

