import torch
from torch import nn


class TemporalTransformer(nn.Module):
    """Small sequence classifier for fixed-width network state windows."""

    def __init__(self, state_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.input_projection = nn.Linear(state_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 512, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, states):
        if states.ndim != 3:
            raise ValueError("states must have shape [batch, sequence, features]")
        if states.shape[1] > self.position_embedding.shape[1]:
            raise ValueError("sequence length exceeds the model position limit")
        encoded = self.input_projection(states)
        encoded = encoded + self.position_embedding[:, : states.shape[1]]
        encoded = self.encoder(encoded)
        pooled = self.output_norm(encoded[:, -1])
        return self.classifier(pooled).squeeze(-1)
