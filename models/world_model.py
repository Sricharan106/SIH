import torch
import torch.nn as nn


class WorldModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_stages=6):

        super().__init__()

        self.hidden_size = hidden_size

        # Temporal memory
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1)
        )

        # Future network state
        self.state_head = nn.Linear(hidden_size, input_size)

        # Attack probability
        self.attack_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

        # MITRE stage
        self.stage_head = nn.Linear(hidden_size, num_stages)

    def forward(self, x):

        # x shape:
        # batch, sequence, features

        lstm_output, _ = self.lstm(x)

        # Attention scores
        attention_scores = self.attention(lstm_output)

        attention_weights = torch.softmax(attention_scores, dim=1)

        # Context vector
        context = torch.sum(attention_weights * lstm_output, dim=1)

        # Predictions

        next_state = self.state_head(context)

        attack_probability = self.attack_head(context)

        stage_logits = self.stage_head(context)

        return (next_state, attack_probability, stage_logits, attention_weights)
