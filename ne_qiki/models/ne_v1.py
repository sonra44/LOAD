import torch
import torch.nn as nn

class NE_v1(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, param_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.param_head = nn.Linear(hidden_dim, param_dim)

    def forward(self, x, action_mask=None):
        out, _ = self.gru(x)
        features = out[:, -1, :]  # последний таймстеп

        logits = self.class_head(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float('-inf'))

        priority = self.priority_head(features).squeeze(-1)
        params = self.param_head(features)

        return logits, priority, params
