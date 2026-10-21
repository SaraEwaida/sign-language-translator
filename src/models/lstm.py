"""BiLSTM baseline for isolated sign classification.

Input:  (B, T, 225) normalized landmarks (64 frames x 75 points x 3 coords)
Output: (B, num_classes) logits
"""
import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, in_dim: int = 225, hidden: int = 256, num_layers: int = 2,
                 num_classes: int = 250, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden), nn.Dropout(dropout), nn.Linear(4 * hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.proj(x))                         # (B, T, 2H)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], -1)  # (B, 4H)
        return self.head(pooled)


if __name__ == "__main__":
    import time

    model = BiLSTMClassifier().eval()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 64, 225)

    with torch.no_grad():
        out = model(x)
        for _ in range(5):  # warm-up
            model(x)
        t0 = time.perf_counter()
        for _ in range(50):
            model(x)
        ms = (time.perf_counter() - t0) / 50 * 1000

    print(f"params {n_params / 1e6:.2f}M | output {tuple(out.shape)} | CPU latency {ms:.1f} ms/sample")