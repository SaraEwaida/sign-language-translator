"""Temporal convolutional network (TCN) for isolated sign classification.

Input:  (B, T, 225) normalized landmarks (64 frames x 75 points x 3 coords)
Output: (B, num_classes) logits

Receptive field: a Conv1d with kernel k and dilation d widens the view by
(k - 1) * d frames. Each block has two convs with kernel 5, so it adds
2 * 4 * d = 8d frames; with d = 1, 2, 4, 8 the total is 8 * 15 = 120, so the
last layer sees 121 frames (non-causal: 60 before and 60 after each frame).
That is wider than the 64-frame clip: every frame in positions 3-60 sees the
whole clip, and the edge frames still see at least 61 of the 64 frames.
Mean + max pooling then combines all positions, so the head gets
information from the full sequence.
"""
import torch
from torch import nn


class TemporalBlock(nn.Module):
    """Two dilated non-causal convs (same length) with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation  # "same" padding keeps T unchanged
        layers = []
        for _ in range(2):
            layers += [
                nn.Conv1d(channels, channels, kernel_size, padding=padding,
                          dilation=dilation, bias=False),  # BatchNorm adds the bias
                nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return x + self.net(x)


class TCNClassifier(nn.Module):
    def __init__(self, in_dim: int = 225, hidden: int = 256, kernel_size: int = 5,
                 dilations: tuple[int, ...] = (1, 2, 4, 8), num_classes: int = 250,
                 dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.blocks = nn.Sequential(
            *[TemporalBlock(hidden, kernel_size, d, dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.proj(x).transpose(1, 2))           # (B, H, T)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], -1)  # (B, 2H)
        return self.head(pooled)


if __name__ == "__main__":
    import time

    model = TCNClassifier().eval()
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
