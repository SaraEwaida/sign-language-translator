"""Transformer encoder for isolated sign classification.

Input:  (B, T, 225) normalized landmarks (64 frames x 75 points x 3 coords)
Output: (B, num_classes) logits

Self-attention: each frame is projected to a query, key and value. A frame's
output is a weighted sum of the values of all 64 frames, with weights
softmax(q . k / sqrt(d)) saying how relevant every other frame is to it.
Unlike the TCN's fixed local kernels, one layer already sees the whole clip,
and the weights depend on the content (e.g. the frames where the hand moves).
4 heads do this in parallel on 64-dim slices, so different heads can track
different relations (one hand, the other hand, timing).

Positional embedding: attention itself is order-blind; shuffling the frames
would give the same pooled output. A learned (64, 256) vector per frame index
is added to the input so the model can tell the start of a sign from its end.
It is learned rather than sinusoidal because the length is always 64.

No padding mask: every clip is resampled to exactly 64 frames in extract.py
(and temporal-crop augmentation resamples back to 64), so there are no padded
positions to hide from attention or from the mean/max pooling. Missing
landmarks are zeros inside real frames, not padding.
"""
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, in_dim: int = 225, hidden: int = 256, num_layers: int = 4,
                 num_heads: int = 4, ff_dim: int = 512, max_len: int = 64,
                 num_classes: int = 250, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, num_heads, dim_feedforward=ff_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # nested tensors only help padded batches; also unsupported with norm_first
        self.encoder = nn.TransformerEncoder(layer, num_layers, norm=nn.LayerNorm(hidden),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden), nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]            # (B, T, H)
        h = self.encoder(h)                                     # (B, T, H)
        pooled = torch.cat([h.mean(dim=1), h.amax(dim=1)], -1)  # (B, 2H)
        return self.head(pooled)


if __name__ == "__main__":
    import time

    model = TransformerClassifier().eval()
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
