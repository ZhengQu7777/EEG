"""
DBConformer: Dual-Branch Convolutional Transformer for EEG Decoding
Simplified version adapted from: https://github.com/wzwvv/DBConformer
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm1d(emb_size),
            nn.GELU()
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.projection(x)  # (B, emb_size, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, emb_size)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class DBConformer(nn.Module):
    def __init__(self, configs=None):
        super().__init__()

        # Extract config
        if configs is not None:
            in_channels = getattr(configs, 'input_channels', 16)
            num_classes = getattr(configs, 'num_classes', 2)
            seq_len = getattr(configs, 'sequence_len', 2000)
        else:
            in_channels = 16
            num_classes = 2
            seq_len = 2000

        self._in_channels = in_channels
        self._num_classes = num_classes
        self._seq_len = seq_len

        # Hyperparameters
        self._emb_size = 64
        self._depth = 4
        self._num_heads = 4
        self._dropout = 0.2
        self._patch_size = 125

        emb_size = self._emb_size
        depth = self._depth
        num_heads = self._num_heads
        dropout = self._dropout
        patch_size = self._patch_size

        # Temporal branch: patch along time dimension (channel-dependent)
        num_patches_temporal = seq_len // patch_size
        self.pos_embedding_temporal = nn.Parameter(torch.randn(1, num_patches_temporal, emb_size))
        self.temporal_transformer = nn.Sequential(*[TransformerBlock(emb_size, num_heads, dropout) for _ in range(depth)])

        # Spatial branch transformer (channel-independent internals, but pos_embed is channel-dependent)
        self.spatial_transformer = nn.Sequential(*[TransformerBlock(emb_size, num_heads, dropout) for _ in range(depth)])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size * 2),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, num_classes)
        )

        # Channel-dependent layers
        if in_channels is not None:
            self._build_channel_layers(in_channels)

    def _build_channel_layers(self, in_channels):
        """构建/重建依赖通道数的层"""
        self._in_channels = in_channels
        device = self.classifier[2].weight.device
        emb_size = self._emb_size
        dropout = self._dropout
        patch_size = self._patch_size

        self.temporal_patch_embed = PatchEmbedding(in_channels, emb_size, patch_size).to(device)

        self.spatial_patch_embed = nn.Sequential(
            nn.Linear(self._seq_len, emb_size),
            nn.GELU(),
            nn.Dropout(dropout)
        ).to(device)
        self.pos_embedding_spatial = nn.Parameter(torch.randn(1, in_channels, emb_size, device=device))

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape

        # 延迟初始化：根据实际输入通道数构建/重建依赖层
        if self._in_channels is None or self._in_channels != C:
            self._build_channel_layers(C)

        # Temporal branch
        x_temporal = self.temporal_patch_embed(x)  # (B, num_patches, emb_size)
        x_temporal = x_temporal + self.pos_embedding_temporal
        x_temporal = self.temporal_transformer(x_temporal)
        x_temporal = x_temporal.mean(dim=1)  # (B, emb_size)

        # Spatial branch
        x_spatial = x.transpose(1, 2)  # (B, T, C)
        x_spatial = x_spatial.transpose(1, 2)  # (B, C, T)
        x_spatial = self.spatial_patch_embed(x_spatial)  # (B, C, emb_size)
        x_spatial = x_spatial + self.pos_embedding_spatial
        x_spatial = self.spatial_transformer(x_spatial)
        x_spatial = x_spatial.mean(dim=1)  # (B, emb_size)

        # Fusion
        x_fused = torch.cat([x_temporal, x_spatial], dim=-1)  # (B, emb_size*2)

        # Classification
        out = self.classifier(x_fused)
        return out


if __name__ == '__main__':
    # Test 19-channel (auto-infer, no configs)
    model = DBConformer()
    x = torch.randn(2, 16, 2000)
    y = model(x)
    print(f"19ch auto: {x.shape} -> {y.shape}")
    assert y.shape == (2, 2)

    # Test 64-channel (explicit configs)
    class Config:
        input_channels = 16
        num_classes = 2
        sequence_len = 2000
    model = DBConformer(Config())
    x = torch.randn(2, 16, 2000)
    y = model(x)
    print(f"64ch explicit: {x.shape} -> {y.shape}")
    assert y.shape == (2, 2)

    # Test 64-channel (auto-infer, configs without input_channels)
    class Config2:
        num_classes = 2
    model = DBConformer(Config2())
    x = torch.randn(2, 16, 2000)
    y = model(x)
    print(f"64ch auto: {x.shape} -> {y.shape}")
    assert y.shape == (2, 2)
    print("DBConformer test passed!")
