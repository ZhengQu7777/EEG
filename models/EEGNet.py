import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class EEGNet(nn.Module):
    """
    EEGNet v4 (Lawhern et al., 2018) in PyTorch.
    Input  : x of shape (B, Ch, Samples)  # e.g., (B, 16, 2000)
    Output : logits of shape (B, num_classes)
    """
    def __init__(
        self,
        num_classes: int = 2,
        Ch: int = 16,                # number of EEG channels
        Samples: int = 2000,         # time samples per trial
        F1: int = 8,                 # temporal filter count
        D: int = 2,                  # depth multiplier for spatial depthwise conv
        kernel_length: int = 64,     # temporal kernel length in block1
        dropout_prob: float = 0.5,
        configs=None,
    ):
        super().__init__()
        if configs is not None:
            Ch = getattr(configs, "input_channels", Ch)
            Samples = getattr(configs, "sequence_len", Samples)
            num_classes = getattr(configs, "num_classes", num_classes)
        self.Ch = Ch
        self.Samples = Samples
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D

        # -------------------------
        # Block 1: Temporal + Spatial Depthwise
        # -------------------------
        # Temporal conv: keep time length EXACT (padding='same' handles even kernels correctly)
        self.conv_temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Spatial depthwise conv across channels
        self.conv_spatial = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(Ch, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(dropout_prob)

        # -------------------------
        # Block 2: Separable Convolution
        # -------------------------
        # Depthwise temporal conv (keep time length)
        self.sep_depthwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            groups=F1 * D,
            padding='same',
            bias=False
        )
        # Pointwise conv
        self.sep_pointwise = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout_prob)

        # -------------------------
        # Classification
        # -------------------------
        # After pool1(÷4) and pool2(÷8), overall ÷32 in time.
        time_len_after = Samples // 32  # integer floor, matches PyTorch pooling
        in_features = self.F2 * time_len_after  # height is 1 after spatial conv
        self.classifier = nn.Linear(in_features, num_classes)

        # Optional: weight init close to Keras defaults (not required)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, Ch, Samples)
        """
        # add an explicit 'conv channel' dimension -> (B, 1, Ch, Samples)
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv_temporal(x)     # (B, F1, Ch, T)
        x = self.bn1(x)
        x = self.conv_spatial(x)      # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)             # (B, F1*D, 1, T/4)
        x = self.drop1(x)

        # Block 2 (Separable)
        x = self.sep_depthwise(x)     # depthwise temporal
        x = self.sep_pointwise(x)     # pointwise -> (B, F2, 1, T)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)             # (B, F2, 1, T/32)
        x = self.drop2(x)

        # Flatten and classify
        x = x.flatten(start_dim=1)    # (B, F2 * floor(T/32))
        logits = self.classifier(x)   # (B, num_classes)
        return logits


if __name__ == "__main__":
    # sanity check with your setting: (N, 16, 2000)
    B, Ch, T = 16, 16, 2000
    model = EEGNet(num_classes=2, Ch=Ch, Samples=T, F1=8, D=2, kernel_length=64, dropout_prob=0.5)
    x = torch.randn(B, Ch, T)
    y = model(x)
    print("Output shape:", y.shape)  # expected: (16, 2)







