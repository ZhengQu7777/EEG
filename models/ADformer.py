# 16通道2分类
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
# from layers.ADformer_EncDec import Encoder, EncoderLayer
# from layers.SelfAttention_Family import ADformerLayer
# from layers.Embed import TokenChannelEmbedding
import numpy as np
import math
import random
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        new_x_t, new_x_c, attn_t, attn_c = self.attention(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
        x_t = [_x_t + self.dropout(_nx_t) for _x_t, _nx_t in zip(x_t, new_x_t)]
        x_c = [_x_c + self.dropout(_nx_c) for _x_c, _nx_c in zip(x_c, new_x_c)]

        y_t = x_t = [self.norm1(_x_t) for _x_t in x_t]
        y_t = [self.dropout(self.activation(self.conv1(_y_t.transpose(-1, 1)))) for _y_t in y_t]
        y_t = [self.dropout(self.conv2(_y_t).transpose(-1, 1)) for _y_t in y_t]

        y_c = x_c = [self.norm3(_x_c) for _x_c in x_c]
        y_c = [self.dropout(self.activation(self.conv3(_y_c.transpose(-1, 1)))) for _y_c in y_c]
        y_c = [self.dropout(self.conv4(_y_c).transpose(-1, 1)) for _y_c in y_c]

        return [self.norm2(_x_t + _y_t) for _x_t, _y_t in zip(x_t, y_t)], \
            [self.norm4(_x_c + _y_c) for _x_c, _y_c in zip(x_c, y_c)], attn_t, attn_c


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]
        attns_t = []
        attns_c = []
        for attn_layer in self.attn_layers:
            x_t, x_c, attn_t, attn_c = attn_layer(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
            attns_t.append(attn_t)
            attns_c.append(attn_c)

        """# concat all the outputs
        if x_t:
            x_t = torch.cat(x_t, dim=1)  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat(x_c, dim=1)  # (batch_size, enc_in_1 + enc_in_2 + ... , d_model)
        else:
            x_c = None"""
        # only concat the routers. router is the last patch/channel of each element in the list

        if x_t:
            x_t = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_t], dim=1)   # (batch_size, len(patch_len_list), d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_c], dim=1)  # (batch_size, len(up_dim_list), d_model)
        else:
            x_c = None

        if self.norm is not None:
            x_t = self.norm(x_t) if x_t is not None else None
            x_c = self.norm(x_c) if x_c is not None else None

        return x_t, x_c, attns_t, attns_c
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
class CrossChannelTokenEmbedding(nn.Module):  # (batch_size, 1, enc_in, seq_len)
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)
        return x  # (batch_size, d_model, patch_num, 1)
class UpDimensionChannelEmbedding(nn.Module):  # B x C x T
    def __init__(self, c_in, t_in, u_dim, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.u_dim = u_dim
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=u_dim,
            kernel_size=3,
            padding=padding,
            bias=False,
        )
        self.fc = nn.Linear(t_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)  # B x u_dim x T
        x = self.fc(x)  # B x u_dim x d_model
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]
class Jitter(nn.Module):
    # apply noise on each element
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            x += torch.randn_like(x) * self.scale
        return x


class Scale(nn.Module):
    # scale each channel by a random scalar
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            x *= 1 + torch.randn(B, C, 1, device=x.device) * self.scale
        return x


class Flip(nn.Module):
    # left-right flip
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            return torch.flip(x, [-1])
        return x


class Shuffle(nn.Module):
    # shuffle channels order
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training and torch.rand(1) < self.prob:
            B, C, T = x.shape
            perm = torch.randperm(C)
            return x[:, perm, :]
        return x


class FrequencyMask(nn.Module):
    def __init__(self, ratio=0.1):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            B, C, T = x.shape
            # Perform rfft
            x_fft = torch.fft.rfft(x, dim=-1)
            # Generate random indices for masking
            mask = torch.rand(x_fft.shape, device=x.device) > self.ratio
            # Apply mask
            x_fft = x_fft * mask
            # Perform inverse rfft
            x = torch.fft.irfft(x_fft, n=T, dim=-1)
        return x
def get_augmentation(augmentation):
    if augmentation.startswith("jitter"):
        if len(augmentation) == 6:
            return Jitter()
        return Jitter(float(augmentation[6:]))
    elif augmentation.startswith("scale"):
        if len(augmentation) == 5:
            return Scale()
        return Scale(float(augmentation[5:]))
    elif augmentation.startswith("drop"):
        if len(augmentation) == 4:
            return nn.Dropout(0.1)
        return nn.Dropout(float(augmentation[4:]))
    elif augmentation.startswith("flip"):
        if len(augmentation) == 4:
            return Flip()
        return Flip(float(augmentation[4:]))
    elif augmentation.startswith("shuffle"):
        if len(augmentation) == 7:
            return Shuffle()
        return Shuffle(float(augmentation[7:]))
    elif augmentation.startswith("frequency"):
        if len(augmentation) == 9:
            return FrequencyMask()
        return FrequencyMask(float(augmentation[9:]))
    elif augmentation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown augmentation {augmentation}")
class TokenChannelEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len_list,
        up_dim_list,
        stride_list,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]

        linear_layers_t = [
            CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        linear_layers_c = [
            UpDimensionChannelEmbedding(
                c_in=enc_in,
                t_in=seq_len,
                u_dim=u_dim,
                d_model=d_model,
            )  # c_in is seq_length here
            for u_dim in up_dim_list
        ]
        self.value_embeddings_t = nn.ModuleList(linear_layers_t)
        self.value_embeddings_c = nn.ModuleList(linear_layers_c)
        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_c = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

        self.learnable_embeddings_t = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.patch_len_list]
        )
        self.learnable_embeddings_c = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.up_dim_list]
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        x_list_t = []
        x_list_c = []
        for padding, value_embedding_t in zip(self.paddings, self.value_embeddings_t):
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_t = self.augmentation[aug_idx](x_copy)
            # temporal dimension
            x_new_t = padding(x_new_t).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new_t = value_embedding_t(x_new_t)  # (batch_size, d_model, 1, patch_num)
            x_new_t = x_new_t.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            x_list_t.append(x_new_t)

        for value_embedding_c in self.value_embeddings_c:
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new_c = self.augmentation[aug_idx](x_copy)
            # add positional embedding to tag each channel
            x_new_c = x_new_c + self.position_embedding_c(x_new_c)
            # channel dimension
            x_new_c = value_embedding_c(x_new_c)  # (batch_size, enc_in, d_model)
            x_list_c.append(x_new_c)

        x_t = [
            x + cxt + self.position_embedding_t(x)
            for x, cxt in zip(x_list_t, self.learnable_embeddings_t)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        x_c = [
            x + cxt
            for x, cxt in zip(x_list_c, self.learnable_embeddings_c)
        ]  # (batch_size, enc_in, d_model), (batch_size, enc_in, d_model), ...
        return x_t, x_c
class ADformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks_t,
        num_blocks_c,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions_t = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_t)
            ]
        )
        self.intra_attentions_c = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_c)
            ]
        )
        if no_inter or num_blocks_t <= 1:
            # print("No inter attention for time")
            self.inter_attention_t = None
        else:
            self.inter_attention_t = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )
        if no_inter or num_blocks_c <= 1:
            # print("No inter attention for channel")
            self.inter_attention_c = None
        else:
            self.inter_attention_c = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        attn_mask_t = ([None] * len(x_t))
        attn_mask_c = ([None] * len(x_c))

        # Intra attention
        x_intra_t = []
        attn_out_t = []
        x_intra_c = []
        attn_out_c = []
        # Temporal dimension
        for x_in_t, layer_t, mask in zip(x_t, self.intra_attentions_t, attn_mask_t):
            _x_out_t, _attn_t = layer_t(x_in_t, x_in_t, x_in_t, attn_mask=mask, tau=tau, delta=delta)
            x_intra_t.append(_x_out_t)  # (B, Ni, D)
            attn_out_t.append(_attn_t)
        # Channel dimension
        for x_in_c, layer_c, mask in zip(x_c, self.intra_attentions_c, attn_mask_c):
            _x_out_c, _attn_c = layer_c(x_in_c, x_in_c, x_in_c, attn_mask=mask, tau=tau, delta=delta)
            x_intra_c.append(_x_out_c)  # (B, C, D)
            attn_out_c.append(_attn_c)

        # Inter attention
        if self.inter_attention_t is not None:
            # Temporal dimension
            routers_t = torch.cat([x[:, -1:] for x in x_intra_t], dim=1)  # (B, n, D)
            x_inter_t, attn_inter_t = self.inter_attention_t(
                routers_t, routers_t, routers_t, attn_mask=None, tau=tau, delta=delta
            )
            x_out_t = [
                torch.cat([x[:, :-1], x_inter_t[:, i : i + 1]], dim=1)  # (B, Ni, D)
                for i, x in enumerate(x_intra_t)
            ]
            attn_out_t += [attn_inter_t]
        else:
            x_out_t = x_intra_t

        if self.inter_attention_c is not None:
            # Channel dimension
            routers_c = torch.cat([x[:, -1:] for x in x_intra_c], dim=1)  # (B, n, D)
            x_inter_c, attn_inter_c = self.inter_attention_c(
                routers_c, routers_c, routers_c, attn_mask=None, tau=tau, delta=delta
            )
            x_out_c = [
                torch.cat([x[:, :-1], x_inter_c[:, i : i + 1]], dim=1)  # (B, C, D)
                for i, x in enumerate(x_intra_c)
            ]
            attn_out_c += [attn_inter_c]
        else:
            x_out_c = x_intra_c

        return x_out_t, x_out_c, attn_out_t, attn_out_c
    
class ADformer(nn.Module):
    def __init__(self, configs=None):
        super(ADformer, self).__init__()
        # --- hard-coded hyperparameters ---
        self.task_name = "classification"
        self.enc_in = getattr(configs, "input_channels", 16)
        self.seq_len = getattr(configs, "sequence_len", 2000)
        self.d_model = 64
        self.n_heads = 4
        self.d_ff = 256
        self.e_layers = 2
        self.dropout = 0.1
        self.activation = "relu"
        self.output_attention = False
        self.no_inter_attn = False
        self.patch_len_list = [100, 200]   # two temporal patch sizes
        self.up_dim_list = [32]            # one channel granularity
        self.stride_list = self.patch_len_list
        self.augmentations = ["none"]
        self.num_class = getattr(configs, "num_classes", 2)

        # Embedding
        self.enc_embedding = TokenChannelEmbedding(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            d_model=self.d_model,
            patch_len_list=self.patch_len_list,
            up_dim_list=self.up_dim_list,
            stride_list=self.stride_list,
            dropout=self.dropout,
            augmentation=self.augmentations,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        num_blocks_t=len(self.patch_len_list),
                        num_blocks_c=len(self.up_dim_list),
                        d_model=self.d_model,
                        n_heads=self.n_heads,
                        dropout=self.dropout,
                        output_attention=self.output_attention,
                        no_inter=self.no_inter_attn,
                    ),
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model),
        )

        # Classifier head
        in_features = self.d_model * (len(self.patch_len_list) + len(self.up_dim_list))
        self.act = F.gelu
        self.drop = nn.Dropout(self.dropout)
        self.projection = nn.Linear(in_features, self.num_class)

    def classification(self, x_enc):
        # x_enc: (B, C=16, T=2000) -> we need (B, T, C)
        x_enc = x_enc.permute(0, 2, 1)  # (B, T, C)
        # Embedding + Encoder
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c)
        # concat temporal & channel features
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)  # (B, N, D)
        # head
        out = self.act(enc_out)
        out = self.drop(out)
        out = out.reshape(out.size(0), -1)  # (B, N*D)
        out = self.projection(out)         # (B, 2)
        return out

    def forward(self, x):
        return self.classification(x)


if __name__ == "__main__":
    # Example run
    model = ADformer()
    # batch of 16 EEG samples: 19 channels, 2000 time points
    x = torch.randn(16, 16, 2000)
    output = model(x)
    print(output.shape)  # should be (16, 2)
