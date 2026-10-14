import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Define the frequency range of the band (unit: Hz)
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}
def compute_band_power(signal, sample_rate):
    fft = torch.fft.rfft(signal, dim=-1)
    power_spectrum = torch.abs(fft) ** 2  
    freqs = torch.fft.rfftfreq(signal.size(-1), d=1./sample_rate).to(signal.device) 

    band_powers = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        fmin, fmax = FREQ_BANDS[band]
        idx = (freqs >= fmin) & (freqs < fmax)
        band_power = power_spectrum[..., idx].mean(dim=-1) 
        band_powers.append(band_power)
    
    band_powers = torch.stack(band_powers, dim=-1)  
    band_powers = band_powers.mean(dim=1) 
    return band_powers 



class CodimNet(nn.Module):
    def __init__(self,configs=None):
        super(CodimNet, self).__init__()
        self.num_classes = getattr(configs, "num_classes", 2)
        input_channels = getattr(configs, "input_channels", 16)
        self.sample_rate_orig = float(getattr(configs, "sfreq", 500.0))
        self.sample_rate_mid = float(getattr(configs, "codimnet_mid_sfreq", self.sample_rate_orig / 2.0))
        self.ch_names = getattr(configs, "ch_names", None)
        self.lead_indices_list = self._get_lead_indices(input_channels, self.ch_names)
        self.lead_branches = nn.ModuleList([
            self._make_cnn_branch(in_channels=len(indices))
            for indices in self.lead_indices_list
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(len(self.lead_indices_list) * 32, self.num_classes)
        self.mid_features_list = [None] * len(self.lead_indices_list)
        self.original_x_list = [None] * len(self.lead_indices_list)

        for branch_idx in range(len(self.lead_branches)):
            def create_hook(idx):
                def hook_fn(m, i, o):
                    self.mid_features_list[idx] = o
                return hook_fn
            target_layer = self.lead_branches[branch_idx][10]
            target_layer.register_forward_hook(create_hook(branch_idx))

    def _get_lead_indices(self, input_channels, ch_names=None):
        if ch_names is not None:
            if len(ch_names) != input_channels:
                raise ValueError("CodimNet ch_names length must match input_channels")
            groups = {"frontal": [], "central": [], "parietal": [], "occipital": [], "temporal": []}
            for idx, raw_name in enumerate(ch_names):
                name = re.sub(r"[^A-Za-z0-9]", "", str(raw_name)).upper()
                if name.startswith(("FP", "AF", "F")) and not name.startswith("FC"):
                    group = "frontal"
                elif name.startswith(("FC", "C")):
                    group = "central"
                elif name.startswith(("CP", "P")) and not name.startswith("PO"):
                    group = "parietal"
                elif name.startswith(("PO", "O")):
                    group = "occipital"
                elif name.startswith(("TP", "T")):
                    group = "temporal"
                else:
                    raise ValueError(f"Cannot assign EEG channel to CodimNet region: {raw_name}")
                groups[group].append(idx)
            # Reduced montages can omit a brain region entirely.
            lead_indices = [
                groups[key] for key in ("frontal", "central", "parietal", "occipital", "temporal")
                if groups[key]
            ]
            if not lead_indices:
                raise ValueError("CodimNet requires at least one EEG channel")
            return lead_indices

        if input_channels == 16:
            return [
                [0, 1, 2, 3, 4],      # Frontal: Fp1, Fp2, F3, Fz, F4
                [6, 7, 8],            # Central: C3, Cz, C4
                [10, 11, 12],         # Parietal: P3, Pz, P4
                [13, 14, 15],         # Occipital: O1, Oz, O2
                [5, 9],               # Temporal: T7, T8
            ]
        if input_channels == 19:
            return [
                [0, 5, 1, 6, 16],           # Frontal Lobe(Fp1, Fp2, F3, F4, FZ)
                [17, 2, 7],                 # Central Lobe(CZ, C3, C4)
                [18, 3, 8],                 # Parietal Lobe(PZ, P3, P4)
                [4, 9],                     # Occipital Lobe (O1, O2)
                [10, 13, 11, 14, 12, 15],   # Temporal Lobe(F7, F8, T3, T4, T5, T6)
            ]
        raise ValueError(f"CodimNet does not define lead groups for {input_channels} channels")

    def _make_cnn_branch(self, in_channels):
        layers = []
        layers.append(nn.Conv1d(in_channels, 32, kernel_size=100, stride=2))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=22, stride=1, groups=32))
        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        # Dense Block 1
        layers.append(DenseBlock(32, num_layers=2, growth_rate=16)) 
        layers.append(nn.Conv1d(32 + 2 * 16, 32, kernel_size=1, stride=1)) 


        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Dense Block 2
        layers.append(DenseBlock(32, num_layers=7, growth_rate=16))
        layers.append(nn.Conv1d(32 + 7 * 16, 16, kernel_size=1, stride=1)) 
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=4))

        # Dense Block 3
        layers.append(DenseBlock(16, num_layers=2, growth_rate=8))  
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)


    def forward(self, x):
        branch_outputs = []
        for idx, (indices, branch) in enumerate(zip(self.lead_indices_list, self.lead_branches)):
            input_x = x[:, indices, :]
            self.original_x_list[idx] = input_x.detach()
            out = branch(input_x) 
            out = self.global_avg_pool(out) 
            branch_outputs.append(out)

        concat_output = torch.cat(branch_outputs, dim=1)


        x = self.dropout(concat_output)    
        x = self.flatten(x)                    
        # if age is not None:
        #     x = torch.cat((x, age.reshape(-1, 1)), dim=1)   
        x = self.fc(x)                          
        ND_pinn_losses = []
        for i in range(len(self.lead_indices_list)):
            mid_feats = self.mid_features_list[i]
            original_feats = self.original_x_list[i]
            if mid_feats is not None:
                ND_loss_i = self.compute_ND_pinn_loss(
                    mid_feats, 
                    original_feats,
                    sample_rate_mid=self.sample_rate_mid,
                    sample_rate_orig=self.sample_rate_orig,
                )
                ND_pinn_losses.append(ND_loss_i)
            else:
                ND_pinn_losses.append(torch.tensor(0.0, device=x.device))


        ND_pinn_loss = torch.stack(ND_pinn_losses).mean()       
        return x, ND_pinn_loss
    def compute_ND_pinn_loss(self, mid_features, original_signal, 
                             sample_rate_mid=100, sample_rate_orig=200):

        mid_band_power = compute_band_power(mid_features, sample_rate_mid)          
        original_band_power = compute_band_power(original_signal, sample_rate_orig) 

        mid_band_power = mid_band_power / (mid_band_power.sum(dim=-1, keepdim=True) + 1e-6)
        original_band_power = original_band_power / (original_band_power.sum(dim=-1, keepdim=True) + 1e-6)

        per_band_mse = (torch.log(mid_band_power + 1e-6) - torch.log(original_band_power + 1e-6)).pow(2).mean(dim=0)
        weights = 1.0 / (per_band_mse + 1e-6)
        weights = weights / weights.sum()

        ND_pinn_loss = (weights * per_band_mse).sum()
        return ND_pinn_loss

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)
        
    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(growth_rate),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x
    
if __name__ == "__main__":
    # 使用模型
    model = CodimNet()
    x = torch.randn(16, 19,2000)  # 示例输入：16 个样本，每个样本 19 个通道，每个通道 2000 个数据点
    logits,_= model(x)
    print("logits.shape =", logits.shape)         # (16, 3)
