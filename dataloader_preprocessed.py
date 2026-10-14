
# 加上RAW_EEG的数据，PINN
import os
import torch
from torch.utils.data import Dataset
import numpy as np
class PreprocessedDataset(Dataset):
    """
    加载 {prefix}_raw_eeg_data.pt 中的 samples [N, C, T] 和 labels。
    指定 exclude_channels 时，按 ch_names 筛选通道，保持样本和剩余通道顺序。
    """
    def __init__(self, data_dir, prefix, exclude_channels=None):
        super().__init__()

        # 构造文件路径
        # psd_path       = os.path.join(data_dir, f"{prefix}_psd_features.pt")
        # wavelet_path   = os.path.join(data_dir, f"{prefix}_wavelet_features.pt")
        # nonlinear_path = os.path.join(data_dir, f"{prefix}_nonlinear_features.pt")
        raw_eeg_data_path = os.path.join(data_dir, f"{prefix}_raw_eeg_data.pt")
        # connectivity_path = os.path.join(data_dir, f"{prefix}_connectivity.pt")

        
        # psd_dict       = torch.load(psd_path)
        # wavelet_dict   = torch.load(wavelet_path)
        # nonlinear_dict = torch.load(nonlinear_path)
        raw_eeg_data_dict = torch.load(raw_eeg_data_path, map_location="cpu", weights_only=False)
        # connectivity_dict = torch.load(connectivity_path)
        # 辅助函数：保证输出一定是 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).type(dtype)
            elif isinstance(x, torch.Tensor):
                return x.type(dtype)
            else:
                raise TypeError(f"Unsupported data type: {type(x)}")

        # 转换 samples 和 labels
        # self.psd       = to_tensor(psd_dict["samples"],       torch.float)  # (N, C, F1)
        # self.wavelet   = to_tensor(wavelet_dict["samples"],   torch.float)  # (N, C, F2)
        # self.nonlinear = to_tensor(nonlinear_dict["samples"], torch.float)  # (N, C, F3)
        self.raw_eeg_data = to_tensor(raw_eeg_data_dict["samples"], torch.float)  # (N, C, F)
        # self.connectivity = to_tensor(connectivity_dict["samples"], torch.float)  # (N, C, F)       
        self.labels    = to_tensor(raw_eeg_data_dict["labels"],        torch.long)   # (N,)
        self.ch_names = raw_eeg_data_dict.get("ch_names")
        self.sfreq = raw_eeg_data_dict.get("sfreq")
        self.standardized = raw_eeg_data_dict.get("standardized", False)

        def normalize_names(names, description):
            if isinstance(names, (str, bytes)):
                raise ValueError(f"{raw_eeg_data_path}: {description} must be a list of names")
            try:
                names = list(names)
            except TypeError as exc:
                raise ValueError(f"{raw_eeg_data_path}: invalid {description}") from exc
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError(f"{raw_eeg_data_path}: invalid names in {description}")
            normalized = [name.strip().casefold() for name in names]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{raw_eeg_data_path}: duplicate names in {description}")
            return normalized

        self.excluded_channels = normalize_names(
            exclude_channels if exclude_channels is not None else [], "exclude_channels"
        )
        if self.raw_eeg_data.ndim != 3 or self.raw_eeg_data.shape[1] == 0:
            raise ValueError(f"{raw_eeg_data_path}: samples must have shape [N, C, T] with C > 0")
        self.original_ch_names = None
        if self.ch_names is not None:
            normalized_names = normalize_names(self.ch_names, "ch_names")
            self.ch_names = list(self.ch_names)
            if len(self.ch_names) != self.raw_eeg_data.shape[1]:
                raise ValueError(f"{raw_eeg_data_path}: ch_names length does not match samples channels")
            self.original_ch_names = list(self.ch_names)
        if self.excluded_channels:
            if self.ch_names is None:
                raise ValueError(f"{raw_eeg_data_path}: ch_names is required to exclude channels")
            missing = sorted(set(self.excluded_channels) - set(normalized_names))
            if missing:
                raise ValueError(f"{raw_eeg_data_path}: excluded channels not found: {missing}")
            keep_indices = [
                index for index, name in enumerate(normalized_names)
                if name not in self.excluded_channels
            ]
            if not keep_indices:
                raise ValueError(f"{raw_eeg_data_path}: channel selection would remove all channels")
            self.raw_eeg_data = self.raw_eeg_data[:, keep_indices, :]
            self.ch_names = [self.ch_names[index] for index in keep_indices]
        
        # 加载个体相关信息（如果存在的话）
        self.subject_ids = None
        self.ages = None  
        self.gender = None
        self.mmse = None
        self.moca = None
        
        if "subject_ids" in raw_eeg_data_dict:
            self.subject_ids = to_tensor(raw_eeg_data_dict["subject_ids"], torch.long)
        if "ages" in raw_eeg_data_dict:
            self.ages = to_tensor(raw_eeg_data_dict["ages"], torch.float)
        if "gender" in raw_eeg_data_dict:
            self.gender = to_tensor(raw_eeg_data_dict["gender"], torch.long)  
        if "mmse" in raw_eeg_data_dict:
            self.mmse = to_tensor(raw_eeg_data_dict["mmse"], torch.float)
        if "moca" in raw_eeg_data_dict:
            self.moca = to_tensor(raw_eeg_data_dict["moca"], torch.float)

        assert len(self.raw_eeg_data) == len(self.labels), \
            f"样本数和标签数不匹配：{len(self.raw_eeg_data)} vs {len(self.labels)}"

    def __getitem__(self, idx):
        result = {
            # "psd":       self.psd[idx],       # FloatTensor (C, F1)
            # "wavelet":   self.wavelet[idx],   # FloatTensor (C, F2)
            # "nonlinear": self.nonlinear[idx], # FloatTensor (C, F3)
            "raw_eeg_data": self.raw_eeg_data[idx], # FloatTensor (C, F3)
            # "connectivity": self.connectivity[idx], # FloatTensor (C, F3)         
            "labels":    self.labels[idx]     # LongTensor scalar
        }
        
        # 添加可选的个体信息
        if self.subject_ids is not None:
            result["subject_ids"] = self.subject_ids[idx]
        if self.ages is not None:
            result["ages"] = self.ages[idx]  
        if self.gender is not None:
            result["gender"] = self.gender[idx]
        if self.mmse is not None:
            result["mmse"] = self.mmse[idx]
        if self.moca is not None:
            result["moca"] = self.moca[idx]
        
        return result

    def __len__(self):
        return self.raw_eeg_data.size(0)


from torch.utils.data import DataLoader

def data_generator_preprocessed(data_path, data_type, hparams, exclude_channels=None):
    """
    从 data_path/data_type（或回退到 all）加载 train/val/test_raw_eeg_data.pt。
    三个集合使用同一通道筛选；筛选后的通道顺序必须一致。
    """
    base_dir = os.path.join(data_path, data_type)
    # External CV folds use a stable ``all`` subdirectory while the dataset
    # name contains the source/session for isolated logging and checkpoints.
    if not os.path.isdir(base_dir) and os.path.isdir(os.path.join(data_path, "all")):
        base_dir = os.path.join(data_path, "all")
    train_ds = PreprocessedDataset(base_dir, "train", exclude_channels)
    val_ds  = PreprocessedDataset(base_dir, "val", exclude_channels)
    test_ds  = PreprocessedDataset(base_dir, "test", exclude_channels)
    if train_ds.excluded_channels:
        expected = [name.strip().casefold() for name in train_ds.ch_names]
        for prefix, dataset in (("val", val_ds), ("test", test_ds)):
            actual = [name.strip().casefold() for name in dataset.ch_names]
            if actual != expected:
                raise ValueError(f"{prefix}: selected channel order differs from train: {dataset.ch_names}")

    bs = hparams["batch_size"]


    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=True,  num_workers=4)
    # Validation selects the best checkpoint; test is evaluated after training.
    val_loader   = DataLoader(val_ds,  batch_size=bs, shuffle=False, drop_last=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, drop_last=False, num_workers=4)

    return train_loader, val_loader, test_loader
