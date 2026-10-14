# EEG 模型训练

使用固定训练、验证、测试集进行二分类。训练期间按验证集准确率选择最佳模型，训练结束后在测试集上评估。

## 安装依赖

以下命令在工程根目录执行。依赖版本已在 Python 3.8.20 环境中验证。

已有 `Stroke` 环境时：

```bash
conda activate Stroke
python -m pip install -r requirements.txt
```

新建环境时，可先运行 `conda create -n Stroke python=3.8`，再执行上面的命令。

使用当前工程验证过的 CUDA 12.1 版 PyTorch 时，在安装 requirements 前运行：

```bash
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## 数据目录

默认读取：

```text
data/stroke_hc/all/
├── train_raw_eeg_data.pt
├── val_raw_eeg_data.pt
└── test_raw_eeg_data.pt
```

当前数据为 session 1 的 32 通道 EEG，每个窗口 2000 个时间点，采样率 256 Hz，已标准化。训练、验证、测试窗口数分别为 406、56、56。

模型输入通道数会根据加载后的数据自动设置。训练入口直接读取已处理的 `.pt` 文件。

## 运行

先检查数据路径和运行参数，此命令不会训练：

```bash
python experiment_runner.py --dry-run
```

运行全部四个模型：

```bash
python experiment_runner.py
```

默认依次运行 `CodimNet`、`ADformer`、`EEGNet`、`DBConformer`，使用 seed=42、60 个 epoch、batch size=128。设备自动选择可用的 `cuda:0` 或 CPU。

常用运行方式：

```bash
# 指定单个模型和 GPU
python experiment_runner.py --models EEGNet --device cuda:0

# 删除指定通道，训练、验证、测试集同步筛选；通道名不区分大小写
python experiment_runner.py --models CodimNet --exclude-channels O1 Oz O2

# 调整训练轮数和批次大小
python experiment_runner.py --models EEGNet --num-epochs 30 --batch-size 32

# 指定数据路径：指向 all/ 的父目录
python experiment_runner.py --data-path data/stroke_hc --dataset all

# 使用 CPU
python experiment_runner.py --models EEGNet --device cpu
```

已完成的运行默认跳过；需要重新训练时添加 `--rerun-completed`。完整参数可通过 `python experiment_runner.py --help` 查看。

## 结果和日志

默认结果目录为 `weights/stroke_hc_fixed_val/fixed_val_seed_42_<模型名>/`，包含最佳和最后模型检查点、测试指标 Excel、混淆矩阵、运行配置及日志。删除通道时，实验目录名会追加对应通道标记。

