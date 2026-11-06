# 脑电数据安全平台

EEG 仓库中的网页平台源码。使用者选择项目和数据、提交训练任务；管理者登记数据、处理审阅与授权、批准模型交付。前端使用原生 HTML/CSS/JavaScript，Node.js 原生 HTTP 服务提供网页和 API。

联合训练页面提供可点击的架构图：合作方 → 训练项目与参与数据 → MP-SPDZ 计算方 → 模型交付。协议切换会同步显示两方或三方计算节点。

## 两种训练流程

| 方式 | 输入 | 流程 |
| --- | --- | --- |
| 本地训练 | 自有数据、Python 项目和启动命令 | 大模型代码审阅 → 模拟数据试运行 → bwrap 沙箱训练 → 产出检查 |
| 联合训练 | 合作方数据、MPC 程序及协议参数 | 大模型代码审阅 → 编译检查 → 数据方授权 → MP-SPDZ 训练 → 批准交付 |

网页通过 `/api` 获取数据，每 2 秒刷新任务状态。联合训练当前选择的是 `.mpc` 程序；训练轮数等写在程序中。Python 工具仍保留部分静态检查和模型转换子命令，但当前任务审阅主流程只调用大模型。

## 运行依赖

- Linux；Node.js 20（当前验证版本），服务端只使用 Node 内置模块，无需 `npm install`。
- Python 环境中安装 PyTorch、NumPy；运行仓库中的 EEG 训练脚本还需要根目录 `requirements.txt` 中的依赖。CUDA 与 PyTorch 版本需适配实际 GPU，可先用 `--device cpu` 验证。
- 本地训练需要 Bubblewrap（`bwrap`）、`unshare`、GNU `time`，以及允许相应用户命名空间的系统配置。
- 计算方证书生成需要 `openssl`。
- 联合训练需要单独安装并编译 [MP-SPDZ](https://github.com/data61/MP-SPDZ)，包括所选协议的计算方二进制和 Python 编译环境。它没有复制进本目录。

## 配置与启动

从仓库根目录执行：

```bash
cd platform
cp config.example.json config.local.json
```

在 `config.local.json` 中填入自己的大模型 API 密钥。该文件已被忽略；仓库中的示例密钥为空。未配置密钥时，任务会转入人工确认。

服务保留了原部署环境的默认路径。首次在其他机器或新的仓库位置运行时，显式配置路径。下面以 `$HOME/miniconda3` 和 `$HOME/MP-SPDZ` 为例，请按本机安装位置修改：

```bash
export EEGSEC_DEFAULT_DATASET="$(cd .. && pwd)/data/stroke_hc/all"
export EEGSEC_CONDA_ROOT="$HOME/miniconda3"
export EEGSEC_PYTHON="$EEGSEC_CONDA_ROOT/envs/Stroke/bin/python"
export MPSPDZ_ROOT="$HOME/MP-SPDZ"
node server.cjs
```

默认监听 `127.0.0.1:4173`。如果端口已有平台在使用，可运行 `PORT=4174 node server.cjs`。

需要后台运行时，先创建日志目录：

```bash
mkdir -p runtime
nohup node server.cjs > runtime/server.log 2>&1 &
```

VS Code Remote SSH 可在“端口”面板转发 4173，或在远程终端执行：

```bash
code --openExternal http://127.0.0.1:4173/
```

也可以在自己的电脑上建立 SSH 转发：

```bash
ssh -N -L 4173:127.0.0.1:4173 用户名@服务器
```

然后用浏览器打开 `http://127.0.0.1:4173/`。服务重启后，需要重新选择账号登录。

## 数据与账号

登记数据集时填写服务器上的目录，目录应包含 `train_raw_eeg_data.pt`、`val_raw_eeg_data.pt`、`test_raw_eeg_data.pt`。网页展示元信息与版本指纹。平台目录没有包含原始脑电、合作方演示数据、模型权重或旧任务运行记录；首次运行需要准备并登记自己的数据。

当前项目列表根据自有数据集路径中的 `data/` 目录推断项目根。例如 `EEG/data/stroke_hc/all` 对应项目 `EEG`。训练快照和项目扫描会排除平台自身及运行目录，避免将平台配置、演示代码或任务产出当作训练项目代码。

默认账号为“研发人员”（使用者）和“数据管理员”（管理者）。也可以在本地配置中加入：

```json
{
  "users": [
    { "id": "developer", "name": "研发人员", "role": "user" },
    { "id": "manager", "name": "数据管理员", "role": "admin" }
  ]
}
```

使用者只能查看自己的任务、下载已放行产出；管理者负责数据管理、审批、监控和审计，不能下载提交人的训练结果。

## 配置项

| 环境变量 | 用途 |
| --- | --- |
| `HOST` / `PORT` | 监听地址与端口，默认 `127.0.0.1:4173` |
| `EEGSEC_PYTHON` | 平台 Python 工具及 MPC 编译使用的解释器 |
| `EEGSEC_CONDA_ROOT` | 本地训练可选择的 Conda 环境根目录 |
| `EEGSEC_DEFAULT_DATASET` | 首次启动时可自动登记的数据集目录 |
| `MPSPDZ_ROOT` | MP-SPDZ 安装目录 |
| `EEGSEC_RUNTIME` | 状态与任务产出目录，默认本目录下的 `runtime/` |
| `EEGSEC_PARTY_MEM_GB` | 每个计算方进程的内存限制，默认 16 GB |
| `EEGSEC_TOOL_MEM_GB` | Python 工具的内存限制，默认 24 GB |

## 当前部署范围

当前是研究与演示原型。登录只选择账号，不校验密码；默认仅监听本机地址。合作方数据仍在同一服务器，授权由管理员代为登记；默认三方协议的 P0、P1、P2 是本机进程，不能将其描述为已经跨机构独立部署。

当前代码仍有待完善的安全边界：本地训练原始日志没有单独做出站审查；超过 2 GiB 的产出会跳过内容扫描；MPC 编译的隔离不等同于本地训练使用的文件系统沙箱。请勿将现有检查机制视为完整的数据防泄漏保证。

## 验证

在仓库根目录执行下面的检查。快照回归使用临时文件验证嵌套运行目录、目录别名、排除规则和脚本执行权限，不运行真实训练：

```bash
node --check platform/server.cjs
node --check platform/app.js
node platform/tests/snapshot.test.cjs
```

## 文件结构

```text
platform/
├── index.html / styles.css / app.js  # 网页及交互架构图
├── server.cjs                      # HTTP API、角色与训练调度
├── config.example.json             # 不含真实密钥的配置模板
├── backend/
│   ├── eegsec.py                   # 元信息、模拟数据、输入准备和产出扫描
│   └── sandbox/sitecustomize.py    # Python 沙箱事件记录
├── samples/
│   ├── eegnet_mpc_train.mpc         # 联合训练示例
│   └── leak_demo.py                # 故意包含外传行为的审阅/拦截演示
└── runtime/                        # 本地生成，不提交 Git
```

任务状态与审计保存在 `runtime/state/state.json`；代码快照、日志、模型在 `runtime/tasks/`。API 用法见 [api-contract.md](api-contract.md)。
