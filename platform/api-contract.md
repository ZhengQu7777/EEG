# 平台 API

接口返回 JSON；错误一般返回 `{"error":"原因"}` 和 4xx 状态码。除账号列表和登录外，接口需要登录 Cookie `eegsec_sid`。下载接口返回文件。

当前登录仅按账号 ID 建立内存会话，不校验密码；服务重启后会话失效。

| 方法与路径 | 权限与用途 |
| --- | --- |
| `GET /api/accounts` | 无需登录，列出可选账号 |
| `POST /api/login` | 无需登录，`{"id":"user"}`；返回账号并设置会话 Cookie |
| `POST /api/logout` | 已登录，清除会话 |
| `GET /api/state` | 已登录；使用者只获得自己的任务，管理者获得全部任务和审计 |
| `GET /api/projects` | 已登录；返回 `{items:{projects:[],samples:[]}}`，包含项目和可选 MPC 文件 |
| `POST /api/datasets` | 管理者；登记服务器数据目录 |
| `POST /api/datasets/:id` | 管理者；修改数据集元信息及归属 |
| `POST /api/code/classes` | 已登录；扫描模型类，保留的辅助接口 |
| `POST /api/tasks` | 使用者；提交任务 |
| `GET /api/tasks/:id/log` | 管理者或任务所有者；最近 400 行日志 |
| `POST /api/tasks/:id/review` | 管理者；`{"model":"deepseek"}`，重新审阅 |
| `POST /api/tasks/:id/confirm` | 管理者；`{"decision":"continue"}` 或 `"block"`，处理待人工确认的任务 |
| `POST /api/tasks/:id/consent` | 管理者；`{"dataset":"DS-02","decision":"approve"}` 或 `"reject"` |
| `POST /api/tasks/:id/return` | 管理者；退回被拦截或需修改的任务，可附 `note` |
| `POST /api/tasks/:id/cancel` | 管理者或任务所有者；终止或撤回任务 |
| `POST /api/tasks/:id/deliver` | 管理者；批准联合训练模型交付 |
| `GET /api/tasks/:id/model` | 任务所有者；下载已批准交付的模型参数 |
| `GET /api/tasks/:id/output?path=...` | 任务所有者；下载已放行且在产出清单中的文件 |

## 数据登记

```json
{
  "path": "/path/to/EEG/data/stroke_hc/all",
  "name": "脑电数据集",
  "owner": "数据管理员",
  "unit": "本机构",
  "classNames": ["健康对照", "卒中"],
  "ownership": "own",
  "partner": ""
}
```

合作方数据使用 `ownership:"partner"`，并填写非空 `partner`。

## 本地任务

```json
{
  "mode": "local",
  "title": "EEGNet 本地训练",
  "datasets": ["DS-01"],
  "code": { "source": "path", "path": "/path/to/EEG" },
  "local": {
    "command": "python experiment_runner.py --models EEGNet --num-epochs 5 --device cpu",
    "env": "Stroke",
    "trialSeconds": 120
  }
}
```

`code` 也支持 `{source:"upload",files:[{path,text}]}`。本地任务只能使用自有数据。提交人从登录会话中获取。

## 联合任务

```json
{
  "mode": "mpc",
  "title": "联合 EEGNet 训练",
  "datasets": ["DS-01", "DS-02"],
  "mpc": {
    "programPath": "sample:eegnet_mpc_train.mpc",
    "params": {
      "protocol": "rep3",
      "precision": "standard",
      "downsample": 4,
      "perClass": 32
    }
  }
}
```

至少选择一个合作方数据集。`programPath` 可引用 `sample:` 示例或项目中的 `.mpc` 路径；接口也支持 `mpc.program:{name,text}` 提供程序源码。当前协议为 `rep3`（三方）或 `semi2k`（两方），精度为 `standard` 或 `high`，降采样为 1/2/4/8；`perClass:0` 表示使用全部窗口。训练轮数、批次等在 `.mpc` 程序中指定。

## 任务状态

- `mode`：`local`、`mpc`。
- 本地步骤：`review → trial → train → check`。
- 联合步骤：`review → compile → consent → train → deliver`。
- `state`：`running`、`waiting`、`queued`、`done`、`blocked`、`needs-fix`、`returned`、`rejected`、`failed`、`cancelled`。

服务重启会将原先 `running` 或 `queued` 的任务标记为失败，需要重新提交。当前 API 没有 `/api/tasks/estimate` 接口。
