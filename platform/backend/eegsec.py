"""脑电数据安全平台的后端工具。

子命令（结果都以 JSON 打印到标准输出）：
  meta     读取数据集目录，返回元信息和版本指纹
  static   对模型代码做静态检查（AST）
  trial    在断网环境中用合成数据试运行模型代码（由 server.cjs 通过 unshare -rn 调用）
  prepare  按任务配置准备 MPC 输入数据（真实数据或同形状的合成数据）
  mpc      生成 MP-SPDZ 训练程序
"""
import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

SPLITS = ("train", "val", "test")


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def fail(message):
    emit({"ok": False, "error": message})
    sys.exit(1)


# ---------------------------------------------------------------------------
# meta
# ---------------------------------------------------------------------------
def cmd_meta(args):
    import torch

    root = Path(args.path)
    files = [root / f"{s}_raw_eeg_data.pt" for s in SPLITS]
    missing = [f.name for f in files if not f.is_file()]
    if missing:
        fail("目录中缺少：" + "、".join(missing))
    digest = hashlib.sha256()
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    windows, subject_label, splits, first = {}, {}, {}, None
    for split, f in zip(SPLITS, files):
        d = torch.load(f, map_location="cpu")
        first = first or d
        labels = [int(v) for v in d["labels"].tolist()]
        sids = d.get("subject_ids")
        sids = [int(v) for v in sids.tolist()] if sids is not None else [None] * len(labels)
        by_label = {}
        for lab in labels:
            by_label[str(lab)] = by_label.get(str(lab), 0) + 1
        splits[split] = {"windows": len(labels), "subjects": len(set(s for s in sids if s is not None)), "byLabel": by_label}
        for lab, sid in zip(labels, sids):
            windows[lab] = windows.get(lab, 0) + 1
            if sid is not None:
                subject_label[sid] = lab
    shape = list(first["samples"].shape)
    classes = {}
    for lab in sorted(windows):
        classes[str(lab)] = [sum(1 for v in subject_label.values() if v == lab), windows[lab]]
    hexd = digest.hexdigest()
    emit({
        "ok": True,
        "channels": shape[1],
        "points": shape[2],
        "sfreq": float(first.get("sfreq", 0) or 0),
        "chNames": list(first.get("ch_names", [])),
        "windowSize": first.get("window_size"),
        "step": first.get("step"),
        "standardization": first.get("standardization"),
        "session": first.get("session"),
        "tag": str(first.get("dataset") or ""),
        "splits": splits,
        "classes": classes,
        "subjects": len(subject_label),
        "windows": sum(windows.values()),
        "fingerprint": f"sha256:{hexd}",
        "fingerprintShort": f"sha256:{hexd[:8]}…{hexd[-6:]}",
        "bytes": sum(f.stat().st_size for f in files),
    })


# ---------------------------------------------------------------------------
# static
# ---------------------------------------------------------------------------
NET_MODULES = {"socket", "requests", "urllib.request", "urllib2", "urllib3", "http.client", "http.server", "httpx",
               "aiohttp", "ftplib", "smtplib", "paramiko", "websocket", "websockets", "telnetlib", "ssl", "grpc"}
CMD_MODULES = {"subprocess", "pty", "ctypes", "cffi"}
DYN_MODULES = {"importlib", "runpy"}
ENC_MODULES = {"base64", "marshal", "pickle", "zlib", "bz2", "lzma", "codecs", "shelve"}
DYN_BUILTINS = {"eval", "exec", "compile", "__import__"}
OS_CMDS = {"system", "popen", "spawnl", "spawnv", "spawnve", "execv", "execve", "execl", "execlp", "execvp", "fork", "forkpty"}
WRITE_FUNCS = {"save", "savez", "savez_compressed", "savetxt", "to_csv", "tofile", "write_bytes", "write_text"}


CODE_SUFFIXES = {".py"}


def cmd_rules(args):
    """对项目目录里的所有 .py 和 .mpc（MP-SPDZ 程序也是 Python 语法）文件做规则检查。"""
    root = Path(args.dir)
    findings, files, lines = [], 0, 0
    for p in sorted(list(root.rglob("*.py")) + list(root.rglob("*.mpc"))):
        if any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts):
            continue
        res = static_findings(p, str(p.relative_to(root)), absolute_write_level="warn")
        findings.extend(res["findings"])
        files += 1
        lines += res["lines"]
    emit({"ok": True, "findings": findings, "files": files, "lines": lines})


def cmd_classes(args):
    """列出项目里定义的 torch.nn.Module 子类（只解析语法，不执行代码）。"""
    root = Path(args.dir)
    found = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) if hasattr(ast, "unparse") else getattr(b, "attr", getattr(b, "id", "")) for b in node.bases]
                if any(b.split(".")[-1] == "Module" for b in bases):
                    found.append({"file": str(rel), "class": node.name, "line": node.lineno})
    emit({"ok": True, "classes": found})


def cmd_static(args):
    emit(static_findings(Path(args.file), args.name or Path(args.file).name))


def static_findings(path, name, absolute_write_level="danger"):
    src = path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    findings = []

    def add(level, rule, node):
        line = getattr(node, "lineno", 0)
        code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        findings.append({"level": level, "rule": rule, "where": f"{name}:{line}", "code": code[:200]})

    try:
        tree = ast.parse(src, filename=name)
    except SyntaxError as exc:
        findings.append({"level": "danger", "rule": "语法错误", "where": f"{name}:{exc.lineno}", "code": str(exc.msg)})
        return {"ok": True, "findings": findings, "lines": len(lines)}

    # 整个训练项目里，子进程、动态加载只作提示；模型文件（联合训练只导入模型）按严格规则
    strict = absolute_write_level == "danger"

    def check_module(mod, node):
        mod = mod or ""
        top = mod.split(".")[0]
        if mod in NET_MODULES or top in NET_MODULES or any(mod.startswith(n + ".") for n in NET_MODULES):
            add("danger", "网络访问", node)
        elif top in CMD_MODULES:
            add("danger" if strict else "warn", "执行系统命令", node)
        elif top in DYN_MODULES:
            add("danger" if strict else "warn", "动态加载代码", node)
        elif top in ENC_MODULES:
            add("warn", "可疑编码或序列化", node)
        elif top == "multiprocessing":
            add("warn", "创建子进程", node)

    def check_path(node, call):
        texts = [a.value for a in list(call.args) + [k.value for k in call.keywords]
                 if isinstance(a, ast.Constant) and isinstance(a.value, str) and ("/" in a.value or "." in a.value)]
        if not texts:
            add("warn", "写入路径无法静态确定", node)
        elif any(t.startswith(("/", "~")) or ".." in t for t in texts):
            add(absolute_write_level, "写入工作目录以外的路径", node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_module(alias.name, node)
        elif isinstance(node, ast.ImportFrom):
            check_module(node.module, node)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in DYN_BUILTINS:
                add("danger" if strict else "warn", "动态执行代码", node)
            elif isinstance(fn, ast.Name) and fn.id == "open":
                mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str) and set(mode.value) & set("wax+"):
                    check_path(node, ast.Call(func=fn, args=node.args[:1], keywords=[]))
            elif isinstance(fn, ast.Attribute):
                if isinstance(fn.value, ast.Name) and fn.value.id == "os" and fn.attr in OS_CMDS:
                    add("danger" if strict else "warn", "执行系统命令", node)
                elif fn.attr in WRITE_FUNCS:
                    check_path(node, node)
    findings.sort(key=lambda f: int(f["where"].rsplit(":", 1)[1] or 0))
    return {"ok": True, "findings": findings, "lines": len(lines)}


# ---------------------------------------------------------------------------
# trial
# ---------------------------------------------------------------------------
CANARY = [7.3125 + 0.0625 * i for i in range(32)]  # float32 可精确表示的金丝雀序列
SUPPORTED_MODULES = {"Linear", "Conv2d", "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d", "ReLU", "Flatten", "BatchNorm2d", "Dropout"}
SUPPORTED_FUNCS = {"flatten", "add", "cat", "relu", "avg_pool2d", "adaptive_avg_pool2d"}
SUPPORTED_METHODS = {"flatten", "view", "size"}
KWARG_ALIASES = {
    "channels": ("channels", "Ch", "ch", "n_channels", "num_channels", "input_channels", "in_chans", "chans"),
    "samples": ("samples", "Samples", "n_samples", "sequence_len", "seq_len", "n_times", "T", "time_points"),
    "classes": ("num_classes", "n_classes", "nb_classes", "classes", "n_outputs"),
}


def pick_model_class(mod, torch):
    own = [c for c in vars(mod).values()
           if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == mod.__name__]
    if not own:
        raise ValueError("代码里没有定义 torch.nn.Module 子类")
    if hasattr(mod, "MODEL_CLASS"):
        return getattr(mod, mod.MODEL_CLASS) if isinstance(mod.MODEL_CLASS, str) else mod.MODEL_CLASS
    return own[-1]


def model_kwargs(cls, channels, samples, classes):
    import inspect
    params = inspect.signature(cls.__init__).parameters
    values = {"channels": channels, "samples": samples, "classes": classes}
    kwargs = {}
    for key, names in KWARG_ALIASES.items():
        for n in names:
            if n in params:
                kwargs[n] = values[key]
                break
    return kwargs


def describe_ops(model, torch):
    import operator
    graph = torch.fx.symbolic_trace(model).graph
    counts = {}
    for node in graph.nodes:
        if node.op == "call_module":
            target = model
            for attr in node.target.split("."):
                target = getattr(target, attr)
            name = type(target).__name__
            note, level = "", "ok" if name in SUPPORTED_MODULES else "danger"
            if name == "Conv2d" and getattr(target, "groups", 1) != 1:
                level, note = "warn", "groups≠1，MP-SPDZ 会当成普通卷积，参数变多"
        elif node.op == "call_function":
            name = "+" if node.target is operator.add else getattr(node.target, "__name__", str(node.target))
            level, note = ("ok" if name in SUPPORTED_FUNCS or name == "+" else "danger"), ""
        elif node.op == "call_method":
            name = f".{node.target}()"
            level, note = ("ok" if node.target in SUPPORTED_METHODS else "danger"), ""
        else:
            continue
        key = (name, level, note)
        counts[key] = counts.get(key, 0) + 1
    return [{"name": k[0], "level": k[1], "note": k[2], "count": v} for k, v in counts.items()]


def cmd_trial(args):
    import resource
    import numpy as np
    import torch

    torch.set_num_threads(2)
    work = Path(args.workdir).resolve()
    out_dir = work / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = str(out_dir)
    events = {"network": [], "process": [], "writes": []}
    report = {"ok": True, "events": events, "canary": {"found": 0, "files": []}}

    def hook(event, a):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto", "socket.sendmsg"):
            events["network"].append(f"{event} {a[1] if len(a) > 1 else ''}"[:200])
            raise PermissionError("试运行环境禁止网络访问")
        if event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "pty.spawn"):
            events["process"].append(f"{event} {a[0] if a else ''}"[:200])
            raise PermissionError("试运行环境禁止执行系统命令")
        if event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
            mode, flags = (a[1] if len(a) > 1 else None), (a[2] if len(a) > 2 else 0)
            writing = (isinstance(mode, str) and bool(set(mode) & set("wax+"))) or \
                      (mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
            if writing:
                p = os.path.realpath(os.fsdecode(a[0]))
                if not (p.startswith(allowed) or p.startswith("/dev/")):
                    events["writes"].append(p[:200])
                    raise PermissionError("试运行环境只允许写输出目录")

    rng = np.random.default_rng(0)
    n, c, t, k = 8, args.channels, args.samples, args.classes
    x = rng.standard_normal((n, 1, c, t)).astype(np.float32)
    for w in (0, 3, 5):
        x[w, 0, 0, 100:100 + len(CANARY)] = CANARY
    y = np.arange(n) % k
    report["input"] = [n, 1, c, t]

    args.file = str(Path(args.file).resolve())
    if args.root:
        sys.path.insert(0, str(Path(args.root).resolve()))
    sys.dont_write_bytecode = True  # 不写 __pycache__，避免误报
    sys.addaudithook(hook)
    os.chdir(out_dir)
    start = time.time()
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_model", args.file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, args.cls) if args.cls else pick_model_class(mod, torch)
        kwargs = model_kwargs(cls, c, t, k)
        model = cls(**kwargs)
        report["modelClass"] = cls.__name__
        report["modelKwargs"] = kwargs
        report["params"] = int(sum(p.numel() for p in model.parameters()))
        try:
            report["ops"] = describe_ops(model, torch)
        except Exception as exc:  # 追踪失败时 MP-SPDZ 也无法导入
            report["ops"] = []
            report["traceError"] = f"{type(exc).__name__}: {exc}"[:300]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        model.train()
        xb, yb = torch.from_numpy(x), torch.from_numpy(y)
        for step in range(2):
            idx = slice(step * 4, step * 4 + 4)
            out = model(xb[idx])
            if out.shape != (4, k):
                raise ValueError(f"模型输出形状是 {tuple(out.shape)}，应为 (批大小, {k})")
            loss = torch.nn.functional.cross_entropy(out, yb[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(round(float(loss), 4))
        report["losses"] = losses
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"[:400]
    report["seconds"] = round(time.time() - start, 2)
    report["maxRssMb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)

    # 产出文件与金丝雀检测
    files = []
    canary32 = np.array(CANARY[:8], dtype=np.float32).tobytes()
    canary64 = np.array(CANARY[:8], dtype=np.float64).tobytes()
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        data = p.read_bytes()
        hit = canary32 in data or canary64 in data
        if not hit and p.suffix in (".npy", ".npz", ".pt", ".pth", ".csv", ".txt"):
            try:
                arr = None
                if p.suffix == ".npy":
                    arr = np.load(p, allow_pickle=False)
                elif p.suffix in (".csv", ".txt"):
                    arr = np.loadtxt(p, delimiter="," if p.suffix == ".csv" else None)
                if arr is not None:
                    hit = int(np.isin(np.round(np.asarray(arr, dtype=np.float64), 4), np.round(CANARY, 4)).sum()) >= 16
            except Exception:
                pass
        files.append({"name": str(p.relative_to(out_dir)), "bytes": len(data), "canary": bool(hit)})
        if hit:
            report["canary"]["found"] += 1
            report["canary"]["files"].append(str(p.relative_to(out_dir)))
    report["files"] = files
    emit(report)


# ---------------------------------------------------------------------------
# synth：生成与真实数据同格式的模拟数据（供本地训练的试运行使用）
# ---------------------------------------------------------------------------
def cmd_synth(args):
    import numpy as np
    import torch

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    made = []
    for ds in cfg["datasets"]:
        out = Path(ds["outDir"])
        out.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            name = f"{split}_raw_eeg_data.pt"
            d = torch.load(Path(ds["path"]) / name, map_location="cpu")
            x = d["samples"]
            fake = rng.standard_normal(tuple(x.shape)).astype(np.float32)
            for w in (0, 3, 5):
                if w < fake.shape[0] and fake.shape[2] >= 100 + len(CANARY):
                    fake[w, 0, 100:100 + len(CANARY)] = CANARY
            d["samples"] = torch.from_numpy(fake).to(x.dtype)
            n = len(d["labels"])
            # 可能暴露身份或来源的字段换成占位值
            for key in ("subject_names", "source_files"):
                if isinstance(d.get(key), list):
                    d[key] = [f"模拟/{key}/{i}" for i in range(len(d[key]))]
            torch.save(d, out / name)
            made.append({"dataset": ds["id"], "file": name, "windows": n})
    emit({"ok": True, "files": made})


# ---------------------------------------------------------------------------
# scan：检查产出文件里有没有金丝雀信号或真实数据片段
# ---------------------------------------------------------------------------
def real_patterns(path, per_split=12):
    import numpy as np
    import torch

    pats = []
    for split in SPLITS:
        f = Path(path) / f"{split}_raw_eeg_data.pt"
        if not f.is_file():
            continue
        x = torch.load(f, map_location="cpu")["samples"].numpy()
        idx = np.linspace(0, x.shape[0] - 1, min(per_split, x.shape[0])).astype(int)
        for i in idx:
            seg = x[i, 0, 100:132]
            pats.append(seg.astype(np.float32).tobytes())
            pats.append(seg.astype(np.float64).tobytes())
    return pats


def cmd_scan(args):
    import numpy as np

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    canary = [np.array(CANARY[:8], dtype=np.float32).tobytes(), np.array(CANARY[:8], dtype=np.float64).tobytes()]
    canary_text = [f"{v:g}" for v in CANARY[:8]]
    real = []
    for p in cfg.get("realDatasets", []):
        real.extend(real_patterns(p))
    results = []
    for f in cfg["files"]:
        p = Path(f["abs"])
        size = p.stat().st_size
        hit_canary = hit_real = False
        if size <= 2 * 1024 ** 3:
            data = p.read_bytes()
            hit_canary = any(c in data for c in canary)
            if not hit_canary and size < 64 * 1024 ** 2:
                text = data.decode("utf-8", errors="ignore")
                hit_canary = sum(1 for t in canary_text if t in text) >= 6
            hit_real = any(r in data for r in real)
        results.append({"path": f["rel"], "bytes": size, "canary": hit_canary, "realData": hit_real})
    emit({"ok": True, "files": results})


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def cmd_prepare(args):
    import numpy as np

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["outDir"])
    out.mkdir(parents=True, exist_ok=True)
    factor = int(cfg.get("downsample", 1)) or 1
    per_class = int(cfg.get("perClass", 0) or 0)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    n_classes = len(cfg["classes"])
    shapes = {}
    for split in SPLITS:
        xs, ys = [], []
        for ds in cfg["datasets"]:
            label_map = {int(k): v for k, v in ds["labelMap"].items()}
            if cfg.get("synthetic"):
                counts = ds["splits"][split]["byLabel"]
                xs.append(np.zeros((sum(counts.values()), ds["channels"], ds["points"]), dtype=np.float64))
                ys.append(np.concatenate([np.full(v, label_map[int(lab)]) for lab, v in counts.items()]).astype(np.int64))
                continue
            import torch
            d = torch.load(Path(ds["path"]) / f"{split}_raw_eeg_data.pt", map_location="cpu")
            xs.append(d["samples"].numpy().astype(np.float64))
            ys.append(np.array([label_map[int(v)] for v in d["labels"].tolist()], dtype=np.int64))
        x, y = np.concatenate(xs), np.concatenate(ys)
        if per_class:
            want = per_class if split == "train" else max(1, per_class // 2)
            idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:want] for c in range(n_classes)])
            idx.sort()
            x, y = x[idx], y[idx]
        if factor > 1:
            t = x.shape[2] // factor * factor
            x = x[:, :, :t].reshape(x.shape[0], x.shape[1], -1, factor).mean(axis=3)
        if cfg.get("synthetic"):
            x = rng.standard_normal(x.shape)
        # MP-SPDZ 的卷积按 NHWC 排列：(样本, 通道, 时间)，最后一维的 1 省略
        np.save(out / f"{split}_samples.npy", x)
        np.save(out / f"{split}_labels.npy", y)
        shapes[split] = {"shape": list(x.shape), "classes": [int((y == c).sum()) for c in range(n_classes)]}
    emit({"ok": True, "shapes": shapes})


# ---------------------------------------------------------------------------
# mpc
# ---------------------------------------------------------------------------
MPC_TEMPLATE = r'''# 由脑电数据安全平台生成，请勿手工修改
import importlib.util
import json

import numpy as np
from Compiler import ml
from Compiler.library import print_ln
from Compiler.types import cfix, sfix, sint

CFG = json.load(open(__CONFIG__))
if CFG.get("split"):
    program.use_split(CFG["split"])
if CFG["precision"] == "high":
    sfix.set_precision(24, 48)
    cfix.set_precision(24, 48)

import sys
import torch

if CFG.get("codeRoot"):
    sys.path.insert(0, CFG["codeRoot"])
spec = importlib.util.spec_from_file_location("user_model", CFG["modelFile"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
net = getattr(mod, CFG["modelClass"])(**CFG["modelKwargs"])
net.train()


def load(name):
    return np.load(CFG["dataDir"] + "/" + name + ".npy")


tx, ty = load("train_samples"), load("train_labels")
vx, vy = load("val_samples"), load("val_labels")
sx, sy = load("test_samples"), load("test_labels")
X, Y = sfix.input_tensor_via(0, tx), sint.input_tensor_via(0, ty, one_hot=True)
VX, VY = sfix.input_tensor_via(0, vx), sint.input_tensor_via(0, vy, one_hot=True)
SX, SY = sfix.input_tensor_via(0, sx), sint.input_tensor_via(0, sy, one_hot=True)

ml.set_n_threads(CFG["threads"])
layers = ml.layers_from_torch(net, list(tx.shape), CFG["batch"], program=program)
if CFG["optimizer"] == "sgd":
    optimizer = ml.SGD(layers)
else:
    optimizer = ml.Adam(layers, amsgrad=True, approx=False)
optimizer.set_learning_rate(CFG["lr"])
optimizer.fit(X, Y, epochs=CFG["epochs"], batch_size=CFG["batch"],
              validation_data=(VX, VY), program=program, print_accuracy=True)
# reveal_correctness 不重新计算损失层，返回的损失是上一次验证的值，所以只输出准确率
n_correct, _ = optimizer.reveal_correctness(SX, SY, CFG["batch"])
print_ln("final_test_acc: %s (%s/%s)", cfix(n_correct, k=63, f=31) / len(sy), n_correct, len(sy))
optimizer.reveal_model_to_binary()
'''


def cmd_mpc(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(MPC_TEMPLATE.replace("__CONFIG__", repr(str(Path(args.config).resolve()))), encoding="utf-8")
    emit({"ok": True, "file": str(out)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("meta"); p.add_argument("path"); p.set_defaults(fn=cmd_meta)
    p = sub.add_parser("static"); p.add_argument("file"); p.add_argument("--name"); p.set_defaults(fn=cmd_static)
    p = sub.add_parser("rules"); p.add_argument("dir"); p.set_defaults(fn=cmd_rules)
    p = sub.add_parser("classes"); p.add_argument("dir"); p.set_defaults(fn=cmd_classes)
    p = sub.add_parser("synth"); p.add_argument("config"); p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("scan"); p.add_argument("config"); p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("trial"); p.add_argument("file"); p.add_argument("--workdir", required=True)
    p.add_argument("--channels", type=int, required=True); p.add_argument("--samples", type=int, required=True)
    p.add_argument("--classes", type=int, default=2); p.add_argument("--root"); p.add_argument("--cls")
    p.set_defaults(fn=cmd_trial)
    p = sub.add_parser("prepare"); p.add_argument("config"); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("mpc"); p.add_argument("config"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_mpc)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
