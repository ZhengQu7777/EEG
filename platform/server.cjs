/*
 * 脑电数据安全平台 · 服务端
 *
 * 零依赖 Node 服务：提供页面、数据集登记、训练任务审批，以及两种训练方式：
 *   本地训练：所选数据全是自有数据。程序员提交自己的训练项目和启动命令，
 *             代码审阅（规则 + 大模型）→ 试运行（模拟数据，bwrap 沙箱）→ 本地训练（真实数据，同一沙箱）→ 产出检查
 *   联合训练：用到合作方数据。代码审阅 → 模型试运行 → 模型转换（MP-SPDZ 编译）→ 数据方授权 → 联合训练 → 模型交付
 *
 * 沙箱（bwrap）：断网；整个文件系统只读，只有工作目录和 /tmp 可写；平台目录、~/.ssh 与未选中的数据集被隐藏；
 * 环境变量清空。Python 审计钩子（backend/sandbox/sitecustomize.py）记录联网、子进程和越界写入。
 *
 * 配置（环境变量）：PORT / HOST、EEGSEC_PYTHON、MPSPDZ_ROOT、EEGSEC_DEFAULT_DATASET、EEGSEC_CONDA_ROOT、
 *   EEGSEC_PARTY_MEM_GB、EEGSEC_RUNTIME。大模型配置在 config.local.json（不纳入版本管理）。
 */
"use strict";
const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const net = require("node:net");
const crypto = require("node:crypto");
const { spawn, spawnSync } = require("node:child_process");

const ROOT = __dirname;
const RUNTIME = process.env.EEGSEC_RUNTIME ? path.resolve(process.env.EEGSEC_RUNTIME) : path.join(ROOT, "runtime");
const STATE_DIR = path.join(RUNTIME, "state");
const TASK_ROOT = path.join(RUNTIME, "tasks");
const CERT_DIR = path.join(RUNTIME, "mpspdz", "Player-Data");
const TOOL = path.join(ROOT, "backend", "eegsec.py");
const HOOK_DIR = path.join(ROOT, "backend", "sandbox");
const PY = process.env.EEGSEC_PYTHON || "/home/qz/miniconda3/envs/Stroke/bin/python";
const CONDA = process.env.EEGSEC_CONDA_ROOT || "/home/qz/miniconda3";
const MPSPDZ = process.env.MPSPDZ_ROOT || "/home/qz/MP-SPDZ";
const DEFAULT_DATASET = process.env.EEGSEC_DEFAULT_DATASET || "/mnt/nvme1n1/qz/projects/EEG/data/stroke_hc/all";
const DEFAULT_PROJECT = path.dirname(path.dirname(path.dirname(DEFAULT_DATASET)));
const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 4173);
const PARTY_MEM_KB = Math.round(Number(process.env.EEGSEC_PARTY_MEM_GB || 16) * 1024 * 1024);
const TOOL_MEM_KB = Math.round(Number(process.env.EEGSEC_TOOL_MEM_GB || 24) * 1024 * 1024);
const STATIC_FILES = { "/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css" };
const FLOWS = {
  local: ["review", "trial", "train", "check"],
  mpc: ["review", "compile", "consent", "train", "deliver"]
};
const PROTOCOLS = {
  rep3: { label: "三方 replicated-ring（半诚实，诚实多数）", binary: "replicated-ring-party.x", parties: 3, split: 3 },
  semi2k: { label: "两方 semi2k（半诚实）", binary: "semi2k-party.x", parties: 2, split: 0 }
};
const ADMIN = "数据管理员";
// 快照时跳过的目录（运行产物与依赖）；数据目录不整体跳过，数据文件靠扩展名和大小过滤
const SKIP_DIRS = new Set([".git", "__pycache__", "weights", "runs", "logs", "outputs", "output", "checkpoints", "wandb", "mlruns", "node_modules", ".ipynb_checkpoints", ".venv", "venv", ".idea", ".vscode"]);
const SKIP_EXT = new Set([".pt", ".pth", ".ckpt", ".npy", ".npz", ".mat", ".edf", ".bdf", ".fif", ".h5", ".hdf5", ".pkl", ".zip", ".tar", ".gz", ".xlsx", ".bin", ".so", ".o"]);
const MAX_CODE_FILE = 2 * 1024 * 1024;
const MAX_CODE_TOTAL = 100 * 1024 * 1024;

let CONFIG = {};
try {
  CONFIG = JSON.parse(fs.readFileSync(path.join(ROOT, "config.local.json"), "utf8")) || {};
} catch { CONFIG = {}; }
const LLM = CONFIG.llm || null;

// 账号与角色：user 为使用者（提交任务、查看自己的任务），admin 为管理者（审批、数据管理、审计）。
// 账号在 config.local.json 的 users 中配置；未配置时使用下面两个默认账号。
const ROLES = { user: "使用者", admin: "管理者" };
const ACCOUNTS = (() => {
  const list = (Array.isArray(CONFIG.users) ? CONFIG.users : [])
    .map((u) => ({ id: String(u.id || "").trim(), name: String(u.name || "").trim(), role: u.role === "admin" ? "admin" : "user" }))
    .filter((u) => u.id && u.name);
  return list.length ? list : [{ id: "user", name: "研发人员", role: "user" }, { id: "admin", name: "数据管理员", role: "admin" }];
})();
const sessions = new Map();

// ---------------------------------------------------------------------------
// 状态
// ---------------------------------------------------------------------------
const state = { datasets: [], tasks: [], audit: [], usage: [], calibration: {}, seq: { dataset: 0, task: 0 } };
const live = new Map();
const queues = { local: [], mpc: [] };
const running = { local: null, mpc: null };
let sandboxOk = false;
let mpspdzVersion = "";

const pad = (n, w = 2) => String(n).padStart(w, "0");
const now = () => new Date().toISOString();
const taskDir = (t) => path.join(TASK_ROOT, t.id);
const dsById = (id) => state.datasets.find((d) => d.id === id);
const flowOf = (t) => FLOWS[t.mode];
const stepIndex = (t, key) => flowOf(t).indexOf(key);

async function readJson(file, fallback) {
  try { return JSON.parse(await fsp.readFile(file, "utf8")); } catch { return fallback; }
}
async function writeJson(file, value) {
  const tmp = `${file}.${process.pid}.tmp`;
  await fsp.writeFile(tmp, JSON.stringify(value, null, 1), "utf8");
  await fsp.rename(tmp, file);
}
let saveTimer = null;
function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => writeJson(path.join(STATE_DIR, "state.json"), state).catch((e) => console.error("保存状态失败", e)), 200);
}
function addAudit(module, actor, event, result, level = "") {
  state.audit.unshift({ time: now(), module, actor, event, result, level });
  if (state.audit.length > 3000) state.audit.length = 3000;
  save();
}
function taskLog(task, line) {
  task.logTail = task.logTail || [];
  task.logTail.push(line);
  if (task.logTail.length > 400) task.logTail.splice(0, task.logTail.length - 400);
  fs.appendFile(path.join(taskDir(task), "task.log"), line + "\n", () => {});
}
const moduleOf = (t) => (t.mode === "mpc" ? "联合训练" : "本地训练");

// ---------------------------------------------------------------------------
// 进程
// ---------------------------------------------------------------------------
function limited(memKb, argv, sandbox) {
  const script = `ulimit -v ${memKb}; exec nice -n 10 "$0" "$@"`;
  return sandbox ? ["unshare", ["-rn", "bash", "-c", script, ...argv]] : ["bash", ["-c", script, ...argv]];
}

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { cwd: opts.cwd || ROOT, env: { ...process.env, ...(opts.env || {}) } });
    if (opts.task) (live.get(opts.task.id) || live.set(opts.task.id, { procs: [] }).get(opts.task.id)).procs.push(child);
    let stdout = "", stderr = "", out = "", err = "";
    const timer = opts.timeoutMs ? setTimeout(() => { child.timedOut = true; child.kill("SIGKILL"); }, opts.timeoutMs) : null;
    const pump = (text, which) => {
      if (!opts.onLine) return;
      let buf = (which === "out" ? out : err) + text;
      const lines = buf.split(/\r?\n|\r/);
      buf = lines.pop();
      if (which === "out") out = buf; else err = buf;
      lines.forEach((l) => opts.onLine(l, which));
    };
    child.stdout.on("data", (c) => { const t = c.toString(); if (stdout.length < 4_000_000) stdout += t; pump(t, "out"); });
    child.stderr.on("data", (c) => { const t = c.toString(); if (stderr.length < 2_000_000) stderr += t; pump(t, opts.mergeStderr ? "out" : "err"); });
    child.on("error", (e) => { stderr += String(e); });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (opts.onLine) { if (out) opts.onLine(out, "out"); if (err) opts.onLine(err, "err"); }
      resolve({ code, signal, stdout, stderr, timedOut: Boolean(child.timedOut) });
    });
  });
}

async function tool(args, opts = {}) {
  const [cmd, argv] = limited(TOOL_MEM_KB, [PY, TOOL, ...args], opts.sandbox);
  const res = await run(cmd, argv, opts);
  const last = res.stdout.trim().split("\n").filter(Boolean).pop();
  try { return JSON.parse(last); } catch {
    return { ok: false, error: res.stderr.trim().split("\n").slice(-3).join(" ") || `工具异常退出（${res.code ?? res.signal}）` };
  }
}

function freePort(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.listen(port, "127.0.0.1", () => srv.close(() => resolve(true)));
  });
}
async function findPortBase(n) {
  for (let i = 0; i < 50; i += 1) {
    const base = 20000 + Math.floor(Math.random() * 20000);
    let ok = true;
    for (let k = 0; k < n && ok; k += 1) ok = await freePort(base + k);
    if (ok) return base;
  }
  throw new Error("找不到可用端口");
}

function condaEnvs() {
  const envs = [];
  const add = (name, dir) => { if (fs.existsSync(path.join(dir, "bin", "python"))) envs.push({ name, bin: path.join(dir, "bin") }); };
  try { fs.readdirSync(path.join(CONDA, "envs")).sort().forEach((n) => add(n, path.join(CONDA, "envs", n))); } catch { /* 无 conda */ }
  add("base", CONDA);
  return envs;
}

// ---------------------------------------------------------------------------
// 沙箱
// ---------------------------------------------------------------------------
function sandboxArgs({ workdir, eventsDir, binds, hide, envBin }) {
  const args = ["--die-with-parent", "--unshare-net", "--unshare-pid", "--unshare-ipc", "--new-session", "--clearenv",
    "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"];
  hide.forEach((h) => { if (fs.existsSync(h)) args.push("--tmpfs", h); });
  args.push("--ro-bind", HOOK_DIR, HOOK_DIR, "--bind", workdir, workdir, "--bind", eventsDir, eventsDir);
  binds.forEach((b) => args.push("--ro-bind", b.src, b.dst));
  const envs = {
    PATH: `${envBin}:/usr/local/bin:/usr/bin:/bin`, HOME: "/tmp/home", MPLCONFIGDIR: "/tmp/mpl", XDG_CACHE_HOME: "/tmp/cache",
    LANG: "C.UTF-8", PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1", PYTHONPATH: HOOK_DIR,
    EEGSEC_EVENTS: path.join(eventsDir, "events.jsonl"), EEGSEC_WRITABLE: [workdir, "/tmp", "/dev", eventsDir].join(":")
  };
  Object.entries(envs).forEach(([k, v]) => args.push("--setenv", k, v));
  args.push("--chdir", workdir);
  return args;
}

function hiddenPaths(selectedIds) {
  const home = os.homedir();
  const list = [ROOT, path.join(home, ".ssh"), path.join(home, ".claude"), path.join(home, ".gnupg")];
  state.datasets.filter((d) => !selectedIds.includes(d.id)).forEach((d) => list.push(d.path));
  return list;
}

// ---------------------------------------------------------------------------
// 代码快照
// ---------------------------------------------------------------------------
function platformStorageRoots() {
  return [ROOT, RUNTIME].map((p) => fs.existsSync(p) ? fs.realpathSync(p) : path.resolve(p));
}

function isPlatformStorage(p, roots) {
  const real = fs.existsSync(p) ? fs.realpathSync(p) : path.resolve(p);
  return roots.some((root) => {
    const rel = path.relative(root, real);
    return rel === "" || (rel !== ".." && !rel.startsWith(`..${path.sep}`) && !path.isAbsolute(rel));
  });
}

async function snapshotFromPath(src, dst) {
  const st = await fsp.stat(src).catch(() => null);
  if (!st || !st.isDirectory()) throw new Error("代码目录不存在");
  const excluded = platformStorageRoots();
  if (isPlatformStorage(src, excluded)) throw new Error("平台目录或运行目录不能作为训练项目");
  let total = 0, files = 0;
  // platform/ 可位于训练项目内部；逐文件复制，避免 fs.cp 拒绝复制到源目录的子目录。
  async function copyDirectory(from, to) {
    await fsp.mkdir(to, { recursive: true });
    for (const entry of await fsp.readdir(from, { withFileTypes: true })) {
      const p = path.join(from, entry.name);
      const s = await fsp.lstat(p);
      if (s.isSymbolicLink()) continue;
      if (isPlatformStorage(p, excluded)) continue;
      const target = path.join(to, entry.name);
      if (s.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) await copyDirectory(p, target);
        continue;
      }
      if (!s.isFile() || SKIP_EXT.has(path.extname(entry.name).toLowerCase()) || s.size > MAX_CODE_FILE) continue;
      total += s.size;
      files += 1;
      if (total > MAX_CODE_TOTAL || files > 5000) throw new Error("代码目录过大（超过 100 MB 或 5000 个文件）");
      await fsp.copyFile(p, target);
      await fsp.chmod(target, s.mode & 0o777);
    }
  }
  await copyDirectory(src, dst);
}

async function snapshotFromUpload(files, dst) {
  if (!Array.isArray(files) || !files.length) throw new Error("没有上传代码文件");
  let total = 0;
  const strip = files.length > 1 && files.every((f) => String(f.path).includes("/")) && new Set(files.map((f) => String(f.path).split("/")[0])).size === 1;
  for (const f of files) {
    let rel = path.normalize(String(f.path || "")).replace(/^(\.\.(\/|\\|$))+/, "");
    if (strip) rel = rel.split(path.sep).slice(1).join(path.sep);
    if (!rel || path.isAbsolute(rel) || rel.split(path.sep).some((p) => p === ".." || SKIP_DIRS.has(p))) continue;
    const text = String(f.text ?? "");
    total += text.length;
    if (total > 20 * 1024 * 1024) throw new Error("上传的代码超过 20 MB");
    await fsp.mkdir(path.join(dst, path.dirname(rel)), { recursive: true });
    await fsp.writeFile(path.join(dst, rel), text, "utf8");
  }
}

async function manifest(dir) {
  const out = {};
  async function walk(d) {
    for (const e of await fsp.readdir(d, { withFileTypes: true })) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) await walk(p);
      else if (e.isFile()) {
        const st = await fsp.stat(p);
        out[path.relative(dir, p)] = st.size <= 64 * 1024 * 1024 ? crypto.createHash("sha256").update(await fsp.readFile(p)).digest("hex") : `size:${st.size}:${st.mtimeMs}`;
      }
    }
  }
  await walk(dir);
  return out;
}

async function producedFiles(workdir, base) {
  const after = await manifest(workdir);
  return Object.entries(after).filter(([rel, h]) => base[rel] !== h).map(([rel]) => rel).sort();
}

// ---------------------------------------------------------------------------
// 大模型审阅
// ---------------------------------------------------------------------------
async function llmReview(task, codeDir) {
  if (!LLM || !LLM.apiKey) return { ok: false, error: "未配置大模型（config.local.json）" };
  const parts = [];
  let total = 0;
  const files = Object.keys(await manifest(codeDir)).filter((f) => /\.(py|mpc|sh|ya?ml|json|cfg|toml|ini|txt)$/i.test(f)).sort();
  for (const rel of files) {
    if (total > 900_000) { parts.push(`（其余文件因长度省略：${rel} 等）`); break; }
    const text = (await fsp.readFile(path.join(codeDir, rel), "utf8")).slice(0, 120_000);
    const numbered = text.split("\n").map((l, i) => `${i + 1}: ${l}`).join("\n");
    parts.push(`===== 文件 ${rel} =====\n${numbered}`);
    total += numbered.length;
  }
  const context = task.mode === "local"
    ? `这是一个脑电模型训练项目，运行命令是：${task.local.command}。平台会先用模拟数据、再用真实数据在断网沙箱里运行它，只有工作目录可写。`
    : `这是联合训练用的 MP-SPDZ 程序（${task.mpc.program}），由多家机构的计算方共同执行，各方输入的数据都应保持秘密分享状态。除了常规风险，还要检查它是否把秘密输入或其中间结果揭示（reveal、print_ln 输出、写入输出文件）给不该看到的一方，以及编译阶段的 Python 代码是否读取或外传文件。正常揭示训练准确率、损失和最终模型参数判为 pass。`;
  const system = "你是代码安全审计员，审查科研训练代码。判断代码是否会把训练数据带出（联网上传、写到工作目录以外、把原始数据编码或藏进模型与日志等产出）、窃取凭据、破坏系统或做与训练无关的事（如挖矿）。正常的训练流程（读取数据、训练、把模型/指标/日志/图表保存到工作目录）判为 pass。只输出 JSON：{\"verdict\":\"pass|suspicious|malicious\",\"summary\":\"一句话中文结论\",\"findings\":[{\"file\":\"相对路径\",\"line\":行号,\"level\":\"danger|warn\",\"reason\":\"问题是什么\",\"suggestion\":\"怎么改\"}]}。line 必须是问题代码所在的行号。没有问题时 findings 为空数组。";
  const body = { model: LLM.model || "deepseek-flash", temperature: 0, response_format: { type: "json_object" },
    messages: [{ role: "system", content: system }, { role: "user", content: `${context}\n\n${parts.join("\n\n")}` }] };
  const start = Date.now();
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 240_000);
    const res = await fetch(`${LLM.baseUrl || "https://api.deepseek.com"}/chat/completions`, {
      method: "POST", signal: ctrl.signal,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${LLM.apiKey}` },
      body: JSON.stringify(body)
    });
    clearTimeout(timer);
    const data = await res.json();
    if (!res.ok) return { ok: false, error: (data.error && data.error.message) || `接口返回 ${res.status}` };
    const out = JSON.parse(data.choices[0].message.content);
    const findings = Array.isArray(out.findings) ? out.findings : [];
    // 附上问题代码的原文（前后各一行），方便直接看到位置
    for (const f of findings) {
      const file = path.join(codeDir, String(f.file || ""));
      const line = Number(f.line);
      if (!line || !file.startsWith(codeDir) || !fs.existsSync(file)) continue;
      const lines = (await fsp.readFile(file, "utf8")).split("\n");
      f.snippet = [line - 1, line, line + 1].filter((n) => n >= 1 && n <= lines.length).map((n) => ({ n, text: lines[n - 1].slice(0, 240), hit: n === line }));
    }
    return { ok: true, model: body.model, seconds: Math.round((Date.now() - start) / 100) / 10, files: files.length, tokens: data.usage && data.usage.total_tokens,
      verdict: ["pass", "suspicious", "malicious"].includes(out.verdict) ? out.verdict : "suspicious", summary: out.summary || "", findings };
  } catch (err) {
    return { ok: false, error: err.name === "AbortError" ? "大模型审阅超时" : String(err.message || err) };
  }
}

// ---------------------------------------------------------------------------
// 数据集
// ---------------------------------------------------------------------------
function datasetFields(body, ds) {
  const ownership = body.ownership === "partner" ? "partner" : "own";
  const partner = String(body.partner || "").trim();
  if (ownership === "partner" && !partner) throw new Error("合作方数据需要填写合作方名称");
  return {
    name: String(body.name ?? ds?.name ?? "").trim() || (ds ? ds.name : ""),
    unit: String(body.unit ?? ds?.unit ?? "").trim(),
    owner: String(body.owner ?? ds?.owner ?? "").trim() || ADMIN,
    ownership, partner: ownership === "partner" ? partner : ""
  };
}

async function registerDataset(body, actor = ADMIN) {
  if (!body.path) throw new Error("请填写数据目录");
  const abs = path.resolve(String(body.path).trim());
  if (state.datasets.some((d) => d.path === abs)) throw new Error("该目录已经登记过");
  const meta = await tool(["meta", abs], { timeoutMs: 15 * 60 * 1000 });
  if (!meta.ok) throw new Error(meta.error || "读取数据失败");
  const labels = Object.keys(meta.classes);
  const given = Array.isArray(body.classNames) ? body.classNames.map((s) => String(s).trim()).filter(Boolean) : [];
  if (given.length && given.length !== labels.length) throw new Error(`数据里有 ${labels.length} 个类别，类别名称需要填 ${labels.length} 个`);
  const fields = datasetFields(body);
  state.seq.dataset += 1;
  const ds = { id: `DS-${pad(state.seq.dataset)}`, ...fields, name: fields.name || meta.tag || path.basename(path.dirname(abs)), path: abs,
    classNames: given.length ? given : labels.map((l) => `类别 ${l}`), registeredAt: now(), meta };
  state.datasets.push(ds);
  addAudit("数据", actor, `登记数据集 ${ds.id}（${ds.name}，${ds.ownership === "own" ? "自有" : `合作方：${ds.partner}`}，${meta.fingerprintShort}）`, "已登记", "ok");
  return ds;
}

function updateDataset(ds, body, actor = ADMIN) {
  const before = ds.ownership === "own" ? "自有" : `合作方：${ds.partner}`;
  Object.assign(ds, datasetFields({ ...ds, ...body }, ds));
  if (Array.isArray(body.classNames) && body.classNames.length === Object.keys(ds.meta.classes).length) ds.classNames = body.classNames.map((s) => String(s).trim());
  const after = ds.ownership === "own" ? "自有" : `合作方：${ds.partner}`;
  addAudit("数据", actor, `修改数据集 ${ds.id}${before !== after ? `（归属：${before} → ${after}）` : ""}`, "已修改", "ok");
  save();
  return ds;
}

// ---------------------------------------------------------------------------
// 项目：代码和数据放在同一个项目目录里，项目根目录取数据集路径中 data/ 的上一级
// ---------------------------------------------------------------------------
const SAMPLE_DIR = path.join(ROOT, "samples");
function projectRoots() {
  const roots = new Set();
  state.datasets.filter((d) => d.ownership !== "partner").forEach((d) => {
    const i = d.path.lastIndexOf(`${path.sep}data${path.sep}`);
    if (i > 0) roots.add(d.path.slice(0, i));
  });
  return [...roots].sort();
}
async function listProjects() {
  const out = [];
  const excluded = platformStorageRoots();
  for (const root of projectRoots()) {
    if (isPlatformStorage(root, excluded)) continue;
    const entries = [], mpc = [];
    let latest = 0;
    const walk = async (dir, depth) => {
      for (const e of await fsp.readdir(dir, { withFileTypes: true }).catch(() => [])) {
        const p = path.join(dir, e.name);
        if (e.isDirectory()) { if (depth < 3 && !isPlatformStorage(p, excluded) && !SKIP_DIRS.has(e.name) && e.name !== "data" && !e.name.startsWith(".")) await walk(p, depth + 1); continue; }
        if (!e.isFile()) continue;
        if (e.name.endsWith(".py")) { latest = Math.max(latest, (await fsp.stat(p)).mtimeMs); if (depth === 0) entries.push(e.name); }
        if (e.name.endsWith(".mpc")) mpc.push(path.relative(root, p));
      }
    };
    await walk(root, 0);
    out.push({ name: path.basename(root), path: root, modified: latest ? new Date(latest).toISOString() : null, entries: entries.sort(), mpc: mpc.sort(),
      datasets: state.datasets.filter((d) => d.path.startsWith(root + path.sep)).map((d) => d.id) });
  }
  const samples = (await fsp.readdir(SAMPLE_DIR).catch(() => [])).filter((f) => f.endsWith(".mpc"));
  return { projects: out, samples };
}
function resolveProgram(ref) {
  if (ref.startsWith("sample:")) {
    const f = path.join(SAMPLE_DIR, path.basename(ref.slice(7)));
    if (!fs.existsSync(f)) throw new Error("示例程序不存在");
    return f;
  }
  const f = path.resolve(ref);
  if (!f.endsWith(".mpc") || !projectRoots().some((r) => f.startsWith(r + path.sep)) || !fs.existsSync(f)) throw new Error("只能选择项目目录里的 .mpc 文件");
  return f;
}

// ---------------------------------------------------------------------------
// 任务创建
// ---------------------------------------------------------------------------
function intIn(v, lo, hi, dflt) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, Math.round(n))) : dflt;
}

function checkDatasets(ids) {
  const sets = (Array.isArray(ids) ? ids : []).map(dsById).filter(Boolean);
  if (!sets.length) throw new Error("请至少选择一份数据");
  const classes = [];
  sets.forEach((d) => d.classNames.forEach((c) => { if (!classes.includes(c)) classes.push(c); }));
  if (classes.length < 2) throw new Error("所选数据只有一个类别，无法训练分类模型");
  if (new Set(sets.map((d) => d.meta.channels)).size > 1) throw new Error("所选数据的通道数不一致");
  if (new Set(sets.map((d) => d.meta.sfreq)).size > 1) throw new Error("所选数据的采样率不一致，需要先在数据方本地统一");
  if (new Set(sets.map((d) => d.meta.points)).size > 1) throw new Error("所选数据的窗口长度不一致");
  return { sets, classes, mode: sets.some((d) => d.ownership === "partner") ? "mpc" : "local" };
}

// 联合训练的平台参数：协议、环大小、数据抽样与降采样；训练轮数等写在 .mpc 程序里
function mpcParams(p = {}) {
  return {
    protocol: PROTOCOLS[p.protocol] ? p.protocol : "rep3",
    precision: p.precision === "high" ? "high" : "standard",
    downsample: [1, 2, 4, 8].includes(Number(p.downsample)) ? Number(p.downsample) : 4,
    perClass: intIn(p.perClass, 0, 100000, 32)
  };
}

async function createTask(body, user) {
  const { sets, classes, mode: derived } = checkDatasets(body.datasets);
  const mode = body.mode === "mpc" || body.mode === "local" ? body.mode : derived;
  if (mode === "local" && derived === "mpc") throw new Error("本地训练只能使用自有数据");
  if (mode === "mpc" && derived === "local") throw new Error("联合训练至少要选择一个合作方的数据集");
  let code = body.code || {};
  const task = {
    id: "", mode,
    title: String(body.title || "").trim(),
    requester: user.name, requesterId: user.id,
    purpose: String(body.purpose || "").trim(),
    createdAt: now(), datasets: sets.map((d) => d.id), classes,
    channels: sets[0].meta.channels, points: sets[0].meta.points,
    step: "review", state: "running", steps: {}, logTail: []
  };
  if (mode === "local") {
    const command = String((body.local || {}).command || "").trim();
    if (!command) throw new Error("请填写运行命令");
    const env = condaEnvs().find((e) => e.name === (body.local || {}).env) || condaEnvs().find((e) => e.name === "Stroke") || condaEnvs()[0];
    if (!env) throw new Error("找不到可用的 Python 环境");
    task.local = { command, env: env.name, envBin: env.bin, trialSeconds: intIn((body.local || {}).trialSeconds, 30, 1800, 120) };
    if (!task.title) {
      const first = code.source === "upload" && code.files && code.files[0] ? String(code.files[0].path).split("/")[0] : path.basename(String(code.path || ""));
      task.title = `${first || "训练代码"} · ${sets.map((d) => d.name).join("、")}`;
    }
  } else {
    const prog = { ...((body.mpc || {}).program || {}) };
    if ((body.mpc || {}).programPath) {
      const file = resolveProgram(String(body.mpc.programPath));
      prog.name = path.basename(file);
      prog.text = await fsp.readFile(file, "utf8");
    }
    const name = path.basename(String(prog.name || "")).replace(/[^\w.\-]/g, "_");
    if (!name.endsWith(".mpc") || !String(prog.text || "").trim()) throw new Error("请上传 MPC 程序（.mpc 文件）");
    const m = String(prog.text).match(/\b(?:EPOCHS|epochs)\s*=\s*(\d+)/);
    task.mpc = { program: name, epochs: m ? Number(m[1]) : null, params: mpcParams((body.mpc || {}).params) };
    code = { source: "upload", files: [{ path: name, text: String(prog.text) }, ...((body.code && body.code.files) || [])] };
    if (!task.title) task.title = `${name} · ${[...new Set(sets.filter((d) => d.ownership === "partner").map((d) => d.partner))].join("、")}`;
  }
  state.seq.task += 1;
  task.id = `TASK-${pad(state.seq.task, 3)}`;
  const codeDir = path.join(taskDir(task), "code");
  await fsp.mkdir(codeDir, { recursive: true });
  try {
    if (code.source === "upload") await snapshotFromUpload(code.files, codeDir);
    else await snapshotFromPath(path.resolve(String(code.path || "")), codeDir);
  } catch (err) {
    state.seq.task -= 1;
    await fsp.rm(taskDir(task), { recursive: true, force: true });
    throw err;
  }
  const mf = await manifest(codeDir);
  const files = Object.keys(mf);
  if (mode === "local" && !files.some((f) => f.endsWith(".py"))) {
    state.seq.task -= 1;
    await fsp.rm(taskDir(task), { recursive: true, force: true });
    throw new Error("代码里没有 .py 文件");
  }
  task.code = {
    source: code.source === "upload" ? "upload" : "path",
    root: code.source === "upload" ? "" : path.resolve(String(code.path)),
    files: files.length,
    sha256: crypto.createHash("sha256").update(JSON.stringify(mf)).digest("hex")
  };
  task.code.short = `sha256:${task.code.sha256.slice(0, 8)}…${task.code.sha256.slice(-6)}`;
  await writeJson(path.join(taskDir(task), "code-manifest.json"), mf);
  state.tasks.unshift(task);
  addAudit(moduleOf(task), task.requester, `提交 ${task.id}（${task.title}），${files.length} 个代码文件，使用 ${task.datasets.join("、")}`, "已提交");
  save();
  pipeline(task);
  return task;
}

// ---------------------------------------------------------------------------
// 流程
// ---------------------------------------------------------------------------
const stopped = (t) => ["blocked", "needs-fix", "failed", "cancelled", "returned", "rejected"].includes(t.state);
function setStep(t, step, st = "running") { t.step = step; t.state = st; save(); }
function stopTask(t, st, reason, actor, level) {
  t.state = st;
  t.reason = reason;
  addAudit(moduleOf(t), actor, `${t.id} ${reason}`, { blocked: "已拦截", "needs-fix": "需修改", failed: "出错" }[st] || st, level);
}

async function pipeline(task) {
  try {
    await stepReview(task);
    if (stopped(task) || task.state === "waiting") return;
    await afterReview(task);
  } catch (err) {
    stopTask(task, "failed", `处理出错：${err.message}`, "平台", "danger");
  }
}

async function afterReview(task) {
  if (task.mode === "local") {
    await stepLocalTrial(task); if (stopped(task)) return;
    setStep(task, "train", "queued");
    queues.local.push(task.id);
    pump("local");
  } else {
    await stepCompile(task); if (stopped(task)) return;
    setStep(task, "consent", "waiting");
    addAudit("联合训练", "平台", `${task.id} 通过代码审阅和编译检查，等待合作方授权`, "待授权", "warn");
  }
}

async function stepReview(task) {
  setStep(task, "review");
  const codeDir = path.join(taskDir(task), "code");
  const llm = await llmReview(task, codeDir);
  task.steps.review = { llm, at: now() };
  if (llm.ok) addAudit(moduleOf(task), `大模型（${llm.model}）`, `${task.id} 代码审阅：${llm.summary}`, { pass: "通过", suspicious: "可疑", malicious: "恶意" }[llm.verdict], { pass: "ok", suspicious: "warn", malicious: "danger" }[llm.verdict]);
  if (llm.ok && llm.verdict === "malicious") {
    stopTask(task, "blocked", `代码审阅判定为恶意：${llm.summary}`, "代码审阅", "danger");
  } else if (!llm.ok || llm.verdict === "suspicious") {
    task.state = "waiting";
    task.reason = !llm.ok ? `大模型审阅不可用（${llm.error}），需要人工确认` : "大模型认为可疑，需要人工确认";
    addAudit(moduleOf(task), "代码审阅", `${task.id} ${task.reason}`, "待确认", "warn");
  } else {
    addAudit(moduleOf(task), "代码审阅", `${task.id} 大模型审阅 ${llm.files} 个文件，未发现问题`, "通过", "ok");
  }
  save();
}

async function confirmReview(task, decision, actor = ADMIN) {
  if (task.step !== "review" || task.state !== "waiting") throw new Error("任务当前不需要人工确认");
  if (decision === "block") {
    stopTask(task, "blocked", "管理员人工确认后拦截", actor, "danger");
    save();
    return;
  }
  task.reason = "";
  task.state = "running";
  task.steps.review.confirmedBy = { by: actor, at: now() };
  addAudit(moduleOf(task), actor, `${task.id} 人工确认代码无问题，继续`, "已确认", "ok");
  save();
  afterReview(task).catch((err) => stopTask(task, "failed", `处理出错：${err.message}`, "平台", "danger"));
}

// ---- 本地训练：试运行、训练、产出检查 ----
async function prepareWork(task, name) {
  const base = path.join(taskDir(task), name);
  const work = path.join(base, "work");
  const events = path.join(base, "events");
  await fsp.rm(base, { recursive: true, force: true });
  await fsp.mkdir(events, { recursive: true });
  await fsp.cp(path.join(taskDir(task), "code"), work, { recursive: true });
  return { base, work, events };
}

function dataBinds(task, work, sourceFor) {
  const binds = [];
  task.datasets.forEach((id) => {
    const ds = dsById(id);
    const src = sourceFor(ds);
    binds.push({ src, dst: ds.path });
    // 再挂到工作目录里项目惯用的相对位置：服务器目录按项目根计算；上传的代码按路径中的 data/ 段推断
    let rel = "";
    if (task.code.root && ds.path.startsWith(task.code.root + path.sep)) rel = path.relative(task.code.root, ds.path);
    else {
      const i = ds.path.lastIndexOf(`${path.sep}data${path.sep}`);
      if (i >= 0) rel = ds.path.slice(i + 1);
    }
    if (rel) {
      const inner = path.join(work, rel);
      fs.mkdirSync(inner, { recursive: true });
      binds.push({ src, dst: inner });
    }
  });
  return binds;
}

async function readEvents(eventsDir) {
  const raw = await fsp.readFile(path.join(eventsDir, "events.jsonl"), "utf8").catch(() => "");
  const ev = { network: [], process: [], writes: [] };
  raw.split("\n").filter(Boolean).forEach((l) => {
    try {
      const e = JSON.parse(l);
      const key = e.kind === "network" ? "network" : e.kind === "process" ? "process" : "writes";
      if (ev[key].length < 50) ev[key].push(e.detail);
    } catch { /* 忽略损坏的行 */ }
  });
  return ev;
}

async function runSandboxed(task, { work, events, binds, timeoutMs, label }) {
  const args = sandboxArgs({ workdir: work, eventsDir: events, binds, hide: hiddenPaths(task.datasets), envBin: task.local.envBin });
  taskLog(task, `[${now()}] 平台：${label}，命令：${task.local.command}`);
  const start = Date.now();
  let usage = null;
  const res = await run("/usr/bin/bwrap", [...args, "/usr/bin/time", "-f", "EEGSEC_USAGE %e %M", "/bin/bash", "-c", task.local.command], {
    task, timeoutMs, mergeStderr: true,
    onLine: (line) => {
      if (!line.trim()) return;
      const u = line.match(/^EEGSEC_USAGE ([\d.]+) (\d+)$/);
      if (u) { usage = { seconds: Number(u[1]), maxRssMb: Math.round(Number(u[2]) / 1024) }; return; }
      taskLog(task, line);
      const tr = task.step === "train" ? task.steps.train : null;
      if (!tr) return;
      const m = line.match(/epoch\s*[:：]?\s*\[?\s*(\d+)\s*\/\s*(\d+)/i);
      if (m) tr.progress = { epoch: Number(m[1]), total: Number(m[2]) };
      // 常见的训练日志格式：Total_loss : x、TRAIN: Acc:x、VAL(Patch): Acc:x、val_acc=x
      if (!tr.progress) return;
      const val = (re) => { const x = line.match(re); if (!x) return null; const v = Number(x[1]); return v > 1.5 ? v / 100 : v; };
      const loss = (line.match(/(?:^|[^a-z_])(?:total_)?loss\s*[:=]\s*([\d.]+)/i) || [])[1];
      const trainAcc = val(/train[^:\n]{0,12}:?\s*acc\s*[:=]\s*([\d.]+)/i);
      const valAcc = val(/val[^:\n]{0,12}:?\s*acc\s*[:=]\s*([\d.]+)/i);
      if (loss == null && trainAcc == null && valAcc == null) return;
      tr.history = tr.history || [];
      let h = tr.history.find((x) => x.epoch === tr.progress.epoch);
      if (!h) { h = { epoch: tr.progress.epoch }; tr.history.push(h); }
      if (loss != null && h.loss == null) h.loss = Number(loss);
      if (trainAcc != null) h.trainAcc = trainAcc;
      if (valAcc != null) h.valAcc = valAcc;
    }
  });
  return { ...res, seconds: Math.round((Date.now() - start) / 1000), maxRssMb: usage ? usage.maxRssMb : null };
}

async function scanFiles(task, work, rels, realDatasets) {
  if (!rels.length) return [];
  const cfgFile = path.join(path.dirname(work), "scan.json");
  await writeJson(cfgFile, { files: rels.map((rel) => ({ rel, abs: path.join(work, rel) })), realDatasets });
  const res = await tool(["scan", cfgFile], { timeoutMs: 30 * 60 * 1000 });
  if (!res.ok) throw new Error(res.error || "产出扫描失败");
  return res.files;
}

async function stepLocalTrial(task) {
  setStep(task, "trial");
  const { base, work, events } = await prepareWork(task, "trial");
  const synthDir = path.join(base, "synthetic");
  const synthCfg = path.join(base, "synth.json");
  await writeJson(synthCfg, { datasets: task.datasets.map((id) => ({ id, path: dsById(id).path, outDir: path.join(synthDir, id) })) });
  const synth = await tool(["synth", synthCfg], { timeoutMs: 30 * 60 * 1000 });
  if (!synth.ok) throw new Error(synth.error || "生成模拟数据失败");
  const binds = dataBinds(task, work, (ds) => path.join(synthDir, ds.id));
  const res = await runSandboxed(task, { work, events, binds, timeoutMs: task.local.trialSeconds * 1000, label: `试运行（模拟数据，限时 ${task.local.trialSeconds} 秒）` });
  const ev = await readEvents(events);
  const base0 = await readJson(path.join(taskDir(task), "code-manifest.json"), {});
  const produced = await producedFiles(work, base0);
  const scanned = await scanFiles(task, work, produced, []);
  const canary = scanned.filter((f) => f.canary).map((f) => f.path);
  task.steps.trial = {
    at: now(), seconds: res.seconds, maxRssMb: res.maxRssMb, exitCode: res.code, timedOut: res.timedOut, events: ev, files: scanned, canary,
    sandbox: "bwrap：断网、只读文件系统、只有工作目录可写、隐藏平台目录与未选数据", logTail: (task.logTail || []).slice(-40)
  };
  const leaks = [];
  if (ev.network.length) leaks.push(`尝试联网 ${ev.network.length} 次（${ev.network[0]}）`);
  if (ev.writes.length) leaks.push(`尝试写工作目录以外的文件 ${ev.writes.length} 次（${ev.writes[0]}）`);
  if (canary.length) leaks.push(`产出文件中检出金丝雀信号（${canary.join("、")}）`);
  if (leaks.length) return stopTask(task, "blocked", `试运行发现外传行为：${leaks.join("；")}`, "试运行沙箱", "danger");
  if (!res.timedOut && res.code !== 0) return stopTask(task, "needs-fix", `试运行异常退出（退出码 ${res.code}），请查看日志`, "试运行沙箱", "warn");
  addAudit("本地训练", "试运行沙箱", `${task.id} 模拟数据试运行${res.timedOut ? `持续 ${task.local.trialSeconds} 秒无异常，已按时限停止` : `正常结束（${res.seconds} 秒）`}，产出 ${produced.length} 个文件`, "通过", "ok");
  save();
}

async function trainLocal(task) {
  setStep(task, "train");
  task.steps.train = { startedAt: now(), phase: "沙箱训练" };
  const { work, events } = await prepareWork(task, "run");
  const binds = dataBinds(task, work, (ds) => ds.path);
  task.datasets.forEach((id) => {
    const ds = dsById(id);
    state.usage.unshift({ time: now(), task: task.id, dataset: id, fingerprint: ds.meta.fingerprintShort, how: "沙箱内只读挂载，本地训练", result: "训练中" });
  });
  addAudit("本地训练", "平台", `${task.id} 在沙箱中开始训练，只读挂载 ${task.datasets.join("、")}`, "已启动", "ok");
  save();
  const res = await runSandboxed(task, { work, events, binds, timeoutMs: 48 * 3600 * 1000, label: "本地训练（真实数据）" });
  const tr = task.steps.train;
  tr.finishedAt = now();
  tr.seconds = res.seconds;
  tr.maxRssMb = res.maxRssMb;
  tr.exitCode = res.code;
  tr.events = await readEvents(events);
  if (task.state === "cancelled") return;
  state.usage.filter((u) => u.task === task.id).forEach((u) => { u.result = res.code === 0 ? "训练完成" : "训练失败"; });
  if (res.code !== 0) return stopTask(task, "failed", `训练异常退出（退出码 ${res.code ?? res.signal}），请查看日志`, "本地训练", "danger");

  setStep(task, "check");
  const base0 = await readJson(path.join(taskDir(task), "code-manifest.json"), {});
  const produced = await producedFiles(work, base0);
  const scanned = await scanFiles(task, work, produced, task.datasets.map((id) => dsById(id).path));
  const hits = scanned.filter((f) => f.realData).map((f) => f.path);
  const ev = tr.events;
  task.steps.check = { at: now(), files: scanned, hits };
  if (hits.length || ev.network.length || ev.writes.length) {
    const why = [hits.length ? `产出中检出原始数据片段（${hits.join("、")}）` : "", ev.network.length ? `训练中尝试联网 ${ev.network.length} 次` : "", ev.writes.length ? `训练中尝试写工作目录以外的文件 ${ev.writes.length} 次` : ""].filter(Boolean).join("；");
    return stopTask(task, "blocked", `产出检查未通过：${why}，产出不予放行`, "产出检查", "danger");
  }
  task.state = "done";
  addAudit("本地训练", "产出检查", `${task.id} 训练完成（${res.seconds} 秒），${produced.length} 个产出文件未检出原始数据，可以下载`, "已放行", "ok");
}

// ---- 联合训练：编译检查、授权、训练、交付（程序由需求方上传的 .mpc 文件提供） ----
function prepareConfig(task, outDir, synthetic) {
  return {
    outDir, synthetic, seed: 0, downsample: task.mpc.params.downsample, perClass: task.mpc.params.perClass, classes: task.classes,
    datasets: task.datasets.map((id) => {
      const ds = dsById(id);
      const labelMap = {};
      ds.classNames.forEach((name, lab) => { labelMap[lab] = task.classes.indexOf(name); });
      return { path: ds.path, labelMap, channels: ds.meta.channels, points: ds.meta.points, splits: ds.meta.splits };
    })
  };
}

function parseCost(text) {
  const cost = {};
  const block = text.split("Program requires at most:")[1];
  if (!block) return cost;
  const names = { "integer triples": "triples", "integer bits": "bits", "integer opens": "opens", "virtual machine rounds": "rounds" };
  block.split("\n").forEach((line) => {
    const m = line.match(/^\s*(\d+)\s+(.+?)\s*$/);
    if (m && names[m[2]]) cost[names[m[2]]] = Number(m[1]);
  });
  return cost;
}
function compileError(res) {
  const lines = (res.stderr + "\n" + res.stdout).split("\n").map((l) => l.trim()).filter(Boolean);
  const err = [...lines].reverse().find((l) => /Error|error:/.test(l)) || lines.pop() || "编译失败";
  const unknown = err.match(/unknown PyTorch module: (.+)$/);
  return unknown ? `MP-SPDZ 不支持的层：${unknown[1]}` : err.replace(/^Compiler\.exceptions\.CompilerError: /, "").slice(0, 300);
}

// 编译需求方上传的 .mpc：数据放在工作目录的 data/（也通过 EEGSEC_DATA_DIR 告知），编译在断网环境中进行
async function compileProgram(task, dir) {
  await fsp.mkdir(path.join(dir, "Programs", "Source"), { recursive: true });
  await fsp.copyFile(path.join(taskDir(task), "code", task.mpc.program), path.join(dir, "Programs", "Source", "train.mpc"));
  const ring = task.mpc.params.precision === "high" ? "128" : "64";
  const [cmd, argv] = limited(TOOL_MEM_KB, [PY, path.join(MPSPDZ, "compile.py"), "-R", ring, "Programs/Source/train.mpc"], sandboxOk);
  const start = Date.now();
  const res = await run(cmd, argv, { cwd: dir, timeoutMs: 60 * 60 * 1000, task,
    env: { EEGSEC_DATA_DIR: path.join(dir, "data"), EEGSEC_PARTIES: String(PROTOCOLS[task.mpc.params.protocol].parties) } });
  return { res, seconds: Math.round((Date.now() - start) / 1000) };
}

async function stepCompile(task) {
  setStep(task, "compile");
  const dir = path.join(taskDir(task), "compile");
  await fsp.mkdir(dir, { recursive: true });
  await writeJson(path.join(dir, "prepare.json"), prepareConfig(task, path.join(dir, "data"), true));
  const prep = await tool(["prepare", path.join(dir, "prepare.json")], { timeoutMs: 30 * 60 * 1000 });
  if (!prep.ok) throw new Error(prep.error || "生成模拟数据失败");
  const { res, seconds } = await compileProgram(task, dir);
  task.steps.compile = { at: now(), seconds, shapes: prep.shapes };
  if (res.code !== 0) {
    task.steps.compile.error = compileError(res);
    return stopTask(task, "needs-fix", `MP-SPDZ 编译失败：${task.steps.compile.error}`, "编译检查", "warn");
  }
  task.steps.compile.cost = parseCost(res.stdout);
  addAudit("联合训练", "编译检查", `${task.id} ${task.mpc.program} 用同形状模拟数据编译通过（${seconds} 秒）`, "通过", "ok");
  save();
}

function consent(task, dsId, decision, actor = ADMIN) {
  if (task.step !== "consent" || task.state !== "waiting") throw new Error("任务当前不在授权步骤");
  if (!task.datasets.includes(dsId)) throw new Error("该任务没有使用这份数据");
  const ds = dsById(dsId);
  const by = ds.ownership === "partner" ? `${ds.partner}（${ds.owner}）` : ds.owner;
  task.steps.consent = task.steps.consent || { decisions: {} };
  task.steps.consent.decisions[dsId] = { decision, by, operator: actor, at: now() };
  if (decision === "reject") {
    stopTask(task, "rejected", `${by} 拒绝 ${dsId} 用于该任务（${actor} 登记）`, by, "danger");
  } else {
    addAudit("联合训练", by, `同意 ${task.id} 使用 ${dsId}（${actor} 登记）`, "已授权", "ok");
    if (task.datasets.every((id) => (task.steps.consent.decisions[id] || {}).decision === "approve")) {
      setStep(task, "train", "queued");
      addAudit("联合训练", "平台", `${task.id} 全部数据方已授权，进入训练队列`, "排队中");
      queues.mpc.push(task.id);
      pump("mpc");
    }
  }
  save();
}

function parseMpcLine(task, line) {
  const tr = task.steps.train;
  let m;
  if ((m = line.match(/done with epoch (\d+)/))) {
    const e = Number(m[1]) + 1;
    if (!tr.history.some((h) => h.epoch === e)) tr.history.push({ epoch: e });
    tr.epoch = Math.max(tr.epoch, e);
  }
  const last = tr.history[tr.history.length - 1];
  if ((m = line.match(/final_test_acc: ([-\d.e]+) \((\d+)\/(\d+)\)/))) tr.test = { acc: Number(m[1]), correct: Number(m[2]), total: Number(m[3]) };
  else if ((m = line.match(/train_acc: ([-\d.e]+) \((\d+)\/(\d+)\)/)) && last) last.trainAcc = Number(m[1]);
  else if ((m = line.match(/(?:^|[^_\w])acc: ([-\d.e]+) \((\d+)\/(\d+)\)/)) && last) last.valAcc = Number(m[1]);
  if ((m = line.match(/(?:^|[^_\w])test loss: ([-\d.e]+)/)) && last) last.valLoss = Number(m[1]);
  if ((m = line.match(/^Time = ([\d.]+) seconds/))) tr.mpcSeconds = Number(m[1]);
  if ((m = line.match(/^Data sent = ([\d.]+) MB in ~(\d+) rounds/))) { tr.dataSentMb = Number(m[1]); tr.rounds = Number(m[2]); }
  if ((m = line.match(/^Global data sent = ([\d.]+) MB/))) tr.globalDataMb = Number(m[1]);
}

async function trainMpc(task) {
  const p = task.mpc.params;
  const proto = PROTOCOLS[p.protocol];
  const dir = path.join(taskDir(task), "mpc");
  const dataDir = path.join(dir, "data");
  setStep(task, "train");
  task.steps.train = { epoch: 0, total: task.mpc.epochs, history: [], startedAt: now(), phase: "准备数据" };
  await fsp.mkdir(dir, { recursive: true });
  await writeJson(path.join(dir, "prepare.json"), prepareConfig(task, dataDir, false));
  const prep = await tool(["prepare", path.join(dir, "prepare.json")], { timeoutMs: 60 * 60 * 1000, task });
  if (!prep.ok) throw new Error(prep.error || "准备数据失败");
  task.steps.train.shapes = prep.shapes;
  task.datasets.forEach((id) => {
    const ds = dsById(id);
    state.usage.unshift({ time: now(), task: task.id, dataset: id, fingerprint: ds.meta.fingerprintShort, how: "本地准备后由计算方 P0 秘密分享输入", result: "训练中" });
    addAudit("数据", ds.owner, `${task.id} 读取 ${id}（${ds.meta.fingerprintShort}）并准备秘密分享输入`, "已读取", "ok");
  });
  task.steps.train.phase = "编译 MPC 程序";
  save();
  const { res, seconds } = await compileProgram(task, dir);
  if (res.code !== 0) throw new Error(`MP-SPDZ 编译失败：${compileError(res)}`);
  task.steps.train.compileSeconds = seconds;
  const pd = path.join(dir, "Player-Data");
  await fsp.mkdir(pd, { recursive: true });
  ensureCerts(proto.parties);
  for (const f of await fsp.readdir(CERT_DIR)) if (/^P\d+\.(pem|key)$/.test(f) || /\.0$/.test(f)) await fsp.copyFile(path.join(CERT_DIR, f), path.join(pd, f));
  const base = await findPortBase(proto.parties);
  Object.assign(task.steps.train, { phase: "安全训练", port: base, parties: [] });
  save();
  addAudit("联合训练", `P0–P${proto.parties - 1}`, `${task.id} 启动 ${proto.binary}（端口 ${base} 起）`, "已启动", "ok");
  const env = { LD_LIBRARY_PATH: `${MPSPDZ}:${process.env.LD_LIBRARY_PATH || ""}` };
  const runs = [];
  for (let i = 0; i < proto.parties; i += 1) {
    const argv = [path.join(MPSPDZ, proto.binary), "-p", String(i), "-pn", String(base), "-h", "localhost"];
    if (proto.parties === 2) argv.push("-N", "2");
    argv.push("train");
    const [cmd, args] = limited(PARTY_MEM_KB, argv, false);
    const party = { name: `P${i}`, status: "运行中" };
    task.steps.train.parties.push(party);
    const logFile = fs.createWriteStream(path.join(dir, `party-P${i}.log`));
    runs.push(run(cmd, args, {
      cwd: dir, env, task,
      onLine: (line, which) => {
        logFile.write(line + "\n");
        if (i === 0 && which === "out" && line.trim()) { taskLog(task, line); parseMpcLine(task, line); save(); }
      }
    }).then((r) => {
      logFile.end();
      party.status = r.code === 0 ? "已结束" : `异常退出（${r.code ?? r.signal}）`;
      party.stderr = r.stderr.trim().split("\n").slice(-3).join(" ").slice(0, 300);
      return r;
    }));
  }
  const results = await Promise.all(runs);
  const tr = task.steps.train;
  results[0].stderr.split("\n").map((l) => l.trim()).filter((l) => /^(Time|Data sent|Global data sent) =/.test(l)).forEach((l) => { taskLog(task, l); parseMpcLine(task, l); });
  tr.finishedAt = now();
  tr.seconds = Math.round((Date.parse(tr.finishedAt) - Date.parse(tr.startedAt)) / 1000);
  if (task.state === "cancelled") return;
  const bad = results.findIndex((r) => r.code !== 0);
  if (bad >= 0) {
    const r = results[bad];
    throw new Error(`计算方 P${bad} 异常：${(r.stderr.trim().split("\n").slice(-2).join(" ") || `退出码 ${r.code ?? r.signal}`).slice(0, 300)}${r.signal === "SIGKILL" ? "（可能超出内存上限）" : ""}`);
  }
  state.usage.filter((u) => u.task === task.id).forEach((u) => { u.result = "训练完成"; });
  setStep(task, "deliver", "waiting");
  addAudit("联合训练", `P0–P${proto.parties - 1}`, `${task.id} 训练结束，测试准确率 ${tr.test ? (tr.test.acc * 100).toFixed(1) + "%" : "—"}，用时 ${tr.seconds} 秒`, "待交付审批", "warn");
}

async function deliver(task, actor = ADMIN) {
  if (task.step !== "deliver" || task.state !== "waiting") throw new Error("任务当前不在交付步骤");
  const src = path.join(taskDir(task), "mpc", "Player-Data", "Binary-Output-P0-0");
  const dst = path.join(taskDir(task), "model_params.bin");
  await fsp.copyFile(src, dst);
  const st = await fsp.stat(dst);
  task.steps.deliver = { at: now(), by: actor, file: "model_params.bin", bytes: st.size };
  task.state = "done";
  addAudit("联合训练", actor, `批准 ${task.id} 的模型交付给 ${task.requester}（${(st.size / 1024).toFixed(1)} KB）`, "已交付", "ok");
  save();
}

// 计算方证书：直接调用 openssl（MP-SPDZ 的 setup-ssl.sh 不支持带空格的路径）
function ensureCerts(n) {
  fs.mkdirSync(CERT_DIR, { recursive: true });
  let made = false;
  for (let i = 0; i < n; i += 1) {
    const pem = path.join(CERT_DIR, `P${i}.pem`), key = path.join(CERT_DIR, `P${i}.key`);
    if (fs.existsSync(pem) && fs.existsSync(key)) continue;
    const r = spawnSync("openssl", ["req", "-newkey", "rsa:2048", "-nodes", "-x509", "-days", "3650", "-out", pem, "-keyout", key, "-subj", `/CN=P${i}`]);
    if (r.status !== 0) throw new Error(`生成计算方证书失败：${String(r.stderr).slice(0, 200)}`);
    made = true;
  }
  if (made || !fs.readdirSync(CERT_DIR).some((f) => f.endsWith(".0"))) spawnSync("openssl", ["rehash", CERT_DIR]);
}

// ---- 队列 ----
async function removePlainInputs(task) {
  const dir = path.join(taskDir(task), "mpc");
  for (const p of [path.join(dir, "data"), path.join(dir, "Player-Data", "Input-Binary-P0-0")]) await fsp.rm(p, { recursive: true, force: true }).catch(() => {});
}
function pump(kind) {
  if (running[kind] || !queues[kind].length) return;
  const task = state.tasks.find((t) => t.id === queues[kind].shift());
  if (!task || task.state !== "queued") return pump(kind);
  running[kind] = task;
  const job = kind === "local" ? trainLocal(task) : trainMpc(task).finally(() => removePlainInputs(task));
  job.catch((err) => stopTask(task, "failed", err.message, "平台", "danger")).finally(() => {
    running[kind] = null;
    live.delete(task.id);
    save();
    pump(kind);
  });
}

function cancel(task, user) {
  if (!["running", "queued", "waiting"].includes(task.state)) throw new Error("任务已经结束");
  const l = live.get(task.id);
  task.state = "cancelled";
  task.reason = user.role === "admin" ? "管理员终止" : "提交人撤回";
  if (l) l.procs.forEach((p) => { try { p.kill("SIGTERM"); } catch { /* 已退出 */ } });
  Object.values(queues).forEach((q) => { const i = q.indexOf(task.id); if (i >= 0) q.splice(i, 1); });
  addAudit(moduleOf(task), user.name, `${user.role === "admin" ? "终止" : "撤回"} ${task.id}`, "已终止");
  save();
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------
const json = (res, status, payload) => {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
  res.end(JSON.stringify(payload));
};
const bodyJson = (req) => new Promise((resolve, reject) => {
  let raw = "";
  req.setEncoding("utf8");
  req.on("data", (c) => { raw += c; if (raw.length > 30_000_000) req.destroy(new Error("请求体过大")); });
  req.on("end", () => { try { resolve(raw ? JSON.parse(raw) : {}); } catch (e) { reject(e); } });
  req.on("error", reject);
});

// 使用者只看到自己提交的任务和与之相关的事件；数据使用记录和完整审计只对管理者开放
function publicState(user) {
  const admin = user.role === "admin";
  const tasks = admin ? state.tasks : state.tasks.filter((t) => t.requesterId === user.id);
  const mine = tasks.map((t) => new RegExp(`${t.id}(?!\\d)`));
  const audit = admin ? state.audit.slice(0, 400)
    : state.audit.filter((a) => a.actor === user.name || mine.some((re) => re.test(a.event))).slice(0, 200);
  return {
    me: { ...user, roleLabel: ROLES[user.role] },
    platform: {
      host: os.hostname(),
      mpspdz: { root: MPSPDZ, version: mpspdzVersion },
      sandbox: "bwrap：断网、只读文件系统、只有工作目录可写",
      llm: LLM && LLM.apiKey ? { model: LLM.model || "deepseek-flash", provider: "DeepSeek" } : null,
      reviewModels: [
        { id: "deepseek", label: "DeepSeek（deepseek-flash）", ready: Boolean(LLM && LLM.apiKey) },
        { id: "glm", label: "智谱 GLM", ready: false },
        { id: "qwen", label: "通义千问 Qwen", ready: false }
      ],
      partyMemGb: PARTY_MEM_KB / 1024 / 1024,
      protocols: Object.fromEntries(Object.entries(PROTOCOLS).map(([k, v]) => [k, { label: v.label, parties: v.parties }])),
      envs: condaEnvs().map((e) => e.name),
      defaultProject: DEFAULT_PROJECT,
      running: { local: running.local ? running.local.id : null, mpc: running.mpc ? running.mpc.id : null },
      queue: { local: queues.local.slice(), mpc: queues.mpc.slice() },
      flows: FLOWS
    },
    datasets: state.datasets,
    tasks: tasks.map((t) => ({ ...t, logTail: undefined, local: t.local ? { ...t.local, envBin: undefined } : undefined })),
    audit,
    usage: admin ? state.usage.slice(0, 200) : []
  };
}

// ---- 登录与权限 ----
const SID = "eegsec_sid";
function currentUser(req) {
  const m = new RegExp(`(?:^|;\\s*)${SID}=([a-f0-9]{32})`).exec(req.headers.cookie || "");
  const id = m && sessions.get(m[1]);
  return ACCOUNTS.find((a) => a.id === id) || null;
}
const denied = (msg, status = 403) => Object.assign(new Error(msg), { status });
function requireRole(user, role) {
  if (user.role !== role) throw denied(role === "admin" ? "该操作需要管理者权限" : "该操作只对使用者开放");
}

async function codeClasses(body) {
  if (body.source === "upload") {
    const tmp = await fsp.mkdtemp(path.join(os.tmpdir(), "eegsec-scan-"));
    try {
      await snapshotFromUpload(body.files, tmp);
      return await tool(["classes", tmp], { timeoutMs: 60000 });
    } finally { await fsp.rm(tmp, { recursive: true, force: true }); }
  }
  const dir = path.resolve(String(body.path || ""));
  if (!fs.existsSync(dir)) throw new Error("代码目录不存在");
  return tool(["classes", dir], { timeoutMs: 60000 });
}

async function handleApi(req, res, url) {
  const parts = url.pathname.split("/").filter(Boolean);
  try {
    if (req.method === "GET" && url.pathname === "/api/accounts") return json(res, 200, { items: ACCOUNTS.map((a) => ({ ...a, roleLabel: ROLES[a.role] })) });
    if (req.method === "POST" && url.pathname === "/api/login") {
      const wanted = String((await bodyJson(req)).id || "");
      const account = ACCOUNTS.find((a) => a.id === wanted);
      if (!account) throw denied("账号不存在", 400);
      const sid = crypto.randomBytes(16).toString("hex");
      sessions.set(sid, account.id);
      addAudit("账号", account.name, `${ROLES[account.role]}登录`, "已登录");
      res.setHeader("Set-Cookie", `${SID}=${sid}; Path=/; HttpOnly; SameSite=Strict`);
      return json(res, 200, { ...account, roleLabel: ROLES[account.role] });
    }
    const user = currentUser(req);
    if (!user) return json(res, 401, { error: "请先登录" });
    if (req.method === "POST" && url.pathname === "/api/logout") {
      const m = new RegExp(`(?:^|;\\s*)${SID}=([a-f0-9]{32})`).exec(req.headers.cookie || "");
      if (m) sessions.delete(m[1]);
      res.setHeader("Set-Cookie", `${SID}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0`);
      return json(res, 200, { ok: true });
    }
    if (req.method === "GET" && url.pathname === "/api/state") return json(res, 200, publicState(user));
    if (req.method === "POST" && url.pathname === "/api/datasets") {
      requireRole(user, "admin");
      const ds = await registerDataset(await bodyJson(req), user.name);
      save();
      return json(res, 201, ds);
    }
    if (req.method === "POST" && parts[1] === "datasets" && parts[2]) {
      requireRole(user, "admin");
      const ds = dsById(parts[2]);
      if (!ds) return json(res, 404, { error: "数据集不存在" });
      return json(res, 200, updateDataset(ds, await bodyJson(req), user.name));
    }
    if (req.method === "GET" && url.pathname === "/api/projects") return json(res, 200, { items: await listProjects() });
    if (req.method === "POST" && url.pathname === "/api/code/classes") return json(res, 200, await codeClasses(await bodyJson(req)));
    if (req.method === "POST" && url.pathname === "/api/tasks") {
      requireRole(user, "user");
      return json(res, 201, await createTask(await bodyJson(req), user));
    }
    if (parts[1] === "tasks" && parts[2]) {
      const task = state.tasks.find((t) => t.id === parts[2]);
      const owner = Boolean(task) && task.requesterId === user.id;
      if (!task || (user.role !== "admin" && !owner)) return json(res, 404, { error: "任务不存在" });
      const action = parts[3];
      if (req.method === "GET" && action === "log") return json(res, 200, { lines: task.logTail || [] });
      if (req.method === "GET" && (action === "model" || action === "output") && !owner) throw denied("训练结果只交付给任务提交人");
      if (req.method === "GET" && action === "model") {
        if (task.mode !== "mpc" || task.state !== "done") return json(res, 403, { error: "模型尚未批准交付" });
        res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Disposition": `attachment; filename="${task.id}-model_params.bin"` });
        return fs.createReadStream(path.join(taskDir(task), "model_params.bin")).pipe(res);
      }
      if (req.method === "GET" && action === "output") {
        if (task.mode !== "local" || task.state !== "done") return json(res, 403, { error: "产出尚未放行" });
        const rel = url.searchParams.get("path") || "";
        const allowed = (task.steps.check.files || []).some((f) => f.path === rel);
        if (!allowed) return json(res, 404, { error: "文件不在产出清单里" });
        const file = path.join(taskDir(task), "run", "work", rel);
        res.writeHead(200, { "Content-Type": "application/octet-stream", "Content-Disposition": `attachment; filename*=UTF-8''${encodeURIComponent(path.basename(rel))}` });
        addAudit("本地训练", user.name, `${task.id} 下载产出 ${rel}`, "已下载");
        return fs.createReadStream(file).pipe(res);
      }
      if (req.method === "POST") {
        const body = await bodyJson(req);
        if (action === "cancel") { cancel(task, user); return json(res, 200, task); }
        requireRole(user, "admin");
        if (action === "review") {
          if (body.model && body.model !== "deepseek") throw new Error("该模型尚未接入");
          const llm = await llmReview(task, path.join(taskDir(task), "code"));
          task.steps.review = { ...(task.steps.review || {}), llm, at: now() };
          if (llm.ok) addAudit(moduleOf(task), `大模型（${llm.model}）`, `${task.id} 重新审阅：${llm.summary}`, { pass: "通过", suspicious: "可疑", malicious: "恶意" }[llm.verdict], { pass: "ok", suspicious: "warn", malicious: "danger" }[llm.verdict]);
          save();
          return json(res, 200, task);
        }
        if (action === "confirm") { await confirmReview(task, body.decision === "block" ? "block" : "continue", user.name); return json(res, 200, task); }
        if (action === "consent") { consent(task, body.dataset, body.decision === "reject" ? "reject" : "approve", user.name); return json(res, 200, task); }
        if (action === "return") {
          if (!["blocked", "needs-fix"].includes(task.state)) throw new Error("只有被拦截或需修改的任务可以退回");
          task.state = "returned";
          task.reason = String(body.note || "").trim() || task.reason;
          addAudit(moduleOf(task), user.name, `${task.id} 退回 ${task.requester} 修改`, "已退回");
          save();
          return json(res, 200, task);
        }
        if (action === "deliver") { await deliver(task, user.name); return json(res, 200, task); }
      }
    }
    return json(res, 404, { error: "接口不存在" });
  } catch (err) {
    return json(res, err.status || 400, { error: err.message || String(err) });
  }
}

function serveStatic(req, res, url) {
  const file = STATIC_FILES[url.pathname];
  if (!file) return json(res, 404, { error: "页面不存在" });
  const types = { ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8" };
  res.writeHead(200, { "Content-Type": types[path.extname(file)], "Cache-Control": "no-store" });
  fs.createReadStream(path.join(ROOT, file)).pipe(res);
}

async function main() {
  await fsp.mkdir(STATE_DIR, { recursive: true });
  await fsp.mkdir(TASK_ROOT, { recursive: true });
  Object.assign(state, await readJson(path.join(STATE_DIR, "state.json"), {}));
  state.calibration = state.calibration || {};
  if (!state.calibration["rep3-standard-8"]) {
    // 本机实测：8 个训练窗口 + 4 验证 + 4 测试，每窗口 500 点，8 线程，1 轮 MPC 计算 23 秒
    state.calibration["rep3-standard-8"] = { secondsPerUnit: 23 / 24, points: 500, at: "2026-10-27T12:40:00+08:00", task: "本机实测" };
  }
  state.datasets.forEach((d) => { if (!d.ownership) { d.ownership = "own"; d.partner = ""; } });
  // 旧记录补读数据文件自带的 dataset 标签
  state.datasets.filter((d) => d.meta && d.meta.tag === undefined).forEach((d) => {
    tool(["meta", d.path], { timeoutMs: 15 * 60 * 1000 }).then((m) => { if (m.ok) { d.meta = m; save(); } });
  });
  // 旧版本的任务记录（按数字步骤）不再兼容，保留在审计日志里
  state.tasks = state.tasks.filter((t) => t.mode);
  // 区分角色之前提交的任务没有提交人账号，归到第一个使用者账号下
  const firstUser = ACCOUNTS.find((a) => a.role === "user");
  state.tasks.filter((t) => !t.requesterId && firstUser).forEach((t) => { t.requesterId = firstUser.id; t.requester = firstUser.name; });
  state.tasks.forEach((t) => {
    if (["running", "queued"].includes(t.state)) { t.state = "failed"; t.reason = "平台服务重启，任务中断，请重新提交"; }
  });
  sandboxOk = spawnSync("unshare", ["-rn", "true"]).status === 0;
  if (spawnSync("bwrap", ["--ro-bind", "/", "/", "true"]).status !== 0) console.error("警告：bwrap 不可用，本地训练无法运行");
  try { mpspdzVersion = ((await fsp.readFile(path.join(MPSPDZ, "CHANGELOG.md"), "utf8")).match(/^## ([\d.]+)/m) || [])[1] || ""; } catch { mpspdzVersion = ""; }
  ensureCerts(3);
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    if (url.pathname.startsWith("/api/")) return handleApi(req, res, url);
    return serveStatic(req, res, url);
  });
  server.listen(PORT, HOST, () => console.log(`脑电数据安全平台：http://${HOST}:${PORT}`));
  if (!state.datasets.length && fs.existsSync(path.join(DEFAULT_DATASET, "train_raw_eeg_data.pt"))) {
    registerDataset({ path: DEFAULT_DATASET, name: "卒中 / 健康对照 静息态脑电", classNames: ["健康对照", "卒中"], ownership: "own" }, "平台")
      .then(save).catch((e) => console.error("自动登记数据集失败：", e.message));
  }
  save();
}

main().catch((err) => { console.error(err); process.exitCode = 1; });
