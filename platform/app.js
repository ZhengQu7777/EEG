/* 脑电数据安全平台 · 前端
 * 所有数据来自 server.cjs 的 /api 接口，每 2 秒刷新一次。 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const esc = (v) => String(v ?? "").replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[c]));
  const pad = (n) => String(n).padStart(2, "0");
  const chip = (text, level) => `<span class="chip ${level || ""}">${esc(text)}</span>`;
  const pct = (v) => (v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`);
  const num = (v, d = 4) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
  const fmtTime = (iso) => {
    if (!iso) return "—";
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? String(iso) : `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };
  const fmtDur = (s) => {
    if (s == null) return "—";
    s = Math.max(0, Math.round(s));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    return h ? `${h} 小时 ${m} 分` : m ? `${m} 分 ${x} 秒` : `${x} 秒`;
  };
  const fmtCount = (n) => (n == null ? "—" : n >= 1e8 ? `${(n / 1e8).toFixed(2)} 亿` : n >= 1e4 ? `${(n / 1e4).toFixed(1)} 万` : String(n));
  const fmtBytes = (b) => (b == null ? "—" : b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(1)} GB` : b >= 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MB` : b >= 1024 ? `${(b / 1024).toFixed(1)} KB` : `${b} B`);

  const MODE = { local: "本地训练", mpc: "联合训练" };
  const STEP = { review: "代码审阅", trial: "试运行", compile: "编译检查", consent: "合作方授权", check: "产出检查", deliver: "模型交付" };
  const stepName = (mode, key) => (key === "train" ? MODE[mode] : STEP[key] || key);
  const ACTIVE = ["running", "waiting", "queued"];
  const STOPPED = ["blocked", "needs-fix", "returned", "rejected", "failed", "cancelled"];
  const TITLES = { home: "工作台", mytasks: "我的任务", overview: "总览", data: "数据", new: "新建训练任务", review: "任务审批", monitor: "训练监控", audit: "历史记录" };
  // 使用者提交任务、查看自己的任务与结果；管理者审批、监控、管理数据和查看审计
  const ROLE_VIEWS = { user: ["home", "new", "mytasks", "monitor", "data"], admin: ["overview", "review", "monitor", "data", "audit"] };
  const ROLE_TITLES = { user: { monitor: "训练结果", data: "数据集" }, admin: { monitor: "训练监控", data: "数据管理" } };
  const isAdmin = () => Boolean(ui.me && ui.me.role === "admin");
  const CODE_EXT = /\.(py|ya?ml|json|txt|md|cfg|toml|ini|sh)$/i;
  const SKIP_DIRS = new Set([".git", "__pycache__", "weights", "runs", "logs", "outputs", "output", "checkpoints", "wandb", "mlruns", "node_modules", ".ipynb_checkpoints", ".venv", "venv", ".idea", ".vscode"]);
  const ENTRY_NAMES = ["train.py", "main.py", "run.py", "experiment_runner.py"];

  function status(t) {
    switch (t.state) {
      case "blocked": return ["已拦截", "danger"];
      case "needs-fix": return ["需修改", "warn"];
      case "returned": return ["已退回", ""];
      case "rejected": return ["授权被拒绝", "danger"];
      case "failed": return ["出错", "danger"];
      case "cancelled": return ["已终止", ""];
      case "done": return [t.mode === "mpc" ? "已交付" : "已完成", "ok"];
      case "queued": return ["排队中", "info"];
      case "waiting":
        if (!isAdmin()) return t.step === "review" ? ["待管理员确认", "warn"] : t.step === "consent" ? ["待数据方授权", "warn"] : ["待批准交付", "warn"];
        return t.step === "review" ? ["待人工确认", "warn"] : t.step === "consent" ? ["待授权", "warn"] : ["待交付审批", "warn"];
      default: return t.step === "train" ? ["训练中", "info"] : [`${stepName(t.mode, t.step)}中`, "info"];
    }
  }
  const modeChip = (t) => chip(MODE[t.mode], t.mode === "mpc" ? "info" : "");
  const flowOf = (t) => ui.data.platform.flows[t.mode];
  const pastTrain = (t) => flowOf(t).indexOf(t.step) >= flowOf(t).indexOf("train");

  const api = {
    async call(method, url, body) {
      const res = await fetch(url, { method, cache: "no-store", headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw Object.assign(new Error(data.error || `请求失败（${res.status}）`), { status: res.status });
      return data;
    },
    get(url) { return this.call("GET", url); },
    post(url, body) { return this.call("POST", url, body || {}); }
  };

  const ui = {
    me: null, view: "overview", data: null, tab: "local", reviewTask: null, reviewFilter: "todo", monitorTask: null, auditFilter: "全部", myTask: null, myFilter: "all",
    localPicked: null, localProject: null, mpcProject: null, mpcProjectNeedsPick: false, mpcProgramNeedsPick: "", projects: null, reviewModel: "deepseek", partner: null, mpcPicked: new Set(), log: [], editing: null, envFilled: false, sig: {}
  };
  const dsById = (id) => (ui.data ? ui.data.datasets.find((d) => d.id === id) : null);
  const taskById = (id) => (ui.data ? ui.data.tasks.find((t) => t.id === id) : null);
  const ownSets = () => ui.data.datasets.filter((d) => d.ownership !== "partner");
  const partnerNames = () => [...new Set(ui.data.datasets.filter((d) => d.ownership === "partner").map((d) => d.partner))];

  // ---------------------------------------------------------------------------
  // 侧栏与总览
  // ---------------------------------------------------------------------------
  function renderEnv() {
    const p = ui.data.platform;
    const line = (id, q) => (id ? `${id} 运行中` : "空闲") + (q ? `，排队 ${q}` : "");
    $("#env-info").innerHTML = `<dt>本地训练</dt><dd>${esc(line(p.running.local, p.queue.local.length))}</dd><dt>联合训练</dt><dd>${esc(line(p.running.mpc, p.queue.mpc.length))}</dd>`;
    $("#me-box").innerHTML = `<div><strong>${esc(ui.me.name)}</strong>${chip(ui.me.roleLabel, isAdmin() ? "info" : "ok")}</div><button class="link" data-logout="1">退出登录</button>`;
    const tasks = ui.data.tasks;
    const training = tasks.filter((t) => t.step === "train" && ["running", "queued"].includes(t.state)).length;
    if (isAdmin()) {
      const todo = tasks.filter(needsAdmin).length;
      $("#nav-review").textContent = todo ? `${todo} 待处理` : "";
      const deliver = tasks.filter((t) => t.step === "deliver" && t.state === "waiting").length;
      $("#nav-monitor").textContent = deliver ? `${deliver} 待交付` : training ? `${training} 训练中` : "";
    } else {
      const fix = tasks.filter(needsMe).length;
      $("#nav-mytasks").textContent = fix ? `${fix} 需修改` : "";
      $("#nav-results").textContent = training ? `${training} 训练中` : "";
    }
  }
  const needsAdmin = (t) => t.state === "waiting" && ["review", "consent"].includes(t.step);
  const needsMe = (t) => ["needs-fix", "returned", "blocked"].includes(t.state);

  function classBar(d) {
    const total = d.meta.windows || 1;
    const bars = Object.entries(d.meta.classes).map(([lab, [, w]], i) => `<i class="c${i % 4}" style="width:${(w / total * 100).toFixed(1)}%"></i>`).join("");
    const legend = Object.entries(d.meta.classes).map(([lab, [s, w]], i) => `<span><i class="c${i % 4}"></i>${esc(d.classNames[Number(lab)])} ${s} 人 / ${w} 窗口</span>`).join("");
    return `<div class="class-bar">${bars}</div><div class="class-legend">${legend}</div>`;
  }
  const ownChip = (d) => (d.ownership === "partner" ? chip(d.partner, "info") : chip("自有", "ok"));

  function renderOverview() {
    const { datasets, tasks, audit } = ui.data;
    const reviewing = tasks.filter((t) => !pastTrain(t) && ACTIVE.includes(t.state)).length;
    const training = tasks.filter((t) => ["train", "check"].includes(t.step) && ACTIVE.includes(t.state)).length;
    const done = tasks.filter((t) => t.state === "done").length;
    const partners = partnerNames().length;
    $("#overview-stats").innerHTML = [
      ["数据集", datasets.length, "", partners ? `${partners} 家合作方` : "全部为自有数据", "data"],
      ["审批中", reviewing, reviewing ? "warn" : "", "审阅、试运行、授权", "review"],
      ["训练中", training, training ? "info" : "", "本地训练与联合训练", "monitor"],
      ["已完成", done, done ? "ok" : "", `共 ${tasks.length} 个任务`, "monitor"]
    ].map(([k, v, c, n, go]) => `<button class="stat" data-nav="${go}"><span>${k}</span><strong class="${c}">${v}</strong><span>${esc(n)}</span></button>`).join("");
    $("#overview-tasks").innerHTML = "<thead><tr><th>任务</th><th>方式</th><th>当前步骤</th><th>状态</th></tr></thead><tbody>" +
      (tasks.length ? tasks.slice(0, 10).map((t) => {
        const [s, l] = status(t);
        return `<tr class="is-selectable" data-open="${esc(t.id)}"><td><strong>${esc(t.title)}</strong><small>${esc(t.id)} · ${fmtTime(t.createdAt)}</small></td><td>${modeChip(t)}</td><td>${esc(stepName(t.mode, t.step))}</td><td>${chip(s, l)}</td></tr>`;
      }).join("") : `<tr><td colspan="4" class="table-empty">还没有训练任务</td></tr>`) + "</tbody>";
    $("#overview-datasets").innerHTML = datasets.length ? datasets.map((d) => `<div class="ds-card"><div class="ds-head"><strong>${esc(d.name)}</strong>${ownChip(d)}</div><small>${esc(d.id)} · ${d.meta.subjects} 人 · ${d.meta.windows} 窗口 · ${d.meta.channels} 通道 · ${d.meta.sfreq} Hz</small>${classBar(d)}</div>`).join("") : `<p class="muted-text">还没有登记数据集</p>`;
    $("#overview-events").innerHTML = audit.slice(0, 6).map((a) => `<li><span class="dot ${a.level || ""}"></span><p>${esc(a.event)}</p><small>${fmtTime(a.time)} · ${esc(a.actor)}</small></li>`).join("") || `<li><span></span><p class="muted-text">暂无事件</p></li>`;
  }

  // 使用者工作台：只统计自己的任务
  function renderHome() {
    const { datasets, tasks, audit } = ui.data;
    const reviewing = tasks.filter((t) => !pastTrain(t) && ACTIVE.includes(t.state)).length;
    const training = tasks.filter((t) => ["train", "check", "deliver"].includes(t.step) && ACTIVE.includes(t.state)).length;
    const fix = tasks.filter(needsMe).length;
    const done = tasks.filter((t) => t.state === "done").length;
    $("#home-stats").innerHTML = [
      ["审批中", reviewing, reviewing ? "warn" : "", "代码审阅、试运行、授权", "mytasks"],
      ["训练中", training, training ? "info" : "", "含等待交付", "monitor"],
      ["需修改", fix, fix ? "warn" : "", "被拦截或退回的任务", "mytasks"],
      ["已完成", done, done ? "ok" : "", `共提交 ${tasks.length} 个任务`, "monitor"]
    ].map(([k, v, c, n, go]) => `<button class="stat" data-nav="${go}"><span>${k}</span><strong class="${c}">${v}</strong><span>${esc(n)}</span></button>`).join("");
    $("#home-tasks").innerHTML = "<thead><tr><th>任务</th><th>方式</th><th>当前步骤</th><th>状态</th></tr></thead><tbody>" +
      (tasks.length ? tasks.slice(0, 10).map((t) => {
        const [s, l] = status(t);
        return `<tr class="is-selectable" data-my="${esc(t.id)}"><td><strong>${esc(t.title)}</strong><small>${esc(t.id)} · ${fmtTime(t.createdAt)}</small></td><td>${modeChip(t)}</td><td>${esc(stepName(t.mode, t.step))}</td><td>${chip(s, l)}</td></tr>`;
      }).join("") : `<tr><td colspan="4" class="table-empty">还没有提交过任务　<button class="link" data-nav="new">新建训练任务</button></td></tr>`) + "</tbody>";
    $("#home-datasets").innerHTML = datasets.length ? datasets.map((d) => `<div class="ds-card"><div class="ds-head"><strong>${esc(d.name)}</strong>${ownChip(d)}</div><small>${esc(d.id)} · ${d.meta.subjects} 人 · ${d.meta.windows} 窗口 · ${d.meta.channels} 通道 · ${d.meta.sfreq} Hz</small></div>`).join("") : `<p class="muted-text">还没有可用的数据集</p>`;
    $("#home-events").innerHTML = audit.slice(0, 6).map((a) => `<li><span class="dot ${a.level || ""}"></span><p>${esc(a.event)}</p><small>${fmtTime(a.time)} · ${esc(a.actor)}</small></li>`).join("") || `<li><span></span><p class="muted-text">暂无动态</p></li>`;
  }

  // ---------------------------------------------------------------------------
  // 数据
  // ---------------------------------------------------------------------------
  const classText = (d) => Object.entries(d.meta.classes).map(([lab, [s, w]]) => `${d.classNames[Number(lab)] || lab} ${s} 人 / ${w} 窗口`).join("；");

  function renderData() {
    const { datasets, usage } = ui.data;
    const admin = isAdmin();
    $("#data-admin").hidden = !admin;
    const perClass = (d) => Object.entries(d.meta.classes).map(([lab, [s]]) => `${d.classNames[Number(lab)] || lab} ${s}`).join(" · ");
    $("#dataset-table").innerHTML = `<thead><tr><th>数据集</th><th>归属</th><th>负责人</th><th>受试者</th><th>各类别人数</th><th>窗口</th><th>通道</th><th>采样率</th><th>版本指纹</th>${admin ? "<th></th>" : ""}</tr></thead><tbody>` +
      (datasets.length ? datasets.map((d) => `<tr><td><strong>${esc(d.name)}</strong><small>${esc(d.id)}${d.meta.tag ? ` · 文件标签 ${esc(d.meta.tag)}` : ""}</small></td><td>${ownChip(d)}</td><td>${esc(d.owner)}</td><td class="num">${d.meta.subjects}</td><td>${esc(perClass(d))}</td><td class="num">${d.meta.windows}</td><td class="num">${d.meta.channels}</td><td class="num">${d.meta.sfreq} Hz</td><td class="mono" title="${esc(d.meta.fingerprint)}">${esc(d.meta.fingerprintShort)}</td>${admin ? `<td><button class="btn" data-edit-ds="${esc(d.id)}">编辑</button></td>` : ""}</tr>`).join("")
        : `<tr><td colspan="10" class="table-empty">还没有登记数据集</td></tr>`) + "</tbody>";
    if (!admin) return;
    $("#usage-table").innerHTML = "<thead><tr><th>时间</th><th>任务</th><th>数据集</th><th>状态</th></tr></thead><tbody>" +
      (usage.length ? usage.map((u) => `<tr><td class="num">${fmtTime(u.time)}</td><td>${esc(u.task)}</td><td>${esc(u.dataset)}<small class="mono">${esc(u.fingerprint)}</small></td><td>${chip(u.result, u.result === "训练完成" ? "ok" : u.result === "训练失败" ? "danger" : "info")}</td></tr>`).join("")
        : `<tr><td colspan="4" class="table-empty">还没有任务读取过真实数据</td></tr>`) + "</tbody>";
  }

  function startEdit(id) {
    const d = dsById(id);
    if (!d) return;
    ui.editing = id;
    $("#reg-title").textContent = `编辑 ${d.id}`;
    $("#reg-cancel").hidden = false;
    $("#reg-path-field").hidden = true;
    $("#reg-name").value = d.name;
    $("#reg-owner").value = d.owner || "";
    $("#reg-classes").value = d.classNames.join(", ");
    $("#reg-ownership").value = d.ownership || "own";
    $("#reg-partner").value = d.partner || "";
    $("#reg-submit").textContent = "保存";
    $("#register-form").scrollIntoView({ block: "center" });
  }
  function endEdit() {
    ui.editing = null;
    $("#register-form").reset();
    $("#reg-title").textContent = "登记数据集";
    $("#reg-cancel").hidden = true;
    $("#reg-path-field").hidden = false;
    $("#reg-submit").textContent = "登记";
    $("#reg-hint").textContent = "";
  }

  async function submitRegister(event) {
    event.preventDefault();
    const btn = $("#reg-submit");
    btn.disabled = true;
    const body = {
      name: $("#reg-name").value, owner: $("#reg-owner").value, ownership: $("#reg-ownership").value, partner: $("#reg-partner").value,
      classNames: $("#reg-classes").value.split(/[,，]/).map((s) => s.trim()).filter(Boolean)
    };
    try {
      if (ui.editing) {
        const ds = await api.post(`/api/datasets/${encodeURIComponent(ui.editing)}`, body);
        toast("已保存", `${ds.id} · ${ds.ownership === "partner" ? ds.partner : "自有数据"}`);
      } else {
        $("#reg-hint").textContent = "正在读取数据…";
        const ds = await api.post("/api/datasets", { ...body, path: $("#reg-path").value });
        toast("已登记", `${ds.id} · ${ds.meta.subjects} 人 · ${ds.meta.windows} 窗口`);
      }
      endEdit();
      ui.sig = {};
      await refresh();
    } catch (err) {
      $("#reg-hint").textContent = err.message;
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // 新建训练任务
  // ---------------------------------------------------------------------------
  const architectureIcons = {
    partner: '<path d="M5 21V5h14v16M3 21h18M9 21v-5h6v5M9 9h6M12 6v6"/>',
    project: '<path d="M3 7a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/><path d="m10 12-3 3 3 3m4-6 3 3-3 3"/>'
  };
  const architectureIcon = (type) => `<svg viewBox="0 0 24 24" aria-hidden="true">${architectureIcons[type]}</svg>`;

  function pickCard(d, checked, kind) {
    const radio = kind === "partner" || kind === "lproj" || kind === "mproj";
    const key = kind === "partner" ? d : radio ? d.path : d.id;
    if (kind === "partner" || kind === "mproj") {
      const body = kind === "partner"
        ? `<strong>${esc(d)}</strong><small>${ui.data.datasets.filter((x) => x.ownership === "partner" && x.partner === d).length} 个可用数据集</small>`
        : `<strong>${esc(d.name)}</strong><small>${d.entries.length} 个入口脚本</small><small class="architecture-path" title="${esc(d.path)}">${esc(d.path)}</small>`;
      return `<label class="pick architecture-choice${checked ? " is-picked" : ""}"><span class="architecture-choice-icon">${architectureIcon(kind === "partner" ? "partner" : "project")}</span><span class="architecture-choice-text">${body}</span><input type="radio" name="${kind}" data-pick-kind="${kind}" data-key="${esc(key)}"${checked ? " checked" : ""} /></label>`;
    }
    if (kind === "mpcds") {
      return `<label class="pick architecture-dataset${checked ? " is-picked" : ""}"><input type="checkbox" name="${kind}" data-pick-kind="${kind}" data-key="${esc(key)}"${checked ? " checked" : ""} /><span><strong>${esc(d.name)}</strong><small>${d.meta.subjects} 人 · ${d.meta.windows} 窗口 · ${d.meta.channels} 通道</small></span></label>`;
    }
    let body;
    if (kind === "partner") body = `<span><strong>${esc(d)}</strong><small>${ui.data.datasets.filter((x) => x.partner === d).length} 个可用数据集</small></span>`;
    else if (radio) body = `<span><strong>${esc(d.name)}</strong><small>${esc(d.path)}</small><small>${d.entries.length} 个入口脚本 · 最近修改 ${fmtTime(d.modified)}</small></span>`;
    else body = `<span><strong>${esc(d.name)}</strong><small>${esc(d.id)} · ${d.meta.subjects} 人 · ${d.meta.windows} 窗口 · ${d.meta.channels} 通道 · ${d.meta.sfreq} Hz</small>${classBar(d)}</span>`;
    return `<label class="pick${checked ? " is-picked" : ""}"><input type="${radio ? "radio" : "checkbox"}" name="${kind}" data-pick-kind="${kind}" data-key="${esc(key)}"${checked ? " checked" : ""} />${body}</label>`;
  }

  const projects = () => (ui.projects && ui.projects.projects) || [];
  const projectByPath = (p) => projects().find((x) => x.path === p);
  const inProject = (d, proj) => proj && d.path.startsWith(proj.path + "/");

  function suggestCommand(proj) {
    const entry = ENTRY_NAMES.find((n) => proj.entries.includes(n)) || proj.entries[0];
    return entry ? `python ${entry}${entry === "experiment_runner.py" ? " --models EEGNet --num-epochs 5 --device cpu" : ""}` : "";
  }

  function renderNew() {
    if (!ui.data) return;
    const p = ui.data.platform;
    if (!ui.envFilled && p.envs.length) {
      $("#local-env").innerHTML = p.envs.map((e) => `<option value="${esc(e)}">${esc(e)}</option>`).join("");
      $("#local-env").value = p.envs.includes("Stroke") ? "Stroke" : p.envs[0];
      ui.envFilled = true;
    }
    $$("[data-tab]").forEach((b) => b.classList.toggle("is-active", b.dataset.tab === ui.tab));
    $$("[data-tab-page]").forEach((s) => { s.hidden = s.dataset.tabPage !== ui.tab; });
    const projs = projects();
    if (!ui.localProject && projs.length) { ui.localProject = projs[0].path; if (!$("#local-command").value) $("#local-command").value = suggestCommand(projs[0]); }
    if (!ui.mpcProject && !ui.mpcProjectNeedsPick && projs.length) ui.mpcProject = projs[0].path;

    // 本地训练：项目 → 数据集 → 运行命令
    const lp = projectByPath(ui.localProject);
    const lsets = ownSets().filter((d) => inProject(d, lp));
    if (!ui.localPicked) ui.localPicked = new Set(lsets.map((d) => d.id));
    const lsig = JSON.stringify([projs.map((x) => x.path), ui.localProject, lsets.map((d) => d.id), [...ui.localPicked]]);
    if (ui.sig.local !== lsig) {
      ui.sig.local = lsig;
      $("#local-projects").innerHTML = projs.length ? projs.map((x) => pickCard(x, x.path === ui.localProject, "lproj")).join("") : `<p class="muted-text">还没有项目。项目根目录由已登记的自有数据集推断（数据集所在 data/ 目录的上一级）。</p>`;
      $("#local-datasets").innerHTML = !lp ? `<p class="muted-text">先选择项目</p>` : lsets.length ? lsets.map((d) => `${pickCard(d, ui.localPicked.has(d.id), "local")}`).join("") : `<p class="muted-text">这个项目里还没有登记过数据集</p>`;
    }
    let lhint = "";
    if (!lp) lhint = "请选择项目";
    else if (![...ui.localPicked].some((id) => lsets.some((d) => d.id === id))) lhint = "请选择数据集";
    else if (!$("#local-command").value.trim()) lhint = "请填写运行命令";
    else if (!p.envs.includes($("#local-env").value)) lhint = "请选择可用的 Python 环境";
    $("#local-submit").disabled = Boolean(lhint);
    $("#local-hint").textContent = lhint || "提交后先做代码审阅和试运行，通过后自动在沙箱里训练。";

    // 联合训练：合作方 → 项目 → 数据集 → MPC 程序
    const partners = partnerNames();
    if (ui.partner && !partners.includes(ui.partner)) ui.partner = null;
    if (!ui.partner && partners.length) ui.partner = partners[0];
    const mp = projectByPath(ui.mpcProject);
    const theirs = ui.data.datasets.filter((d) => d.ownership === "partner" && d.partner === ui.partner);
    const ours = ownSets().filter((d) => inProject(d, mp));
    const msig = JSON.stringify([partners, ui.partner, projs, ui.mpcProject, theirs, ours, [...ui.mpcPicked], ui.mpcProgramNeedsPick, (ui.projects || {}).samples]);
    if (ui.sig.mpc !== msig) {
      ui.sig.mpc = msig;
      $("#mpc-partners").innerHTML = partners.length ? partners.map((n) => pickCard(n, n === ui.partner, "partner")).join("")
        : `<p class="muted-text">还没有合作方数据集。到“数据”页登记数据集时，把归属设为合作方。</p>`;
      $("#mpc-projects").innerHTML = projs.length ? projs.map((x) => pickCard(x, x.path === ui.mpcProject, "mproj")).join("") : `<p class="muted-text">还没有项目</p>`;
      $("#mpc-datasets").innerHTML = !ui.partner ? `<p class="muted-text">先选择合作方</p>` :
        theirs.map((d) => pickCard(d, ui.mpcPicked.has(d.id), "mpcds")).join("") +
        (ours.length ? `<p class="pick-label">本方数据（可一起参与）</p>${ours.map((d) => pickCard(d, ui.mpcPicked.has(d.id), "mpcds")).join("")}` : "");
      const opts = [];
      if (mp) mp.mpc.forEach((rel) => opts.push([`${mp.path}/${rel}`, `${rel}`]));
      ((ui.projects && ui.projects.samples) || []).forEach((f) => opts.push([`sample:${f}`, `平台示例：${f}`]));
      const cur = $("#mpc-program").value;
      $("#mpc-program").innerHTML = (ui.mpcProgramNeedsPick ? `<option value="">请重新选择 MPC 程序（上次：${esc(ui.mpcProgramNeedsPick)}）</option>` : "") +
        (opts.length ? opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("") : `<option value="">项目里没有 .mpc 文件</option>`);
      const keep = opts.find(([v]) => v === cur) || opts.find(([v]) => v.startsWith("sample:")) || opts[0];
      if (ui.mpcProgramNeedsPick) $("#mpc-program").value = "";
      else if (keep) $("#mpc-program").value = keep[0];
    }
    const picked = [...ui.mpcPicked].map(dsById).filter(Boolean);
    let mhint = "";
    if (!ui.partner) mhint = "请选择合作方";
    else if (!mp) mhint = "请选择项目";
    else if (!picked.some((d) => d.ownership === "partner")) mhint = "请至少选择一个合作方数据集";
    else if (!$("#mpc-program").value) mhint = ui.mpcProgramNeedsPick ? "上次任务未保存程序来源，请重新选择 MPC 程序" : "请选择 MPC 程序";
    $("#mpc-submit").disabled = Boolean(mhint);
    $("#mpc-hint").textContent = mhint || "提交后先做代码审阅和编译检查，通过后等待合作方授权。";
    renderMpcTopology(mp, picked, mhint);
  }

  function renderMpcTopology(project, picked, hint) {
    const protocol = $("#p-protocol").value;
    const count = protocol === "semi2k" ? 2 : 3;
    const names = Array.from({ length: count }, (_, i) => `P${i}`);
    const windows = picked.reduce((sum, d) => sum + (Number(d.meta.windows) || 0), 0);
    const summary = [ui.partner || "待选择合作方", project ? project.name : "待选择项目", picked.length ? `${picked.length} 个数据集 · ${windows} 个窗口` : "待选择参与数据"].join(" / ");
    const sig = JSON.stringify([protocol, summary, hint]);
    if (ui.sig.topology === sig) return;
    ui.sig.topology = sig;
    const ready = !hint;
    $("#mpc-topology").classList.toggle("is-ready", ready);
    $("#topology-status").textContent = ready ? "可提交审批" : "配置中";
    $("#topology-status").className = `chip ${ready ? "ok" : ""}`;
    $("#topology-data-count").textContent = `已选 ${picked.length} 个`;
    $("#topology-summary").textContent = summary;
    $("#topology-data-link").classList.toggle("is-connected", Boolean(project && picked.some((d) => d.ownership === "partner")));
    $("#topology-run-link").classList.toggle("is-connected", ready);
    $("#topology-parties").style.setProperty("--party-count", count);
    $("#topology-parties").innerHTML = names.map((name) => `<div class="architecture-party"><span class="architecture-party-dot"></span><strong>${name}</strong><small>计算方</small></div>`).join("");
    $("#topology-protocol").textContent = protocol === "semi2k" ? "两方 semi2k" : "三方 replicated-ring";
  }

  function onPick(el) {
    const kind = el.dataset.pickKind, key = el.dataset.key;
    if (kind === "lproj") { ui.localProject = key; ui.localPicked = null; const pj = projectByPath(key); if (pj) $("#local-command").value = suggestCommand(pj); }
    else if (kind === "mproj") { ui.mpcProject = key; ui.mpcProjectNeedsPick = false; ui.mpcPicked = new Set([...ui.mpcPicked].filter((id) => { const d = dsById(id); return d && (d.ownership === "partner" || inProject(d, projectByPath(key))); })); }
    else if (kind === "partner") { ui.partner = key; ui.mpcPicked = new Set(); }
    else if (kind === "local") { if (el.checked) ui.localPicked.add(key); else ui.localPicked.delete(key); }
    else if (kind === "mpcds") { if (el.checked) ui.mpcPicked.add(key); else ui.mpcPicked.delete(key); }
    ui.sig = {};
    renderNew();
  }

  async function submit(mode) {
    const btn = $(mode === "local" ? "#local-submit" : "#mpc-submit");
    btn.disabled = true;
    try {
      const body = mode === "local"
        ? { mode, datasets: [...ui.localPicked], code: { source: "path", path: ui.localProject },
          local: { command: $("#local-command").value, env: $("#local-env").value, trialSeconds: Number($("#local-trial").value) } }
        : { mode, datasets: [...ui.mpcPicked], mpc: { programPath: $("#mpc-program").value,
          params: { protocol: $("#p-protocol").value, precision: $("#p-precision").value, downsample: Number($("#p-downsample").value), perClass: Number($("#p-perclass").value) } } };
      const task = await api.post("/api/tasks", body);
      toast("已提交审批", `${task.id} · ${MODE[task.mode]}`);
      ui.myTask = task.id;
      ui.myFilter = "all";
      await refresh();
      navigate("mytasks");
    } catch (err) {
      toast("提交失败", err.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------------------
  // 任务审批
  // ---------------------------------------------------------------------------
  const REVIEW_FILTERS = [
    ["todo", "待处理", needsAdmin],
    ["stopped", "未通过", (t) => STOPPED.includes(t.state) && !pastTrain(t)],
    ["passed", "已通过", (t) => pastTrain(t)],
    ["all", "全部", () => true]
  ];

  function renderReview() {
    const tasks = ui.data.tasks;
    $("#review-filters").innerHTML = REVIEW_FILTERS.map(([k, label, fn]) => `<button class="filter${ui.reviewFilter === k ? " is-active" : ""}" data-review-filter="${k}">${label} ${tasks.filter(fn).length}</button>`).join("");
    const rows = tasks.filter(REVIEW_FILTERS.find(([k]) => k === ui.reviewFilter)[2]);
    if (!rows.some((t) => t.id === ui.reviewTask)) ui.reviewTask = rows.length ? rows[0].id : null;
    $("#review-table").innerHTML = "<thead><tr><th>任务</th><th>提交人</th><th>方式</th><th>当前步骤</th><th>状态</th></tr></thead><tbody>" +
      (rows.length ? rows.map((t) => {
        const [s, l] = status(t);
        return `<tr class="is-selectable${t.id === ui.reviewTask ? " is-selected" : ""}" data-review="${esc(t.id)}"><td><strong>${esc(t.title)}</strong><small>${esc(t.id)} · ${fmtTime(t.createdAt)}</small></td><td>${esc(t.requester)}</td><td>${modeChip(t)}</td><td>${esc(stepName(t.mode, t.step))}</td><td>${chip(s, l)}</td></tr>`;
      }).join("") : `<tr><td colspan="5" class="table-empty">${ui.reviewFilter === "todo" ? "没有待处理的任务" : "没有符合条件的任务"}</td></tr>`) + "</tbody>";
    renderReviewDetail();
  }

  function sectionStatus(t, key) {
    const f = flowOf(t);
    const i = f.indexOf(key), cur = f.indexOf(t.step);
    if (t.state === "done" || cur > i) return ["通过", "ok"];
    if (cur < i) return ["未开始", ""];
    return status(t);
  }

  // editable：管理者可换模型重审、人工确认；使用者只读审阅意见，用来修改代码
  function reviewBody(t, editable = true) {
    const rv = t.steps.review;
    if (!rv) return `<p class="muted-text">正在审阅…</p>`;
    const llm = rv.llm || {};
    const verdict = { pass: ["通过", "ok"], suspicious: ["可疑", "warn"], malicious: ["发现问题", "danger"] }[llm.verdict] || ["不可用", "warn"];
    const models = (ui.data.platform.reviewModels || []).map((m) => `<option value="${esc(m.id)}"${m.id === ui.reviewModel ? " selected" : ""}>${esc(m.label)}</option>`).join("");
    let html = editable ? `<div class="review-bar"><select id="review-model" aria-label="审阅模型">${models}</select><button class="btn" data-act="rereview">重新审阅</button>${chip(verdict[0], verdict[1])}</div>`
      : `<div class="review-bar"><span class="muted-text">审阅结论</span>${chip(verdict[0], verdict[1])}</div>`;
    html += `<p class="summary"><b>结论：</b>${esc(llm.ok ? llm.summary : llm.error)}</p>`;
    if (llm.ok) html += `<p class="muted-text">审阅范围：${llm.files} 个文件 · ${llm.seconds} 秒 · ${fmtTime(rv.at)}</p>`;
    if (llm.ok && llm.findings.length) {
      html += llm.findings.map((f) => {
        const high = f.level === "danger";
        return `<div class="finding ${high ? "danger" : "warn"}"><div class="finding-head"><span class="where">${esc(f.file)}${f.line ? ` 第 ${f.line} 行` : ""}</span>${chip(high ? "风险：高" : "风险：中", high ? "danger" : "warn")}</div>` +
          (f.snippet && f.snippet.length ? `<pre class="snippet">${f.snippet.map((x) => `<span class="${x.hit ? "hit" : ""}"><b>${x.n}</b>${esc(x.text)}</span>`).join("\n")}</pre>` : "") +
          `<div class="why"><b>问题：</b>${esc(f.reason)}</div>${f.suggestion ? `<div class="why"><b>建议：</b>${esc(f.suggestion)}</div>` : ""}</div>`;
      }).join("");
    }
    if (rv.confirmedBy) html += `<p class="muted-text">${fmtTime(rv.confirmedBy.at)} ${esc(rv.confirmedBy.by || "管理员")}人工确认后继续。</p>`;
    if (editable && t.step === "review" && t.state === "waiting") html += `<div class="actions"><button class="btn primary" data-act="confirm" data-decision="continue">确认无问题，继续</button><button class="btn danger" data-act="confirm" data-decision="block">拦截</button></div>`;
    return html;
  }

  const countChip = (list) => (list && list.length ? chip(`${list.length} 次`, "danger") : chip("无", "ok"));

  function trialBody(t) {
    const tr = t.steps.trial;
    if (!tr) return t.step === "trial" ? `<p class="muted-text">正在试运行…</p>` : "";
    const ev = tr.events || {};
    const files = tr.files || [];
    const bytes = files.reduce((n, f) => n + (f.bytes || 0), 0);
    const result = tr.timedOut ? chip("运行正常，到时停止", "ok") : tr.exitCode === 0 ? chip("正常结束（退出码 0）", "ok") : chip(`异常退出（退出码 ${tr.exitCode}）`, "warn");
    const procs = (ev.process || []).map((x) => String(x).replace(/[[\]']/g, "").split(",")[0]).filter(Boolean);
    const rows = [
      ["运行结果", result],
      ["运行时长", esc(fmtDur(tr.seconds))],
      ["峰值内存", tr.maxRssMb != null ? esc(fmtBytes(tr.maxRssMb * 1024 * 1024)) : "—"],
      ["联网尝试", countChip(ev.network)],
      ["越界写入", countChip(ev.writes)],
      ["启动的子进程", procs.length ? `${procs.length} 个 <small>${esc([...new Set(procs)].join("、"))}</small>` : "无"],
      ["产出文件", `${files.length} 个，共 ${esc(fmtBytes(bytes))}`],
      ["产出夹带输入数据", tr.canary && tr.canary.length ? chip(`检出：${tr.canary.join("、")}`, "danger") : chip("未检出", "ok")]
    ];
    return `<table class="table mini"><tbody>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</tbody></table>` +
      (tr.logTail && tr.logTail.length ? `<h4 class="sub-title">运行输出</h4><pre class="log small">${esc(tr.logTail.join("\n"))}</pre>` : "");
  }

  function compileBody(t) {
    const c = t.steps.compile;
    if (!c) return t.step === "compile" ? `<p class="muted-text">正在用同形状的模拟数据编译…</p>` : "";
    if (c.error) return `<p class="reason warn">${esc(c.error)}</p>`;
    return `<dl class="kv"><dt>编译用时</dt><dd>${c.seconds} 秒</dd>${c.cost && c.cost.triples ? `<dt>乘法（上限）</dt><dd>${fmtCount(c.cost.triples)}</dd><dt>通信轮数（上限）</dt><dd>${fmtCount(c.cost.rounds)}</dd>` : ""}</dl>`;
  }

  function consentBody(t, editable = true) {
    const f = flowOf(t);
    if (f.indexOf(t.step) < f.indexOf("consent") && t.state !== "done") return "";
    const dec = (t.steps.consent || {}).decisions || {};
    const canAct = editable && t.step === "consent" && t.state === "waiting";
    return `<table class="table mini"><tbody>${t.datasets.map((id) => {
      const d = dsById(id) || { name: id, owner: "—", ownership: "own" }, x = dec[id];
      const who = d.ownership === "partner" ? d.partner : "本方";
      const st = x ? (x.decision === "approve" ? chip("已同意", "ok") : chip("已拒绝", "danger")) : chip("待审批", "warn");
      const btns = canAct && !x ? `<button class="btn primary" data-act="approve" data-ds="${esc(id)}">同意</button> <button class="btn danger" data-act="reject" data-ds="${esc(id)}">拒绝</button>` : "";
      return `<tr><td>${esc(who)}<small>${esc(d.name)}</small></td><td>${st}</td><td>${btns}</td></tr>`;
    }).join("")}</tbody></table>`;
  }

  // 相同内容保留原节点，避免轮询关闭下拉框或重置详情日志的滚动位置。
  const detailHtml = new WeakMap();
  function updateDetail(selector, html) {
    const el = $(selector);
    if (detailHtml.get(el) === html) return;
    el.innerHTML = html;
    detailHtml.set(el, html);
  }

  function renderReviewDetail() {
    const t = taskById(ui.reviewTask);
    if (!t) { updateDetail("#review-detail", `<p class="muted-text">选择左侧的任务查看审批详情。</p>`); return; }
    const [s, l] = status(t);
    const sec = (key, title, body) => { const [st, sl] = sectionStatus(t, key); return `<div class="detail-section"><div class="section-head"><h4>${title}</h4>${chip(st, sl)}</div>${body}</div>`; };
    const kv = [["提交人", esc(t.requester)], ["方式", modeChip(t)], ["提交时间", fmtTime(t.createdAt)], ["数据", t.datasets.map((id) => esc((dsById(id) || {}).name || id)).join("、")]];
    if (t.mode === "local") kv.push(["代码", `${t.code.files} 个文件 · <span class="mono">${esc(t.code.short || "")}</span>`], ["运行命令", `<code class="cmd">${esc(t.local.command)}</code>`]);
    else kv.push(["MPC 程序", `<span class="mono">${esc(t.mpc.program)}</span>`]);
    let actions = "";
    if (["blocked", "needs-fix"].includes(t.state)) actions += `<button class="btn" data-act="return">退回修改</button>`;
    if (ACTIVE.includes(t.state) && !(t.step === "review" && t.state === "waiting")) actions += `<button class="btn danger" data-act="cancel">终止任务</button>`;
    if (pastTrain(t)) actions += `<button class="btn" data-goto-monitor="${esc(t.id)}">查看训练</button>`;
    updateDetail("#review-detail", `
      <div class="panel-head"><h3>${esc(t.title)}</h3>${chip(s, l)}</div>
      <dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
      ${t.reason ? `<p class="reason ${["needs-fix", "waiting"].includes(t.state) ? "warn" : ""}">${esc(t.reason)}</p>` : ""}
      ${sec("review", "代码审阅", reviewBody(t))}
      ${t.mode === "local" ? sec("trial", "试运行", trialBody(t)) : sec("compile", "编译检查", compileBody(t)) + sec("consent", "合作方授权", consentBody(t))}
      ${actions ? `<div class="actions">${actions}</div>` : ""}`);
  }

  // ---------------------------------------------------------------------------
  // 我的任务（使用者）：看进度、看审阅意见、撤回或修改后重新提交，不能审批
  // ---------------------------------------------------------------------------
  const MY_FILTERS = [
    ["all", "全部", () => true],
    ["active", "进行中", (t) => ACTIVE.includes(t.state)],
    ["fix", "需修改", needsMe],
    ["done", "已完成", (t) => t.state === "done"]
  ];

  function renderMyTasks() {
    const tasks = ui.data.tasks;
    $("#my-filters").innerHTML = MY_FILTERS.map(([k, label, fn]) => `<button class="filter${ui.myFilter === k ? " is-active" : ""}" data-my-filter="${k}">${label} ${tasks.filter(fn).length}</button>`).join("");
    const rows = tasks.filter(MY_FILTERS.find(([k]) => k === ui.myFilter)[2]);
    if (!rows.some((t) => t.id === ui.myTask)) ui.myTask = rows.length ? rows[0].id : null;
    $("#my-table").innerHTML = "<thead><tr><th>任务</th><th>方式</th><th>当前步骤</th><th>状态</th></tr></thead><tbody>" +
      (rows.length ? rows.map((t) => {
        const [s, l] = status(t);
        return `<tr class="is-selectable${t.id === ui.myTask ? " is-selected" : ""}" data-my="${esc(t.id)}"><td><strong>${esc(t.title)}</strong><small>${esc(t.id)} · ${fmtTime(t.createdAt)}</small></td><td>${modeChip(t)}</td><td>${esc(stepName(t.mode, t.step))}</td><td>${chip(s, l)}</td></tr>`;
      }).join("") : `<tr><td colspan="4" class="table-empty">${tasks.length ? "没有符合条件的任务" : "还没有提交过任务"}</td></tr>`) + "</tbody>";
    renderMyDetail();
  }

  function myNotice(t) {
    if (t.state === "waiting") {
      if (t.step === "review") return ["代码审阅发现需要确认的内容，已转管理员人工确认。", "warn"];
      if (t.step === "consent") return ["代码已通过审阅和编译检查，正在等待数据方授权。", "warn"];
      return ["训练已结束，等待管理员批准交付模型。", "warn"];
    }
    if (["needs-fix", "returned"].includes(t.state)) return [`需要修改：${t.reason || "请根据审阅意见修改代码"}。修改后重新提交。`, "warn"];
    if (t.state === "blocked") return [`已被拦截：${t.reason || "代码存在数据外带风险"}。请根据审阅意见修改后重新提交。`, ""];
    if (["rejected", "failed", "cancelled"].includes(t.state)) return [t.reason || status(t)[0], ""];
    if (t.state === "done") return [t.mode === "mpc" ? "模型已批准交付，可在“训练结果”中下载模型参数。" : "训练完成，产出已通过检查，可在“训练结果”中下载。", "ok"];
    return null;
  }

  function myProgress(t) {
    return `<ol class="step-track">${flowOf(t).map((key) => {
      const [st, sl] = sectionStatus(t, key);
      const at = (t.steps[key] || {}).at;
      return `<li><span class="dot ${sl}"></span><strong>${esc(stepName(t.mode, key))}</strong>${chip(st, sl)}<small>${at ? fmtTime(at) : ""}</small></li>`;
    }).join("")}</ol>`;
  }

  function renderMyDetail() {
    const t = taskById(ui.myTask);
    if (!t) { updateDetail("#my-detail", `<p class="muted-text">选择左侧的任务查看处理进度。</p>`); return; }
    const [s, l] = status(t);
    const kv = [["方式", modeChip(t)], ["提交时间", fmtTime(t.createdAt)], ["数据", t.datasets.map((id) => esc((dsById(id) || {}).name || id)).join("、")]];
    if (t.mode === "local") kv.push(["代码", `${t.code.files} 个文件 · <span class="mono">${esc(t.code.short || "")}</span>`], ["运行命令", `<code class="cmd">${esc(t.local.command)}</code>`]);
    else kv.push(["MPC 程序", `<span class="mono">${esc(t.mpc.program)}</span>`]);
    const note = myNotice(t);
    const sec = (title, body) => (body ? `<div class="detail-section"><div class="section-head"><h4>${title}</h4></div>${body}</div>` : "");
    let actions = "";
    if (ACTIVE.includes(t.state) && !pastTrain(t)) actions += `<button class="btn danger" data-act="cancel" data-task="${esc(t.id)}">撤回任务</button>`;
    if (needsMe(t)) actions += `<button class="btn primary" data-resubmit="${esc(t.id)}">修改后重新提交</button>`;
    if (pastTrain(t)) actions += `<button class="btn" data-goto-monitor="${esc(t.id)}">查看训练结果</button>`;
    updateDetail("#my-detail", `
      <div class="panel-head"><h3>${esc(t.title)}</h3>${chip(s, l)}</div>
      <dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>
      ${note ? `<p class="reason ${note[1]}">${esc(note[0])}</p>` : ""}
      ${sec("处理进度", myProgress(t))}
      ${t.steps.review ? sec("审阅意见", reviewBody(t, false)) : ""}
      ${t.mode === "local" ? sec("试运行结果", trialBody(t)) : sec("编译检查", compileBody(t)) + sec("数据方授权", consentBody(t, false))}
      ${actions ? `<div class="actions">${actions}</div>` : ""}`);
  }

  function resubmit(id) {
    const t = taskById(id);
    if (!t) return;
    // 先填充环境等选项，再恢复上次保存的值。
    renderNew();
    if (t.mode === "local") {
      ui.tab = "local";
      if (t.code.root) ui.localProject = t.code.root;
      ui.localPicked = new Set(t.datasets);
      $("#local-command").value = t.local.command;
      $("#local-env").value = t.local.env || "";
      $("#local-trial").value = t.local.trialSeconds ?? 120;
    } else {
      ui.tab = "mpc";
      const p = t.datasets.map(dsById).find((d) => d && d.ownership === "partner");
      if (p) ui.partner = p.partner;
      ui.mpcPicked = new Set(t.datasets);
      ui.mpcProject = null;
      ui.mpcProjectNeedsPick = true;
      // 服务端只保存程序文件名；同名程序可能来自不同项目，不能自动猜选。
      ui.mpcProgramNeedsPick = t.mpc.program || "未记录";
      const params = t.mpc.params || {};
      $("#p-protocol").value = params.protocol ?? "rep3";
      $("#p-precision").value = params.precision ?? "standard";
      $("#p-downsample").value = params.downsample ?? 4;
      $("#p-perclass").value = params.perClass ?? 32;
    }
    ui.sig = {};
    navigate("new");
    renderNew();
    toast("已带入上次的设置", t.mode === "mpc" ? "数据集和参数已恢复；请确认项目并重新选择 MPC 程序" : "运行命令、Python 环境和试运行时长已恢复，请确认后提交");
  }

  // ---------------------------------------------------------------------------
  // 训练监控
  // ---------------------------------------------------------------------------
  function renderMonitor() {
    const list = ui.data.tasks.filter(pastTrain);
    if (!list.some((t) => t.id === ui.monitorTask)) ui.monitorTask = list.length ? list[0].id : null;
    $("#monitor-table").innerHTML = "<thead><tr><th>任务</th><th>方式</th><th>进度</th><th>结果</th><th>状态</th></tr></thead><tbody>" +
      (list.length ? list.map((t) => {
        const [s, l] = status(t), tr = t.steps.train || {};
        const prog = t.mode === "mpc" ? `${tr.epoch || 0} / ${tr.total || "?"} 轮` : tr.progress ? `${tr.progress.epoch} / ${tr.progress.total} 轮` : tr.seconds != null ? fmtDur(tr.seconds) : "—";
        const result = t.mode === "mpc" ? (tr.test ? `测试准确率 ${pct(tr.test.acc)}` : "—") : t.steps.check ? `${t.steps.check.files.length} 个产出文件` : "—";
        return `<tr class="is-selectable${t.id === ui.monitorTask ? " is-selected" : ""}" data-monitor="${esc(t.id)}"><td><strong>${esc(t.title)}</strong><small>${esc(t.id)}</small></td><td>${modeChip(t)}</td><td class="num">${esc(prog)}</td><td>${esc(result)}</td><td>${chip(s, l)}</td></tr>`;
      }).join("") : `<tr><td colspan="5" class="table-empty">还没有通过审批的任务</td></tr>`) + "</tbody>";
    const t = taskById(ui.monitorTask);
    $("#monitor-local").hidden = !t || t.mode !== "local";
    $("#monitor-mpc").hidden = !t || t.mode !== "mpc";
    if (!t) return;
    if (t.mode === "local") renderLocal(t); else renderMpc(t);
  }

  function setLog(sel) {
    const log = $(sel);
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
    log.textContent = ui.log.length ? ui.log.join("\n") : "暂无输出";
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function renderLocal(t) {
    const tr = t.steps.train || {}, ck = t.steps.check;
    const [s, l] = status(t);
    const live = t.state === "running" && t.step === "train";
    const elapsed = tr.startedAt ? (live ? (Date.now() - Date.parse(tr.startedAt)) / 1000 : tr.seconds) : null;
    $("#local-title").textContent = t.title;
    $("#local-chip").innerHTML = chip(s, l) + (["running", "queued"].includes(t.state) ? ` <button class="btn danger" data-act="cancel" data-task="${esc(t.id)}">终止</button>` : "");
    $("#local-metrics").innerHTML = [
      ["当前步骤", stepName("local", t.step)],
      ["训练轮次", tr.progress ? `${tr.progress.epoch} / ${tr.progress.total}` : "—"],
      ["已用时间", fmtDur(elapsed)]
    ].map(([k, v]) => `<div class="metric"><span>${k}</span><strong>${esc(v)}</strong></div>`).join("");
    $("#local-progress").style.width = tr.progress ? `${Math.min(100, (tr.progress.epoch / tr.progress.total) * 100)}%` : (t.state === "done" ? "100%" : "0%");
    renderChart("#local-chart", tr.history || [], [["loss", "line-loss", "pt-loss"], ["valAcc", "line-acc", "pt-acc"], ["trainAcc", "line-train", "pt-train"]], tr.progress && tr.progress.total);
    setLog("#local-log");
    if (ck && ck.files.length) {
      const ok = t.state === "done";
      const get = ok && !isAdmin();
      $("#local-outputs").innerHTML = `<table class="table mini"><tbody>${ck.files.map((f) => `<tr><td><span class="small-path">${esc(f.path.split("/").pop())}</span><small>${fmtBytes(f.bytes)}</small></td><td>${f.realData ? chip("含原始数据", "danger") : chip("安全", "ok")}</td><td>${get ? `<a class="link" href="/api/tasks/${encodeURIComponent(t.id)}/output?path=${encodeURIComponent(f.path)}">下载</a>` : ""}</td></tr>`).join("")}</tbody></table>${ok ? (isAdmin() ? `<p class="muted-text">已放行给提交人 ${esc(t.requester)}</p>` : "") : `<p class="reason">${esc(t.reason || "产出未放行")}</p>`}`;
    } else {
      $("#local-outputs").innerHTML = `<p class="muted-text">${t.step === "train" ? "训练结束后检查产出，没有原始数据才能下载。" : "没有产出文件"}</p>`;
    }
    const ev = tr.events || {};
    $("#local-safety").innerHTML = [["网络", chip("已断开", "ok")], ["数据", chip("只读", "ok")], ["联网尝试", countChip(ev.network)], ["越界写入", countChip(ev.writes)]]
      .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  }

  // series：[[字段, 线条样式, 点样式]]，所有序列共用一个纵轴（损失与 0–1 的准确率）
  function renderChart(sel, hist, series, total) {
    const W = 640, H = 200, L = 34, R = 10, T = 10, B = 22;
    const maxEpoch = Math.max(2, total || 0, ...hist.map((h) => h.epoch));
    const maxY = Math.max(1, ...hist.flatMap((h) => series.map(([k]) => h[k] || 0)));
    const x = (e) => L + ((e - 1) / Math.max(1, maxEpoch - 1)) * (W - L - R);
    const y = (v) => T + (1 - v / maxY) * (H - T - B);
    const grid = [0, maxY / 2, maxY].map((v) => `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="axis-text" x="2" y="${y(v) + 3}">${v.toFixed(2)}</text>`).join("");
    const ticks = [...new Set([1, Math.ceil(maxEpoch / 2), maxEpoch])].map((e) => `<text class="axis-text" x="${x(e) - 4}" y="${H - 5}">${e}</text>`).join("");
    const line = ([k, cls, pc]) => { const p = hist.filter((h) => h[k] != null); return `<polyline class="${cls}" points="${p.map((h) => `${x(h.epoch).toFixed(1)},${y(h[k]).toFixed(1)}`).join(" ")}"/>` + p.map((h) => `<circle class="${pc}" cx="${x(h.epoch).toFixed(1)}" cy="${y(h[k]).toFixed(1)}" r="3"/>`).join(""); };
    $(sel).innerHTML = grid + ticks + series.map(line).join("") + (hist.length ? "" : `<text class="axis-text" x="${W / 2 - 40}" y="${H / 2}">等待第一轮结束</text>`);
  }

  function renderMpc(t) {
    const tr = t.steps.train || {}, hist = tr.history || [], last = hist[hist.length - 1] || {};
    const [s, l] = status(t);
    const live = t.state === "running" && t.step === "train";
    const elapsed = tr.startedAt ? (live ? (Date.now() - Date.parse(tr.startedAt)) / 1000 : tr.seconds) : null;
    $("#mpc-title").textContent = t.title;
    $("#mpc-chip").innerHTML = chip(s, l) + (["running", "queued"].includes(t.state) ? ` <button class="btn danger" data-act="cancel" data-task="${esc(t.id)}">终止</button>` : "");
    $("#mpc-metrics").innerHTML = [
      ["已完成轮次", `${tr.epoch || 0} / ${tr.total || "?"}`], ["验证准确率", pct(last.valAcc)], ["验证损失", num(last.valLoss)],
      ["已用时间", fmtDur(elapsed)], ["测试准确率", tr.test ? `${pct(tr.test.acc)}（${tr.test.correct}/${tr.test.total}）` : "—"], ["阶段", live && tr.phase ? tr.phase : status(t)[0]]
    ].map(([k, v]) => `<div class="metric"><span>${k}</span><strong>${esc(v)}</strong></div>`).join("");
    $("#mpc-progress").style.width = tr.total ? `${Math.min(100, ((tr.epoch || 0) / tr.total) * 100)}%` : "0%";
    renderChart("#mpc-chart", hist, [["valLoss", "line-loss", "pt-loss"], ["valAcc", "line-acc", "pt-acc"]], tr.total);
    const proto = ui.data.platform.protocols[t.mpc.params.protocol];
    const owners = [...new Set(t.datasets.map((id) => { const d = dsById(id); return d && d.ownership === "partner" ? d.partner : "本方"; }))];
    const parties = tr.parties && tr.parties.length ? tr.parties : Array.from({ length: proto.parties }, (_, i) => ({ name: `P${i}`, status: t.state === "queued" ? "等待启动" : "—" }));
    $("#party-table").innerHTML = "<tbody>" + parties.map((x, i) => `<tr><td>${esc(x.name)}<small>${esc(owners[i] || (i === 0 ? "本方" : "计算方"))}</small></td><td>${chip(x.status, x.status === "运行中" ? "info" : x.status === "已结束" ? "ok" : String(x.status).startsWith("异常") ? "danger" : "")}</td></tr>`).join("") + "</tbody>";
    let delivery;
    if (t.step === "deliver" && t.state === "waiting") delivery = isAdmin()
      ? `<p class="muted-text">训练已结束，批准后提交人 ${esc(t.requester)} 可下载模型参数。</p><div class="actions"><button class="btn primary" data-act="deliver" data-task="${esc(t.id)}">批准交付</button></div>`
      : `<p class="muted-text">训练已结束，等待管理员批准交付。</p>`;
    else if (t.state === "done") delivery = isAdmin()
      ? `<p class="muted-text">${fmtTime(t.steps.deliver.at)} 已批准交付给 ${esc(t.requester)}</p>`
      : `<p class="muted-text">${fmtTime(t.steps.deliver.at)} 已批准交付</p><div class="actions"><a class="btn primary" href="/api/tasks/${encodeURIComponent(t.id)}/model">下载模型参数</a></div>`;
    else if (STOPPED.includes(t.state)) delivery = `<p class="reason">${esc(t.reason || "训练未完成")}</p>`;
    else delivery = `<p class="muted-text">${isAdmin() ? "训练结束后在这里批准交付。" : "训练结束并经管理员批准后，在这里下载模型参数。"}</p>`;
    $("#mpc-delivery").innerHTML = delivery;
    const base = tr.baselineSeconds, mpcS = tr.mpcSeconds, total = tr.seconds;
    if (mpcS) {
      const max = Math.max(total || 0, mpcS);
      const bar = (label, v, cls, note) => `<div class="cmp-row"><span>${label}</span><div class="cmp-track"><i class="${cls}" style="width:${Math.max(0.6, (v / max) * 100).toFixed(1)}%"></i></div><b>${esc(v < 10 ? v.toFixed(2) : Math.round(v))} 秒</b>${note ? `<small>${esc(note)}</small>` : ""}</div>`;
      $("#mpc-compare").innerHTML = (base != null ? bar("明文训练", base, "plain", "同模型、同数据、同轮数") : "") +
        bar("MPC 计算", mpcS, "mpc", base ? `约 ${Math.round(mpcS / base)} 倍` : "") +
        (total ? bar("MPC 全程", total, "total", `含编译 ${tr.compileSeconds || 0} 秒`) : "") +
        `<p class="muted-text">通信量：${esc(fmtBytes((tr.globalDataMb || 0) * 1024 * 1024))}（各方合计）</p>`;
    } else {
      $("#mpc-compare").innerHTML = `<p class="muted-text">训练结束后显示与明文训练的耗时对比。</p>`;
    }
    setLog("#mpc-log");
  }

  async function refreshLog() {
    if (ui.view !== "monitor" || !ui.monitorTask) return;
    try { ui.log = (await api.get(`/api/tasks/${encodeURIComponent(ui.monitorTask)}/log`)).lines; } catch { ui.log = []; }
  }

  // ---------------------------------------------------------------------------
  // 审计
  // ---------------------------------------------------------------------------
  function renderAudit() {
    const filters = ["全部", "数据", "本地训练", "联合训练"];
    $("#audit-filters").innerHTML = filters.map((f) => `<button class="filter${ui.auditFilter === f ? " is-active" : ""}" data-audit="${f}">${f}</button>`).join("");
    const rows = ui.data.audit.filter((a) => ui.auditFilter === "全部" || a.module === ui.auditFilter);
    $("#audit-table").innerHTML = "<thead><tr><th>时间</th><th>模块</th><th>主体</th><th>事件</th><th>结果</th></tr></thead><tbody>" +
      (rows.length ? rows.map((a) => `<tr><td class="num">${fmtTime(a.time)}</td><td>${esc(a.module)}</td><td>${esc(a.actor)}</td><td>${esc(a.event)}</td><td>${chip(a.result, a.level)}</td></tr>`).join("") : `<tr><td colspan="5" class="table-empty">暂无事件</td></tr>`) + "</tbody>";
  }

  // ---------------------------------------------------------------------------
  // 刷新、导航、事件
  // ---------------------------------------------------------------------------
  async function refresh() {
    try {
      ui.data = await api.get("/api/state");
      const switched = !ui.me || ui.me.id !== ui.data.me.id;
      ui.me = ui.data.me;
      if (switched) applyRole();
      if (!isAdmin() && (!ui.projects || ui.view === "new" && Date.now() - (ui.projectsAt || 0) > 10000)) {
        ui.projects = (await api.get("/api/projects")).items;
        ui.projectsAt = Date.now();
      }
      $("#banner").hidden = true;
      await refreshLog();
      renderEnv(); renderData(); renderMonitor();
      if (isAdmin()) { renderOverview(); renderReview(); renderAudit(); }
      else { renderHome(); renderNew(); renderMyTasks(); }
    } catch (err) {
      if (err.status === 401) { showLogin(); return; }
      $("#banner").hidden = false;
      $("#banner").textContent = `无法连接平台服务（${err.message}）。请确认服务器上的 node server.cjs 正在运行。`;
    }
  }

  // ---------------------------------------------------------------------------
  // 登录与角色
  // ---------------------------------------------------------------------------
  function applyRole() {
    $("#login").hidden = true;
    $("#app").hidden = false;
    $$("[data-role]").forEach((el) => { el.hidden = el.dataset.role !== ui.me.role; });
    navigate(ROLE_VIEWS[ui.me.role].includes(ui.view) ? ui.view : ROLE_VIEWS[ui.me.role][0]);
  }

  async function showLogin() {
    ui.me = null;
    ui.data = null;
    $("#app").hidden = true;
    $("#login").hidden = false;
    try {
      const { items } = await api.get("/api/accounts");
      const what = { user: "提交训练任务，查看自己任务的审批进度和训练结果", admin: "审批任务、登记数据集、监控训练、查看历史记录" };
      $("#login-accounts").innerHTML = items.map((a) => `<button class="login-item" data-login="${esc(a.id)}"><span><strong>${esc(a.name)}</strong>${chip(a.roleLabel, a.role === "admin" ? "info" : "ok")}</span><small>${esc(what[a.role])}</small></button>`).join("");
      $("#login-hint").textContent = "";
    } catch (err) {
      $("#login-hint").textContent = `无法连接平台服务（${err.message}）`;
    }
  }

  async function login(id) {
    try {
      await api.post("/api/login", { id });
      ui.me = null;
      ui.view = "";
      ui.projects = null;
      ui.sig = {};
      await refresh();
    } catch (err) {
      $("#login-hint").textContent = err.message;
    }
  }

  async function logout() {
    try { await api.post("/api/logout"); } catch { /* 会话已失效 */ }
    Object.assign(ui, { reviewTask: null, myTask: null, monitorTask: null, localPicked: null, localProject: null, mpcProject: null, mpcProjectNeedsPick: false, mpcProgramNeedsPick: "", partner: null, mpcPicked: new Set(), projects: null, log: [], sig: {} });
    showLogin();
  }

  function navigate(view) {
    const allowed = ui.me ? ROLE_VIEWS[ui.me.role] : Object.keys(TITLES);
    if (!allowed.includes(view)) view = allowed[0];
    ui.view = view;
    $$("[data-page]").forEach((p) => p.classList.toggle("is-active", p.dataset.page === view));
    $$(".nav-item").forEach((n) => n.classList.toggle("is-active", n.dataset.nav === view));
    $("#page-title").textContent = (ui.me && ROLE_TITLES[ui.me.role][view]) || TITLES[view];
    try { history.replaceState(null, "", view === "overview" ? location.pathname : `#${view}`); } catch { /* 忽略 */ }
    window.scrollTo({ top: 0 });
    if (view === "monitor") refreshLog().then(() => ui.data && renderMonitor());
  }

  let toastTimer;
  function toast(title, message) {
    $("#toast-title").textContent = title;
    $("#toast-message").textContent = message;
    $("#toast").classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $("#toast").classList.remove("is-visible"), 3600);
  }

  async function act(action, el) {
    const id = el.dataset.task || (isAdmin() ? ui.reviewTask : ui.myTask);
    try {
      if (action === "approve" || action === "reject") { await api.post(`/api/tasks/${encodeURIComponent(id)}/consent`, { dataset: el.dataset.ds, decision: action }); toast(action === "approve" ? "已同意" : "已拒绝", id); }
      else if (action === "rereview") {
        const model = ($("#review-model") || {}).value || "deepseek";
        toast("正在重新审阅", "请稍候…");
        await api.post(`/api/tasks/${encodeURIComponent(id)}/review`, { model });
        toast("审阅完成", id);
      }
      else if (action === "confirm") { await api.post(`/api/tasks/${encodeURIComponent(id)}/confirm`, { decision: el.dataset.decision }); toast(el.dataset.decision === "block" ? "已拦截" : "已确认", id); }
      else if (action === "return") { await api.post(`/api/tasks/${encodeURIComponent(id)}/return`); toast("已退回", id); }
      else if (action === "cancel") { await api.post(`/api/tasks/${encodeURIComponent(id)}/cancel`); toast(isAdmin() ? "已终止" : "已撤回", id); }
      else if (action === "deliver") { await api.post(`/api/tasks/${encodeURIComponent(id)}/deliver`); toast("已批准交付", id); }
    } catch (err) { toast("操作失败", err.message); }
    refresh();
  }

  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-nav],[data-open],[data-review],[data-review-filter],[data-act],[data-monitor],[data-goto-monitor],[data-audit],[data-edit-ds],[data-tab],[data-login],[data-logout],[data-my],[data-my-filter],[data-resubmit]");
    if (!el) return;
    const d = el.dataset;
    if (d.login) login(d.login);
    else if (d.logout) logout();
    else if (d.my) { ui.myTask = d.my; if (ui.view !== "mytasks") navigate("mytasks"); renderMyTasks(); }
    else if (d.myFilter) { ui.myFilter = d.myFilter; renderMyTasks(); }
    else if (d.resubmit) resubmit(d.resubmit);
    else if (d.nav) { navigate(d.nav); refresh(); }
    else if (d.tab) { ui.tab = d.tab; renderNew(); }
    else if (d.open) {
      const t = taskById(d.open);
      if (t && pastTrain(t)) { ui.monitorTask = t.id; ui.log = []; navigate("monitor"); } else { ui.reviewTask = d.open; ui.reviewFilter = "all"; navigate("review"); }
      refresh();
    } else if (d.review) { ui.reviewTask = d.review; renderReview(); }
    else if (d.reviewFilter) { ui.reviewFilter = d.reviewFilter; renderReview(); }
    else if (d.act) act(d.act, el);
    else if (d.monitor) { ui.monitorTask = d.monitor; ui.log = []; renderMonitor(); refreshLog().then(renderMonitor); }
    else if (d.gotoMonitor) { ui.monitorTask = d.gotoMonitor; ui.log = []; navigate("monitor"); renderMonitor(); }
    else if (d.audit) { ui.auditFilter = d.audit; renderAudit(); }
    else if (d.editDs) startEdit(d.editDs);
  });
  document.addEventListener("change", (e) => {
    const el = e.target, d = el.dataset || {};
    if (d.pickKind) onPick(el);
    else if (el.id === "mpc-program") { if (el.value) ui.mpcProgramNeedsPick = ""; renderNew(); }
    else if (el.id === "local-env") renderNew();
    else if (el.id === "p-protocol") renderNew();
    else if (el.id === "review-model") ui.reviewModel = el.value;
  });
  document.addEventListener("input", (e) => { if (e.target.id === "local-command") renderNew(); });
  $("#local-submit").addEventListener("click", () => submit("local"));
  $("#mpc-submit").addEventListener("click", () => submit("mpc"));
  $("#register-form").addEventListener("submit", submitRegister);
  $("#reg-cancel").addEventListener("click", endEdit);

  const clock = () => { const d = new Date(); $("#clock").textContent = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
  clock();
  setInterval(clock, 1000);
  ui.view = location.hash.replace("#", "");
  refresh();
  setInterval(() => { if (ui.me) refresh(); }, 2000);
})();
