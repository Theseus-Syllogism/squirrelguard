/* squirrel-guard console (web GUI). Talks to web.py on the same loopback address.
 * Shared with the persons UI: ui.css, and ui.js for the theme switch, toasts and the confirm dialog (window.sgui).
 * Everything from the server is put on the page with textContent, never as HTML. */
try { const t = localStorage.getItem("sg-replay-theme"); if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; } catch (e) { /* private mode */ }

(() => {
  "use strict";
  const TOKEN = document.querySelector('meta[name="sg-token"]').content;
  const PAGES = ["overview", "install", "config", "endpoints", "secrets", "checks", "service", "events"];
  let main, conn, jobTimer = null, pageSeq = 0;

  /* ------------------------------------------------------------ helpers */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k in el && k !== "list" && k !== "type") el[k] = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
    return el;
  }
  const toast = (m, kind, ms) => window.sgui ? window.sgui.toast(m, kind, ms) : console.log(m);
  const chip = (tone, text) => h("span", { class: "chip " + tone }, text);
  const show = (v) => typeof v === "boolean" ? (v ? "yes" : "no") : Array.isArray(v) ? v.join(", ") : v == null ? "" : String(v);

  async function api(path, body) {
    const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-SG-Token": TOKEN }, body: JSON.stringify(body) };
    let res;
    try { res = await fetch(path, opt); }
    catch (e) { conn.textContent = "Console stopped: run `sg gui` again"; throw new Error("The console is not answering. Is `sg gui` still running?"); }
    conn.textContent = "";
    const type = res.headers.get("Content-Type") || "";
    const data = type.includes("json") ? await res.json() : { error: (await res.text()).trim() };
    if (!res.ok) throw new Error(data.error || data.output || "The request failed (" + res.status + ").");
    return data;
  }
  async function busy(btn, fn) {                       // spinner on the button, no double clicks, errors as toasts
    if (btn.disabled) return;
    btn.classList.add("is-busy"); btn.disabled = true; btn.setAttribute("aria-busy", "true");
    try { return await fn(); }
    catch (e) { toast(e.message, "err", 7000); }
    finally { btn.classList.remove("is-busy"); btn.disabled = false; btn.removeAttribute("aria-busy"); }
  }
  const confirmBox = (o) => window.sgui ? window.sgui.confirmDialog(o) : Promise.resolve(true);
  function head(title, ...right) { return h("div", { class: "head" }, h("h2", {}, title), h("span", { class: "grow" }), ...right); }

  /* ------------------------------------------------------------ overview */
  async function overview() {
    const st = await api("/api/status");
    const c = st.checks, cards = [];
    const stat = (label, value, link, text, attn) => h("div", { class: "stat" + (attn ? " attn" : "") }, h("b", {}, value), h("span", {}, label), link ? h("div", {}, h("a", { href: link }, text)) : null);
    cards.push(stat("This machine", st.role_title, "#install", "Change role"));
    cards.push(stat("Installation", `${st.steps_done} of ${st.steps_total} steps`, "#install", st.next_step ? "Continue" : "Review", !!st.next_step));
    cards.push(stat("Checks", c.fail ? `${c.fail} failed` : c.warn ? `${c.warn} warning${c.warn > 1 ? "s" : ""}` : "All passed", "#checks", "Open checks", c.fail > 0));
    if (st.service) {
      cards.push(stat("Detector service", st.service.active ? "Running" : st.service.installed ? "Stopped" : "Not installed", "#service", "Service and log"));
      cards.push(stat("Deterrent", st.dry_run ? "Dry run" : "Armed", "#config", st.dry_run ? "Logs only: change in Configuration" : "Plays on confirmed detections", !st.dry_run));
    }
    if (st.containers) cards.push(stat("Containers", `${st.containers.filter((r) => r.state === "running").length} of ${st.containers.length} running`, "#service", "Open list"));
    const e = st.last_event;
    cards.push(stat("Last event", e ? `${e.label} → ${e.action}` : "None yet", "#events", e ? String(e.time).slice(0, 19).replace("T", " ") : "Open events"));
    return [head("Overview", h("button", { class: "ghost", onclick: () => go("overview", true) }, "Refresh")),
      st.next_step ? h("div", { class: "notice" }, h("b", {}, "Next step: "), st.next_step + ". ", h("a", { href: "#install" }, "Open the install menu")) : null,
      h("div", { class: "stat-row" }, cards),
      h("p", { class: "muted" }, "Folder: ", h("code", {}, st.root), " · every action here is also a command: ", h("code", {}, "sg --help"))];
  }

  /* ------------------------------------------------------------ install */
  async function install() {
    const [st, data] = await Promise.all([api("/api/status"), api("/api/steps")]);
    const steps = data.steps, firstTodo = steps.find((s) => s.done === false && !s.manual);
    const dry = h("input", { type: "checkbox", id: "dry" });
    const out = h("pre", { class: "out", tabindex: "0", "aria-live": "polite" }), outTitle = h("span", {}, "Output");
    const outBox = h("section", { class: "outbox", hidden: true }, h("h3", {}, outTitle), out);

    async function runSteps(ids, btn) {
      await busy(btn, async () => {
        const r = await api("/api/run", { steps: ids, dry_run: dry.checked });
        outBox.hidden = false; out.textContent = ""; outTitle.replaceChildren(h("span", { class: "spin" }), " Running…");
        outBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
        await follow(r.job, out);
        const job = follow.last;
        outTitle.replaceChildren(chip(job.rc === 0 ? "ok" : "fail", job.rc === 0 ? (dry.checked ? "dry run finished" : "finished") : "did not finish"), " Output");
        toast(job.rc === 0 ? (dry.checked ? "Dry run finished: nothing was changed." : "Step finished.") : "The step did not finish: see the output.", job.rc === 0 ? "ok" : "err");
        if (!dry.checked) { const keep = out.textContent, y = scrollY; await go("install", true); const o = document.querySelector(".outbox"); if (o) { o.hidden = false; o.querySelector("pre").textContent = keep; o.querySelector("h3").replaceChildren(chip(job.rc === 0 ? "ok" : "fail", job.rc === 0 ? "finished" : "did not finish"), " Output"); } scrollTo(0, y); }
      });
    }
    async function follow(id, pre) {
      let from = 0;
      for (;;) {
        const j = await api(`/api/job?id=${encodeURIComponent(id)}&from=${from}`);
        if (j.lines.length) { const stick = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8; pre.append(j.lines.join("\n") + "\n"); if (stick) pre.scrollTop = pre.scrollHeight; }
        from = j.next;
        if (j.done) { follow.last = j; return; }
        await new Promise((r) => { jobTimer = setTimeout(r, 600); });
      }
    }

    const items = steps.map((s, i) => {
      const state = s.manual ? (s.done ? chip("ok", "done") : chip("info", "by hand")) : s.done === true ? chip("ok", "done") : s.done === false ? chip("warn", "to do") : chip("", "not checked");
      let action = null, ask = null;
      if (s.needs) {
        const inp = h("input", { type: "password", autocomplete: "off", placeholder: s.needs_help, "aria-label": s.needs_help });
        const save = h("button", { class: s.done ? "" : "primary", onclick: () => busy(save, async () => {
          if (!inp.value.trim()) { inp.focus(); throw new Error("Enter the value first."); }
          await api("/api/run", { steps: [s.id], answer: inp.value }); inp.value = ""; toast("Stored privately. The value is not shown again."); await go("install", true);
        }) }, s.done ? "Replace" : "Save");
        inp.addEventListener("keydown", (e) => { if (e.key === "Enter") save.click(); });
        ask = h("div", { class: "ask" }, inp, save);
      } else if (!s.manual) {
        const btn = h("button", { class: s === firstTodo ? "primary" : "", onclick: async () => {
          if (s.done && !dry.checked && !await confirmBox({ title: "Run this step again?", text: `“${s.title}” is already done. Running it again is safe but may take a while.`, ok: "Run again", danger: false, icon: "info" })) return;
          runSteps([s.id], btn);
        } }, s.done ? "Run again" : "Run");
        action = h("div", { class: "do" }, btn);
      }
      return h("li", { class: "step" + (s.done ? " done" : "") + (s === firstTodo ? " next" : "") },
        h("div", { class: "num", "aria-hidden": "true" }, s.done ? "✓" : String(i + 1)),
        h("div", {}, h("h3", {}, s.title, " ", state), h("p", {}, s.why), s.manual ? h("p", {}, h("b", {}, "To do by hand: "), s.manual) : null,
          s.shown.length ? h("pre", {}, s.shown.map((c) => "$ " + c).join("\n")) : null, ask),
        action);
    });

    const roles = h("div", { class: "rolepick", role: "group", "aria-label": "What this machine does" }, Object.entries(st.roles).map(([k, r]) => {
      const b = h("button", { "aria-pressed": String(k === st.role), onclick: () => k !== st.role && busy(b, async () => { await api("/api/settings", { key: "role", value: k }); toast("Role: " + r.title); await go("install", true); }) }, h("b", {}, r.title), h("span", {}, r.what));
      return b;
    }));
    const todo = steps.filter((s) => !s.done && !s.manual && !s.needs).map((s) => s.id);
    const all = h("button", { class: "primary", disabled: !todo.length, onclick: async () => {
      if (!dry.checked && !await confirmBox({ title: `Run ${todo.length} step${todo.length > 1 ? "s" : ""}?`, text: "They run in order and stop at the first one that fails. Steps that need a value from you are not included.", ok: "Run", danger: false, icon: "info" })) return;
      runSteps(todo, all);
    } }, todo.length ? `Run remaining (${todo.length})` : "Nothing left to run");
    let runtime = null;
    if (st.role === "detector") {
      runtime = h("div", { class: "rolepick", role: "group", "aria-label": "How the detector runs" }, Object.entries(st.runtimes).map(([k, title]) => {
        const b = h("button", { "aria-pressed": String(k === st.runtime), onclick: () => k !== st.runtime && busy(b, async () => { await api("/api/settings", { key: "runtime", value: k }); toast("Runs as: " + title); await go("install", true); }) },
          h("b", {}, k === "docker" ? "Container" : "On this machine"), h("span", {}, title));
        return b;
      }));
    }
    let folder = null;
    if (st.role === "recorder") {
      const inp = h("input", { value: st.settings.recorder_dir, "aria-label": "Stack folder", size: 28 });
      const b = h("button", { onclick: () => busy(b, async () => { await api("/api/settings", { key: "recorder_dir", value: inp.value }); toast("Stack folder saved."); await go("install", true); }) }, "Save");
      folder = h("p", {}, h("label", {}, "Stack folder: ", inp), " ", b);
    }
    return [head("Install", h("label", {}, dry, " Dry run (show only)"), all),
      h("p", { class: "lead" }, "Choose what this machine does, then work down the list. Each step says what it will do before it does it; finished steps can be run again safely."),
      roles, runtime, folder, h("ol", { class: "steps" }, items), outBox];
  }

  /* ------------------------------------------------------------ configuration */
  async function config() {
    const data = await api("/api/config");
    if (!data.exists) return [head("Configuration"), h("div", { class: "notice warn-box" }, "There is no configuration file yet. ", h("a", { href: "#install" }, "Create it from the install menu"), " (the values below are the defaults and are saved to a new file when you change one).")].concat(sections(data));
    return [head("Configuration"), h("p", { class: "lead" }, "Changes are checked, then saved to ", h("code", {}, data.path), " with your comments kept. The previous file stays beside it as a .bak. Restart the service for a change to take effect."), ...sections(data)];
  }
  function sections(data) {
    const groups = new Map();
    for (const f of data.fields) { if (!groups.has(f.section)) groups.set(f.section, []); groups.get(f.section).push(f); }
    return [...groups].map(([name, fields]) => h("section", { class: "section" }, h("h3", {}, "[" + name + "]"), fields.map(fieldRow)));
  }
  function fieldRow(f) {
    const cur = f.in_file ? f.value : f.default, id = "f-" + f.id.replace(/[^a-z0-9]/gi, "-");
    const err = h("div", { class: "err", role: "alert", hidden: true });
    let input, get;
    if (f.kind === "bool") { input = h("input", { type: "checkbox", id, checked: !!cur }); get = () => (input.checked ? "yes" : "no"); }
    else {
      const num = f.kind === "int" || f.kind === "float";
      input = h("input", { id, type: num ? "number" : "text", value: show(cur), spellcheck: false, step: num ? "any" : null, min: f.rule.min, max: f.rule.max, inputmode: num ? "decimal" : null });
      get = () => input.value;
    }
    input.disabled = !f.editable;
    const start = get();
    const save = h("button", { class: "primary", hidden: true }, "Save"), undo = h("button", { class: "ghost", hidden: true }, "Undo");
    const row = h("div", { class: "frow" },
      h("label", { class: "name", for: id }, f.key, f.in_file ? null : h("span", { class: "tag" }, " default")),
      h("div", { class: "ctl" }, input, save, undo),
      h("div", { class: "help" }, f.help || "", f.editable ? null : " Change this one in a text editor."), err);
    const mark = () => { const d = get() !== start; row.classList.toggle("dirty", d); save.hidden = undo.hidden = !d || f.kind === "bool"; if (!d) { err.hidden = true; input.removeAttribute("aria-invalid"); } };
    const commit = () => busy(save, async () => {
      try {
        await api("/api/config", { id: f.id, value: get() });
        toast(`Saved ${f.id}.`);
        const fresh = (await api("/api/config")).fields.find((x) => x.id === f.id);
        const nr = fieldRow(fresh); nr.classList.add("saved"); row.replaceWith(nr);
      } catch (e) { err.textContent = e.message; err.hidden = false; input.setAttribute("aria-invalid", "true"); input.focus(); if (f.kind === "bool") input.checked = !input.checked; }
    });
    input.addEventListener("input", mark);
    if (f.kind === "bool") input.addEventListener("change", commit);
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && get() !== start) commit(); else if (e.key === "Escape") { input.value = start; mark(); } });
    save.addEventListener("click", commit);
    undo.addEventListener("click", () => { input.value = start; mark(); input.focus(); });
    return row;
  }

  /* ------------------------------------------------------------ endpoints */
  async function endpointsPage() {
    const data = await api("/api/endpoints");
    const word = { ok: "works", warn: "warning", fail: "failed", info: "note" };
    const result = (r) => h("div", { class: "ep-result" }, chip(r.status, word[r.status]), " ", r.detail,
      r.fix && r.status !== "ok" ? h("div", { class: "fix" }, h("b", {}, "What to do: "), r.fix) : null);
    const cards = data.endpoints.map((e) => {
      const out = h("div", { "aria-live": "polite" });
      const post = (what, value) => api("/api/endpoint", { id: e.id, what, value });
      const hooks = [];
      const shown = (r) => { if (r && r.test) { out.replaceChildren(result(r.test)); hooks.forEach((f) => f(r.test)); } };
      const addr = h("input", { value: e.source === "set" ? e.address : "", placeholder: e.address ? e.address + (e.source === "fallback" ? "  (from the second-opinion model)" : "  (default)") : (e.kind === "mqtt" ? "host or host:port" : "http://host:port"),
        "aria-label": e.title + " address", spellcheck: "false", autocomplete: "off" });
      const saveA = h("button", { class: "primary", onclick: () => busy(saveA, async () => { shown(await post("address", addr.value)); toast(addr.value ? "Address saved." : "Address cleared."); }) }, "Save");
      addr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") saveA.click(); });
      const test = h("button", { onclick: () => busy(test, async () => shown(await post("test"))) }, "Test");
      const rows = [h("div", { class: "ep-row" }, h("label", {}, "Address"), h("div", { class: "actions" }, addr, saveA, test))];
      if (e.model_key) {
        const list = h("datalist", { id: "models-" + e.id });
        const model = h("input", { value: e.model, list: list.id, placeholder: "not set: the server picks", "aria-label": e.title + " model name", autocomplete: "off" });
        const saveM = h("button", { onclick: () => busy(saveM, async () => { await post("model", model.value); toast(model.value ? "Model saved." : "Model cleared."); }) }, "Save");
        hooks.push((r) => r.models && list.replaceChildren(...r.models.map((m) => h("option", { value: m }))));   // test fills the choices
        rows.push(h("div", { class: "ep-row" }, h("label", {}, "Model"), h("div", { class: "actions" }, model, list, saveM)));
      }
      if (e.api_key) {
        const key = h("input", { type: "password", autocomplete: "off", placeholder: e.api_key_set ? "enter a new key to replace" : "only if the server was started with a key", "aria-label": e.title + " API key" });
        const saveK = h("button", { onclick: () => busy(saveK, async () => { if (!key.value.trim()) { key.focus(); throw new Error("Enter the key first."); } shown(await post("key", key.value)); key.value = ""; toast("API key stored."); await go("endpoints", true); }) }, e.api_key_set ? "Replace" : "Set");
        const delK = e.api_key_set ? h("button", { class: "danger", onclick: async () => {
          if (!await confirmBox({ title: "Remove the API key?", text: "The stored key is deleted. A server that requires one will refuse requests until you set it again.", ok: "Remove" })) return;
          busy(delK, async () => { await post("key", null); toast("API key removed."); await go("endpoints", true); });
        } }, "Remove") : null;
        rows.push(h("div", { class: "ep-row" }, h("label", {}, "API key ", chip(e.api_key_set ? "ok" : "", e.api_key_set ? "set" : "not set")), h("div", { class: "actions" }, key, saveK, delK)));
      }
      const note = e.source === "none" ? h("div", { class: "muted" }, "Not set: " + (e.fallback || "nothing uses it") + ".") : null;
      const sec = h("section", { class: "section ep", id: "ep-" + e.id }, h("h3", {}, e.title), h("p", { class: "help" }, e.help, " ", h("span", { class: "muted" }, "Stored in " + e.stored_in + ".")), note, rows, out);
      sec.runTest = async () => shown(await post("test"));
      return sec;
    });
    const all = h("button", { onclick: () => busy(all, async () => { for (const c of cards) await c.runTest(); }) }, "Test all");
    return [head("Endpoints", data.endpoints.length ? all : null),
      h("p", { class: "lead" }, "The servers this machine talks to: model servers (llama.cpp or any OpenAI-compatible server), Frigate, the message broker, the timestamp service. Addresses take effect when the service or containers restart. API keys go to the private secrets file and are never shown again."),
      cards.length ? cards : h("div", { class: "notice" }, "This role talks to no outside servers that can be set here.")];
  }

  /* ------------------------------------------------------------ secrets */
  async function secretsPage() {
    const list = await api("/api/env");
    const rows = list.map((e) => {
      const inp = h("input", { type: "password", autocomplete: "off", placeholder: e.set ? "enter a new value to replace" : "enter a value", "aria-label": e.name });
      const set = h("button", { onclick: () => busy(set, async () => { if (!inp.value.trim()) { inp.focus(); throw new Error("Enter a value first."); } await api("/api/env", { name: e.name, value: inp.value }); toast(e.name + " stored."); await go("secrets", true); }) }, e.set ? "Replace" : "Set");
      inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") set.click(); });
      const del = e.set ? h("button", { class: "danger", onclick: async () => {
        if (!await confirmBox({ title: `Remove ${e.name}?`, text: "The stored value is deleted from the secrets file. Anything that needs it stops working until you set it again.", ok: "Remove" })) return;
        busy(del, async () => { await api("/api/env", { name: e.name, value: null }); toast(e.name + " removed."); await go("secrets", true); });
      } }, "Remove") : null;
      return h("tr", {}, h("td", {}, h("code", {}, e.name), h("div", { class: "muted" }, e.help)), h("td", {}, chip(e.set ? "ok" : "", e.set ? "set" : "not set")), h("td", {}, h("div", { class: "actions" }, inp, set, del)));
    });
    return [head("Secrets"), h("p", { class: "lead" }, "Passwords and the camera stream address live in a private file (.env, readable only by you), never in the configuration file. Stored values are never shown again: you can only replace or remove them."),
      h("div", { class: "table-wrap" }, h("table", {}, h("tr", {}, h("th", {}, "Name"), h("th", {}, "Status"), h("th", {}, "Change")), rows))];
  }

  /* ------------------------------------------------------------ checks */
  async function checks() {
    const list = await api("/api/checks");
    const word = { ok: "passed", warn: "warning", fail: "failed", info: "note" };
    const n = (s) => list.filter((c) => c.status === s).length;
    const again = h("button", { onclick: () => busy(again, async () => { await go("checks", true); toast("Checks run again.", "info", 1800); }) }, "Run again");
    return [head("Checks", again),
      h("p", { class: "lead" }, n("fail") ? `${n("fail")} check${n("fail") > 1 ? "s" : ""} failed. Each one says how to fix it.` : n("warn") ? "Nothing failed; the warnings below are worth a look." : "Everything this machine needs is in place."),
      h("section", { class: "section" }, h("ul", { class: "checks" }, list.map((c) => h("li", {}, h("div", {}, chip(c.status, word[c.status])),
        h("div", {}, h("b", {}, c.title), h("div", { class: "d" }, c.detail), c.fix && c.status !== "ok" ? h("div", { class: "fix" }, h("b", {}, "What to do: "), c.fix) : null)))))];
  }

  /* ------------------------------------------------------------ service */
  async function service() {
    const d = await api("/api/service");
    if (d.role !== "detector") {
      const rows = d.containers;
      return [head("Service", h("button", { class: "ghost", onclick: () => go("service", true) }, "Refresh")),
        rows === null ? h("div", { class: "notice warn-box" }, "The stack is not installed on this machine, or docker cannot be asked. ", h("a", { href: "#install" }, "Open the install menu")) :
          h("div", { class: "table-wrap" }, h("table", {}, h("tr", {}, h("th", {}, "Container"), h("th", {}, "State"), h("th", {}, "Since")),
            rows.length ? rows.map((r) => h("tr", {}, h("td", {}, r.name), h("td", {}, chip(r.state === "running" ? "ok" : "warn", r.state)), h("td", {}, r.status))) : h("tr", { class: "empty" }, h("td", { colSpan: 3 }, "No containers yet."))))];
    }
    const s = d.service;
    const act = (label, action, cls, ask) => { const b = h("button", { class: cls, disabled: !s.installed, onclick: async () => {
      if (ask && !await confirmBox({ title: `${label} the detector?`, text: ask, ok: label, danger: action === "stop", icon: "info" })) return;
      busy(b, async () => { await api("/api/service", { action }); toast(`${label}: done.`); await go("service", true); });
    } }, label); return b; };
    const log = h("pre", { class: "out", tabindex: "0" }, d.logs.join("\n") || "No log lines yet.");
    queueMicrotask(() => { log.scrollTop = log.scrollHeight; });
    return [head("Service", h("button", { class: "ghost", onclick: () => go("service", true) }, "Refresh")),
      h("p", {}, "Detector service: ", chip(s.active ? "ok" : s.installed ? "warn" : "", s.active ? "running" : s.installed ? "stopped" : "not installed"), s.enabled ? " starts at login" : ""),
      s.installed ? null : h("div", { class: "notice" }, "The background service is not installed. ", h("a", { href: "#install" }, "Install it from the install menu"), ", or run the detector by hand."),
      h("div", { class: "actions" }, s.active ? null : act("Start", "start", "primary"), s.active ? act("Restart", "restart", "", "Detection pauses for a few seconds while it restarts.") : null,
        s.active ? act("Stop", "stop", "danger", "Nothing is detected or deterred while the service is stopped.") : null),
      h("h3", {}, "Recent log"), log];
  }

  /* ------------------------------------------------------------ events */
  async function events() {
    const list = await api("/api/events");
    const rows = list.map((e) => h("tr", {}, h("td", {}, e.snapshot ? h("a", { href: "/snapshot/" + encodeURIComponent(e.snapshot), target: "_blank", rel: "noopener", title: "Open the snapshot" }, h("img", { class: "snap", loading: "lazy", alt: "Snapshot", src: "/snapshot/" + encodeURIComponent(e.snapshot) })) : ""),
      h("td", {}, String(e.time || "").slice(0, 19).replace("T", " ")), h("td", {}, show(e.label)), h("td", { class: "num" }, show(e.score)), h("td", {}, show(e.verifier) || "none"),
      h("td", {}, chip(e.action === "rejected" ? "" : e.action === "dry_run" || e.action === "dry-run" ? "info" : e.confirmed ? "ok" : "", show(e.action)))));
    return [head("Events", h("button", { class: "ghost", onclick: () => go("events", true) }, "Refresh")),
      h("div", { class: "table-wrap" }, h("table", {}, h("tr", {}, h("th", {}, "Snapshot"), h("th", {}, "Time"), h("th", {}, "Detected"), h("th", { class: "num" }, "Score"), h("th", {}, "Second opinion"), h("th", {}, "Result")),
        rows.length ? rows : h("tr", { class: "empty" }, h("td", { colSpan: 6 }, "No events recorded yet. They appear here once the detector has seen something."))))];
  }

  /* ------------------------------------------------------------ router */
  const RENDER = { overview, install, config, endpoints: endpointsPage, secrets: secretsPage, checks, service, events };
  async function go(page, keepScroll) {
    if (!PAGES.includes(page)) page = "overview";
    clearTimeout(jobTimer);
    const seq = ++pageSeq;
    document.querySelectorAll("#nav a").forEach((a) => a.getAttribute("href") === "#" + page ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
    try {
      const nodes = await RENDER[page]();
      if (seq !== pageSeq) return;                     // the operator moved on while this was loading
      main.replaceChildren(...nodes.flat().filter(Boolean));
      document.title = page[0].toUpperCase() + page.slice(1) + " · squirrel-guard console";
      if (!keepScroll) scrollTo(0, 0);
    } catch (e) {
      if (seq !== pageSeq) return;
      main.replaceChildren(h("div", { class: "notice warn-box", role: "alert" }, h("b", {}, "This page could not be loaded. "), e.message, " ", h("button", { onclick: () => go(page) }, "Try again")));
    }
  }
  document.addEventListener("DOMContentLoaded", () => {
    main = document.getElementById("main"); conn = document.getElementById("conn");
    addEventListener("hashchange", () => go(location.hash.slice(1)));
    go(location.hash.slice(1) || "overview");
  });
})();
