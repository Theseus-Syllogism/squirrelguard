/* squirrel-guard UI behaviour (persons UI). Progressive enhancement: every page works without this file.
 *
 * - theme: system / light / dark, stored as 'sg-replay-theme' (shared with the 3D replay viewer, same origin)
 * - forms: data-confirm -> styled confirm dialog; busy spinner + no double submit; merge confirm names the target
 * - toasts: ?done=<code> after a redirect (saved, merged, deleted, enrolled, removed) -> one toast, then the
 *   parameter is removed from the address bar so a reload does not repeat it
 * - images: lazy loading, shimmer while loading, fade in, "no image" placeholder, click to zoom
 * - tables: wrapped in a scrolling card; a table with no data rows gets an empty-state row
 * - keyboard: "/" focuses search, Esc leaves it
 */
(() => {
  "use strict";
  const KEY = "sg-replay-theme";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const store = { get: () => { try { return localStorage.getItem(KEY); } catch { return null; } },
                  set: (v) => { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch { /* private mode */ } } };
  const ICON = {
    ok: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/></svg>',
    err: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.5"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5v.5"/></svg>',
    trash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/></svg>',
    merge: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="6" r="2.5"/><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="12" r="2.5"/><path d="M6 8.5v7M8.3 7.2 15.6 11M8.3 16.8 15.6 13"/></svg>',
  };

  /* ---------------------------------------------------------------- theme */
  const applyTheme = (v) => { if (v === "light" || v === "dark") document.documentElement.dataset.theme = v; else delete document.documentElement.dataset.theme; };
  const themeLabel = { "": "Theme: system", light: "Theme: light", dark: "Theme: dark" };
  function initTheme() {
    const btn = $("#theme-toggle");
    if (!btn) return;
    const sync = () => { const v = store.get() || ""; btn.title = btn.ariaLabel = themeLabel[v] + " (click to change)"; btn.dataset.mode = v || "system"; };
    sync();
    btn.addEventListener("click", () => {
      const order = ["", "light", "dark"], cur = store.get() || "", next = order[(order.indexOf(cur) + 1) % 3];
      store.set(next); applyTheme(next); sync(); toast(themeLabel[next], "info", 1600);
    });
  }

  /* ---------------------------------------------------------------- toasts */
  function toast(msg, kind = "ok", ms = 4200) {
    let box = $(".toasts");
    if (!box) { box = document.createElement("div"); box.className = "toasts"; box.setAttribute("role", "status"); box.setAttribute("aria-live", "polite"); document.body.append(box); }
    const t = document.createElement("div");
    t.className = "toast " + kind;
    t.innerHTML = (ICON[kind] || ICON.ok) + '<div class="msg"></div><button class="x" aria-label="Dismiss">×</button>';
    t.querySelector(".msg").textContent = msg;
    const close = () => { t.classList.add("out"); setTimeout(() => t.remove(), 220); };
    t.querySelector(".x").addEventListener("click", close);
    box.append(t);
    if (ms) setTimeout(close, ms);
  }
  const DONE = { saved: "Changes saved.", merged: "Merged. The sightings now belong to this identity.", deleted: "Deleted.",
                 enrolled: "Face enrolled. It will be used for recognition from the next visit.", removed: "Enrolled face removed.",
                 split: "Split into a new identity.", exported: "Evidence package created, signed and logged.",
                 verified: "Integrity check passed: hashes, signature and timestamp match.", reviewed: "Review recorded.",
                 release_started: "Making the release copy. It appears below when ready (reload in a minute or two).",
                 release_running: "A release copy is already being made." };
  function doneToast() {
    const u = new URL(location.href), d = u.searchParams.get("done");
    if (!d) return;
    if (d === "verify_failed") toast("Integrity check FAILED: see the custody log below.", "err", 0);
    else toast(DONE[d] || "Done.", "ok");
    u.searchParams.delete("done");
    history.replaceState(null, "", u.pathname + (u.searchParams.toString() ? "?" + u.searchParams : "") + u.hash);
  }

  /* ---------------------------------------------------------------- confirm dialog */
  function confirmDialog({ title, text, ok = "Confirm", danger = true, icon = "trash" }) {
    return new Promise((resolve) => {
      const d = document.createElement("dialog");
      d.innerHTML = `<form method="dialog"><div class="dlg-body"><div class="dlg-icon ${danger ? "" : "warn"}">${ICON[icon] || ICON.trash}</div>
        <div><h4></h4><p></p></div></div><div class="dlg-foot"><button value="cancel" class="ghost">Cancel</button>
        <button value="ok" class="${danger ? "solid-danger" : "primary"}"></button></div></form>`;
      d.querySelector("h4").textContent = title;
      d.querySelector("p").textContent = text;
      d.querySelector('button[value="ok"]').textContent = ok;
      d.addEventListener("close", () => { resolve(d.returnValue === "ok"); d.remove(); });
      d.addEventListener("click", (e) => { if (e.target === d) d.close("cancel"); });   // click on the backdrop
      document.body.append(d);
      d.showModal();
      d.querySelector('button[value="cancel"]').focus();                                  // the safe choice has focus
    });
  }

  /* ---------------------------------------------------------------- forms */
  function initForms() {
    $$("form").forEach((f) => {
      f.addEventListener("submit", async (e) => {
        if (f.dataset.confirmed === "1") return;                                          // second pass, already confirmed
        const sub = e.submitter || f.querySelector("button");
        if (f.dataset.confirm) {
          e.preventDefault();
          let text = f.dataset.confirm;
          const into = f.querySelector('select[name="into"]');
          if (into) text = text.replace("{target}", into.options[into.selectedIndex]?.textContent.trim() || "the selected item");
          const ok = await confirmDialog({ title: f.dataset.confirmTitle || "Are you sure?", text, ok: f.dataset.confirmOk || (sub && sub.textContent.trim()) || "Confirm",
                                           danger: f.dataset.confirmKind !== "warn", icon: f.dataset.confirmIcon || "trash" });
          if (!ok) return;
          f.dataset.confirmed = "1";
          busy(sub);
          f.requestSubmit ? f.requestSubmit(sub) : f.submit();
          return;
        }
        if (f.dataset.busy === "1") { e.preventDefault(); return; }                        // no double submit
        f.dataset.busy = "1";
        if ((f.method || "get").toLowerCase() === "post") busy(sub);
      });
    });
    // un-stick after a back navigation from the bfcache
    addEventListener("pageshow", (e) => { if (e.persisted) $$("form").forEach((f) => { f.dataset.busy = ""; f.dataset.confirmed = ""; }); $$("button.is-busy").forEach((b) => { b.classList.remove("is-busy"); b.disabled = false; }); });
  }
  function busy(btn) { if (!btn) return; btn.classList.add("is-busy"); btn.setAttribute("aria-busy", "true"); setTimeout(() => { btn.disabled = true; }, 0); }

  /* ---------------------------------------------------------------- images */
  function initImages() {
    $$("main img").forEach((img, i) => {
      if (!img.hasAttribute("loading") && i >= 8) img.loading = "lazy";      // the first screenful loads at once
      img.decoding = "async";
      const done = () => { img.classList.remove("loading"); img.classList.add("ready"); };
      const fail = () => {
        img.classList.remove("loading"); img.classList.add("broken");
        const ph = document.createElement("span"); ph.className = "img-missing"; ph.textContent = "no image";
        if (img.height > 20) { ph.style.height = img.height + "px"; ph.style.width = Math.max(48, Math.round(img.height * 0.75)) + "px"; }
        img.replaceWith(ph);
      };
      if (img.complete && img.naturalWidth) { img.classList.add("fade", "ready"); }
      else if (img.complete) { fail(); }
      else { img.classList.add("fade", "loading"); img.addEventListener("load", done, { once: true }); img.addEventListener("error", fail, { once: true }); }
      if (!img.closest("a") && (img.classList.contains("t") || img.closest(".card, .ev, .grid, td"))) {
        img.classList.add("zoomable"); img.tabIndex = 0; img.setAttribute("role", "button"); img.setAttribute("aria-label", "Enlarge image");
        img.addEventListener("click", () => lightbox(img));
        img.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); lightbox(img); } });
      }
    });
  }
  function lightbox(img) {
    const d = document.createElement("dialog");
    d.className = "lightbox";
    d.innerHTML = '<figure style="margin:0"><img alt=""><figcaption>Click anywhere or press Esc to close</figcaption></figure>';
    d.querySelector("img").src = img.currentSrc || img.src;
    d.addEventListener("click", () => d.close());
    d.addEventListener("close", () => d.remove());
    document.body.append(d);
    d.showModal();
  }

  /* ---------------------------------------------------------------- tables */
  function initTables() {
    $$("main table").forEach((t) => {
      if (!t.parentElement.classList.contains("table-wrap") && !t.closest(".facegeom")) {
        const w = document.createElement("div"); w.className = "table-wrap"; t.replaceWith(w); w.append(t);
      }
      const rows = $$("tr", t), dataRows = rows.filter((r) => !r.querySelector("th"));
      const head = rows.find((r) => r.querySelector("th"));
      if (head && t.closest(".table-wrap")) {                                   // labels for the stacked (phone) layout
        const labels = $$("th", head).map((th) => th.textContent.trim());
        dataRows.forEach((r) => [...r.children].forEach((td, i) => {
          if (labels[i]) td.dataset.label = labels[i];
          if (td.querySelector("img") && !td.textContent.trim()) td.classList.add("media");
        }));
        if (labels.length >= 4) t.closest(".table-wrap").classList.add("stack");
      }
      if (rows.length && !dataRows.length) {
        const cols = Math.max(1, ...rows.map((r) => r.children.length));
        const tr = document.createElement("tr"); tr.className = "empty";
        tr.innerHTML = `<td colspan="${cols}">Nothing here yet.</td>`;
        (t.tBodies[0] || t).append(tr);
      }
      $$("td", t).forEach((td) => { if (/^\s*-?\d+([.,]\d+)?\s*(%|cm|px|s)?\s*$/.test(td.textContent) && !td.querySelector("img, form")) td.classList.add("num"); });
    });
  }

  /* ---------------------------------------------------------------- page title, keyboard */
  function initPage() {
    const h = $("main h2");
    if (h) document.title = h.childNodes[0]?.textContent.trim().replace(/\s+/g, " ") + " · Persons";
    const q = $("#q");
    addEventListener("keydown", (e) => {
      if (e.key === "/" && q && document.activeElement !== q && !/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) { e.preventDefault(); q.focus(); q.select(); }
      else if (e.key === "Escape" && document.activeElement === q) q.blur();
    });
  }

  document.addEventListener("DOMContentLoaded", () => { initTheme(); initPage(); initTables(); initImages(); initForms(); doneToast(); });
  window.sgui = { toast, confirmDialog };
})();
