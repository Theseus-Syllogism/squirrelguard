// Investigation pane ("single pane of glass") for the trajectory replay: click the person in the 3D view and a
// small draggable sub-dashboard opens over the scene with everything known about the visit, its related sightings
// (node-link graph), correlated clips and the identity's timeline. No page navigation: clips play inline, related
// 3D replays open in the same viewer, Why? pages open in a new tab.
// Data: persons-ui /api/visit/<frigate event> and /api/identity/<id> (same sign-in as the persons UI; ?api= points
// elsewhere for tests). Demo samples have no stored visit: the pane shows what their replay JSON holds.
// Everything from the API is inserted as text (textContent / attributes), never as HTML.

const SVGNS = 'http://www.w3.org/2000/svg';
const NARROW = '(max-width: 560px)';
const store = { get(k) { try { return localStorage.getItem('sg-replay-pane-' + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem('sg-replay-pane-' + k, v); } catch { /* not remembered */ } } };

function el(tag, props, ...kids) { return fill(document.createElement(tag), props, kids); }
function sv(tag, props, ...kids) { return fill(document.createElementNS(SVGNS, tag), props, kids); }
function fill(e, props, kids) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.setAttribute('class', v);
    else if (k === 'text') e.textContent = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids.flat(3)) if (c != null && c !== false && c !== '') e.append(c instanceof Node ? c : String(c));
  return e;
}
function safeURL(u) { // only http(s) or same-origin paths from the API become links / media sources
  if (!u || typeof u !== 'string') return null;
  try { const x = new URL(u, location.href); return ['http:', 'https:'].includes(x.protocol) ? x.href : null; } catch { return null; }
}
const fmtTime = (iso) => { if (!iso) return ''; const d = new Date(iso); return Number.isNaN(+d) ? iso : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); };
const fmtDay = (iso) => new Date(iso).toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
const fmtDur = (s) => s == null ? '' : s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`;
const num = (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? '–' : Number(v).toFixed(d));
const BANDS = ['strong', 'likely', 'possible', 'unlikely'];
function chip(band, pct, extra) {
  if (!band && pct == null) return null;
  return el('span', { class: `chip band-${BANDS.includes(band) ? band : 'none'}`, title: 'estimated probability; a ranking aid, not a fact' },
    `${band || ''}${pct != null ? ` ${pct}%` : ''}${extra ? ` ${extra}` : ''}`.trim());
}
const kindChip = (k) => k ? el('span', { class: `chip kind-${k}` }, k) : null;
const EDGE_SHORT = { reid: 're-ID', face: 'face', gait: 'gait', identity: 'same ID', assigned: 'assigned', possible: 'possible', time: 'time', cooccur: 'seen together' };
function edgeLabel(st) {
  const e = (st && st.edges) || [];
  return e.slice(0, 2).map((x) => EDGE_SHORT[x.kind] + (x.kind === 'reid' || x.kind === 'face' || x.kind === 'possible' || x.kind === 'assigned' ? (x.value != null ? ` ${Number(x.value).toFixed(2).replace(/^0/, '')}` : '')
    : x.kind === 'gait' ? ` ${Number(x.value).toFixed(2).replace(/^0/, '')}` : x.kind === 'time' ? ` ${Math.round(Math.abs(x.value) / 60)}m` : '')).join(' · ');
}
function edgeKinds(n) { // short edge label in the graph: the evidence beyond plain appearance, else re-ID
  const e = ((n.strength && n.strength.edges) || []).map((x) => x.kind);
  if (n.type === 'vehicle') {
    const t = n.strength.edges.find((x) => x.kind === 'time'), co = n.strength.edges.find((x) => x.kind === 'cooccur');
    return t ? `${Math.max(1, Math.round(Math.abs(t.value) / 60))} min` + (co ? ` · ${co.value}×` : '') : '';
  }
  if (n.type === 'identity') return 'assigned';
  if (n.type === 'possible_identity') return 'possible';
  const extra = e.filter((k) => k === 'face' || k === 'gait').map((k) => EDGE_SHORT[k]);
  return extra.length ? extra.join(' · ') : 're-ID';
}
function section(title, ...kids) {
  const body = kids.flat().filter((k) => k != null && k !== false);
  return body.length ? el('section', { class: 'pn-sec' }, el('h3', {}, title), body) : null;
}
function kv(rows) { // [[label, value], ...] -> definition grid (skips empty values)
  const r = rows.filter(([, v]) => v != null && v !== '' && v !== false);
  return r.length ? el('dl', { class: 'pn-kv' }, r.map(([k, v]) => [el('dt', {}, k), el('dd', {}, v)])) : null;
}
function meter(value, max, mark, label) { // horizontal bar with an optional threshold mark
  const w = Math.max(0, Math.min(1, value / (max || 1))) * 100;
  return el('div', { class: 'pn-meter', role: 'img', 'aria-label': label },
    el('span', { class: 'fill', style: `width:${w.toFixed(1)}%` }),
    mark != null ? el('span', { class: 'mark', style: `left:${Math.min(100, mark / (max || 1) * 100).toFixed(1)}%` }) : null);
}
function thumb(urls, alt, cls = 'pn-thumb') { // first URL that loads; placeholder otherwise
  const list = urls.map(safeURL).filter(Boolean);
  if (!list.length) return el('div', { class: `${cls} ph`, 'aria-hidden': 'true' });
  const img = el('img', { class: cls, alt: alt || '', loading: 'lazy', src: list[0] });
  let i = 0;
  img.addEventListener('error', () => { i += 1; if (i < list.length) img.src = list[i]; else img.replaceWith(el('div', { class: `${cls} ph`, 'aria-hidden': 'true' })); });
  return img;
}
function linkBtn(href, text, title) {
  const u = safeURL(href);
  return u ? el('a', { class: 'pn-btn', href: u, target: '_blank', rel: 'noopener', title }, text) : null;
}
function videoPlayer(src, fallbackHref) {
  const u = safeURL(src);
  const wrap = el('div', { class: 'pn-video' });
  if (!u) return wrap;
  const v = el('video', { controls: true, playsinline: true, preload: 'metadata', src: u, autoplay: true });
  v.muted = true; // autoplay needs muted; the controls unmute
  v.addEventListener('error', () => {
    wrap.replaceChildren(el('p', { class: 'pn-note' }, 'The clip could not be loaded here (Frigate may have pruned it, or this browser is not signed in to Frigate). ',
      linkBtn(fallbackHref, 'Open in Frigate')));
  });
  wrap.append(v);
  return wrap;
}

export function eventOf(data, dataURL) {
  if (data && typeof data.event === 'string') return data.event;
  const m = /([^/?#]+)_traj\.json(?:$|[?#])/.exec(dataURL || '');
  return m ? decodeURIComponent(m[1]) : null;
}

export function createPane(host) {
  const pane = document.getElementById('pane');
  const head = document.getElementById('pane-head');
  const titleEl = document.getElementById('pane-title');
  const subEl = document.getElementById('pane-sub');
  const avatarEl = document.getElementById('pane-avatar');
  const chipsEl = document.getElementById('pane-chips');
  const tabsEl = document.getElementById('pane-tabs');
  const bodyEl = document.getElementById('pane-body');
  const cardEl = document.getElementById('pane-card');
  const noticeEl = document.getElementById('pane-notice');
  const apiBase = new URL(host.apiBase || '/api/', location.href);
  const S = { open: false, tab: 'overview', sceneEvent: null, focusEvent: null, visit: null, err: null, loading: false,
    timeline: null, timelineFor: null, timelineErr: null, clickT: null, playing: null, cardFor: null };
  let invoker = null, reqId = 0;

  const isReal = () => !!S.focusEvent;
  const hasRig = () => { const d = host.getData(); return !!(d && d.rig && d.rig.angles_deg); };
  const tabs = () => (isReal()
    ? [['overview', 'Overview'], ['face', 'Face'], ['related', 'Related'], ['clips', 'Clips'], ['timeline', 'Timeline']]
    : [['overview', 'Overview'], ['scene', 'Scene & model'], ['related', 'Related']])
    .concat(hasRig() ? [['body', 'Body & gait']] : []);
  let playheads = []; // [{line, x0, x1, t0, t1}] of the open Body & gait charts, moved by onTime()

  async function api(path) {
    const r = await fetch(new URL(path, apiBase), { credentials: 'same-origin', headers: { Accept: 'application/json' } });
    if (r.status === 401) throw new Error('sign-in required: open the persons UI first');
    let j = null;
    try { j = await r.json(); } catch { /* not JSON */ }
    if (!r.ok) throw new Error((j && j.error) || `${r.status} ${r.statusText}`);
    return j;
  }
  async function loadVisit(eid) {
    const my = ++reqId;
    S.loading = true; S.err = null; S.visit = null; S.timeline = null; S.timelineFor = null; S.timelineErr = null;
    closeCard(); render();
    try {
      const v = await api(`visit/${encodeURIComponent(eid)}`);
      if (my !== reqId) return;
      S.visit = v;
    } catch (e) {
      if (my !== reqId) return;
      S.err = e.message || String(e);
    }
    S.loading = false;
    render();
  }
  async function loadTimeline(iid) {
    S.timelineFor = iid; S.timeline = null; S.timelineErr = null;
    render();
    try {
      const j = await api(`identity/${encodeURIComponent(iid)}`);
      if (S.timelineFor === iid) S.timeline = j;
    } catch (e) { if (S.timelineFor === iid) S.timelineErr = e.message || String(e); }
    render();
  }

  // -- open / close / position ---------------------------------------------------------------------------------
  function open(opts = {}) {
    invoker = opts.invoker || document.activeElement;
    S.clickT = opts.t ?? null;
    const ev = eventOf(host.getData(), host.getDataURL());
    const fresh = !S.open || ev !== S.sceneEvent || S.focusEvent !== ev;
    S.sceneEvent = ev;
    if (fresh) { S.focusEvent = ev; S.tab = 'overview'; }
    if (opts.tab && tabs().some(([k]) => k === opts.tab)) S.tab = opts.tab;   // ?pane_tab= deep link
    S.deepTab = opts.tab || null;   // kept until the recording has loaded (onData would reset it)
    S.open = true;
    pane.hidden = false;
    placeInitial();
    if (fresh && ev) loadVisit(ev); else render();
    requestAnimationFrame(() => document.getElementById('pane-close').focus());
    host.onToggle && host.onToggle(true);
  }
  function close() {
    S.open = false;
    stopVideos();
    pane.hidden = true;
    closeCard();
    host.onToggle && host.onToggle(false);
    if (invoker && invoker.focus && document.contains(invoker)) invoker.focus();
  }
  function stopVideos() { pane.querySelectorAll('video').forEach((v) => { try { v.pause(); } catch { /* */ } }); }
  function onData() { // the 3D scene switched to another recording
    const ev = eventOf(host.getData(), host.getDataURL());
    if (ev !== S.sceneEvent) S.clickT = null;   // "picked at" belonged to the previous recording
    S.sceneEvent = ev;
    if (!S.open) return;
    if (ev && ev === S.focusEvent && (S.visit || S.loading)) { render(); return; }
    S.focusEvent = ev; S.tab = isReal() ? S.tab : 'overview';
    if (S.deepTab && tabs().some(([k]) => k === S.deepTab)) S.tab = S.deepTab;
    S.deepTab = null;
    if (!tabs().some(([k]) => k === S.tab)) S.tab = 'overview';
    if (ev) loadVisit(ev); else { S.visit = null; S.err = null; render(); }
  }
  function placeInitial() {
    if (matchMedia(NARROW).matches) { pane.style.left = pane.style.top = pane.style.maxHeight = ''; return; }
    const app = pane.parentElement.getBoundingClientRect();
    let pos = null;
    try { pos = JSON.parse(store.get('pos') || 'null'); } catch { pos = null; }
    const w = pane.offsetWidth, top0 = (document.getElementById('top').offsetHeight || 60) + 22;
    const x = pos ? pos.x : app.width - w - 16, y = pos ? pos.y : top0;
    setPos(x, y);
  }
  function setPos(x, y) { // clamp inside the viewer; keep the time bar at the bottom uncovered when there is room
    const app = pane.parentElement.getBoundingClientRect();
    const w = pane.offsetWidth, bottomBar = (document.getElementById('bottom').offsetHeight || 80) + 24;
    const top = Math.round(Math.max(8, Math.min(app.height - 160, y)));
    pane.style.left = `${Math.round(Math.max(8, Math.min(app.width - w - 8, x)))}px`;
    pane.style.top = `${top}px`;
    pane.style.maxHeight = `${Math.max(240, app.height - top - bottomBar)}px`;
  }
  // drag by the header (desktop); the phone bottom sheet stays put
  let drag = null;
  head.addEventListener('pointerdown', (e) => {
    if (matchMedia(NARROW).matches || e.button !== 0 || e.target.closest('button, a, [role=tab]')) return;
    drag = { dx: e.clientX - pane.offsetLeft, dy: e.clientY - pane.offsetTop, id: e.pointerId };
    head.setPointerCapture(e.pointerId);
    pane.classList.add('dragging');
  });
  head.addEventListener('pointermove', (e) => { if (drag && e.pointerId === drag.id) setPos(e.clientX - drag.dx, e.clientY - drag.dy); });
  const endDrag = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null; pane.classList.remove('dragging');
    store.set('pos', JSON.stringify({ x: pane.offsetLeft, y: pane.offsetTop }));
  };
  head.addEventListener('pointerup', endDrag);
  head.addEventListener('pointercancel', endDrag);
  head.addEventListener('dblclick', (e) => { if (!e.target.closest('button, a')) { store.set('pos', ''); placeInitialDefault(); } });
  function placeInitialDefault() { const app = pane.parentElement.getBoundingClientRect(); setPos(app.width - pane.offsetWidth - 16, (document.getElementById('top').offsetHeight || 60) + 22); }
  window.addEventListener('resize', () => { if (S.open) placeInitial(); });
  document.getElementById('pane-close').addEventListener('click', close);
  document.getElementById('pane-min').addEventListener('click', () => {
    const m = pane.classList.toggle('min');
    document.getElementById('pane-min').setAttribute('aria-expanded', String(!m));
    document.getElementById('pane-min').textContent = m ? '▴' : '▾';
  });
  pane.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); if (!cardEl.hidden) closeCard(true); else close(); }
  });

  // -- tabs ------------------------------------------------------------------------------------------------------
  function renderTabs() {
    const list = tabs();
    tabsEl.replaceChildren(...list.map(([k, label]) => el('button', {
      role: 'tab', id: `pn-tab-${k}`, 'aria-selected': String(S.tab === k), 'aria-controls': 'pane-body',
      tabindex: S.tab === k ? '0' : '-1', onclick: () => { selectTab(k); },
    }, label, k === 'related' && S.visit ? el('span', { class: 'count' }, String(S.visit.related.filter((r) => r.type === 'sighting').length)) : null)));
  }
  function selectTab(k, focus) {
    S.tab = k; closeCard(); stopVideos(); render();
    bodyEl.scrollTop = 0;
    if (focus) document.getElementById(`pn-tab-${k}`).focus();
    if (k === 'timeline' && S.visit && S.visit.identity && S.timelineFor == null) loadTimeline(S.visit.identity.id);
  }
  tabsEl.addEventListener('keydown', (e) => {
    const list = tabs().map(([k]) => k), i = list.indexOf(S.tab);
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      selectTab(list[(i + (e.key === 'ArrowRight' ? 1 : list.length - 1)) % list.length], true);
    } else if (e.key === 'Home') { e.preventDefault(); selectTab(list[0], true); } else if (e.key === 'End') { e.preventDefault(); selectTab(list[list.length - 1], true); }
  });

  // -- render ----------------------------------------------------------------------------------------------------
  function render() {
    if (!S.open) return;
    if (!tabs().some(([k]) => k === S.tab)) S.tab = 'overview';
    renderHead();
    renderTabs();
    bodyEl.setAttribute('aria-labelledby', `pn-tab-${S.tab}`);
    playheads = [];
    const view = { overview, face, related, clips, timeline, scene, body }[S.tab] || overview;
    bodyEl.replaceChildren(...[view()].flat().filter(Boolean));
  }

  function renderHead() {
    const v = S.visit, d = host.getData() || {};
    chipsEl.replaceChildren();
    noticeEl.replaceChildren();
    noticeEl.hidden = true;
    if (!isReal()) {
      avatarEl.replaceChildren(el('span', { class: 'glyph', 'aria-hidden': 'true' }, '◎'));
      titleEl.textContent = host.sampleName() || 'Sample recording';
      const M = d.metrics || {};
      subEl.textContent = ['demo sample, not a stored visit', M.duration_s != null ? fmtDur(M.duration_s) : null,
        M.path_length_m != null ? `${M.path_length_m.toFixed(1)} m walked` : null, S.clickT != null ? `picked at ${S.clickT.toFixed(1)} s` : null].filter(Boolean).join(' · ');
      chipsEl.append(el('span', { class: 'chip' }, 'sample'));
      return;
    }
    avatarEl.replaceChildren(v ? thumb([v.links && v.links.crop, v.links && v.links.thumbnail], '', 'pn-avatar-img') : el('span', { class: 'glyph', 'aria-hidden': 'true' }, '…'));
    if (!v) {
      titleEl.textContent = S.loading ? 'Loading visit…' : 'Visit';
      subEl.textContent = `event ${S.focusEvent}` + (d.camera_name ? ` · ${d.camera_name}` : '');
      return;
    }
    titleEl.replaceChildren(v.identity ? v.identity.name : 'Unassigned visit');
    if (v.identity) chipsEl.append(kindChip(v.identity.kind));
    const c = chip(v.explain.band, v.explain.pct);
    if (c) chipsEl.append(c);
    subEl.textContent = [fmtTime(v.start), v.camera, v.duration_s ? fmtDur(v.duration_s) : null,
      S.clickT != null && S.focusEvent === S.sceneEvent ? `picked at ${S.clickT.toFixed(1)} s` : null].filter(Boolean).join(' · ');
    if (S.focusEvent !== S.sceneEvent) {
      noticeEl.hidden = false;
      noticeEl.append('Showing a related visit, not the one in the 3D scene. ',
        v.links.replay_data ? el('button', { class: 'pn-btn', onclick: () => host.openReplay(v.links.replay_data) }, 'Open its 3D replay') : null, ' ',
        S.sceneEvent ? el('button', { class: 'pn-btn', onclick: () => { S.focusEvent = S.sceneEvent; loadVisit(S.sceneEvent); } }, 'Back to the scene’s visit') : null);
    }
    const D = host.getData && host.getData();
    if (S.focusEvent === S.sceneEvent && v.links.replay_fused && !(D && D.fusion)) { // docs/fusion.md
      noticeEl.hidden = false;
      noticeEl.append('Other cameras saw this visit too. ',
        el('button', { class: 'pn-btn', onclick: () => host.openReplay(v.links.replay_fused) }, 'Open the all-camera replay'));
    }
  }

  function stateBlock() { // loading / error placeholder for API-backed tabs
    if (S.loading) return el('p', { class: 'pn-note', role: 'status' }, 'Loading the stored visit…');
    if (S.err) return el('p', { class: 'pn-note warn', role: 'status' }, `Could not load the stored visit for event ${S.focusEvent}: ${S.err}.`);
    return null;
  }

  // Motion facts from the replay JSON itself (both real visits and samples)
  function motion() {
    if (S.focusEvent !== S.sceneEvent) return null;
    const d = host.getData() || {}, M = d.metrics || {}, G = (d.gaze && d.gaze.summary) || null;
    const stops = (M.stops || []).map((s) => `${num(s.t0, 1)}–${num(s.t1, 1)} s`).join(', ');
    return section('Movement (this replay)',
      kv([['Duration', M.duration_s != null ? fmtDur(M.duration_s) : null],
        ['Walked', M.path_length_m != null ? `${M.path_length_m.toFixed(1)} m, mean ${num(M.speed_mean_mps)} m/s, max ${num(M.speed_max_mps)} m/s` : null],
        ['Stops', stops || (M.stops ? 'none' : null)],
        ['Heading', M.heading_entry_deg != null ? `in ${Math.round(M.heading_entry_deg)}°, out ${Math.round(M.heading_exit_deg)}°` : null],
        ['Step rate', M.cadence_spm != null ? `${Math.round(M.cadence_spm)} steps/min${M.cadence_assumed ? ' (assumed)' : ''}` : null],
        ['Frames', M.frames_total != null ? `${M.frames_observed} of ${M.frames_total} observed` : null],
        ['Head direction', G ? `facing the camera ${Math.round((G.facing_camera_share || 0) * 100)} % of the time; looked off the path ${Math.round((G.looking_off_path_share || 0) * 100)} %`
          + (G.largest_off_path_deg != null ? ` (largest ${Math.round(G.largest_off_path_deg)}° at ${num(G.largest_off_path_t, 1)} s)` : '') : null]]),
      G ? el('p', { class: 'pn-note' }, 'Head direction is from nose, eyes and ears (about ±25°), not eye gaze.') : null);
  }

  function overview() {
    if (!isReal()) {
      const d = host.getData() || {};
      return [el('p', { class: 'pn-note info' }, 'This is a demo sample with no record in the visitor database. Related sightings, faces, prior clips and the identity timeline appear when you open the 3D replay of a real visit (the “3D replay” link on a sighting’s Why? page).'),
        section('Recording', kv([['Source', d.source], ['Frames', d.t ? `${d.t.length} at ${num(d.fps, 1)} fps` : null]])),
        motion()];
    }
    const v = S.visit;
    if (!v) return [stateBlock(), motion()];
    const x = v.explain, a = v.attributes || {}, rt = v.routine;
    const desc = a.described || {};
    const clothes = [[desc.upper_color, desc.upper_clothing].filter(Boolean).join(' '), [desc.lower_color, desc.lower_clothing].filter(Boolean).join(' ')].filter(Boolean).join('; ');
    return [
      x.headline ? el('p', { class: 'pn-headline' }, x.headline, x.estimated ? el('small', {}, ' (estimate)') : null) : null,
      x.reasons.length ? el('div', { class: 'pn-reasons', role: 'list', 'aria-label': 'reason codes' },
        x.reasons.map((r) => el('span', { class: 'code', role: 'listitem', title: r.code }, r.text))) : null,
      section('Why this match', x.evidence.length ? el('ul', { class: 'pn-list' }, x.evidence.map((e) => el('li', {}, e))) : null),
      section('What they did', (x.did.length || (v.behavior && v.behavior.entered_zones)) ? el('ul', { class: 'pn-list' },
        x.did.map((e) => el('li', {}, e))) : null,
      (v.actions || []).length ? kv([['Actions', v.actions.map((t) => `${t.tag.replace(/_/g, ' ')} ${Math.round(t.p * 100)} % at ${Math.round(t.t)} s`).join('; ')]]) : null,
      (v.objects || []).length ? kv([['Objects', v.objects.map((o) => `${o.label}${o.carried ? ' (carried)' : ''} ${Math.round((o.score || 0) * 100)} %`).join(', ')]]) : null),
      section('How they looked (apparent, estimates)', kv([
        ['Clothing', clothes || null],
        ['Activity', desc.activity],
        ['Age', a.apparent_age ? `${a.apparent_age[0]}–${a.apparent_age[1]} (apparent)` : null],
        ['Gender', a.apparent_gender ? `${a.apparent_gender.value} (apparent, ${Math.round((a.apparent_gender.conf || 0) * 100)} %)` : null],
        ['PULC', a.pulc ? a.pulc.line : null],
        ['Height', v.height ? `${Math.round(v.height.cm)} cm` + (v.height.iqr_cm ? ` ± ${Math.max(2, Math.round(v.height.iqr_cm / 2))}` : '') : null],
        ['Walk', v.gait && v.gait.usable ? `${Math.round(v.gait.cadence_spm)} steps/min` + (v.gait.step_m ? `, step ${v.gait.step_m} m` : '') : null],
        ['Gait signature', v.gait_signature ? 'recorded (compared in Related)' : null],
        ['Expression', v.face && v.face.expression ? `${v.face.expression} (description only)` : null]])),
      rt ? section('Routine', el('p', {}, rt.unusual ? el('span', { class: 'chip band-unlikely' }, 'unusual') : rt.ready ? el('span', { class: 'chip band-strong' }, 'ordinary') : el('span', { class: 'chip' }, 'learning'),
        ` ${num(rt.bits, 1)} bits of surprise against ${rt.history} earlier visits (alert at ${rt.threshold})`),
      meter(rt.bits || 0, Math.max(rt.threshold || 1, rt.bits || 0) * 1.15, rt.threshold, `routine surprise ${rt.bits} bits, threshold ${rt.threshold}`),
      (rt.parts || []).length ? el('ul', { class: 'pn-list small' }, rt.parts.map((p) => el('li', {}, `${p.code}: ${num(p.bits, 1)} bits`, p.text ? ` — ${p.text}` : ''))) : null,
      rt.note ? el('p', { class: 'pn-note' }, rt.note) : null) : null,
      motion(),
      (v.alerts.length || v.decisions.length) ? section('Alerts and decisions',
        v.alerts.map((al) => el('p', { class: 'pn-alert' }, el('b', {}, al.kind.replace(/_/g, ' ')), ` · ${fmtTime(al.ts)} · ${al.message}`, al.sent ? '' : ' (not sent)')),
        el('ul', { class: 'pn-list small' }, v.decisions.map((t) => el('li', {}, `${fmtTime(t.ts)} ${t.stage}`, t.band ? [' ', chip(t.band, t.pct)] : null, t.message ? ` — ${t.message}` : '')))) : null,
      v.evidence && v.evidence.card ? section('Evidence card', el('a', { href: safeURL(v.evidence.card), target: '_blank', rel: 'noopener' }, thumb([v.evidence.card], 'evidence card', 'pn-evidence'))) : null,
      el('div', { class: 'pn-actions' }, linkBtn(v.links.why, 'Why? ↗', 'full explanation page (new tab)'), linkBtn(v.links.identity, 'Identity page ↗'),
        linkBtn(v.links.explore, 'Frigate ↗'), el('button', { class: 'pn-btn', onclick: () => { selectTab('clips'); playClip(v.links.clip, v.links.explore, `this visit`); } }, '▶ Play clip')),
    ];
  }

  function face() {
    const v = S.visit;
    if (!v) return stateBlock();
    const f = v.face;
    if (!f) return el('p', { class: 'pn-note' }, 'No face was recorded for this visit (none seen, or face identification was not running).');
    const d = f.decision || {};
    const sim = d.sim, thr = d.threshold;
    return [
      el('div', { class: 'pn-faces' }, f.crops.length ? f.crops.map((u, i) => thumb([u], `face crop ${i + 1} of this visit`, 'pn-face')) : el('p', { class: 'pn-note' }, 'No face crop stored.'),
        f.enrolled_crop ? el('figure', { class: 'pn-enrolled' }, thumb([f.enrolled_crop], 'enrolled face it matched', 'pn-face'), el('figcaption', {}, 'enrolled face (consented)')) : null),
      section('Face decision', d.status ? el('p', {}, el('b', {}, d.status.replace(/_/g, ' ')),
        sim != null ? ` · best enrolled face cosine ${num(sim)}` + (d.second_sim != null ? `, next ${num(d.second_sim)}` : '') : '') : null,
      sim != null && thr != null ? meter(Math.max(0, sim), Math.max(1, sim), thr, `face cosine ${sim}, threshold ${thr}`) : null,
      sim != null && thr != null ? el('p', { class: 'pn-note' }, `Mark = threshold ${num(thr)}: a face only decides the identity at or above it, with liveness passed, and only for enrolled household/known people.`) : null,
      (d.reasons || []).length ? el('div', { class: 'pn-reasons' }, d.reasons.map((r) => el('span', { class: 'code', title: r.code }, r.text))) : null),
      section('Quality', kv([
        ['Usable faces', f.usable ? `${f.n_faces} of ${f.faces_seen ?? '?'} seen` : `none usable${f.reason ? ` (${f.reason})` : ''}`],
        ['Quality', f.quality != null ? num(f.quality) : null],
        ['Face size', f.size_px ? `~${Math.round(f.size_px)} px` : f.max_size_px ? `largest ${Math.round(f.max_size_px)} px` : null],
        ['Liveness', f.liveness ? `${f.liveness.ok ? 'passed' : 'NOT passed'} (${num(f.liveness.score)})` : null],
        ['Consistency', f.consistency != null ? num(f.consistency) : null],
        ['Frigate face', f.frigate_face ? `${f.frigate_face.name} (${num(f.frigate_face.score)})` : null],
        ['Apparent expression', f.expression ? `${f.expression}: a description only, never used for identity or alerts` : null]])),
      faceModelSection(v),
      el('p', { class: 'pn-note' }, 'Faces name only enrolled, consenting household/known people. Face similarity between other visits appears in Related as correlation evidence, never as a name.'),
    ];
  }

  // 3D face (squirrel_guard/facefit.py) + this identity's prior 3D models (persons UI /api/identity/<id>/faces)
  const faceModels = new Map(); // identity -> index JSON | Promise | {error}
  function faceModelSection(v) {
    const ff = v.face_fit;
    if (!ff) return section('3D face', el('p', { class: 'pn-note' }, 'No fitted face for this visit (no usable face frames, or recorded before the face fit was running).'));
    const theme = document.documentElement.dataset.theme || '';
    const src = (ev, ref) => `face3d.html?identity=${encodeURIComponent(ff.identity)}&event=${encodeURIComponent(ev)}&embed=1`
      + (ref ? `&ref=${encodeURIComponent(ref)}` : '') + (theme ? `&theme=${theme}` : '') + (apiBase.pathname !== '/api/' ? `&api=${encodeURIComponent(apiBase.href)}` : '');
    const frame = el('iframe', { class: 'pn-face3d', src: src(S.focusEvent), title: '3D fitted face of this visit (drag to rotate)', loading: 'lazy' });
    const list = el('div', { class: 'pn-models' }, el('p', { class: 'pn-note' }, 'Loading prior 3D models…'));
    const fill = (ix) => {
      if (!ix || ix.error) { list.replaceChildren(el('p', { class: 'pn-note' }, `Prior models unavailable${ix && ix.error ? `: ${ix.error}` : ''}.`)); return; }
      const others = ix.models.filter((m) => m.event !== S.focusEvent);
      const me = ix.models.find((m) => m.event === S.focusEvent);
      list.replaceChildren(
        kv([['Prior 3D models', `${ix.models.length} for ${ix.identity.name}` + (ix.mean ? ` · models agree within ${ix.mean.consistency_rms_pct_median} % of the eye spacing (median)` : '')],
          ['This visit vs mean', me && me.vs_mean ? `${me.vs_mean.rms_pct} % RMS (p90 ${me.vs_mean.p90_pct} %)` : (ix.mean ? null : 'needs two fitted faces')]]),
        others.length ? el('ul', { class: 'pn-modellist' }, others.slice(0, 12).map((m) => el('li', {},
          el('span', {}, `${fmtTime(m.ts)} · ${m.camera} · ${m.n_frames} frames` + (m.vs_mean ? ` · ${m.vs_mean.rms_pct} % vs mean` : '')),
          el('span', { class: 'row' },
            el('button', { class: 'pn-btn sm', onclick: () => { frame.src = src(m.event, S.focusEvent); } }, 'View vs this'),
            el('a', { class: 'pn-btn sm', href: m.why, target: '_blank', rel: 'noopener' }, 'Why? ↗'))))) : null,
        el('p', { class: 'pn-note' }, ix.note));
    };
    const cached = faceModels.get(ff.identity);
    if (cached && !(cached instanceof Promise)) fill(cached);
    else {
      const p = cached || api(`identity/${encodeURIComponent(ff.identity)}/faces`).catch((e) => ({ error: e.message || String(e) }));
      faceModels.set(ff.identity, p);
      p.then((ix) => { faceModels.set(ff.identity, ix); fill(ix); });
    }
    return section('3D face', frame,
      kv([['Fit', `${ff.n_frames} frames` + (ff.reproj_px_med != null ? `, ${num(ff.reproj_px_med)} px median error` : '') + (ff.yaw_range ? `, yaw ${ff.yaw_range[0]}° to ${ff.yaw_range[1]}°` : '')],
        ['Scale', ff.metric ? 'millimetres from two calibrated cameras' : 'relative (one camera): proportions only'],
        ['Open', el('a', { href: src(S.focusEvent).replace('&embed=1', ''), target: '_blank', rel: 'noopener' }, 'full-size 3D viewer ↗')]]),
      list);
  }

  // -- related: radial node-link graph + list ---------------------------------------------------------------------
  function relatedNodes() {
    const v = S.visit;
    if (!v) return [];
    const r = v.related || [];
    const ids = r.filter((x) => x.type === 'identity' || x.type === 'possible_identity');
    const sights = r.filter((x) => x.type === 'sighting').slice(0, 10);
    const cars = r.filter((x) => x.type === 'vehicle').slice(0, 4);
    return [...ids, ...sights, ...cars];
  }
  function related() {
    if (!isReal()) {
      return [graphSVG([], null),
        el('p', { class: 'pn-note info' }, 'Related-sighting analysis (same identity, re-ID neighbours, face and gait similarity, vehicles nearby) appears on real visits. Demo samples have no stored visits to compare with.')];
    }
    const v = S.visit;
    if (!v) return stateBlock();
    const nodes = relatedNodes();
    const all = v.related || [];
    return [
      graphSVG(nodes, v),
      el('div', { class: 'pn-legend' }, BANDS.map((b) => el('span', {}, el('i', { class: `sw band-${b}` }), b)),
        el('span', {}, el('i', { class: 'sw ring-identity' }), 'identity'), el('span', {}, el('i', { class: 'sw ring-vehicle' }), 'vehicle')),
      el('p', { class: 'pn-note' }, `Closer and thicker = stronger correlation (estimated). Earlier sightings from the last ${v.window_days} days; labels name the evidence: re-ID (appearance), face, gait, same ID, time.`),
      section(`All related (${all.length})`, el('ul', { class: 'pn-rel' }, all.map((n) => el('li', {},
        el('button', { class: 'pn-rel-btn', onclick: (e) => showCard(n, e.currentTarget) },
          thumb([n.crop, n.links && n.links.thumbnail], '', 'pn-mini'),
          el('span', { class: 'who' }, nodeTitle(n), el('small', {}, nodeSub(n))),
          chip(n.strength.band, n.strength.pct)))))),
    ];
  }
  const TYPE_WORD = { identity: 'identity', possible_identity: 'possible identity', sighting: 'earlier visit', vehicle: 'vehicle' };
  const nodeTitle = (n) => n.name;
  function nodeSub(n) {
    if (n.type === 'sighting') return `${TYPE_WORD[n.type]} · ${fmtTime(n.start)} · ${n.camera}`;
    if (n.type === 'vehicle') return `vehicle · ${[n.plate, n.look].filter(Boolean).join(' · ')} · ${fmtTime(n.start)}`;
    return `${TYPE_WORD[n.type]} · ${n.sightings} sighting${n.sightings === 1 ? '' : 's'}`;
  }

  function graphSVG(nodes, v) {
    const W = 480, H = 360, cx = W / 2, cy = H / 2, rMin = 96, rMax = 138, XS = 1.32;
    const s = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'pn-graph', role: 'group', 'aria-label': 'related items graph: this visit in the centre' });
    const defs = sv('defs');
    s.append(defs);
    const gEdges = sv('g', { class: 'edges' }), gLabels = sv('g', { class: 'labels' }), gNodes = sv('g', { class: 'nodes' });
    s.append(gEdges, gLabels, gNodes);
    // even angular spread, grouped: identities at the top, vehicles at the bottom, visits around
    const n = nodes.length;
    const pos = new Map();
    const order = [...nodes.filter((x) => x.type === 'identity' || x.type === 'possible_identity'),
      ...nodes.filter((x) => x.type === 'sighting'), ...nodes.filter((x) => x.type === 'vehicle')];
    const idCount = order.filter((x) => x.type.includes('identity')).length;
    order.forEach((nd, i) => {
      const a = -Math.PI / 2 - (idCount > 1 ? (idCount - 1) * Math.PI / n : 0) + (i / Math.max(1, n)) * 2 * Math.PI;
      const p = Math.max(0, Math.min(1, nd.strength.p));
      const r = rMin + (1 - p) * (rMax - rMin);
      pos.set(nd, [cx + r * Math.cos(a) * XS, cy + r * Math.sin(a)]);
    });
    const clip = (id, r) => { defs.append(sv('clipPath', { id }, sv('circle', { r, cx: 0, cy: 0 }))); return `url(#${id})`; };
    // membership: visit -> its identity node, when that identity is in the graph
    const idNode = new Map(order.filter((x) => x.type.includes('identity')).map((x) => [x.id, x]));
    for (const nd of order) {
      if (nd.type !== 'sighting' || !idNode.has(nd.identity_id)) continue;
      const [x1, y1] = pos.get(nd), [x2, y2] = pos.get(idNode.get(nd.identity_id));
      gEdges.append(sv('line', { x1, y1, x2, y2, class: 'member' }));
    }
    let k = 0;
    for (const nd of order) {
      const [x, y] = pos.get(nd), p = nd.strength.p, band = nd.strength.band || 'none';
      const line = sv('line', { x1: cx, y1: cy, x2: x, y2: y, class: `edge band-${band}`, 'stroke-width': (1 + 4.5 * p).toFixed(2) });
      gEdges.append(line);
      const r = 12 + 7 * p, id = `pnc${++k}`;
      const lbl = edgeKinds(nd);
      if (lbl) { // along the edge, midway between the centre disc and the node, kept upright
        const half = lbl.length * 2.7, dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1;
        const mid = Math.max(30 + half, d - r - 6 - half); // next to the node, where the edges are furthest apart
        let ang = Math.atan2(dy, dx) * 180 / Math.PI;
        if (ang > 90 || ang < -90) ang += 180;
        gLabels.append(sv('text', { x: (cx + dx / d * mid).toFixed(1), y: (cy + dy / d * mid).toFixed(1), class: 'elabel', 'text-anchor': 'middle',
          'dominant-baseline': 'central', transform: `rotate(${ang.toFixed(1)} ${(cx + dx / d * mid).toFixed(1)} ${(cy + dy / d * mid).toFixed(1)})` }, lbl));
      }
      const g = sv('g', { class: `node t-${nd.type} band-${band}`, transform: `translate(${x.toFixed(1)},${y.toFixed(1)})`, tabindex: '0', role: 'button',
        'aria-label': `${nodeTitle(nd)}, ${nodeSub(nd)}, correlation ${nd.strength.band} ${nd.strength.pct} percent` });
      g.append(sv('title', {}, `${nodeTitle(nd)} — ${nodeSub(nd)}`));
      g.append(sv('circle', { r: r + 2.5, class: 'ring' }));
      const img = safeURL(nd.crop || (nd.links && nd.links.thumbnail));
      if (img) {
        const im = sv('image', { href: img, x: -r, y: -r, width: 2 * r, height: 2 * r, preserveAspectRatio: 'xMidYMid slice', 'clip-path': clip(id, r) });
        im.addEventListener('error', () => im.remove());
        g.append(sv('circle', { r, class: 'fillc' }), im);
      } else {
        g.append(sv('circle', { r, class: 'fillc' }), sv('text', { class: 'glyph', 'text-anchor': 'middle', 'dominant-baseline': 'central' }, nd.type === 'vehicle' ? 'V' : nd.type.includes('identity') ? 'ID' : '•'));
      }
      { // node label outside the node, pointing away from the centre
        const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1, ux = dx / d, uy = dy / d;
        const anchor = ux > 0.35 ? 'start' : ux < -0.35 ? 'end' : 'middle';
        g.append(sv('text', { class: 'nlabel', x: (ux * (r + 6)).toFixed(1), y: (uy * (r + 6) + (Math.abs(uy) > 0.6 ? uy * 6 : 0)).toFixed(1), 'text-anchor': anchor,
          'dominant-baseline': 'central' }, nd.type === 'sighting' ? fmtTime(nd.start).replace(/,.*$/, '') : nd.type === 'vehicle' ? (nd.plate || shorten(nd.look || 'vehicle', 12)) : shorten(nd.name, 14)));
      }
      const act = (e) => { showCard(nd, g); e.preventDefault(); };
      g.addEventListener('click', act);
      g.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') act(e); });
      g.addEventListener('pointerenter', () => line.classList.add('hot'));
      g.addEventListener('pointerleave', () => line.classList.remove('hot'));
      gNodes.append(g);
    }
    // centre: this visit
    const cr = 26, c = sv('g', { class: 'node centre', transform: `translate(${cx},${cy})` });
    c.append(sv('circle', { r: cr + 3, class: 'ring' }), sv('circle', { r: cr, class: 'fillc' }));
    const cimg = v && safeURL(v.links && (v.links.crop || v.links.thumbnail));
    if (cimg) { const im = sv('image', { href: cimg, x: -cr, y: -cr, width: 2 * cr, height: 2 * cr, preserveAspectRatio: 'xMidYMid slice', 'clip-path': clip('pncc', cr) }); im.addEventListener('error', () => im.remove()); c.append(im); }
    else c.append(sv('text', { class: 'glyph', 'text-anchor': 'middle', 'dominant-baseline': 'central' }, v ? '•' : '◎'));
    c.append(sv('text', { class: 'nlabel strong', y: cr + 14, 'text-anchor': 'middle' }, v ? 'this visit' : 'this sample'));
    gNodes.append(c);
    if (!nodes.length && v) gNodes.append(sv('text', { x: cx, y: cy + 62, class: 'nlabel', 'text-anchor': 'middle' }, 'nothing related in the window'));
    if (!v) gEdges.append(sv('circle', { cx, cy, r: (rMin + rMax) / 2, class: 'placeholder' }));
    return el('div', { class: 'pn-graph-wrap' }, s);
  }
  const shorten = (t, n) => (t && t.length > n ? t.slice(0, n - 1) + '…' : t || '');

  // -- mini card for a node / list row -----------------------------------------------------------------------------
  function showCard(n, from) {
    S.cardFor = from || null;
    const st = n.strength || {};
    const L = n.links || {};
    const slot = el('div', { class: 'pn-card-video' });
    const isVisit = n.type === 'sighting';
    const actions = el('div', { class: 'pn-actions' },
      L.clip ? el('button', { class: 'pn-btn', onclick: () => { slot.replaceChildren(videoPlayer(L.clip, L.explore)); } }, '▶ Play clip') : null,
      L.replay_data ? el('button', { class: 'pn-btn', onclick: () => host.openReplay(L.replay_data) }, 'Open 3D replay') : null,
      isVisit ? el('button', { class: 'pn-btn', onclick: () => { S.focusEvent = n.event; S.tab = 'overview'; loadVisit(n.event); } }, 'Focus here') : null,
      (n.type === 'identity' || n.type === 'possible_identity') ? el('button', { class: 'pn-btn', onclick: () => { closeCard(); S.tab = 'timeline'; loadTimeline(n.id); } }, 'Show timeline') : null,
      linkBtn(L.why, 'Why? ↗'), linkBtn(L.identity, 'Identity page ↗'), linkBtn(L.vehicle, 'Vehicle page ↗'), linkBtn(L.explore, 'Frigate ↗'));
    cardEl.replaceChildren(
      el('div', { class: 'pn-card-head' },
        thumb([n.crop, L.thumbnail], '', 'pn-card-thumb'),
        el('div', { class: 'who' }, el('b', {}, nodeTitle(n)), ' ', kindChip(n.kind), el('small', {}, nodeSub(n))),
        el('button', { class: 'pn-x', 'aria-label': 'close card', onclick: () => closeCard(true) }, '×')),
      el('p', {}, 'Correlation ', chip(st.band, st.pct), el('small', {}, ' estimated; a ranking aid, not a decision')),
      el('ul', { class: 'pn-list small' }, (st.edges || []).map((e) => el('li', { class: `eff-${e.effect}` }, e.text, e.effect && e.effect !== 'base' ? el('small', {}, ` (${e.effect})`) : null))),
      isVisit && (n.band || n.pct != null) ? el('p', { class: 'pn-note' }, 'Its own identity decision: ', chip(n.band, n.pct)) : null,
      slot, actions);
    cardEl.hidden = false;
    cardEl.setAttribute('aria-label', `details: ${nodeTitle(n)}`);
    requestAnimationFrame(() => { const b = cardEl.querySelector('.pn-actions .pn-btn'); (b || cardEl.querySelector('.pn-x')).focus(); });
  }
  function closeCard(refocus) {
    cardEl.querySelectorAll('video').forEach((x) => x.pause());
    cardEl.hidden = true;
    cardEl.replaceChildren();
    if (refocus && S.cardFor && document.contains(S.cardFor)) S.cardFor.focus();
    S.cardFor = null;
  }

  // -- clips -------------------------------------------------------------------------------------------------------
  function playClip(src, explore, what) {
    S.playing = { src, explore, what };
    const box = document.getElementById('pn-player');
    if (box) box.replaceChildren(el('p', { class: 'pn-note' }, `Playing: ${what}`), videoPlayer(src, explore));
  }
  function clips() {
    const v = S.visit;
    if (!v) return stateBlock();
    const strong = (v.related || []).filter((r) => r.type === 'sighting' && r.strength.p >= 0.4);
    const items = [{ self: true, name: 'This visit', start: v.start, camera: v.camera, links: v.links, crop: v.links.crop, strength: null }, ...strong];
    const player = el('div', { id: 'pn-player', class: 'pn-player' }, S.playing ? [el('p', { class: 'pn-note' }, `Playing: ${S.playing.what}`), videoPlayer(S.playing.src, S.playing.explore)]
      : el('p', { class: 'pn-note' }, 'Pick a clip to play it here (Frigate recording; needs the same sign-in).'));
    return [player,
      el('div', { class: 'pn-clips' }, items.map((c) => el('div', { class: 'pn-clip' + (c.self ? ' self' : '') },
        el('button', { class: 'pn-clip-play', 'aria-label': `play clip: ${c.self ? 'this visit' : c.name} ${fmtTime(c.start)}`,
          onclick: () => playClip(c.links.clip, c.links.explore, `${c.self ? 'this visit' : c.name}, ${fmtTime(c.start)}`) },
        thumb([c.links.thumbnail, c.crop], '', 'pn-clip-img'), el('span', { class: 'play', 'aria-hidden': 'true' }, '▶')),
        el('div', { class: 'meta' }, el('b', {}, c.self ? 'This visit' : c.name), ' ', c.strength ? chip(c.strength.band, c.strength.pct) : null,
          el('small', {}, `${fmtTime(c.start)} · ${c.camera}`),
          c.strength ? el('small', {}, edgeLabel(c.strength)) : null,
          el('span', { class: 'row' }, c.links.replay_data && !c.self ? el('button', { class: 'pn-btn sm', onclick: () => host.openReplay(c.links.replay_data) }, '3D') : null,
            linkBtn(c.links.why, 'Why? ↗')))))),
      strong.length ? null : el('p', { class: 'pn-note' }, `No earlier clip correlates at “possible” or better (last ${v.window_days} days).`)];
  }

  // -- timeline ----------------------------------------------------------------------------------------------------
  function timeline() {
    const v = S.visit;
    if (!v) return stateBlock();
    if (!v.identity) return el('p', { class: 'pn-note' }, 'This visit is not filed under an identity.');
    if (S.timelineFor == null) { loadTimeline(v.identity.id); return el('p', { class: 'pn-note', role: 'status' }, 'Loading…'); }
    if (S.timelineErr) return el('p', { class: 'pn-note warn' }, `Could not load the timeline: ${S.timelineErr}`);
    const T = S.timeline;
    if (!T) return el('p', { class: 'pn-note', role: 'status' }, 'Loading the identity’s visits…');
    const vs = T.visits || [];
    const other = T.identity.id !== v.identity.id;
    const head = el('p', {}, el('b', {}, T.identity.name), ' ', kindChip(T.identity.kind),
      el('small', {}, ` · ${T.identity.sightings} visit${T.identity.sightings === 1 ? '' : 's'} · first ${fmtTime(T.identity.first_seen)} · last ${fmtTime(T.identity.last_seen)}`),
      other ? el('button', { class: 'pn-btn sm', onclick: () => loadTimeline(v.identity.id) }, 'Back to this visit’s identity') : null);
    const facts = kv([['Height', T.height_cm_median ? `${Math.round(T.height_cm_median)} cm (median of ${T.height_n})` : null],
      ['Walk', T.cadence_spm_median ? `${Math.round(T.cadence_spm_median)} steps/min` : null],
      ['Enrolled faces', T.enrolled_faces ? String(T.enrolled_faces) : null],
      ['Vehicles nearby', (T.vehicles || []).length ? T.vehicles.map((x) => `${x.name}${x.look ? ` (${x.look})` : ''} ×${x.times}`).join(', ') : null]]);
    // strip: one dot per visit on a time axis
    const W = 440, H = 58, ts = vs.map((x) => +new Date(x.start)), t0 = Math.min(...ts), t1 = Math.max(...ts);
    const X = (t) => 14 + (t1 > t0 ? (t - t0) / (t1 - t0) : 0.5) * (W - 28);
    const strip = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'pn-strip', role: 'img', 'aria-label': `${vs.length} visits from ${fmtTime(vs.length ? vs[vs.length - 1].start : null)} to ${fmtTime(vs.length ? vs[0].start : null)}` },
      sv('line', { x1: 10, x2: W - 10, y1: 30, y2: 30, class: 'axis' }),
      vs.map((x) => sv('circle', { cx: X(+new Date(x.start)).toFixed(1), cy: 30, r: x.event === S.focusEvent ? 7 : 4.5, class: `dot band-${x.band || 'none'}${x.event === S.focusEvent ? ' cur' : ''}${x.unusual ? ' unusual' : ''}` },
        sv('title', {}, `${fmtTime(x.start)} · ${x.camera}${x.band ? ` · ${x.band} ${x.pct}%` : ''}`))),
      vs.length ? [sv('text', { x: 10, y: 52, class: 'tick' }, fmtTime(vs[vs.length - 1].start)), sv('text', { x: W - 10, y: 52, class: 'tick', 'text-anchor': 'end' }, fmtTime(vs[0].start))] : null);
    let day = null;
    const rows = [];
    for (const x of vs) {
      const dkey = fmtDay(x.start);
      if (dkey !== day) { day = dkey; rows.push(el('li', { class: 'day' }, dkey)); }
      rows.push(el('li', { class: x.event === S.focusEvent ? 'cur' : '' },
        el('span', { class: 't' }, new Date(x.start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
        thumb([x.crop], '', 'pn-mini'),
        el('span', { class: 'who' }, x.camera, x.unusual ? el('span', { class: 'chip band-unlikely' }, 'unusual') : null, x.height_cm ? el('small', {}, `${Math.round(x.height_cm)} cm`) : null),
        chip(x.band, x.pct),
        el('span', { class: 'row' },
          el('button', { class: 'pn-btn sm', 'aria-label': `play clip ${fmtTime(x.start)}`, onclick: (e) => { const li = e.currentTarget.closest('li'); let s = li.nextElementSibling; if (s && s.classList.contains('vid')) { s.remove(); return; } s = el('li', { class: 'vid' }, videoPlayer(x.links.clip, x.links.explore)); li.after(s); } }, '▶'),
          x.links.replay_data && x.event !== S.sceneEvent ? el('button', { class: 'pn-btn sm', onclick: () => host.openReplay(x.links.replay_data) }, '3D') : null,
          x.event !== S.focusEvent ? el('button', { class: 'pn-btn sm', onclick: () => { S.focusEvent = x.event; S.tab = 'overview'; loadVisit(x.event); } }, 'Focus') : null,
          linkBtn(x.links.why, 'Why? ↗'))));
    }
    return [head, facts, strip, el('ul', { class: 'pn-timeline' }, rows), vs.length >= 200 ? el('p', { class: 'pn-note' }, 'Newest 200 visits shown.') : null];
  }

  // -- samples: what the replay JSON holds about the scene and the models -------------------------------------------
  function scene() {
    const d = host.getData() || {};
    const C = d.calibration || {}, cam = d.camera || {}, R = d.replay || {}, B = d.body3d, st = (B && B.stats) || {};
    const K = cam.K;
    return [
      section('Camera calibration', kv([
        ['Method', C.method === 'walk' ? 'from the walker (height assumed)' : C.method === 'given' ? 'given / solved offline' : C.method],
        ['Camera height', cam.C ? `${cam.C[2].toFixed(2)} m` : null],
        ['Pitch', C.pitch_deg != null ? `${num(C.pitch_deg, 1)}° (${C.pitch_note || ''})` : null],
        ['Roll', C.roll_deg != null ? `${num(C.roll_deg, 1)}°` : null],
        ['Lens', K ? `focal ${Math.round(K[0][0])} px, ${cam.size ? cam.size.join('×') : ''}` + (C.hfov_assumed_deg ? `, hfov ${C.hfov_assumed_deg}° assumed` : '') + (cam.D && cam.D[0] ? `, k1 ${num(cam.D[0])}` : '') : null],
        ['Residual', C.residual_cm_median_abs != null ? `${C.residual_cm_median_abs} cm median, ${C.residual_cm_p90_abs} cm p90 (${C.observations} observations)` : null]]),
      C.solved_by ? el('details', {}, el('summary', {}, 'How it was solved'), el('p', { class: 'pn-note' }, C.solved_by)) : null),
      section('Replay media', kv([
        ['Cut-outs', R.cutouts && R.cutouts.items ? `${R.cutouts.items.length} person cut-outs` + (R.stats && R.stats.mask_missed != null ? `, ${R.stats.mask_missed} masks missed` : '') : 'none'],
        ['Projection', R.ground ? `ground mesh ${R.ground.cells} cells to ${R.ground.max_range_m} m, backdrop ${R.ground.backdrop_cells}` : 'none'],
        ['Splat', d.assets && d.assets.splat ? (d.splat_note || 'Gaussian-splat scan') : null],
        ['Head direction', d.gaze ? `${d.gaze.focus_deg}° focus / ${d.gaze.fov_deg}° field cone; ${d.gaze.note}` : null]])),
      B ? section(B.root_mode === 'rig' ? '3D body (SMPL on the IK rig)' : '3D body (WHAM)', kv([
        ['Method', B.method], ['Model', B.model], ['Frames with a body', st.replay_frames_with_body != null ? `${st.replay_frames_with_body} of ${st.replay_frames}` : null],
        ['Fit to the rig', st.marker_err_cm_median != null ? `${st.marker_err_cm_median} cm median marker error (${st.marker_err_cm_p90} cm p90), fitted in ${(st.fit_ms / 1000).toFixed(1)} s` : null],
        ['Fitted stature', st.stature_m ? `${st.stature_m} m (path assumed ${st.stature_assumed_m} m)` : null],
        ['Image fit', st.image_fit && st.image_fit.reproj_px_after ? `joint reprojection ${st.image_fit.reproj_px_after.median} px median (was ${st.image_fit.reproj_px_before.median})` : null],
        ['Legs', st.image_fit && st.image_fit.frames_legs_replaced != null ? `${st.image_fit.frames_legs_replaced} of ${st.image_fit.frames} frames used a walking prior (legs out of view)` : null],
        ['Mesh', B.mesh ? `${B.mesh.vertex_count} vertices, ${B.mesh.face_count} faces` : 'joints only'],
        ['Licence', B.licence]])) : el('p', { class: 'pn-note' }, 'No 3D body in this recording (pick a “+ 3D body” sample).'),
    ];
  }

  // -- Body & gait: the IK rig (squirrel_guard.rig) ---------------------------------------------------------------
  const SEG_LABEL = { stature: 'Stature', thigh: 'Thigh (hip-knee)', shank: 'Shank (knee-ankle)', upper_arm: 'Upper arm',
    forearm: 'Forearm', spine: 'Pelvis to shoulders', sh_half: 'Shoulder width', hip_half: 'Hip width', neck: 'Neck to head centre',
    toe: 'Ankle to toe', heel: 'Ankle to heel', ankle_h: 'Ankle height' };
  function angleChart(title, t, L, R, tNow) {
    const W = 320, H = 110, pl = 30, pr = 6, pt = 8, pb = 18;
    const vals = [...L, ...R].filter((v) => v != null);
    if (vals.length < 3) return el('p', { class: 'pn-note' }, `${title}: too few frames with this joint in view.`);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    lo = Math.floor(Math.min(lo, 0) / 10) * 10; hi = Math.ceil(Math.max(hi, lo + 20) / 10) * 10;
    const t0 = t[0], t1 = t[t.length - 1] || t0 + 1;
    const X = (tv) => pl + (tv - t0) / Math.max(1e-6, t1 - t0) * (W - pl - pr);
    const Y = (v) => pt + (hi - v) / (hi - lo) * (H - pt - pb);
    const path = (A) => { let d = '', pen = false; A.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? 'L' : 'M'}${X(t[i]).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; }); return d; };
    const grid = [];
    for (let v = lo; v <= hi; v += (hi - lo > 80 ? 30 : 15)) {
      grid.push(sv('line', { x1: pl, x2: W - pr, y1: Y(v), y2: Y(v), class: 'gr' }), sv('text', { x: pl - 4, y: Y(v) + 3, class: 'ax', 'text-anchor': 'end' }, `${v}°`));
    }
    const head = sv('line', { x1: X(tNow), x2: X(tNow), y1: pt, y2: H - pb, class: 'ph' });
    playheads.push({ line: head, X, t0, t1 });
    const rom = (A) => { const v = A.filter((x) => x != null).sort((a, b) => a - b); return v.length > 5 ? Math.round(v[Math.floor(v.length * 0.95)] - v[Math.floor(v.length * 0.05)]) : null; };
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'pn-angles', role: 'img',
      'aria-label': `${title} over time: left range ${rom(L) ?? 'n/a'} degrees, right ${rom(R) ?? 'n/a'} degrees` },
    ...grid, sv('path', { d: path(L), class: 'ln l' }), sv('path', { d: path(R), class: 'ln r' }), head,
    sv('text', { x: pl, y: H - 4, class: 'ax' }, `${t0.toFixed(0)} s`), sv('text', { x: W - pr, y: H - 4, class: 'ax', 'text-anchor': 'end' }, `${t1.toFixed(1)} s`));
    return el('figure', { class: 'pn-fig' }, el('figcaption', {}, title, ' ', el('span', { class: 'lg l' }, 'L'), ' ', el('span', { class: 'lg r' }, 'R')), svg);
  }
  function body() {
    const d = host.getData() || {}, G = d.rig || {}, A = G.angles_deg || {}, g = G.gait || {};
    const t = d.t || [], tNow = host.getT ? host.getT() : 0;
    const err = Object.entries(G.error || {}).map(([k, v]) => (v && v.median_err_px != null ? `${k} ${Math.round(v.median_err_px)} px` : null)).filter(Boolean).join(', ');
    const segs = G.segments_m || {};
    const lr = (o) => (o ? `L ${o.L ?? 'n/a'}°, R ${o.R ?? 'n/a'}°` : null);
    const cad = g.cadence_spm != null ? `${g.cadence_spm} steps/min` + (g.step_time_s_cv != null && g.step_time_s_cv > 0.25 ? ` (irregular: step-time variation ${Math.round(g.step_time_s_cv * 100)} %)` : '') : null;
    return [
      section('Rig', kv([
        ['Stature', G.stature_m != null ? `${G.stature_m.toFixed(2)} m (rig scale; the world scale rests on the camera calibration)` : null],
        ['Cameras', (G.cameras || []).join(', ')],
        ['Frames fitted', G.frames_fitted != null ? `${G.frames_fitted} of ${t.length}` : null],
        ['Fit error', err ? `${err} (median, keypoints vs the rig re-projected)` : null]]),
      el('details', {}, el('summary', {}, 'How the rig works'), el('p', { class: 'pn-note' },
        'A procedural skeleton with fixed bone lengths for the whole visit (anthropometric proportions, left = right) and joint angles kept in human ranges (knees and elbows bend one way), fitted to whole-body keypoints (RTMW, 133 points incl. heels and toes) in every calibrated camera at once, smoothed over time, feet on the floor.'))),
      section('Joint angles', angleChart('Knee flexion', t, A.knee_L || [], A.knee_R || [], tNow),
        angleChart('Hip flexion', t, A.hip_flex_L || [], A.hip_flex_R || [], tNow),
        angleChart('Elbow flexion', t, A.elbow_L || [], A.elbow_R || [], tNow)),
      partsSection(G.parts, t, tNow),
      section('Gait', kv([
        ['Walking', g.walking_frames != null ? `${g.walking_frames} of ${g.frames} frames` + (g.speed_mps_walking_median != null ? `, ${g.speed_mps_walking_median} m/s median` : '') : null],
        ['Cadence', cad], ['Step length', g.step_length_m_median != null ? `${g.step_length_m_median} m median` : null],
        ['Knee range', lr(g.knee_rom_deg)], ['Knee symmetry', g.knee_symmetry != null ? `${g.knee_symmetry} (1 = equal)` : null],
        ['Hip range', lr(g.hip_flex_rom_deg)], ['Hip symmetry', g.hip_flex_symmetry != null ? `${g.hip_flex_symmetry} (1 = equal)` : null],
        ['Legs in view', g.leg_frames_observed ? `L ${g.leg_frames_observed.L}, R ${g.leg_frames_observed.R} walking frames` : null]]),
      g.note ? el('p', { class: 'pn-note' }, g.note) : null,
      el('p', { class: 'pn-note' }, 'Descriptive measurements of this visit only; ranges from few frames (legs hidden, short walks) are unreliable and are not a health or identity assessment.')),
      section('Segment lengths', kv(Object.entries(segs).map(([k, v]) => [SEG_LABEL[k] || k, `${(k.endsWith('_half') ? 2 * v : v).toFixed(2)} m`]))),
    ];
  }
  function partsSection(P, t, tNow) {
    if (!P) return null;
    const ur = P.usable_rate || {}, pct = (v) => (v != null ? `${Math.round(v * 100)} %` : 'n/a');
    const ypr = P.head_ypr_deg || [], col = (k) => ypr.map((v) => (v ? v[k] : null));
    return section('Head & hands', kv([
      ['Face usable', `${pct(ur.face)} of fitted frames (head pose and jaw measured there; elsewhere not shown)`],
      ['Hands usable', `L ${pct(ur.L)}, R ${pct(ur.R)} (wrist rotation and finger curl)`]]),
    angleChart('Head yaw (L line) and pitch (R line)', t, col(0), col(1), tNow),
    angleChart('Jaw opening', t, P.jaw_deg || [], [], tNow),
    el('p', { class: 'pn-note' }, 'Orientation only, from the face and hand keypoints of every calibrated camera; never used to identify anyone. Reliable when faces are about 60 px or larger.'));
  }
  function onTime(tv) {
    for (const p of playheads) {
      const x = p.X(Math.min(Math.max(tv, p.t0), p.t1));
      p.line.setAttribute('x1', x); p.line.setAttribute('x2', x);
    }
  }

  return { open, close, onData, isOpen: () => S.open, render, onTime };
}
