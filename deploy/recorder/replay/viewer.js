// Trajectory replay: three.js scene for squirrel_guard.trajectory.estimate() output.
// World frame of the JSON: metres, Z up, ground Z = 0. three.js is Y up, so everything sits in `world`,
// a group rotated so that world (x, y, z) -> three (x, z, -y).
// Optional replay media (squirrel_guard.replay_assets): the camera's picture projected on the ground + a backdrop,
// and the person as video cut-outs on cards facing the camera. Both are textured per pixel by projecting the
// fragment's world position through the calibrated camera (K, D incl. lens distortion, R, C), so in Camera view
// they reproduce the camera's picture exactly (undistorted); the meshes only decide where to draw.
// Optional "body3d" (scripts/body3d_rig.py / the profiler: SMPL dressed onto the IK rig; or scripts/body3d_wham.py,
// offline WHAM): per-frame SMPL joints in the same world frame, and a
// body mesh from two binary files (float16 vertices per frame relative to the frame's root + a shared uint16
// triangle list). Those files are SMPL-derived and may be absent: the viewer then draws the 3D joints only.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// Investigation pane (pane.js, docs/replay-pane.md): clicking the person (stick figure, body, cut-out, marker; or the
// Inspect button / I key) opens a sub-dashboard of the stored visit and what it correlates with.
import { createPane } from './pane.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const col = (name) => new THREE.Color(css(name));
const CHI2_95 = 5.991;

// -- renderer / scene ---------------------------------------------------------------------------------------
const canvas = $('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
scene.add(new THREE.HemisphereLight(0xffffff, 0x667788, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(5, 10, 4);
scene.add(sun);

const W = (x, y, z) => new THREE.Vector3(x, y, z); // world-frame vector (used inside `world`)
// invisible, generous pick target around the person (outside `built`: still clickable with the overlays hidden,
// e.g. when only the video cut-out or the body mesh is shown); radius 0.4 m, 1.8 m tall, axis = world Z
const hit = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.4, 1.8, 12).rotateX(Math.PI / 2), new THREE.MeshBasicMaterial({ visible: false }));
world.add(hit);

let camMode = false; // true while looking out of the Wyze camera (principal-point view offset applied)
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  if (camMode && data) setCamOffset();
  camera.updateProjectionMatrix();
  document.documentElement.style.setProperty('--bottom-h', `${$('bottom').offsetHeight}px`);
}

// -- helpers ------------------------------------------------------------------------------------------------
function eig2(c) { // 2x2 symmetric -> {r1, r2, angle} of the 95 % ellipse
  const a = c[0][0], b = c[0][1], d = c[1][1];
  const tr = a + d, det = a * d - b * b;
  const s = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = tr / 2 + s, l2 = Math.max(1e-9, tr / 2 - s);
  const ang = Math.abs(b) < 1e-12 ? (a >= d ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
  return { r1: Math.sqrt(CHI2_95 * l1), r2: Math.sqrt(CHI2_95 * l2), angle: ang };
}
function ellipseShape(c) {
  const e = eig2(c);
  const sh = new THREE.Shape();
  sh.absellipse(0, 0, e.r1, e.r2, 0, Math.PI * 2, false, 0);
  return { shape: sh, e };
}
function placeFlat(mesh, x, y, z, angle) { // a Shape built in XY -> on the ground at height z, rotated about Z
  mesh.position.set(x, y, z);
  mesh.rotation.z = angle;
}
function timeColor(u) { return col('--c-start').lerp(col('--c-end'), u); }
function lerpIdx(ts, t) { // -> [i, frac]
  if (t <= ts[0]) return [0, 0];
  if (t >= ts[ts.length - 1]) return [ts.length - 2, 1];
  let lo = 0, hi = ts.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (ts[m] <= t) lo = m; else hi = m; }
  return [lo, (t - ts[lo]) / (ts[lo + 1] - ts[lo] || 1)];
}
function footShape(side) { // a small foot outline ~0.26 m long, pointing +X, toes forward
  const s = new THREE.Shape();
  const m = side === 'L' ? 1 : -1;
  s.moveTo(-0.13, -0.045 * m);
  s.bezierCurveTo(-0.14, 0.05 * m, -0.06, 0.06 * m, 0.0, 0.05 * m);
  s.bezierCurveTo(0.07, 0.045 * m, 0.12, 0.055 * m, 0.13, 0.02 * m);
  s.bezierCurveTo(0.135, -0.03 * m, 0.08, -0.05 * m, 0.0, -0.045 * m);
  s.bezierCurveTo(-0.06, -0.04 * m, -0.12, -0.08 * m, -0.13, -0.045 * m);
  return s;
}

// -- scene builders ------------------------------------------------------------------------------------------
const built = new THREE.Group();
world.add(built);
let data = null, playing = false, tCur = 0, tEnd = 1, lastTick = 0, skelFrames = null;
let skelLines, skelJoints, marker, markerEllipse, obsFlags, gridHelper;
let rigFeet = null; // rig heel/toe segments (data.rig.markers 15-22), toggled by #rigfeet
const FOOT_BONES = [[15, 19], [15, 17], [19, 17], [17, 18], [16, 22], [16, 20], [22, 20], [20, 21]]; // ankle/heel/toes
let gazeData = null, gazeGroup = null;   // head-direction view cone (data.gaze)
let bodyFrames = null, bodyLines, bodyJoints; // body3d joints (COCO-17, world frame, per entry of t or null)

const BONES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]];

function build(d) {
  built.clear();
  const t = d.t, mean = d.mean, cov = d.cov, obs = d.observed;
  const n = t.length;
  obsFlags = obs;
  const t0 = t[0], t1 = t[n - 1];
  const cx = mean.reduce((s, p) => s + p[0], 0) / n, cy = mean.reduce((s, p) => s + p[1], 0) / n;

  // ground grid (1 m), 30 x 30 m around the path
  const grid = new THREE.GridHelper(30, 30, col('--c-grid'), col('--c-grid2'));
  grid.rotation.x = Math.PI / 2; // GridHelper lies in XZ; inside `world` we want XY
  grid.position.set(Math.round(cx), Math.round(cy), 0);
  grid.material.transparent = true;
  built.add(grid);
  gridHelper = grid;
  const axes = new THREE.AxesHelper(1);
  axes.position.set(Math.round(cx) - 15, Math.round(cy) - 15, 0.001);
  built.add(axes);

  // zones
  for (const z of d.zones || []) {
    const sh = new THREE.Shape(z.polygon.map(([x, y]) => new THREE.Vector2(x, y)));
    const m = new THREE.Mesh(new THREE.ShapeGeometry(sh), new THREE.MeshBasicMaterial({ color: col('--c-zone'), transparent: true, opacity: 0.18, side: THREE.DoubleSide }));
    m.position.z = 0.003;
    built.add(m);
    const o = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(z.polygon.map(([x, y]) => W(x, y, 0.004))), new THREE.LineBasicMaterial({ color: col('--c-zone') }));
    built.add(o);
  }

  // camera position + frustum (pinhole from K and size; distortion ignored); extra cameras of a multi-camera
  // replay (cameras_extra, same world frame) are drawn the same way, fainter, with their name
  const drawCam = (c, primary) => {
    const C = c.C, R = c.R, K = c.K, [iw, ih] = c.size;
    const camPos = W(C[0], C[1], C[2]);
    const camBox = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.08, 0.08), new THREE.MeshStandardMaterial({ color: col('--c-cam') }));
    camBox.position.copy(camPos);
    built.add(camBox);
    const rayW = (u, v) => { // pixel -> world unit direction (R is world_from_camera)
      const xc = (u - K[0][2]) / K[0][0], yc = (v - K[1][2]) / K[1][1];
      const dir = [R[0][0] * xc + R[0][1] * yc + R[0][2], R[1][0] * xc + R[1][1] * yc + R[1][2], R[2][0] * xc + R[2][1] * yc + R[2][2]];
      return W(...dir).normalize();
    };
    const corners = [[0, 0], [iw, 0], [iw, ih], [0, ih]].map(([u, v]) => rayW(u, v));
    const fl = primary ? 2.0 : 1.4;
    const fpts = [];
    for (let i = 0; i < 4; i++) {
      const a = camPos.clone().add(corners[i].clone().multiplyScalar(fl));
      const b = camPos.clone().add(corners[(i + 1) % 4].clone().multiplyScalar(fl));
      fpts.push(camPos.clone(), a, a, b);
    }
    built.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(fpts), new THREE.LineBasicMaterial({ color: col('--c-cam'), transparent: true, opacity: primary ? 0.8 : 0.5 })));
    if (c.name) {
      const cv = document.createElement('canvas');
      cv.width = 256; cv.height = 64;
      const g = cv.getContext('2d');
      g.font = '600 34px system-ui, sans-serif'; g.fillStyle = col('--c-cam').getStyle(); g.textBaseline = 'middle';
      g.fillText(c.name, 4, 32);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthTest: false }));
      sp.scale.set(0.6, 0.15, 1);
      sp.position.copy(camPos).add(new THREE.Vector3(0, 0.15, 0));
      built.add(sp);
    }
    if (!primary) return;
    // footprint of the view on the ground (where the corner rays hit Z = 0, capped at 25 m)
    const gpts = corners.map((r) => {
      const s = r.z < -1e-3 ? Math.min(25, -C[2] / r.z) : 25;
      return camPos.clone().add(r.clone().multiplyScalar(s)).setZ(0.002);
    });
    built.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(gpts), new THREE.LineDashedMaterial({ color: col('--c-cam'), dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.5 })).computeLineDistances());
  };
  drawCam(d.camera, true);
  for (const c of d.cameras_extra || []) drawCam(c, false);

  // smoothed path: tube per run of observed / filled-in samples, vertex colour = time
  let i = 0;
  while (i < n - 1) {
    let j = i;
    while (j + 1 < n && obs[j + 1] === obs[i]) j++;
    const lo = Math.max(0, i - 1), hi = Math.min(n - 1, j + 1);
    if (hi - lo >= 1) {
      const pts = [];
      for (let k = lo; k <= hi; k++) pts.push(W(mean[k][0], mean[k][1], 0.03));
      const curve = new THREE.CatmullRomCurve3(pts, false, 'centripetal', 0.0);
      const seg = Math.max(2, (hi - lo) * 3);
      const geo = new THREE.TubeGeometry(curve, seg, obs[i] ? 0.06 : 0.035, 10, false);
      const pos = geo.attributes.position, colors = new Float32Array(pos.count * 3);
      const uv = geo.attributes.uv;
      for (let v = 0; v < pos.count; v++) {
        const tt = t[lo] + uv.getX(v) * (t[hi] - t[lo]);
        const c = timeColor((tt - t0) / (t1 - t0 || 1));
        colors.set([c.r, c.g, c.b], v * 3);
      }
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, transparent: !obs[i], opacity: obs[i] ? 1 : 0.35 });
      built.add(new THREE.Mesh(geo, mat));
      if (!obs[i]) { // dashed ground line marks the filled-in stretch
        const dl = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts.map((p) => p.clone().setZ(0.006))),
          new THREE.LineDashedMaterial({ color: timeColor((t[(lo + hi) >> 1] - t0) / (t1 - t0 || 1)), dashSize: 0.12, gapSize: 0.08 }));
        dl.computeLineDistances();
        built.add(dl);
      }
    }
    i = j + 1;
  }

  // 95 % ellipses every ~0.5 s (and at the end of each gap)
  const step = Math.max(1, Math.round(0.5 / ((t1 - t0) / (n - 1) || 0.125)));
  const ellMat = new THREE.MeshBasicMaterial({ color: col('--c-ellipse'), transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false });
  const ellEdge = new THREE.LineBasicMaterial({ color: col('--c-ellipse'), transparent: true, opacity: 0.6 });
  for (let k = 0; k < n; k += step) {
    const { shape, e } = ellipseShape(cov[k]);
    const m = new THREE.Mesh(new THREE.ShapeGeometry(shape, 32), ellMat);
    placeFlat(m, mean[k][0], mean[k][1], 0.008, e.angle);
    built.add(m);
    const o = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(shape.getPoints(32).map((p) => W(p.x, p.y, 0))), ellEdge);
    placeFlat(o, mean[k][0], mean[k][1], 0.009, e.angle);
    built.add(o);
  }

  // footprints
  const vel = d.velocity;
  for (const f of d.footprints || []) {
    const [k] = lerpIdx(t, f.t);
    const v = vel ? vel[k] : [1, 0];
    const ang = Math.atan2(v[1], v[0]);
    const m = new THREE.Mesh(new THREE.ShapeGeometry(footShape(f.side), 8),
      new THREE.MeshBasicMaterial({ color: col(f.side === 'L' ? '--c-foot-l' : '--c-foot-r'), transparent: true, opacity: f.estimated ? 0.6 : 0.95, side: THREE.DoubleSide }));
    placeFlat(m, f.x, f.y, 0.012, ang);
    m.scale.set(1.3, 1.3, 1);
    built.add(m);
    if (f.sx && f.sy) { // thin uncertainty ring around the estimated footprint
      const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(new THREE.EllipseCurve(0, 0, 2 * f.sx, 2 * f.sy, 0, 2 * Math.PI).getPoints(20).map((p) => W(p.x, p.y, 0))),
        new THREE.LineBasicMaterial({ color: col(f.side === 'L' ? '--c-foot-l' : '--c-foot-r'), transparent: true, opacity: 0.25 }));
      placeFlat(ring, f.x, f.y, 0.011, 0);
      built.add(ring);
    }
  }

  // prediction: translucent widening cone (rings of 95 % ellipses joined into a surface) + mean line
  const P = d.prediction;
  if (P && P.t && P.t.length) {
    const rings = [[mean[n - 1][0], mean[n - 1][1], cov[n - 1]]].concat(P.t.map((_, k) => [P.mean[k][0], P.mean[k][1], P.cov[k]]));
    const R_ = 24, verts = [], idx = [];
    rings.forEach(([x, y, c], r) => {
      const e = eig2(c), ca = Math.cos(e.angle), sa = Math.sin(e.angle);
      for (let a = 0; a < R_; a++) {
        const th = (a / R_) * Math.PI * 2, ex = e.r1 * Math.cos(th), ey = e.r2 * Math.sin(th);
        verts.push(x + ex * ca - ey * sa, y + ex * sa + ey * ca, 0.015 + 0.002 * r);
      }
      if (r > 0) for (let a = 0; a < R_; a++) {
        const b = (a + 1) % R_, p = (r - 1) * R_, q = r * R_;
        idx.push(p + a, q + a, p + b, q + a, q + b, p + b);
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    built.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col('--c-pred'), transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false })));
    const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      Array.from({ length: R_ }, (_, a) => W(verts[(rings.length - 1) * R_ * 3 + a * 3], verts[(rings.length - 1) * R_ * 3 + a * 3 + 1], 0.02))),
      new THREE.LineBasicMaterial({ color: col('--c-pred') }));
    built.add(edge);
    const ml = new THREE.Line(new THREE.BufferGeometry().setFromPoints([W(mean[n - 1][0], mean[n - 1][1], 0.03)].concat(P.mean.map((p) => W(p[0], p[1], 0.03)))),
      new THREE.LineDashedMaterial({ color: col('--c-pred'), dashSize: 0.15, gapSize: 0.1 }));
    ml.computeLineDistances();
    built.add(ml);
  }

  // ground truth (synthetic only)
  if (d.truth && d.truth.xy) {
    built.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(d.truth.xy.map((p) => W(p[0], p[1], 0.005))), new THREE.LineBasicMaterial({ color: col('--c-truth') })));
    $('truthrow').hidden = false;
  }

  // stick figure + current-position marker (updated per frame)
  skelFrames = d.skeleton || null;
  skelLines = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(BONES.length * 6), 3)),
    new THREE.LineBasicMaterial({ color: col('--c-skel'), transparent: true, opacity: 1 }));
  skelLines.frustumCulled = false;
  built.add(skelLines);
  skelJoints = new THREE.InstancedMesh(new THREE.SphereGeometry(0.035, 10, 8), new THREE.MeshStandardMaterial({ color: col('--c-skel'), transparent: true }), 17);
  skelJoints.frustumCulled = false;
  built.add(skelJoints);
  rigFeet = null;
  $('rigfeetlbl').hidden = !(d.rig && d.rig.markers);
  if (d.rig && d.rig.markers) {
    rigFeet = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(FOOT_BONES.length * 6), 3)),
      new THREE.LineBasicMaterial({ color: col('--c-skel'), transparent: true, opacity: 1 }));
    rigFeet.frustumCulled = false;
    built.add(rigFeet);
  }
  // aerial photo as the ground (map-calibrated cameras: data.aerial = {url, extent_m: [xmin, ymin, xmax, ymax]} in
  // the same east/north metres as the path; USGS NAIP, loaded by this browser straight from USGS)
  if (d.aerial && d.aerial.url && d.aerial.extent_m) {
    const [x0, y0, x1, y1] = d.aerial.extent_m;
    const tex = new THREE.TextureLoader().setCrossOrigin('anonymous').load(d.aerial.url, () => render());
    tex.colorSpace = THREE.SRGBColorSpace;
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.9, depthWrite: false }));
    plane.position.set((x0 + x1) / 2, (y0 + y1) / 2, -0.01);  // PlaneGeometry lies in XY, facing +Z: no rotation needed
    plane.renderOrder = -1;
    built.add(plane);
    document.querySelectorAll('.lg-aerial').forEach((el) => { el.hidden = false; });
  }
  // view cone: apex at the eyes, pointing along the head yaw (world frame, Z up); two nested cones
  gazeData = d.gaze && d.gaze.yaw ? d.gaze : null;
  gazeGroup = null;
  if (gazeData) {
    const L = 3.0, mk = (half, op) => {
      const g = new THREE.ConeGeometry(L * Math.tan(half * Math.PI / 180), L, 32, 1, true);
      g.translate(0, -L / 2, 0); g.rotateZ(Math.PI / 2);          // apex at the origin, opening along +X
      return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: col('--c-gaze'), transparent: true, opacity: op,
        side: THREE.DoubleSide, depthWrite: false }));
    };
    gazeGroup = new THREE.Group();
    gazeGroup.add(mk(gazeData.focus_deg / 2, 0.22), mk(Math.min(80, gazeData.fov_deg / 2), 0.035));
    gazeGroup.children[1].scale.setScalar(0.6);                  // wide field: shorter, so it does not hide the scene
    const ray = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(L, 0, 0)]),
      new THREE.LineBasicMaterial({ color: col('--c-gaze') }));
    gazeGroup.add(ray);
    gazeGroup.visible = false;
    built.add(gazeGroup);
    document.querySelectorAll('.lg-gaze').forEach((el) => { el.hidden = false; });
  }
  // 3D body joints (body3d), same bones as the stick figure, no re-centring: they are already in the world frame
  bodyFrames = (d.body3d && d.body3d.joints) || null;
  if (bodyFrames) {
    bodyLines = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(BONES.length * 6), 3)),
      new THREE.LineBasicMaterial({ color: col('--c-body'), transparent: true }));
    bodyLines.frustumCulled = false;
    bodyJoints = new THREE.InstancedMesh(new THREE.SphereGeometry(0.03, 10, 8), new THREE.MeshStandardMaterial({ color: col('--c-body'), transparent: true }), 17);
    bodyJoints.frustumCulled = false;
    built.add(bodyLines, bodyJoints);
  }
  if (body) body.mesh.material.color = col('--c-body');
  marker = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), new THREE.MeshStandardMaterial({ color: col('--c-end') }));
  built.add(marker);
  markerEllipse = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(new THREE.EllipseCurve(0, 0, 1, 1, 0, 2 * Math.PI).getPoints(40).map((p) => W(p.x, p.y, 0))), new THREE.LineBasicMaterial({ color: col('--c-end') }));
  built.add(markerEllipse);

  // timeline
  tEnd = P && P.t.length ? P.t[P.t.length - 1] : t1;
  const slider = $('time');
  slider.min = t0; slider.max = tEnd; slider.step = 0.02; slider.value = t0;
  tCur = t0;
  const M = d.metrics || {};
  $('plen').textContent = M.path_length_m != null ? `${M.path_length_m.toFixed(1)} m, mean ${M.speed_mean_mps} m/s` : '–';
  $('stops').textContent = M.stops && M.stops.length ? M.stops.map((s) => `${s.t0}–${s.t1} s`).join(', ') : 'none';
  if (M.zone_seconds) { $('zonesro').hidden = false; $('zones').textContent = Object.entries(M.zone_seconds).map(([k, v]) => `${k} ${v} s`).join(', '); }
  $('status').textContent = `${n} frames, ${obs.filter(Boolean).length} observed, ${(d.footprints || []).length} footprints` +
    (M.cadence_assumed ? ', cadence assumed' : '') + (d.source ? ` · ${d.source.split(':')[0]}` : '') + mediaNote + bodyNote;
  update(tCur);
  syncScene();
  fitView();
}

function update(tv) {
  tCur = tv;
  const t = data.t, n = t.length;
  const [i, f] = lerpIdx(t, Math.min(tv, t[n - 1]));
  const x = data.mean[i][0] * (1 - f) + data.mean[i + 1][0] * f, y = data.mean[i][1] * (1 - f) + data.mean[i + 1][1] * f;
  const inPred = tv > t[n - 1];
  let px = x, py = y, pc = data.cov[Math.round(i + f)];
  if (inPred && data.prediction) {
    const [k, g] = lerpIdx(data.prediction.t, tv);
    const pm = data.prediction.mean;
    px = pm[k][0] * (1 - g) + pm[k + 1][0] * g; py = pm[k][1] * (1 - g) + pm[k + 1][1] * g; pc = data.prediction.cov[Math.round(k + g)];
  }
  marker.position.set(px, py, 0.06);
  hit.position.set(px, py, 0.9);
  const e = eig2(pc);
  markerEllipse.scale.set(e.r1, e.r2, 1);
  placeFlat(markerEllipse, px, py, 0.02, e.angle);
  const k = Math.round(i + f);
  const observed = !inPred && obsFlags[k];
  // skeleton: nearest frame at/before t with joints; faded when filled in or stale
  let shown = false;
  if (skelFrames && !inPred) {
    let j = k;
    while (j >= 0 && !skelFrames[j]) j--;
    if (j >= 0 && k - j < 12) {
      const J = skelFrames[j];
      // older replays: slide the flat-card figure onto the path; a rig IS the position (the path follows it)
      const anch = !data.rig && J[15] && J[16];
      const dx = anch ? x - (J[15][0] + J[16][0]) / 2 : 0, dy = anch ? y - (J[15][1] + J[16][1]) / 2 : 0;
      const arr = skelLines.geometry.attributes.position.array;
      BONES.forEach(([a, b], bi) => {
        const A = J[a], B = J[b];
        const o = bi * 6;
        if (A && B) arr.set([A[0] + dx, A[1] + dy, A[2], B[0] + dx, B[1] + dy, B[2]], o);
        else arr.set([0, 0, -1, 0, 0, -1], o);
      });
      skelLines.geometry.attributes.position.needsUpdate = true;
      const m4 = new THREE.Matrix4();
      J.forEach((p, ji) => {
        m4.makeTranslation(p ? p[0] + dx : 0, p ? p[1] + dy : 0, p ? p[2] : -1);
        skelJoints.setMatrixAt(ji, m4);
      });
      skelJoints.instanceMatrix.needsUpdate = true;
      const op = j === k ? 1 : 0.35;
      skelLines.material.opacity = op; skelJoints.material.opacity = op;
      const RM = rigFeet && data.rig.markers[j];
      if (RM) {
        const fa = rigFeet.geometry.attributes.position.array;
        FOOT_BONES.forEach(([a, b], bi) => {
          const A = RM[a], B = RM[b];
          if (A && B) fa.set([A[0] + dx, A[1] + dy, A[2], B[0] + dx, B[1] + dy, B[2]], bi * 6);
          else fa.set([0, 0, -1, 0, 0, -1], bi * 6);
        });
        rigFeet.geometry.attributes.position.needsUpdate = true;
        rigFeet.material.opacity = op;
      }
      shown = true;
    }
  }
  skelLines.visible = skelJoints.visible = shown && bodyWants().stick;
  if (rigFeet) rigFeet.visible = shown && bodyWants().stick && $('rigfeet').checked;
  pane.onTime && pane.onTime(tv);
  updateBody(k, inPred);
  // view cone: nearest head reading at/before this frame (up to ~1 s old), faded when stale or uncertain
  if (gazeGroup) {
    let j = inPred ? -1 : k;
    while (j >= 0 && (gazeData.yaw[j] == null)) j--;
    const ok = j >= 0 && k - j < 8;
    gazeGroup.visible = ok;
    if (ok) {
      gazeGroup.position.set(x, y, gazeData.eye_z);
      gazeGroup.rotation.set(0, 0, gazeData.yaw[j] * Math.PI / 180);
      const c = Math.max(0.3, Math.min(1, gazeData.conf[j] || 0)) * (j === k ? 1 : 0.5);
      gazeGroup.children[0].material.opacity = 0.22 * c;
      gazeGroup.children[1].material.opacity = 0.035 * c;
    }
  }
  // readout
  $('tlabel').textContent = `${tv.toFixed(1)} s`;
  const sp = inPred ? null : data.speed[k], ss = data.speed_std ? data.speed_std[k] : null;
  $('spd').textContent = sp == null ? '–' : `${sp.toFixed(2)} m/s` + (ss != null ? ` ± ${ss.toFixed(2)}` : '');
  $('obs').textContent = inPred ? 'predicted' : observed ? 'observed' : 'filled in (no detection)';
  $('obs').style.color = inPred ? css('--c-pred') : observed ? css('--ok') : css('--warn');
  $('pos').textContent = `${px.toFixed(2)}, ${py.toFixed(2)} m (±${Math.sqrt(pc[0][0]).toFixed(2)}, ±${Math.sqrt(pc[1][1]).toFixed(2)})`;
  $('time').value = tv;
  updateCutouts(tv);
  render();
}

// -- views ---------------------------------------------------------------------------------------------------
function fitView() {
  world.updateMatrixWorld(true);
  const box = new THREE.Box3();
  data.mean.forEach((p) => box.expandByPoint(world.localToWorld(W(p[0], p[1], 0))));
  box.expandByPoint(world.localToWorld(W(data.camera.C[0], data.camera.C[1], data.camera.C[2])));
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3()).length();
  camMode = false;
  camera.clearViewOffset();
  camera.up.set(0, 1, 0);
  camera.position.copy(c).add(new THREE.Vector3(-0.6, 0.9, 0.8).normalize().multiplyScalar(Math.max(6, s * 0.9)));
  camera.fov = 50; camera.updateProjectionMatrix();
  controls.target.copy(c);
  controls.update();
  render();
}
function lookAt(v) { // ?look=x,y,z,tx,ty,tz: eye and target in world metres (Z up)
  if (v.length !== 6 || v.some((x) => !Number.isFinite(x))) return;
  world.updateMatrixWorld(true);
  camMode = false;
  camera.clearViewOffset();
  camera.up.set(0, 1, 0);
  camera.fov = 50;
  camera.position.copy(world.localToWorld(W(v[0], v[1], v[2])));
  controls.target.copy(world.localToWorld(W(v[3], v[4], v[5])));
  camera.updateProjectionMatrix();
  controls.update();
  render();
}
function setCamOffset() { // principal point off-centre: show the image window of a view centred on (cx, cy)
  const K = data.camera.K, [iw, ih] = data.camera.size, fw = ih * camera.aspect;
  if (Math.abs(K[0][2] - iw / 2) < 0.5 && Math.abs(K[1][2] - ih / 2) < 0.5) { camera.clearViewOffset(); return; }
  camera.setViewOffset(fw, ih, iw / 2 - K[0][2], ih / 2 - K[1][2], fw, ih);
}
function cameraView() { // look out of the Wyze camera: forward = R[:,2], up = -R[:,1] (OpenCV y is down)
  world.updateMatrixWorld(true);
  const C = data.camera.C, R = data.camera.R, K = data.camera.K, ih = data.camera.size[1];
  const pos = world.localToWorld(W(C[0], C[1], C[2]));
  const fwd = world.localToWorld(W(R[0][2], R[1][2], R[2][2])).sub(world.localToWorld(W(0, 0, 0))).normalize();
  const up = world.localToWorld(W(-R[0][1], -R[1][1], -R[2][1])).sub(world.localToWorld(W(0, 0, 0))).normalize();
  camera.position.copy(pos);
  camera.up.copy(up);
  camera.fov = 2 * Math.atan(ih / 2 / K[1][1]) * 180 / Math.PI;
  camMode = true;
  setCamOffset();
  camera.updateProjectionMatrix();
  controls.target.copy(pos.clone().add(fwd.multiplyScalar(5)));
  controls.update();
  render();
}

// -- replay media: camera projection, video cut-outs, Gaussian splat -------------------------------------------
const media = new THREE.Group(); // lives outside `built`, so a theme rebuild keeps the loaded textures
world.add(media);
const store = { get(k) { try { return localStorage.getItem('sg-replay-' + k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem('sg-replay-' + k, v); } catch { /* private mode: not remembered */ } } };
let proj = null, projLoading = null; // {group, mats}
let cut = null, cutLoading = null;   // {items, ts, cur, ghosts}
let splat = null, splatLoading = null;
let opacityTouched = false, mediaNote = '';

// pinhole + OpenCV distortion (k1, k2, p1, p2, k3) in GLSL; returns normalised image coords (v down) and validity
const PROJ_GLSL = `
uniform mat3 uRcw; uniform vec3 uC; uniform vec4 uK; uniform float uD[5]; uniform float uR2max;
vec3 camProject(vec3 p) {
  vec3 q = uRcw * (p - uC);
  if (q.z < 1e-4) return vec3(0.0);
  vec2 n = q.xy / q.z;
  float r2 = dot(n, n);
  if (r2 > uR2max) return vec3(0.0);
  float rad = 1.0 + r2 * (uD[0] + r2 * (uD[1] + r2 * uD[4]));
  vec2 d = n * rad + vec2(2.0 * uD[2] * n.x * n.y + uD[3] * (r2 + 2.0 * n.x * n.x),
                          uD[2] * (r2 + 2.0 * n.y * n.y) + 2.0 * uD[3] * n.x * n.y);
  return vec3(uK.x * d.x + uK.z, uK.y * d.y + uK.w, 1.0);
}`;
const VERT = `varying vec3 vW;
void main() { vW = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

function camUniforms(d) {
  const c = d.camera, R = c.R, K = c.K, [w, h] = c.size;
  const D = (c.D || []).concat([0, 0, 0, 0, 0]).slice(0, 5);
  const r2 = (d.replay && d.replay.r2max) || 1.1 * (Math.max(K[0][2], w - K[0][2]) ** 2 / K[0][0] ** 2 + Math.max(K[1][2], h - K[1][2]) ** 2 / K[1][1] ** 2);
  return {
    uRcw: { value: new THREE.Matrix3().set(R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1], R[0][2], R[1][2], R[2][2]) },
    uC: { value: new THREE.Vector3(c.C[0], c.C[1], c.C[2]) },
    uK: { value: new THREE.Vector4(K[0][0] / w, K[1][1] / h, K[0][2] / w, K[1][2] / h) },
    uD: { value: D }, uR2max: { value: r2 },
  };
}
const DEFAULT_DATA = 'demo/demo.json';
let dataURL = params.get('data') || DEFAULT_DATA; // the recording on screen (changes when a related replay opens)
let gen = 0;                                       // bumped per loaded recording: late media loads of an older one are dropped
const assetURL = (name) => new URL(name, new URL(dataURL, location.href)).href;
function loadTexture(url) {
  return new Promise((ok, fail) => new THREE.TextureLoader().load(url, (tx) => {
    tx.flipY = false; tx.generateMipmaps = false; tx.minFilter = THREE.LinearFilter; tx.magFilter = THREE.LinearFilter;
    tx.needsUpdate = true; ok(tx);
  }, undefined, () => fail(new Error(`could not load ${url.split('/').pop()}`))));
}
function meshFrom(m, mat) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices, 3));
  g.setIndex(m.indices);
  const mesh = new THREE.Mesh(g, mat);
  mesh.frustumCulled = false;
  return mesh;
}

// camera projection: textured ground (clipped to max range) + backdrop
function ensureProjection() {
  if (proj || projLoading || !data.assets || !data.assets.ground || !data.assets.background) return projLoading;
  const my = gen;
  projLoading = (async () => {
    const [gj, tex] = await Promise.all([fetch(assetURL(data.assets.ground)).then((r) => {
      if (!r.ok) throw new Error(`ground mesh: ${r.status}`); return r.json(); }), loadTexture(assetURL(data.assets.background))]);
    if (my !== gen) return;
    const uni = camUniforms(data);
    const mk = (range) => new THREE.ShaderMaterial({
      uniforms: { ...uni, map: { value: tex }, uOpacity: { value: 1 }, uRange: { value: range } },
      vertexShader: VERT,
      fragmentShader: PROJ_GLSL + `
uniform sampler2D map; uniform float uOpacity; uniform float uRange; varying vec3 vW;
void main() {
  if (uRange > 0.0 && length(vW.xy - uC.xy) > uRange) discard;
  vec3 q = camProject(vW);
  if (q.z < 0.5 || q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) discard;
  gl_FragColor = vec4(texture2D(map, q.xy).rgb, uOpacity);
}`,
      side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 });
    const group = new THREE.Group(), mats = [];
    const range = (data.replay && data.replay.ground && data.replay.ground.max_range_m) || 30;
    if (gj.ground) { const m = mk(range); mats.push(m); group.add(meshFrom(gj.ground, m)); } // no z offset: polygonOffset only (a shift moves the range clip off the backdrop at grazing angles)
    if (gj.backdrop) { const m = mk(0); mats.push(m); group.add(meshFrom(gj.backdrop, m)); }
    group.renderOrder = -1;
    media.add(group);
    proj = { group, mats };
    syncScene();
  })().catch((e) => { $('err').textContent = `projection: ${e.message || e}`; projLoading = null; });
  return projLoading;
}

// video cut-outs: one mesh with every card (trail / ghosts, per-vertex alpha) + one for the current card
function ensureCutouts() {
  const R = data.replay && data.replay.cutouts;
  if (cut || cutLoading || !data.assets || !data.assets.atlas || !R || !R.items || !R.items.length) return cutLoading;
  const my = gen;
  cutLoading = (async () => {
    const tex = await loadTexture(assetURL(data.assets.atlas));
    if (my !== gen) return;
    const [fw, fh] = data.replay.frame_size;
    const items = R.items;
    const mkGeo = (list) => {
      const n = list.length, pos = new Float32Array(n * 12), crop = new Float32Array(n * 16), uv = new Float32Array(n * 16), idx = [];
      list.forEach((it, k) => {
        it.corners.forEach((p, j) => { pos.set(p, k * 12 + j * 3); });
        for (let j = 0; j < 4; j++) {
          crop.set([it.crop[0] / fw, it.crop[1] / fh, it.crop[2] / fw, it.crop[3] / fh], k * 16 + j * 4);
          uv.set(it.uv, k * 16 + j * 4);
        }
        const b = k * 4;
        idx.push(b, b + 3, b + 1, b + 1, b + 3, b + 2);
      });
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setAttribute('aCrop', new THREE.BufferAttribute(crop, 4));
      g.setAttribute('aUV', new THREE.BufferAttribute(uv, 4));
      g.setAttribute('aAlpha', new THREE.BufferAttribute(new Float32Array(n * 4), 1));
      g.setIndex(idx);
      return g;
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: { ...camUniforms(data), map: { value: tex } },
      vertexShader: `attribute vec4 aCrop; attribute vec4 aUV; attribute float aAlpha;
varying vec3 vW; varying vec4 vCrop; varying vec4 vUV; varying float vA;
void main() { vW = position; vCrop = aCrop; vUV = aUV; vA = aAlpha;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: PROJ_GLSL + `
uniform sampler2D map; varying vec3 vW; varying vec4 vCrop; varying vec4 vUV; varying float vA;
void main() {
  if (vA < 0.004 || vW.z < -0.01) discard;          // hidden card / below the ground
  vec3 q = camProject(vW);
  if (q.z < 0.5) discard;
  vec2 c = (q.xy - vCrop.xy) / (vCrop.zw - vCrop.xy); // position inside the crop
  if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0) discard;
  vec4 t = texture2D(map, mix(vUV.xy, vUV.zw, c));
  float a = t.a * vA;
  if (a < 0.03) discard;
  gl_FragColor = vec4(t.rgb, a);
}`,
      transparent: true, depthWrite: false, side: THREE.DoubleSide });
    const ghosts = new THREE.Mesh(mkGeo(items), mat);
    const cur = new THREE.Mesh(mkGeo([items[0]]), mat);
    ghosts.frustumCulled = cur.frustumCulled = false;
    ghosts.renderOrder = 1; cur.renderOrder = 2;
    media.add(ghosts, cur);
    cut = { items, ts: items.map((it) => it.t), ghosts, cur, shown: -2 };
    syncScene();
  })().catch((e) => { $('err').textContent = `cut-outs: ${e.message || e}`; cutLoading = null; });
  return cutLoading;
}

const TRAIL = 6;
function updateCutouts(tv) {
  if (!cut) return;
  const mode = $('cutouts').value, ts = cut.ts, n = ts.length;
  let i = -1; // nearest cut-out at or before t, if not too stale
  if (mode !== 'off' && tv >= ts[0] - 1e-6) {
    let lo = 0, hi = n - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ts[m] <= tv + 1e-6) lo = m; else hi = m - 1; }
    const gap = lo + 1 < n ? ts[lo + 1] - ts[lo] : 0.5;
    if (tv - ts[lo] <= Math.max(0.6, gap + 0.05)) i = lo;
  }
  cut.cur.visible = i >= 0;
  if (i >= 0 && i !== cut.shown) {
    const it = cut.items[i], g = cut.cur.geometry, [fw, fh] = data.replay.frame_size;
    it.corners.forEach((p, j) => g.attributes.position.array.set(p, j * 3));
    for (let j = 0; j < 4; j++) {
      g.attributes.aCrop.array.set([it.crop[0] / fw, it.crop[1] / fh, it.crop[2] / fw, it.crop[3] / fh], j * 4);
      g.attributes.aUV.array.set(it.uv, j * 4);
      g.attributes.aAlpha.array[j] = 1;
    }
    for (const k of ['position', 'aCrop', 'aUV', 'aAlpha']) g.attributes[k].needsUpdate = true;
  }
  cut.shown = i;
  const a = cut.ghosts.geometry.attributes.aAlpha.array;
  a.fill(0);
  if (mode === 'ghost') a.fill(0.3);
  if (mode === 'trail' && i >= 0) {
    for (let k = 1; k <= TRAIL && i - k >= 0; k++) {
      if (ts[i] - ts[i - k] > 3) break;
      a.fill(0.55 * (1 - k / (TRAIL + 1)), (i - k) * 4, (i - k) * 4 + 4);
    }
  }
  if (i >= 0) a.fill(0, i * 4, i * 4 + 4);
  cut.ghosts.geometry.attributes.aAlpha.needsUpdate = true;
  cut.ghosts.visible = mode === 'ghost' || mode === 'trail';
}

// Gaussian splat: ?splat= (+ splat_pos/rot/scale) or scene.json next to the page
async function splatConfig() {
  const num = (k) => params.get(k) && params.get(k).split(',').map(Number);
  let cfg = null;
  if (params.get('splat')) cfg = { url: new URL(params.get('splat'), location.href).href };
  else if (data && data.assets && data.assets.splat) cfg = { url: assetURL(data.assets.splat) }; // already in the replay's world frame
  else {
    const r = await fetch('scene.json', { cache: 'no-cache' });
    if (r.ok) {
      const j = await r.json();
      if (j && j.splat) cfg = { url: new URL(j.splat, r.url).href, position: j.position, rotation: j.rotation, scale: j.scale };
    }
  }
  if (!cfg) return null;
  cfg.position = num('splat_pos') || cfg.position || [0, 0, 0];
  cfg.rotation = num('splat_rot') || cfg.rotation || [0, 0, 0];
  cfg.scale = Number(params.get('splat_scale') || cfg.scale || 1);
  return cfg;
}
function ensureSplat() {
  if (splat || splatLoading) return splatLoading;
  const my = gen;
  splatLoading = (async () => {
    const cfg = await splatConfig();
    if (my !== gen) return;
    if (!cfg) { $('status').textContent = 'no 3D scan for this sample (add assets.splat, ?splat= or scene.json)'; return; }
    $('status').textContent = 'loading splat scene…';
    const GS = await import('@mkkellogg/gaussian-splats-3d');
    // Instant reveal: the default fade-in advances per rendered frame, and this page renders on demand
    const viewer = new GS.DropInViewer({ sharedMemoryForWorkers: false, gpuAcceleratedSort: false, dynamicScene: false,
      sceneRevealMode: GS.SceneRevealMode.Instant });
    // The viewer culls splats in its own scene space and ignores parent transforms, so it is NOT put inside
    // `world`: the Z-up -> Y-up rotation of `world` is composed into the splat scene's own transform instead.
    // File point p -> world (Z up): Rxyz(rotation) * (scale * p) + position.
    const [rx, ry, rz] = cfg.rotation.map((v) => v * Math.PI / 180), sc = cfg.scale;
    const q = world.quaternion.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(rx, ry, rz, 'XYZ')));
    const pos = new THREE.Vector3(...cfg.position).applyQuaternion(world.quaternion);
    await viewer.addSplatScene(cfg.url, { splatAlphaRemovalThreshold: 5, showLoadingUI: false, position: pos.toArray(),
      rotation: q.toArray(), scale: [sc, sc, sc] });
    if (my !== gen) { try { viewer.dispose(); } catch { /* older viewers: let it be collected */ } return; }
    scene.add(viewer);
    splat = { viewer, cfg };
    $('status').textContent = `splat scene loaded (${cfg.url.split('/').pop()})`;
    syncScene();
  })().catch((e) => { $('err').textContent = `splat: ${e.message || e}`; splatLoading = null; });
  return splatLoading;
}

// 3D body (body3d): joints always available, mesh when its files load
let body = null, bodyLoading = null, bodyNote = ''; // {mesh, verts: Float32Array, slot: Int32Array, nv}
function bodyWants() { // which person drawings the Body select asks for
  if (!bodyFrames) return { stick: true, mesh: false, joints: false };
  const m = $('body').value;
  return { stick: m === 'stick' || m === 'both', mesh: m === 'mesh' || m === 'both', joints: m === 'joints' || ((m === 'mesh' || m === 'both') && !body) };
}
let F16 = null;
function halfToFloat(u16) { // float16 bits -> Float32Array (lookup table, built once)
  if (!F16) {
    F16 = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const sg = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
      F16[h] = e === 0 ? sg * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : sg * Infinity) : sg * 2 ** (e - 15) * (1 + f / 1024);
    }
  }
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = F16[u16[i]];
  return out;
}
function ensureBody() {
  const M = data.body3d && data.body3d.mesh;
  if (body || bodyLoading || !M || !bodyWants().mesh) return bodyLoading;
  const my = gen;
  bodyLoading = (async () => {
    const get = async (f) => { const r = await fetch(assetURL(f)); if (!r.ok) throw new Error(`${f}: ${r.status}`); return r.arrayBuffer(); };
    const [fb, vb] = await Promise.all([get(M.faces), get(M.vertices)]);
    if (my !== gen) return;
    const nv = M.vertex_count, idx = new Uint16Array(fb), verts = halfToFloat(new Uint16Array(vb));
    if (verts.length !== M.index.length * nv * 3) throw new Error('mesh size does not match body3d.mesh.index');
    const slot = new Int32Array(data.t.length).fill(-1);
    M.index.forEach((k, i) => { slot[k] = i; });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(nv * 3), 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: col('--c-body'), roughness: 0.75, metalness: 0, side: THREE.DoubleSide }));
    mesh.frustumCulled = false;
    mesh.visible = false;
    media.add(mesh);
    body = { mesh, verts, slot, nv, shown: -2 };
    update(tCur);
    syncScene();
  })().catch(() => { // SMPL-derived files are not deployed everywhere: joints only, and say so
    if (my !== gen) return;
    bodyNote = ' · 3D body: joints only (mesh files not available)';
    $('status').textContent += bodyNote;
    data.body3d.mesh = null;
    update(tCur);
  });
  return bodyLoading;
}
function updateBody(k, inPred) {
  const want = bodyWants();
  let j = -1; // nearest body frame at/before k, not too stale
  if (bodyFrames && !inPred) {
    j = k;
    while (j >= 0 && !bodyFrames[j]) j--;
    if (j >= 0 && k - j >= 12) j = -1;
  }
  const op = j === k ? 1 : 0.35;
  if (bodyFrames) {
    const show = want.joints && j >= 0;
    bodyLines.visible = bodyJoints.visible = show;
    if (show) {
      const J = bodyFrames[j], arr = bodyLines.geometry.attributes.position.array;
      BONES.forEach(([a, b], bi) => arr.set([...J[a], ...J[b]], bi * 6));
      bodyLines.geometry.attributes.position.needsUpdate = true;
      const m4 = new THREE.Matrix4();
      J.forEach((p, ji) => { bodyJoints.setMatrixAt(ji, m4.makeTranslation(p[0], p[1], p[2])); });
      bodyJoints.instanceMatrix.needsUpdate = true;
      bodyLines.material.opacity = bodyJoints.material.opacity = op;
    }
  }
  if (!body) return;
  let s = -1; // nearest mesh slot at/before the body frame
  if (want.mesh && j >= 0) { s = j; while (s >= 0 && body.slot[s] < 0) s--; if (s >= 0 && j - s >= 12) s = -1; }
  body.mesh.visible = s >= 0;
  if (s >= 0 && s !== body.shown) {
    const i = body.slot[s], nv = body.nv, R = data.body3d.root[s], pos = body.mesh.geometry.attributes.position.array;
    const src = body.verts.subarray(i * nv * 3, (i + 1) * nv * 3);
    for (let v = 0; v < nv * 3; v += 3) { pos[v] = src[v] + R[0]; pos[v + 1] = src[v + 1] + R[1]; pos[v + 2] = src[v + 2] + R[2]; }
    body.mesh.geometry.attributes.position.needsUpdate = true;
    body.mesh.geometry.computeVertexNormals();
    body.shown = s;
  }
  body.mesh.material.transparent = op < 1;
  body.mesh.material.opacity = op;
}

// scene + layer state -> visibility, opacity, legend
function syncScene() {
  if (!data) return;
  const mode = $('scene').value, wantProj = mode === 'projection' || mode === 'both', wantSplat = mode === 'splat' || mode === 'both';
  if (wantProj) ensureProjection();
  if (wantSplat) ensureSplat();
  if ($('cutouts').value !== 'off') ensureCutouts();
  if (bodyFrames) ensureBody();
  const op = Number($('opacity').value);
  if (proj) {
    proj.group.visible = wantProj && op > 0;
    for (const m of proj.mats) {
      m.uniforms.uOpacity.value = op;
      m.transparent = op < 1; m.depthWrite = op >= 1; m.needsUpdate = true;
    }
  }
  if (splat) splat.viewer.visible = wantSplat;
  $('opacitylbl').hidden = !wantProj;
  if (gridHelper) gridHelper.material.opacity = wantProj && op > 0.5 ? 0.45 : wantSplat ? 0.6 : 1;
  built.visible = $('overlays').checked;
  if (objGroup) objGroup.visible = $('objects').checked;
  document.querySelectorAll('.lg-proj').forEach((el) => { el.hidden = !(wantProj && proj); });
  document.querySelectorAll('.lg-splat').forEach((el) => { el.hidden = !(wantSplat && splat); });
  document.querySelectorAll('.lg-cut').forEach((el) => { el.hidden = $('cutouts').value === 'off' || !cut; });
  document.querySelectorAll('.lg-ghost').forEach((el) => { el.hidden = !cut || !['trail', 'ghost'].includes($('cutouts').value); });
  const bw = bodyWants();
  document.querySelectorAll('.lg-body').forEach((el) => { el.hidden = !(bw.mesh && body); });
  document.querySelectorAll('.lg-bodyj').forEach((el) => { el.hidden = !bw.joints; });
  updateCutouts(tCur);
  render();
}
function initMedia() { // enable what the data offers; URL params > remembered choice > defaults
  const A = data.assets || {}, R = data.replay || {};
  const hasProj = !!(A.ground && A.background), hasCut = !!(A.atlas && R.cutouts && R.cutouts.items && R.cutouts.items.length);
  const sel = $('scene');
  for (const o of sel.options) if (o.value === 'projection' || o.value === 'both') o.disabled = !hasProj;
  const pick = (el, v) => { if (v && [...el.options].some((o) => o.value === v && !o.disabled)) el.value = v; };
  pick(sel, store.get('scene'));
  pick(sel, params.get('scene'));
  $('cutouts').disabled = !hasCut;
  if (hasCut) { pick($('cutouts'), store.get('cutouts')); pick($('cutouts'), params.get('cutouts')); }
  else { // say why instead of a silently greyed-out control
    $('cutouts').options[0].textContent = 'Not in this recording';
    $('cutouts').title = 'This visit was analysed without video cut-outs (no "assets.atlas" in its data file).';
  }
  if (!hasProj) sel.title = 'Camera projection needs a background image and ground mesh in the data file; this visit has none.';
  const op = params.get('opacity') ?? store.get('opacity');
  if (op != null && op !== '' && !Number.isNaN(Number(op))) { $('opacity').value = op; opacityTouched = true; }
  else $('opacity').value = sel.value === 'both' ? 0.35 : 1;
  if (params.get('overlays') === '0') $('overlays').checked = false;
  if (data.body3d && data.body3d.joints) {
    $('bodylbl').hidden = false;
    pick($('body'), store.get('body'));
    pick($('body'), params.get('body'));
  }
  const rng = R.ground && R.ground.max_range_m;
  if (rng) document.querySelectorAll('#legend .range').forEach((el) => { el.textContent = String(rng); });
  if (hasCut || hasProj) mediaNote = ` · ${hasCut ? `${R.cutouts.items.length} cut-outs` : 'no cut-outs'}${hasProj ? ', camera projection' : ''}`;
  if (data.body3d && data.body3d.joints) mediaNote += data.body3d.root_mode === 'rig' ? ' · 3D body (SMPL on the IK rig)' : ' · 3D body (WHAM)';
  $('status').textContent += mediaNote;
  syncScene();
}
$('scene').addEventListener('change', () => {
  const v = $('scene').value;
  if (!opacityTouched) $('opacity').value = v === 'both' ? 0.35 : 1;
  store.set('scene', v);
  syncScene();
});
$('opacity').addEventListener('input', () => { opacityTouched = true; store.set('opacity', $('opacity').value); syncScene(); });
$('cutouts').addEventListener('change', () => { store.set('cutouts', $('cutouts').value); syncScene(); });
$('overlays').addEventListener('change', syncScene);
$('objects').addEventListener('change', syncScene);
$('rigfeet').addEventListener('change', () => { if (data) update(tCur); render(); });
$('body').addEventListener('change', () => { store.set('body', $('body').value); syncScene(); update(tCur); });

// -- UI --------------------------------------------------------------------------------------------------------
$('time').addEventListener('input', (ev) => { update(Number(ev.target.value)); });
$('play').addEventListener('click', () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; $('play').setAttribute('aria-pressed', String(playing)); });
$('fit').addEventListener('click', fitView);
$('camview').addEventListener('click', cameraView);
$('theme').addEventListener('click', () => {
  const r = document.documentElement, dark = r.dataset.theme === 'dark' || (!r.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  r.dataset.theme = dark ? 'light' : 'dark';
  renderer.setClearColor(col('--scene-bg'));
  build(data);
});

// render synchronously on every change (scrub, buttons, orbit); the animation loop only runs while playing.
// Keeps software GL and phones cool, and makes headless screenshots deterministic.
function render() { renderer.render(scene, camera); }
controls.addEventListener('change', render);
window.addEventListener('resize', () => { resize(); render(); });
function tick(now) {
  requestAnimationFrame(tick);
  if (!data) return;
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  if (playing) {
    let tv = tCur + dt;
    if (tv > tEnd) tv = data.t[0];
    update(tv);
  }
  if (controls.update() || (splat && splat.viewer.visible)) render(); // splats sort asynchronously: keep drawing
}

// -- investigation pane: click (or tap) the person -> sub-dashboard over the scene (pane.js) ---------------------------
const pane = createPane({
  apiBase: params.get('api') || '/api/',
  getData: () => data,
  getDataURL: () => dataURL,
  getT: () => tCur,
  sampleName: () => { const o = $('sample').selectedOptions[0]; return o && o.value === dataURL && !o.dataset.own ? o.textContent : null; },
  openReplay: (url) => loadData(url, { push: true }),
  onToggle: (open) => { $('inspect').setAttribute('aria-expanded', String(open)); },
});
const raycaster = new THREE.Raycaster();
raycaster.params.Line.threshold = 0.06;
const ndc = new THREE.Vector2();
function drawn(o) { for (let x = o; x; x = x.parent) if (!x.visible) return false; return true; }
function pickPerson(clientX, clientY) { // -> intersection with the person (any drawing of them) or null
  if (!data) return null;
  const targets = [marker, skelLines, skelJoints, bodyFrames && bodyLines, bodyFrames && bodyJoints, body && body.mesh, cut && cut.cur]
    .filter((o) => o && drawn(o));
  for (const o of targets) { // these move every frame: cached bounds would make the ray miss them
    if (o.isInstancedMesh) o.boundingSphere = null;
    if (o.geometry) { o.geometry.boundingSphere = null; o.geometry.boundingBox = null; }
  }
  targets.push(hit);
  const r = canvas.getBoundingClientRect();
  ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
  scene.updateMatrixWorld(true);
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(targets, false);
  return hits.length ? hits[0] : null;
}
let down = null;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId }; });
// open on 'click' (not pointerup): on touch screens the click synthesised after a tap would otherwise land on the
// pane that just appeared under the finger
canvas.addEventListener('click', (e) => {
  const d = down; down = null;
  if (!d || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 6 || performance.now() - d.t > 700) return; // a drag = orbit
  if (pickPerson(e.clientX, e.clientY)) { canvas.style.cursor = ''; $('picktip').hidden = true; pane.open({ invoker: canvas, t: tCur }); }
});
let hoverReq = 0, hoverEv = null;
canvas.addEventListener('pointermove', (e) => {
  if (e.pointerType !== 'mouse' || e.buttons) return;
  hoverEv = e;
  if (!hoverReq) hoverReq = requestAnimationFrame(() => {
    hoverReq = 0;
    const on = !!pickPerson(hoverEv.clientX, hoverEv.clientY);
    canvas.style.cursor = on ? 'pointer' : '';
    const tip = $('picktip');
    tip.hidden = !on;
    if (on) { const r = $('app').getBoundingClientRect(); tip.style.left = `${hoverEv.clientX - r.left + 14}px`; tip.style.top = `${hoverEv.clientY - r.top + 10}px`; }
  });
});
canvas.addEventListener('pointerleave', () => { $('picktip').hidden = true; });
window.sgReplay = { // for headless tests: where the person is on screen (page coordinates), and the loaded recording
  personPoint(z = 0.9) {
    hit.updateMatrixWorld(true);
    const v = world.localToWorld(W(hit.position.x, hit.position.y, z)).project(camera), r = canvas.getBoundingClientRect();
    return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height };
  },
  dataURL: () => dataURL,
};
$('inspect').addEventListener('click', () => { if (pane.isOpen()) pane.close(); else pane.open({ invoker: $('inspect'), t: tCur }); });
canvas.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); pane.open({ invoker: canvas, t: tCur }); } });
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest('input, select, textarea, #pane')) return;
  if (e.key === 'i' || e.key === 'I') { e.preventDefault(); if (pane.isOpen()) pane.close(); else pane.open({ invoker: document.activeElement, t: tCur }); }
  else if (e.key === 'Escape' && pane.isOpen()) pane.close();
});

// -- loading a recording (first load, sample picker, related replays from the pane, back/forward) -----------------------
function disposeTree(o) {
  o.traverse((x) => {
    if (x.geometry) x.geometry.dispose();
    for (const m of [].concat(x.material || [])) {
      for (const u of Object.values(m.uniforms || {})) if (u && u.value && u.value.isTexture) u.value.dispose();
      if (m.map) m.map.dispose();
      m.dispose();
    }
  });
}
// 3D object models (scripts/rig_objects.py): vertex-coloured GLB meshes already in world metres, Z up, so they go
// straight into the Z-up world (via `media`, which resetMedia clears); one label sprite per object
let objGroup = null;
function labelSprite(text, size = 0.11) {
  const cv = document.createElement('canvas');
  cv.width = 512; cv.height = 96;
  const g = cv.getContext('2d');
  g.font = '600 52px system-ui, sans-serif'; g.textBaseline = 'middle';
  const w = Math.min(512, g.measureText(text).width + 24);
  const x0 = (512 - w) / 2;
  g.fillStyle = 'rgba(17,24,39,0.72)'; g.fillRect(x0, 12, w, 72);
  g.fillStyle = '#f9fafb'; g.fillText(text, x0 + 12, 48);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthTest: false }));
  sp.scale.set(size * 512 / 96, size, 1);
  return sp;
}
async function loadObjects(my) {
  const name = data && data.assets && data.assets.objects;
  $('objlbl').hidden = !name;
  if (!name) return;
  try {
    const r = await fetch(assetURL(name));
    if (!r.ok) throw new Error(`${r.status}`);
    const meta = await r.json();
    const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
    if (my !== gen) return;
    objGroup = new THREE.Group();
    objGroup.name = 'objects';
    media.add(objGroup);
    const loader = new GLTFLoader();
    await Promise.all(meta.objects.map(async (o) => {
      const g = await loader.loadAsync(assetURL(o.file));
      if (my !== gen) return;
      g.scene.traverse((m) => {
        if (m.isMesh) m.material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
      });
      g.scene.userData.object = o;
      objGroup.add(g.scene);
      const lb = labelSprite(o.label);
      lb.position.copy(W(o.centre[0], o.centre[1], o.centre[2] + o.size[2] / 2 + 0.12));
      objGroup.add(lb);
    }));
    if (my !== gen) return;
    mediaNote += ` · ${meta.objects.length} 3D objects`;
    syncScene();
  } catch (e) {
    console.warn('objects:', e);
    $('objlbl').hidden = true;
  }
}

function resetMedia() { // forget everything the previous recording loaded (its textures, meshes, splat, legend rows)
  for (const o of [...media.children]) { media.remove(o); disposeTree(o); }
  if (splat) { scene.remove(splat.viewer); try { splat.viewer.dispose(); } catch { /* collected */ } }
  proj = projLoading = cut = cutLoading = splat = splatLoading = body = bodyLoading = objGroup = null;
  mediaNote = bodyNote = '';
  document.querySelectorAll('.lg-aerial, .lg-gaze, #truthrow').forEach((el) => { el.hidden = true; });
  $('bodylbl').hidden = true; $('zonesro').hidden = true;
  for (const o of $('scene').options) o.disabled = false;
  $('scene').title = ''; $('cutouts').disabled = false; $('cutouts').title = ''; $('cutouts').options[0].textContent = 'Off';
}
function syncSampleSelect(url) { // a sighting's own file (not one of the samples) shows as "This sighting"
  const smp = $('sample'), own = [...smp.options].find((o) => o.dataset.own);
  if ([...smp.options].some((o) => !o.dataset.own && o.value === url)) { smp.value = url; return; }
  own.value = url; own.hidden = false; smp.value = url;
}
async function loadData(url, { push = false, first = false } = {}) {
  gen += 1;
  const my = gen;
  dataURL = url;
  resetMedia();
  syncSampleSelect(url);
  $('err').textContent = '';
  $('status').textContent = 'loading…';
  if (push) {
    const q = new URLSearchParams(location.search);
    q.set('data', url); ['t', 'look', 'view'].forEach((k) => q.delete(k));
    history.pushState({ data: url }, '', `?${q}`);
  }
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    const d = await r.json();
    if (my !== gen) return;
    if (!d.t || !d.mean || !d.camera) throw new Error('not a trajectory JSON (needs t, mean, camera)');
    data = d;
    build(data);
    initMedia();
    loadObjects(my);
    if (first) {
      if (params.get('autoplay') !== '0') { playing = true; $('play').textContent = 'Pause'; $('play').setAttribute('aria-pressed', 'true'); }
      if (params.get('t')) { playing = false; $('play').textContent = 'Play'; $('play').setAttribute('aria-pressed', 'false'); update(Number(params.get('t'))); }
      if (params.get('view') === 'camera') cameraView();
      else if (params.get('look')) lookAt(params.get('look').split(',').map(Number));
      if (params.get('pane') === '1') pane.open({ invoker: $('inspect'), t: tCur, tab: params.get('pane_tab') });
    }
  } catch (e) {
    if (my !== gen) return;
    $('status').textContent = '';
    $('err').textContent = `could not load ${url}: ${e.message || e}`;
  }
  pane.onData();
}
window.addEventListener('popstate', () => { loadData(new URLSearchParams(location.search).get('data') || DEFAULT_DATA); });

async function main() {
  resize();
  if (window.innerWidth < 560) $('legend').open = false; // phones: legend collapsed until tapped
  renderer.setClearColor(col('--scene-bg'));
  $('sample').addEventListener('change', () => { if ($('sample').value && $('sample').value !== dataURL) loadData($('sample').value, { push: true }); });
  await loadData(dataURL, { first: true });
  requestAnimationFrame(tick);
}
main();
