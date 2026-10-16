// RoomDiff static viewer: before / after point clouds and change boxes (plain three.js).
//
// Data (written by roomdiff/viewer_export.py): <scene folder>/scene.json, which names
// before.ply, after.ply and changes.json; crops are relative to changes.json.
// index.html?scene=scenes/<name> loads scenes/<name>/scene.json, otherwise ./scene.json.
// ?select=<change id> opens the viewer on that change. Scripts drive the viewer through
// window.roomdiff (see automationApi), e.g. scripts/shoot_viewer.py.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const TYPES = ['added', 'removed', 'moved', 'unverified'];
const TYPE_COLORS = { added: '#22a447', removed: '#e03131', moved: '#2f6fdf', unverified: '#e6b400' };
// Unverified regions are often large patches of floor or table top that only one video
// observed: their boxes are drawn dimmed, their labels appear only when selected, and the
// home view does not zoom out for them.
const UNVERIFIED_OPACITY = 0.55;
const TINT_COLORS = { before: '#3b82f6', after: '#f97316' };
const BACKGROUNDS = { dark: '#15171c', light: '#f3f4f6' };
const EVIDENCE_LABELS = {
  free_rays: 'free rays',
  observing_views: 'observing views',
  appearance_similarity: 'appearance similarity',
};
const FOV = 50; // vertical field of view, degrees
const HOME_ELEVATION = THREE.MathUtils.degToRad(35);
const FLY_MS = 900;
const DIMMED = 0.25; // opacity of the boxes that are not selected
const LABEL_HEIGHT = 0.038; // label height as a fraction of about the view height
const FIT_MARGIN = 0.06; // home view: empty border as a fraction of the view size

const $ = (id) => document.getElementById(id);
const statusEl = $('status');

const state = {
  sceneUrl: null,
  changesUrl: null,
  box: null, // robust scene bounding box, recon world frame
  viewDir: null, // horizontal viewing direction for the home view, display frame
  clouds: {}, // before / after THREE.Points
  baseSize: 0.025, // point size (m) at slider value 1
  focusScale: 1, // point size factor while a change is focused (close-ups use smaller points)
  sample: [], // display-frame sample of both clouds (home view fit)
  items: [], // one per change: {change, type, group, label, bounds, entry, materials}
  filters: Object.fromEntries(TYPES.map((t) => [t, true])),
  selected: -1,
  tween: null,
};

let renderer, scene3d, camera, controls, world;

// ------------------------------------------------------------------ helpers

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const vec = (array) => new THREE.Vector3(...array);

function fmt(x) {
  return Number.isInteger(x) ? String(x) : x.toFixed(2);
}

function setStatus(text) {
  statusEl.textContent = text;
  statusEl.hidden = !text;
}

function fail(error) {
  console.error(error);
  let text = `Could not load the scene: ${error.message || error}.`;
  if (location.protocol === 'file:') text += ' ' + window.SERVE_HINT;
  window.showError(text);
}

function sceneJsonUrl() {
  const scene = new URLSearchParams(location.search).get('scene');
  if (!scene) return new URL('scene.json', location.href);
  if (/^([a-z][a-z0-9+.-]*:|\/\/)/i.test(scene)) {
    throw new Error('the scene parameter must be a relative path such as scenes/<name>');
  }
  const path = scene.endsWith('.json') ? scene : scene.replace(/\/*$/, '/scene.json');
  return new URL(path, location.href);
}

async function fetchJson(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (error) {
    throw new Error(`${url.pathname}: ${error.message}`);
  }
  if (!response.ok) throw new Error(`${url.pathname}: HTTP ${response.status}`);
  return response.json();
}

// -------------------------------------------------------------- three.js

function initThree() {
  const canvas = $('canvas');
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  } catch (error) {
    throw new Error('WebGL is not available in this browser');
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  scene3d = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 1000);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.addEventListener('start', () => (state.tween = null)); // the user takes over
  world = new THREE.Group(); // recon world frame, rotated so that up_axis becomes +Y
  scene3d.add(world);

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = $('view');
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe($('view'));
  resize();
  renderer.setAnimationLoop(animate);
}

function animate(now) {
  const tween = state.tween;
  if (tween) {
    const k = Math.min(1, (now - tween.start) / FLY_MS);
    const e = k < 0.5 ? 4 * k ** 3 : 1 - (-2 * k + 2) ** 3 / 2; // ease in-out cubic
    controls.target.lerpVectors(tween.from.target, tween.to.target, e);
    camera.position.lerpVectors(tween.from.position, tween.to.position, e);
    if (k === 1) state.tween = null;
  }
  controls.update();
  renderer.render(scene3d, camera);
}

function flyTo(view) {
  state.tween = {
    start: performance.now(),
    from: { target: controls.target.clone(), position: camera.position.clone() },
    to: view,
  };
}

// Moves the camera to `view` at once, without the flight.
function jump(view) {
  state.tween = null;
  controls.target.copy(view.target);
  camera.position.copy(view.position);
}

// Smallest half field of view (vertical or horizontal), radians.
function halfFov() {
  const vertical = THREE.MathUtils.degToRad(FOV / 2);
  return Math.min(vertical, Math.atan(Math.tan(vertical) * camera.aspect));
}

// Display-frame position of a recon world-frame point.
function toDisplay(point) {
  return world.localToWorld(point.clone());
}

// Overview: the scene centre seen from behind the cameras' main direction and above.
function homeView() {
  state.focusScale = 1;
  updateClouds();
  const target = toDisplay(state.box.getCenter(new THREE.Vector3()));
  const offset = state.viewDir.clone().multiplyScalar(-Math.cos(HOME_ELEVATION));
  offset.y = Math.sin(HOME_ELEVATION);
  const corners = [];
  for (const x of [state.box.min.x, state.box.max.x]) {
    for (const y of [state.box.min.y, state.box.max.y]) {
      for (const z of [state.box.min.z, state.box.max.z]) corners.push(toDisplay(new THREE.Vector3(x, y, z)));
    }
  }
  let distance = fitDistance(corners, target, offset);
  if (state.sample.length) distance = fitDistanceToPoints(state.sample, target, offset, distance);
  const boxCorners = [];
  for (const item of state.items) { // every confirmed change box stays in frame
    if (!item.bounds || item.type === 'unverified') continue;
    for (const x of [item.bounds.min.x, item.bounds.max.x]) {
      for (const y of [item.bounds.min.y, item.bounds.max.y]) {
        for (const z of [item.bounds.min.z, item.bounds.max.z]) boxCorners.push(new THREE.Vector3(x, y, z));
      }
    }
  }
  if (boxCorners.length) distance = Math.max(distance, fitDistance(boxCorners, target, offset));
  return { target, position: target.clone().addScaledVector(offset, distance) };
}

// Refines `distance` (and recentres `target` in place) so that the 1st-99th percentiles of
// the projected `sample` points fit the view with a margin: the box above is often set by
// stray points far from the room, which would leave the room small in the middle.
function fitDistanceToPoints(sample, target, direction, distance) {
  const forward = direction.clone().negate();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  const tanV = Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * (1 - 2 * FIT_MARGIN);
  const tanH = tanV * camera.aspect;
  const quantile = (sorted, f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
  const rel = new THREE.Vector3();
  for (let iteration = 0; iteration < 4; iteration++) {
    const xs = [];
    const ys = [];
    for (const point of sample) {
      rel.subVectors(point, target);
      const z = distance + rel.dot(forward);
      if (z < 0.05) continue;
      xs.push(rel.dot(right) / z);
      ys.push(rel.dot(up) / z);
    }
    if (xs.length < 10) break;
    xs.sort((a, b) => a - b);
    ys.sort((a, b) => a - b);
    const [x0, x1, y0, y1] = [quantile(xs, 0.01), quantile(xs, 0.99), quantile(ys, 0.01), quantile(ys, 0.99)];
    target.addScaledVector(right, 0.5 * (x0 + x1) * distance).addScaledVector(up, 0.5 * (y0 + y1) * distance);
    const factor = Math.max((x1 - x0) / (2 * tanH), (y1 - y0) / (2 * tanV));
    distance *= factor;
    if (Math.abs(factor - 1) < 0.02) break;
  }
  return distance;
}

// Distance from `target` along the unit vector `direction` (towards the camera) at which
// all display-frame `points` fit the view, with a margin; the camera looks at the target.
function fitDistance(points, target, direction) {
  const forward = direction.clone().negate();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0, 1, 0));
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  const tanV = Math.tan(THREE.MathUtils.degToRad(FOV / 2));
  const tanH = tanV * camera.aspect;
  let distance = 0.5;
  for (const point of points) {
    const rel = point.clone().sub(target);
    const depth = rel.dot(forward); // beyond the target = further from the camera
    distance = Math.max(distance, Math.abs(rel.dot(right)) / tanH - depth,
      Math.abs(rel.dot(up)) / tanV - depth);
  }
  return 1.12 * distance;
}

// Close-up of a display-frame box, keeping the current viewing direction but from above
// (steeper for small objects, so that nearby surfaces do not fill the view); the points
// shrink with the distance so that a wallet is not hidden behind its own point squares.
function focusView(bounds) {
  const target = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(0.5 * bounds.getSize(new THREE.Vector3()).length(), 0.5);
  const direction = camera.position.clone().sub(controls.target).normalize();
  direction.y = Math.max(direction.y, radius < 0.6 ? 0.7 : 0.5);
  direction.normalize();
  const distance = (1.4 * radius) / Math.sin(halfFov());
  state.focusScale = THREE.MathUtils.clamp(distance / 3.5, 0.5, 1);
  updateClouds();
  return { target, position: target.clone().addScaledVector(direction, distance) };
}

// Labels whose boxes are close in the floor plane are stacked at different heights, with a
// thin leader line back to the box, so that they do not hide each other in the overview.
function spreadLabels() {
  const spacing = 1.1; // m, floor-plane distance below which two labels are stacked
  const step = 0.35; // m, vertical step between stacked labels
  const placed = [];
  const items = state.items.filter((item) => item.label && item.type !== 'unverified')
    .sort((a, b) => a.label.position.y - b.label.position.y);
  for (const item of items) {
    const anchor = item.label.position.clone();
    let y = anchor.y;
    // A label within `step` of a placed neighbour goes one step above it. The tolerance
    // matters: (other.y + step) - other.y can round to slightly less than step, which
    // made this loop endless; the counter is a second guard.
    for (let changed = true, rounds = 0; changed && rounds < 100; rounds++) {
      changed = false;
      for (const other of placed) {
        const near = Math.hypot(anchor.x - other.x, anchor.z - other.z) < spacing;
        if (near && Math.abs(y - other.y) < step - 1e-6) {
          y = other.y + step;
          changed = true;
        }
      }
    }
    placed.push({ x: anchor.x, y, z: anchor.z });
    if (y <= anchor.y + 1e-6) continue;
    item.label.position.y = y;
    const geometry = new THREE.BufferGeometry().setFromPoints([anchor, item.label.position.clone()]);
    const material = new THREE.LineBasicMaterial({ color: TYPE_COLORS[item.type], transparent: true, opacity: 0.8 });
    scene3d.add(new THREE.Line(geometry, material));
    item.materials.push(material);
  }
}

function resetView() {
  select(-1);
  flyTo(homeView());
}

// ------------------------------------------------------------------ clouds

// `loaded` holds the bytes received per cloud (the total is unknown for compressed responses)
function loadPly(url, side, loaded) {
  return new PLYLoader().loadAsync(url.href, (event) => {
    loaded[side] = event.loaded;
    setStatus(`Loading point clouds… ${((loaded.before + loaded.after) / 1e6).toFixed(1)} MB`);
  });
}

function addCloud(geometry, side) {
  const material = new THREE.PointsMaterial({ size: state.baseSize });
  const cloud = new THREE.Points(geometry, material);
  state.clouds[side] = cloud;
  world.add(cloud);
}

function updateClouds() {
  const show = document.querySelector('input[name="show"]:checked').value;
  const tint = $('tint').checked;
  const scale = Number($('size').value);
  $('size-value').textContent = `${scale.toFixed(2)}×`;
  for (const [side, cloud] of Object.entries(state.clouds)) {
    const material = cloud.material;
    cloud.visible = show === 'both' || show === side;
    material.vertexColors = !tint && cloud.geometry.hasAttribute('color');
    material.color.set(tint ? TINT_COLORS[side] : '#ffffff');
    material.size = state.baseSize * scale * state.focusScale;
    material.needsUpdate = true;
  }
}

function updateBackground() {
  const light = $('light').checked;
  document.body.classList.toggle('light', light);
  scene3d.background = new THREE.Color(light ? BACKGROUNDS.light : BACKGROUNDS.dark);
}

// ------------------------------------------------------------ change boxes

// Box of one change side: the oriented box `obb` (center, extents [w, h, d], yaw_deg about
// +Y) when present, else the axis-aligned `aabb`. Recon world frame.
function sideBox(side) {
  if (side?.obb) {
    const { center, extents, yaw_deg: yaw } = side.obb;
    return { center: vec(center), size: vec(extents), yaw: THREE.MathUtils.degToRad(yaw || 0) };
  }
  if (!side?.aabb) return null;
  const lo = vec(side.aabb[0]);
  const hi = vec(side.aabb[1]);
  return { center: lo.clone().add(hi).multiplyScalar(0.5), size: hi.sub(lo), yaw: 0 };
}

function boxLines(box, color, dashed) {
  const size = box.size.clone().max(new THREE.Vector3(0.01, 0.01, 0.01));
  const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(size.x, size.y, size.z));
  const dash = Math.max(size.x, size.y, size.z) / 12;
  const material = dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: dash, gapSize: 0.7 * dash })
    : new THREE.LineBasicMaterial({ color });
  const lines = new THREE.LineSegments(geometry, material);
  lines.position.copy(box.center);
  lines.rotation.y = box.yaw;
  if (dashed) lines.computeLineDistances();
  return lines;
}

function arrowBetween(from, to, color) {
  const start = from.clone();
  const direction = to.clone().sub(start);
  const length = direction.length();
  if (length < 1e-3) return null;
  const head = Math.min(0.25 * length, 0.15);
  return new THREE.ArrowHelper(direction.normalize(), start, length, color, head, 0.5 * head);
}

function makeLabel(text, color) {
  const font = 'bold 34px system-ui, sans-serif';
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  ctx.font = font;
  canvas.width = Math.ceil(ctx.measureText(text).width) + 24;
  canvas.height = 48;
  ctx.font = font; // resizing the canvas resets the context
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(0, 0, canvas.width, canvas.height, 10);
  ctx.fill();
  ctx.fillStyle = color === TYPE_COLORS.unverified ? '#1f2328' : '#ffffff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 12, canvas.height / 2 + 1);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({ map: texture, sizeAttenuation: false });
  const sprite = new THREE.Sprite(material);
  sprite.name = text;
  sprite.scale.set((LABEL_HEIGHT * canvas.width) / canvas.height, LABEL_HEIGHT, 1);
  sprite.userData.baseScale = sprite.scale.clone();
  sprite.center.set(0.5, 0); // anchored at its bottom centre, above the box
  return sprite;
}

// 3D objects of one change: before / after boxes (moved: dashed darker before box and an
// arrow between the box centres), plus a label above them. Returns null if the change has
// no box.
function buildChange3d(change, type) {
  const color = new THREE.Color(TYPE_COLORS[type]);
  const group = new THREE.Group();
  const boxes = {};
  for (const side of ['before', 'after']) {
    boxes[side] = sideBox(change[side]);
    if (!boxes[side]) continue;
    const dashed = type === 'moved' && side === 'before';
    group.add(boxLines(boxes[side], dashed ? color.clone().multiplyScalar(0.6) : color, dashed));
  }
  if (!group.children.length) return null;
  if (type === 'moved' && boxes.before && boxes.after) {
    const arrow = arrowBetween(boxes.before.center, boxes.after.center, color);
    if (arrow) group.add(arrow);
  }
  world.add(group);
  world.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(group); // display frame
  const label = makeLabel(`${change.id} ${change.label ?? ''}`.trim().slice(0, 40), TYPE_COLORS[type]);
  label.position.set((bounds.min.x + bounds.max.x) / 2, bounds.max.y, (bounds.min.z + bounds.max.z) / 2);
  scene3d.add(label);

  // drawn on top of the points so that boxes inside dense clouds stay visible
  const materials = [label.material];
  group.traverse((object) => object.material && materials.push(object.material));
  for (const material of materials) {
    material.transparent = true;
    material.depthTest = false;
  }
  group.traverse((object) => (object.renderOrder = 1));
  label.renderOrder = 2;
  return { group, label, bounds, materials };
}

// --------------------------------------------------------------- side panel

function evidenceText(change) {
  const parts = [];
  const transform = change.transform || {};
  if (typeof transform.translation_m === 'number') parts.push(`moved ${fmt(transform.translation_m)} m`);
  if (typeof transform.rotation_deg === 'number') {
    const angle = `${Math.round(transform.rotation_deg)}°`;
    parts.push(transform.rotation_reliable === false ? `rotation (unreliable) ${angle}` : `rotated ${angle}`);
  }
  const evidence = change.evidence || {};
  const known = Object.keys(EVIDENCE_LABELS).filter((key) => key in evidence);
  const others = Object.keys(evidence).filter((key) => !(key in EVIDENCE_LABELS));
  for (const key of [...known, ...others]) {
    if (typeof evidence[key] !== 'number') continue;
    parts.push(`${EVIDENCE_LABELS[key] ?? key.replaceAll('_', ' ')} ${fmt(evidence[key])}`);
  }
  return parts.join(' · ');
}

function badge(type) {
  const node = el('span', `badge ${type}`, type);
  node.style.setProperty('--type', TYPE_COLORS[type]);
  return node;
}

function buildEntry(change, type, index) {
  const entry = el('li', 'change');
  entry.tabIndex = 0;
  const confidence = typeof change.confidence === 'number' ? `${Math.round(100 * change.confidence)} %` : '';
  const head = el('div', 'change-head');
  head.append(badge(type), el('span', 'change-title', `${change.id} ${change.label ?? ''}`), el('span', 'conf', confidence));
  if (confidence) head.lastChild.title = 'confidence';
  entry.append(head);
  if (change.description) entry.append(el('p', 'desc', change.description));
  const evidence = evidenceText(change);
  if (evidence) entry.append(el('p', 'evidence', evidence));
  if (typeof change.evidence?.vlm_verdict === 'string') {
    entry.append(el('p', 'verdict', `VLM: ${change.evidence.vlm_verdict}`));
  }
  entry.append(el('div', 'crops'));
  entry.addEventListener('click', () => select(index));
  entry.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      select(index);
    }
  });
  return entry;
}

function showCrops(item) {
  const box = item.entry.querySelector('.crops');
  if (box.childElementCount) return;
  for (const side of ['before', 'after']) {
    const src = item.change.crops?.[side];
    if (!src) continue;
    const figure = el('figure');
    const img = el('img');
    img.alt = `${side} crop of change ${item.change.id}`;
    img.onerror = () => figure.replaceWith(el('p', 'missing', `${side} crop not found: ${src}`));
    img.src = new URL(src, state.changesUrl).href;
    figure.append(img, el('figcaption', '', side));
    box.append(figure);
  }
}

function buildHeader(scene, doc, counts) {
  const name = scene.name || 'RoomDiff scene';
  $('title').textContent = name;
  document.title = `${name} · RoomDiff`;
  const box = $('counts');
  for (const type of TYPES) {
    const count = el('span', counts[type] ? 'count' : 'count zero');
    const dot = el('span', 'dot');
    dot.style.setProperty('--type', TYPE_COLORS[type]);
    count.append(dot, `${counts[type]} ${type}`);
    box.append(count);
  }
  $('summary').textContent = doc.summary || '';
}

function buildFilters(counts) {
  const row = $('filters');
  for (const type of TYPES) {
    const label = el('label');
    const input = el('input');
    input.type = 'checkbox';
    input.checked = true;
    input.addEventListener('change', () => {
      state.filters[type] = input.checked;
      updateFilters();
    });
    const dot = el('span', 'dot');
    dot.style.setProperty('--type', TYPE_COLORS[type]);
    label.append(input, ' ', dot, `${type} (${counts[type]})`);
    row.append(label);
  }
}

function updateFilters() {
  for (const item of state.items) {
    const on = state.filters[item.type];
    item.entry.hidden = !on;
    if (item.group) item.group.visible = on;
  }
  updateLabels();
  if (state.selected >= 0 && state.items[state.selected].entry.hidden) select(-1);
}

// A label is shown for every visible confirmed change and for the selected change.
function updateLabels() {
  state.items.forEach((item, i) => {
    if (!item.label) return;
    item.label.visible = !item.entry.hidden && (item.type !== 'unverified' || i === state.selected);
  });
}

// -------------------------------------------------------------- selection

function select(index) {
  state.selected = index;
  state.items.forEach((item, i) => {
    item.entry.classList.toggle('selected', i === index);
    const base = item.type === 'unverified' ? UNVERIFIED_OPACITY : 1;
    const opacity = i === index ? 1 : index < 0 ? base : DIMMED;
    for (const material of item.materials) material.opacity = opacity;
    if (item.label) item.label.scale.copy(item.label.userData.baseScale).multiplyScalar(i === index ? 1.3 : 1);
  });
  updateLabels();
  if (index < 0) return;
  const item = state.items[index];
  showCrops(item);
  item.entry.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  if (item.bounds) flyTo(focusView(item.bounds));
}

function step(delta) {
  const visible = state.items.map((_, i) => i).filter((i) => !state.items[i].entry.hidden);
  if (!visible.length) return;
  const position = visible.indexOf(state.selected);
  const next = position < 0
    ? (delta > 0 ? 0 : visible.length - 1)
    : (position + delta + visible.length) % visible.length;
  select(visible[next]);
}

function onKey(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const actions = { n: () => step(1), p: () => step(-1), r: resetView };
  const action = actions[event.key.toLowerCase()];
  if (!action) return;
  event.preventDefault();
  action();
}

// ------------------------------------------------------------- automation

// window.roomdiff: hook for scripts that drive the viewer (scripts/shoot_viewer.py).
// Camera moves are immediate; `ready` is true once the scene is loaded.
function automationApi() {
  return {
    ready: true,
    count: state.items.length,
    items: () => state.items.map((it, i) => ({
      index: i, id: it.change.id, type: it.type, label: it.change.label, hidden: it.entry.hidden,
    })),
    jumpHome() {
      select(-1);
      jump(homeView());
    },
    jumpTo(index) {
      const item = state.items[index];
      if (!item) throw new Error(`jumpTo: there is no change at index ${index}`);
      select(index);
      item.entry.scrollIntoView({ block: 'nearest' }); // at once, ending the smooth scroll
      if (item.bounds) jump(focusView(item.bounds));
    },
    // turns the camera about the vertical axis through the target (same distance, elevation)
    orbit(deg) {
      const offset = camera.position.clone().sub(controls.target);
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(deg));
      jump({ target: controls.target.clone(), position: controls.target.clone().add(offset) });
    },
    setClouds(mode) {
      if (!['before', 'after', 'both'].includes(mode)) {
        throw new Error(`setClouds: expected before, after or both, got ${mode}`);
      }
      document.querySelector(`input[name="show"][value="${mode}"]`).checked = true;
      updateClouds();
    },
    // stops the render loop (false restarts it), so that only render() draws: software
    // WebGL (headless Chromium) takes ~0.3 s per frame, and a running loop delays shots
    pauseLoop(paused = true) {
      renderer.setAnimationLoop(paused ? null : animate);
    },
    render() {
      controls.update();
      renderer.render(scene3d, camera);
    },
  };
}

// ------------------------------------------------------------------- main

async function main() {
  initThree();
  state.sceneUrl = sceneJsonUrl();
  setStatus('Loading scene…');
  const scene = await fetchJson(state.sceneUrl);
  state.changesUrl = new URL(scene.changes || 'changes.json', state.sceneUrl);
  const loaded = { before: 0, after: 0 };
  setStatus('Loading point clouds…');
  const [doc, before, after] = await Promise.all([
    fetchJson(state.changesUrl),
    loadPly(new URL(scene.before_ply || 'before.ply', state.sceneUrl), 'before', loaded),
    loadPly(new URL(scene.after_ply || 'after.ply', state.sceneUrl), 'after', loaded),
  ]);

  // orientation: the estimated up axis becomes +Y, so the floor is horizontal
  const up = vec(scene.up_axis || [0, -1, 0]).normalize();
  world.quaternion.setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
  world.updateMatrixWorld(true);

  addCloud(before, 'before');
  addCloud(after, 'after');
  state.sample = [];
  for (const geometry of [before, after]) { // display-frame sample for the home view fit
    const positions = geometry.getAttribute('position');
    const stride = Math.max(1, Math.floor(positions.count / 8000));
    for (let i = 0; i < positions.count; i += stride) {
      state.sample.push(toDisplay(new THREE.Vector3().fromBufferAttribute(positions, i)));
    }
  }
  if (scene.bbox_m) {
    state.box = new THREE.Box3(vec(scene.bbox_m[0]), vec(scene.bbox_m[1]));
  } else {
    state.box = new THREE.Box3();
    for (const geometry of [before, after]) {
      geometry.computeBoundingBox();
      state.box.union(geometry.boundingBox);
    }
  }
  const radius = 0.5 * state.box.getSize(new THREE.Vector3()).length() || 1;
  camera.near = radius / 500;
  camera.far = radius * 50;
  camera.updateProjectionMatrix();
  state.viewDir = vec(scene.view_dir || [0, 0, 1]).applyQuaternion(world.quaternion).setY(0);
  if (state.viewDir.lengthSq() < 1e-6) state.viewDir.set(0, 0, -1);
  state.viewDir.normalize();
  state.baseSize = 2.5 * (scene.stats?.voxel_m || 0.01);

  const changes = Array.isArray(doc.changes) ? doc.changes : [];
  const counts = Object.fromEntries(TYPES.map((t) => [t, 0]));
  const list = $('changes');
  changes.forEach((change, index) => {
    const type = TYPES.includes(change.type) ? change.type : 'unverified';
    counts[type] += 1;
    const entry = buildEntry(change, type, index);
    list.append(entry);
    const objects = buildChange3d(change, type) || { materials: [] };
    state.items.push({ change, type, entry, ...objects });
  });
  if (!changes.length) list.append(el('li', 'empty', 'No changes in changes.json.'));
  spreadLabels();
  updateLabels(); // unverified labels stay hidden until their region is selected
  buildHeader(scene, doc, counts);
  buildFilters(counts);

  for (const input of document.querySelectorAll('input[name="show"], #tint')) {
    input.addEventListener('change', updateClouds);
  }
  $('size').addEventListener('input', updateClouds);
  $('light').addEventListener('change', updateBackground);
  window.addEventListener('keydown', onKey);
  updateClouds();
  updateBackground();

  const home = homeView();
  controls.target.copy(home.target);
  camera.position.copy(home.position);
  setStatus('');

  window.roomdiff = automationApi();
  const selectId = new URLSearchParams(location.search).get('select');
  if (selectId !== null) {
    const index = state.items.findIndex((item) => String(item.change.id) === selectId);
    if (index >= 0) window.roomdiff.jumpTo(index);
    else console.warn(`select=${selectId}: there is no change with this id`);
  }
}

main().catch(fail);
