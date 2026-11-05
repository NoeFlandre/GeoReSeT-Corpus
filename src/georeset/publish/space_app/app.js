import { parquetRead } from "https://esm.sh/hyparquet@1.9.1";

const { MapboxOverlay, H3HexagonLayer } = deck;
const DATA = "data/";
const API = "https://huggingface.co/api/datasets/";

const PCA = "What places talk about (PCA)";
const LAYERS = [
  { id: "relevant", group: "Volume", label: "Land-use sentences", kind: "count" },
  { id: "photos", group: "Volume", label: "Scene photos", kind: "count" },
  { id: "language", group: "Language", label: "Most frequent", kind: "language" },
  { id: "pca_landuse", group: PCA, row: "Land-use sentences", label: "All languages", kind: "pca", corpus: "landuse" },
  { id: "pca_english", group: PCA, row: "Land-use sentences", label: "English only", kind: "pca", corpus: "english" },
  { id: "pca_scenes", group: PCA, row: "Scene photos", label: "Their descriptions", kind: "pca", corpus: "scenes" },
];
const RAMP = ["#dbe8fa", "#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d2b5e"];
// Source filter of the sentence layer and samples: field of the data, ``source`` value of the samples.
const SOURCES = [
  { id: "relevant", label: "All sources", source: null },
  { id: "website", label: "Websites", source: "website" },
  { id: "description", label: "OSM descriptions", source: "osm_description" },
  { id: "wikipedia", label: "Wikipedia", source: "wikipedia" },
  { id: "wikivoyage", label: "Wikivoyage", source: "wikivoyage" },
];
const PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
  "#9c755f", "#86bcb6", "#d37295", "#8cd17d"];
const OTHER = "#bab0ac";
const AXIS_COLOURS = ["#e15759", "#59a14f", "#4e79a7"];
const ZOOMS = [[2.5, 2], [4, 3], [5.5, 4], [Infinity, 5]];

const state = { layer: "relevant", source: "relevant", res: 2, manifest: null, tables: {}, chunks: new Map(), samples: new Map(), selected: null, labels: undefined };
const ALPHA = 170; // translucent cells: roads, coasts and water stay readable underneath
const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en");
const names = new Intl.DisplayNames(["en"], { type: "language" });

function language(code) {
  if (!code) return "–";
  try { return names.of(code.split("_")[0]) || code; } catch { return code; }
}
function short(n) {
  if (n >= 1e6) return `${+(n / 1e6).toPrecision(3)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toPrecision(3)}K`;
  return `${+n.toPrecision(3)}`;
}
function hex(colour) {
  const v = parseInt(colour.slice(1), 16);
  return [v >> 16, (v >> 8) & 255, v & 255];
}
function ramp(colours, t) {
  const x = Math.min(Math.max(t, 0), 1) * (colours.length - 1);
  const i = Math.min(Math.floor(x), colours.length - 2);
  const [a, b] = [hex(colours[i]), hex(colours[i + 1])];
  return a.map((c, k) => Math.round(c + (b[k] - c) * (x - i)));
}
const unpack = (v) => [v >> 16 & 255, v >> 8 & 255, v & 255];

async function table(path) {
  const response = await fetch(DATA + path);
  if (!response.ok) return [];
  const file = await response.arrayBuffer();
  return new Promise((resolve) => parquetRead({ file, rowFormat: "object", onComplete: resolve }));
}

function layerOf(id) { return LAYERS.find((l) => l.id === id); }

function field(layer) { return layer.id === "relevant" ? state.source : layer.id; }

function value(cell, layer) {
  if (layer.kind === "count") return res5() ? cell[field(layer)] : cell[`m_${field(layer)}`];
  return cell[layer.id];
}
const res5 = () => state.res === 5;

function colour(layer) {
  const top = state.manifest.ranges[state.res];
  const languages = state.manifest.languages;
  return (cell) => {
    const v = value(cell, layer);
    if (layer.kind === "count") return [...ramp(RAMP, Math.log1p(v) / Math.log1p(top[field(layer)])), ALPHA];
    if (layer.kind === "language") {
      const i = languages.indexOf(cell.language);
      return [...hex(i < 0 ? OTHER : PALETTE[i]), ALPHA];
    }
    return [...unpack(v), ALPHA + 30];
  };
}

function visible(layer) {
  const rows = res5() ? [...state.chunks.values()].flat() : state.tables[state.res] || [];
  return rows.filter((cell) => {
    const v = layer.kind === "language" ? cell.language : value(cell, layer);
    return v !== null && v !== undefined && !(typeof v === "number" && (Number.isNaN(v) || (layer.kind === "count" && v <= 0)));
  });
}

function tooltip({ object }) {
  if (!object) return null;
  const layer = layerOf(state.layer);
  const v = value(object, layer);
  let line;
  const label = layer.id === "relevant" && state.source !== "relevant" ? `${sourceOf().label} sentences` : layer.label;
  if (layer.kind === "count") line = `${label}: <b>${fmt.format(Math.round(v * 10) / 10)}</b>${res5() ? "" : " per res-5 cell"}`;
  else if (layer.kind === "language") line = `Most frequent: <b>${language(object.language)}</b>`;
  else line = `${layer.label}: click for the three axes`;
  return { html: `${line}<br><span style="opacity:.7">${fmt.format(object.relevant)} land-use sentences · click for details</span>` };
}

let overlay;
function render() {
  const layer = layerOf(state.layer);
  const cells = new H3HexagonLayer({
    id: `cells-${state.res}`,
    data: visible(layer),
    getHexagon: (d) => d.cell,
    getFillColor: colour(layer),
    updateTriggers: { getFillColor: [state.layer, state.source, state.res] },
    stroked: false, extruded: false, pickable: true, highPrecision: "auto",
    autoHighlight: true, highlightColor: [20, 20, 20, 70],
    onClick: ({ object }) => object && select(object),
    beforeId: state.labels,
  });
  const selected = new H3HexagonLayer({
    id: "selected",
    data: state.selected ? [state.selected] : [],
    getHexagon: (d) => d,
    filled: false, stroked: true, extruded: false,
    getLineColor: [20, 20, 20, 255], lineWidthUnits: "pixels", getLineWidth: 2.5,
    beforeId: state.labels,
  });
  overlay.setProps({ layers: [cells, selected] });
  $("resolution").textContent = `H3 resolution ${state.res}${res5() ? " (data layer)" : ": mean of resolution-5 cells"}`;
}

function legend() {
  const layer = layerOf(state.layer);
  const box = $("legend");
  if (layer.kind === "count") {
    const top = state.manifest.ranges[state.res][field(layer)];
    box.innerHTML = `<div class="ramp" style="background:linear-gradient(90deg,${RAMP.join(",")})"></div>
      <div class="ramp-labels"><span>1</span><span>${short(top)}+ ${res5() ? "per cell" : "per res-5 cell"}</span></div>`;
  } else if (layer.kind === "language") {
    const items = state.manifest.languages.map((code, i) => `<span class="swatch"><i style="background:${PALETTE[i]}"></i>${language(code)}</span>`);
    box.innerHTML = `<div class="swatches">${items.join("")}<span class="swatch"><i style="background:${OTHER}"></i>Other</span></div>`;
  } else {
    const pca = state.manifest.pca[layer.corpus];
    box.innerHTML = `<div class="axes-legend">${pca.axes.map(([low, high], i) =>
      `<div><b style="background:${AXIS_COLOURS[i]}"></b><span>${["Red", "Green", "Blue"][i]} (${Math.round(pca.explained[i] * 100)}%): ${low} → ${high}</span></div>`).join("")}</div>
      <p class="muted" style="margin:6px 0 0;font-size:12px">Similar colours, similar ${pca.what}.</p>`;
  }
}

function controls() {
  const box = $("layers");
  const groups = [...new Set(LAYERS.map((l) => l.group))];
  const chip = (l) => `<button class="chip" role="radio" data-layer="${l.id}" aria-checked="${l.id === state.layer}">${l.label}</button>`;
  const shown = (l) => l.kind !== "pca" || state.manifest.pca[l.corpus];
  // A group's chips, under a small label per row where its layers have one (the PCA maps: what they embed).
  const chips = (g) => {
    const layers = LAYERS.filter((l) => l.group === g && shown(l));
    const rows = [...new Set(layers.map((l) => l.row))];
    return rows.map((row) => `${row ? `<p class="row">${row}</p>` : ""}<div class="chips">${
      layers.filter((l) => l.row === row).map(chip).join("")}</div>`).join("");
  };
  box.innerHTML = groups.map((g) => `<div class="group"><h2>${g}</h2>${chips(g)}</div>`).join("");
  box.querySelector(".group .chips").insertAdjacentHTML("afterend", `<div id="sources" class="chips sources" role="radiogroup" aria-label="Source">${
    SOURCES.map((s) => `<button class="chip small" role="radio" data-source="${s.id}" aria-checked="${s.id === state.source}">${s.label}</button>`).join("")
  }</div>`);
  box.onclick = (event) => {
    const { layer: id, source } = event.target.dataset || {};
    if (source) {
      state.source = source;
      state.layer = "relevant";
    } else if (id) {
      state.layer = id;
    } else return;
    box.querySelectorAll("[data-layer]").forEach((c) => c.setAttribute("aria-checked", c.dataset.layer === state.layer));
    box.querySelectorAll("[data-source]").forEach((c) => c.setAttribute("aria-checked", c.dataset.source === state.source));
    $("sources").classList.toggle("dim", state.layer !== "relevant");
    legend(); render(); remember();
    if (state.selected) select(state.cellRow);
  };
  $("sources").classList.toggle("dim", state.layer !== "relevant");
}

async function loadChunks(map) {
  if (!res5()) return;
  // The chunks under the view: those of a 5 × 5 grid of points over it (chunks are far larger than a cell).
  const b = map.getBounds();
  const steps = [0, 0.25, 0.5, 0.75, 1];
  const lats = steps.map((t) => b.getSouth() + t * (b.getNorth() - b.getSouth()));
  const lngs = steps.map((t) => b.getWest() + t * (b.getEast() - b.getWest()));
  const parents = lats.flatMap((lat) => lngs.map((lng) => h3.latLngToCell(lat, ((lng + 540) % 360) - 180, state.manifest.chunk_resolution)));
  const wanted = new Set(parents.filter((p) => state.manifest.chunks.includes(p)));
  const missing = [...wanted].filter((p) => !state.chunks.has(p));
  if (!missing.length) return;
  busy(true);
  await Promise.all(missing.map(async (p) => state.chunks.set(p, await table(`r5/${p}.parquet`))));
  busy(false);
  render();
}

function busy(on) { $("loading").hidden = !on; }

async function samples(cell) {
  const parent = h3.cellToParent(cell, state.manifest.chunk_resolution);
  if (!state.manifest.chunks.includes(parent)) return [];
  if (!state.samples.has(parent)) state.samples.set(parent, await table(`samples/${parent}.parquet`));
  const res = h3.getResolution(cell);
  return state.samples.get(parent).filter((row) => (res === 5 ? row.cell === cell : h3.cellToParent(row.cell, res) === cell));
}

function axes(cell) {
  return Object.entries(state.manifest.pca).map(([corpus, pca]) => {
    const v = cell[`pca_${corpus}`];
    if (v === null || v === undefined) return "";
    const rgb = unpack(v);
    return `<h4>What it talks about: ${pca.what}</h4>` + pca.axes.map(([low, high], i) =>
      `<div class="axis"><div class="axis-names"><span>${low}</span><span>${high}</span></div>
      <div class="axis-track"><i style="left:${(rgb[i] / 255) * 100}%;background:${AXIS_COLOURS[i]}"></i></div></div>`).join("");
  }).join("");
}

function sourceOf() { return SOURCES.find((s) => s.id === state.source); }

async function select(cell) {
  state.selected = cell.cell;
  state.cellRow = cell;
  render();
  const res = h3.getResolution(cell.cell);
  const total = (k) => fmt.format(cell[k] || 0);
  $("drawer").hidden = false;
  $("drawer-body").innerHTML = `<h3>H3 cell, resolution ${res}</h3>
    <p class="muted" style="margin:0">${res === 5 ? "About 250 km²." : "Totals over the cell; colours are the mean of its resolution-5 cells."}</p>
    <h4>Numbers</h4>
    <dl class="stats"><dt>Land-use sentences</dt><dd>${total("relevant")}</dd>
    ${SOURCES.slice(1).map((s) => `<dt class="sub">${s.label}</dt><dd>${total(s.id)}</dd>`).join("")}
    <dt>Scene photos</dt><dd>${total("photos")}</dd>
    <dt>Most frequent language</dt><dd>${language(cell.language)}</dd></dl>
    ${axes(cell)}<div id="examples"><h4>Land-use sentences</h4><p class="muted">Loading…</p></div>`;
  const rows = await samples(cell.cell);
  if (state.selected !== cell.cell) return;
  const wanted = sourceOf().source;
  const sentences = rows.filter((r) => r.text && (!wanted || r.source === wanted)).slice(0, 3);
  const scene = rows.find((r) => r.scene);
  const link = (url) => (url ? ` · <a href="${url}" target="_blank" rel="noopener">source</a>` : "");
  const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const title = wanted ? `Land-use sentences: ${sourceOf().label}` : "Land-use sentences";
  $("examples").innerHTML = `<h4>${title}</h4>${sentences.length ? sentences.map((r) =>
    `<div class="sample"><p>${esc(r.text)}</p><small>${r.source.replace("_", " ")} · ${language(r.language)}${link(r.url)}</small></div>`).join("")
    : '<p class="muted">None here.</p>'}${scene ? `<h4>A scene photographed here</h4><div class="sample"><p>${esc(scene.scene)}</p><small>${esc(scene.lulc_label)}</small></div>` : ""}`;
}

function remember(map = window.__map) {
  const c = map.getCenter();
  history.replaceState(null, "", `#${state.layer}/${map.getZoom().toFixed(2)}/${c.lat.toFixed(3)}/${c.lng.toFixed(3)}`);
}
function recalled() {
  const [layer, zoom, lat, lng] = location.hash.slice(1).split("/");
  return { layer: layerOf(layer) ? layer : null, zoom: +zoom, centre: lat && lng ? [+lng, +lat] : null };
}

async function source() {
  const m = state.manifest;
  const link = `https://huggingface.co/datasets/${m.dataset}/tree/${m.revision}`;
  $("source").innerHTML = `Data: <a href="https://huggingface.co/datasets/${m.dataset}" target="_blank" rel="noopener">${m.dataset}</a>
    @ <a href="${link}" target="_blank" rel="noopener">${m.revision.slice(0, 7)}</a>, built ${m.built.slice(0, 10)}.
    Search by <a href="https://nominatim.openstreetmap.org" target="_blank" rel="noopener">Nominatim</a>.`;
  // Stale only if a folder the explorer is built from has a newer commit: card edits do not count.
  try {
    const tree = await (await fetch(`${API}${m.dataset}/tree/main?expand=true`)).json();
    const now = Object.fromEntries(tree.map((entry) => [entry.path, entry.lastCommit?.id]));
    const changed = Object.entries(m.folders || {}).filter(([folder, commit]) => now[folder] !== commit).map(([folder]) => folder);
    if (changed.length) {
      $("source").insertAdjacentHTML("beforeend", `<span class="stale">The dataset's ${changed.join(", ")} changed since this build.</span>`);
    }
  } catch { /* offline: no check */ }
}

async function search(event, map) {
  event.preventDefault();
  const q = $("search-input").value.trim();
  if (!q) return;
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`;
  const [hit] = await (await fetch(url, { headers: { Accept: "application/json" } })).json();
  if (!hit) return;
  const [s, n, w, e] = hit.boundingbox.map(Number);
  map.fitBounds([[w, s], [e, n]], { maxZoom: 7, padding: 40 });
}

async function main() {
  state.manifest = await (await fetch(DATA + "manifest.json")).json();
  const saved = recalled();
  if (saved.layer) state.layer = saved.layer;
  const map = new maplibregl.Map({
    container: "map",
    style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    center: saved.centre || [10, 25], zoom: Number.isFinite(saved.zoom) ? saved.zoom : 1.6,
    bounds: saved.centre ? undefined : [[-170, -50], [180, 72]],
    fitBoundsOptions: { padding: window.innerWidth > 700 ? { left: 320, top: 20, right: 20, bottom: 20 } : 10 },
    minZoom: 1, maxZoom: 9, attributionControl: { compact: true },
  });
  window.__map = map;
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
  overlay = new MapboxOverlay({ interleaved: true, layers: [], getTooltip: tooltip });
  map.addControl(overlay);
  await new Promise((resolve) => (map.loaded() ? resolve() : map.once("load", resolve)));
  // Cells go under the first label layer, so place names stay on top.
  state.labels = map.getStyle().layers.find((l) => l.type === "symbol")?.id;
  controls(); legend(); source();
  busy(true);
  const coarse = await Promise.all(state.manifest.resolutions.filter((r) => r < 5).map(async (r) => [r, await table(`r${r}.parquet`)]));
  coarse.forEach(([r, rows]) => { state.tables[r] = rows; });
  busy(false);
  const update = () => {
    const res = ZOOMS.find(([z]) => map.getZoom() < z)[1];
    if (res !== state.res) { state.res = res; legend(); }
    render(); loadChunks(map); remember(map);
  };
  map.on("moveend", update);
  update();
  $("search").addEventListener("submit", (e) => search(e, map));
  $("close").onclick = () => { $("drawer").hidden = true; state.selected = null; render(); };
  $("collapse").onclick = () => {
    const panel = $("panel");
    const collapsed = panel.classList.toggle("collapsed");
    $("collapse").textContent = collapsed ? "+" : "−";
    $("collapse").setAttribute("aria-expanded", !collapsed);
  };
}

main();
