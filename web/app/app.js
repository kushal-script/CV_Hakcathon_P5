const CFG = window.P5_CONFIG || {};
const API = (CFG.apiBase || "").replace(/\/$/, "");
const ROUTE_COLOURS = ["#a8382a", "#5c4a2c", "#d4952a", "#3e6a44", "#8a5a2b", "#b56a4f", "#6e5a37", "#4d4234"];
const PIN = { cam: "#2b2318", seen: "#a8382a", best: "#a8382a", other: "#d4952a", station: "#5c4a2c", own: "#a8382a" };
const LEVEL = {
  class: ["Model identified", "good"], family: ["Model family, year uncertain", "caution"], make: ["Make only", "caution"],
  abstain: ["Not certain", "caution"], ood_make: ["Make only, model outside the catalogue", "alert"],
  ood: ["Outside the known catalogues", "alert"], no_vehicle: ["No vehicle found", "alert"],
};

function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const $ = (s) => document.querySelector(s);
const pct = (v, d = 1) => (v === null || v === undefined ? "–" : `${(100 * v).toFixed(d)}%`);
const clock = (s) => { if (s == null) return "–"; const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`; };
const jpg = (b64) => (b64 ? `data:image/jpeg;base64,${b64}` : "");
const fmtTime = (epoch) => (epoch == null ? "–" : new Date(epoch * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" }));

function notice(msg, ms = 3600, bad = false) {
  const n = $("#notice");
  n.textContent = msg; n.hidden = false; n.classList.toggle("bad", bad);
  clearTimeout(notice.t); notice.t = setTimeout(() => (n.hidden = true), ms);
}

const session = { user: null, region: null };
const lanes = { n: 0 };
function traffic(on) {
  lanes.n = Math.max(0, lanes.n + (on ? 1 : -1));
  $("#lanes").classList.toggle("on", lanes.n > 0);
  document.querySelector(".side-art")?.classList.toggle("busy-now", lanes.n > 0);
}

class AuthError extends Error {}
async function api(path, { method = "GET", form, json } = {}) {
  const opts = { method, credentials: "same-origin", headers: {} };
  if (method !== "GET") opts.headers["X-Requested-With"] = "vahana";
  if (form) opts.body = form;
  if (json) { opts.body = JSON.stringify(json); opts.headers["Content-Type"] = "application/json"; }
  traffic(true);
  let res;
  try { res = await fetch(API + path, opts); } finally { traffic(false); }
  if (res.status === 401) { location.replace("/login?ended=1&next=" + encodeURIComponent(location.pathname)); throw new AuthError("Please sign in."); }
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : await res.text();
  if (res.status === 403 && data && data.detail === "Change your password first.") { location.replace("/login?change=1"); throw new AuthError(data.detail); }
  if (!res.ok) throw new Error((data && data.detail) || (typeof data === "string" && data) || `Request failed (${res.status})`);
  return data;
}

async function signOut() {
  const entry = $("#entry");
  entry.hidden = false; entry.classList.remove("open", "leaving");
  $("#entry-name").textContent = "Signing out"; $("#entry-kn").textContent = "ನಿರ್ಗಮನ"; $("#entry-sub").textContent = "";
  await fetch(API + "/auth/logout", { method: "POST", credentials: "same-origin", headers: { "X-Requested-With": "vahana" } }).catch(() => {});
  setTimeout(() => location.replace("/login"), 450);
}
$("#logout").addEventListener("click", signOut);
$("#menu-btn").addEventListener("click", (e) => {
  e.stopPropagation();
  const m = $("#menu"); m.hidden = !m.hidden; $("#menu-btn").setAttribute("aria-expanded", String(!m.hidden));
});
document.addEventListener("click", (e) => { if (!$("#menu").hidden && !$("#menu").contains(e.target)) { $("#menu").hidden = true; $("#menu-btn").setAttribute("aria-expanded", "false"); } });
$("#change-pw").addEventListener("click", () => {
  $("#menu").hidden = true; $("#pw-error").hidden = true; $("#pw-form").reset();
  $("#pw-form [name=username]").value = session.user ? session.user.username : "";
  $("#pw-sheet").showModal(); $("#pw-current").focus();
});
$("#pw-close").addEventListener("click", () => $("#pw-sheet").close());
$("#pw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const err = $("#pw-error"); err.hidden = true;
  if ($("#pw-new").value !== $("#pw-repeat").value) { err.textContent = "The new passwords do not match."; err.hidden = false; return; }
  try {
    const res = await fetch(API + "/auth/password", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json", "X-Requested-With": "vahana" },
      body: JSON.stringify({ current: $("#pw-current").value, new: $("#pw-new").value }) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Could not change the password.");
    $("#pw-sheet").close(); notice("Password changed. Other devices were signed out.");
  } catch (x) { err.textContent = x.message; err.hidden = false; }
});

async function health() {
  const el = $("#health");
  try {
    const r = await fetch(API + "/api/health").then((x) => x.json());
    el.textContent = r.status === "ok" ? "Service online" : "Starting"; el.className = `health ${r.status === "ok" ? "ok" : ""}`;
  } catch { el.textContent = "Service unreachable"; el.className = "health down"; }
}

const NEUTRAL_WATER = { light: "#d9cfbb", dark: "#2b241a" };
function isBluish(c) {
  const m = /^#([0-9a-f]{6})$/i.exec(c) || /^#([0-9a-f]{3})$/i.exec(c);
  let r, g, b;
  if (m) {
    const hex = m[1].length === 3 ? m[1].split("").map((x) => x + x).join("") : m[1];
    [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  } else {
    const rgb = /rgba?\(([^)]+)\)/i.exec(c);
    if (rgb) [r, g, b] = rgb[1].split(",").map((x) => parseFloat(x));
    else {
      const hsl = /hsla?\(\s*([\d.]+)[ ,]+([\d.]+)%[ ,]+([\d.]+)%/i.exec(c);
      if (!hsl) return false;
      const hue = parseFloat(hsl[1]), sat = parseFloat(hsl[2]);
      return hue >= 175 && hue <= 265 && sat > 12;
    }
  }
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  if (max - min < 18) return false;
  let hue;
  if (max === r) hue = ((g - b) / (max - min)) % 6; else if (max === g) hue = (b - r) / (max - min) + 2; else hue = (r - g) / (max - min) + 4;
  hue = (hue * 60 + 360) % 360;
  return hue >= 175 && hue <= 265;
}
function neutralise(colour) {
  if (typeof colour !== "string" || !isBluish(colour)) return colour;
  return NEUTRAL_WATER.light;
}
function stripBlue(style) {
  for (const layer of style.layers || []) {
    for (const group of ["paint", "layout"]) {
      const p = layer[group];
      if (!p) continue;
      for (const key of Object.keys(p)) {
        if (!/color/.test(key)) continue;
        const v = p[key];
        if (typeof v === "string") p[key] = neutralise(v);
        else if (v && Array.isArray(v.stops)) v.stops = v.stops.map(([z, c]) => [z, neutralise(c)]);
        else if (Array.isArray(v)) p[key] = JSON.parse(JSON.stringify(v), (k, x) => neutralise(x));
      }
    }
    if (/water/.test(layer.id) && layer.paint) {
      if ("fill-color" in layer.paint) layer.paint["fill-color"] = NEUTRAL_WATER.light;
      if ("line-color" in layer.paint) layer.paint["line-color"] = NEUTRAL_WATER.light;
    }
  }
  return style;
}

const map = { m: null, ready: null, picking: null, data: {}, loaded: false };
function circle(lat, lon, radius, n = 48) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    pts.push([lon + (radius / (111320 * Math.cos((lat * Math.PI) / 180))) * Math.cos(a), lat + (radius / 110540) * Math.sin(a)]);
  }
  return pts;
}
const fc = (features = []) => ({ type: "FeatureCollection", features });
function setLayer(name, data) {
  map.data[name] = data;
  const s = map.loaded && map.m.getSource(name);
  if (s) s.setData(data);
}
function clearLayers(...names) { names.forEach((n) => setLayer(n, fc())); }
function fit(coords, maxZoom = 15) {
  if (!coords.length || !map.m) return;
  const b = coords.reduce((acc, c) => acc.extend(c), new maplibregl.LngLatBounds(coords[0], coords[0]));
  map.m.fitBounds(b, { padding: 70, maxZoom, duration: 600 });
}
function setKey(items) {
  const key = $("#map-key");
  key.replaceChildren(...items.map(([kind, colour, label]) => h("div", { class: "k" }, h("span", { class: kind, style: { background: colour } }), label)));
  key.hidden = !items.length;
}

async function initMap() {
  let style;
  try {
    style = stripBlue(await (await fetch(CFG.mapStyle || "https://tiles.openfreemap.org/styles/positron")).json());
  } catch {
    style = { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#efe7d6" } }] };
  }
  style.sources.satellite = { type: "raster", tiles: ["https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2021_3857/default/g/{z}/{y}/{x}.jpg"],
    tileSize: 256, maxzoom: 15, attribution: "Sentinel-2 cloudless 2021 by EOX IT Services GmbH (contains modified Copernicus Sentinel data)" };
  style.layers.splice(1, 0, { id: "satellite", type: "raster", source: "satellite", layout: { visibility: "none" } });
  const font = style.layers.map((l) => l.layout && l.layout["text-font"]).find(Boolean);
  const labelFont = font ? { "text-font": font } : {};
  const bb = session.region && session.region.bbox;
  const bounds = bb ? [[bb[0] - 0.08, bb[1] - 0.06], [bb[2] + 0.08, bb[3] + 0.06]] : null;
  const own = session.user && session.user.station;
  map.m = new maplibregl.Map({ container: "map", style, center: own && own.lon != null ? [own.lon, own.lat] : CFG.center || [77.5946, 12.9716],
    zoom: own && own.lon != null ? 12.6 : CFG.zoom || 11.2, maxPitch: 70, minZoom: 9.4, ...(bounds ? { maxBounds: bounds } : {}), attributionControl: { compact: true } });
  map.m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  map.ready = new Promise((res) => map.m.on("load", res));
  await map.ready;
  const add = (id, type, paint, filter, layout) => map.m.addLayer({ id, type, source: id.split(":")[0], paint, ...(filter ? { filter } : {}), ...(layout ? { layout } : {}) });
  for (const s of ["boundary", "stations", "iso", "routes", "gps", "areas", "heat", "points"]) map.m.addSource(s, { type: "geojson", data: fc() });
  add("boundary", "line", { "line-color": "#a8382a", "line-width": 1.6, "line-opacity": 0.55, "line-dasharray": [3, 2] });
  add("iso", "fill", { "fill-color": ["get", "colour"], "fill-opacity": ["get", "opacity"] });
  add("iso:line", "line", { "line-color": ["get", "colour"], "line-width": 1, "line-opacity": 0.6 });
  add("gps", "line", { "line-color": "#5c4a2c", "line-width": 2, "line-dasharray": [1, 2] });
  add("routes", "line", { "line-color": ["get", "colour"], "line-width": ["get", "width"], "line-opacity": ["get", "opacity"] }, ["!", ["get", "dashed"]], { "line-cap": "round", "line-join": "round" });
  add("routes:dash", "line", { "line-color": ["get", "colour"], "line-width": ["get", "width"], "line-opacity": ["get", "opacity"], "line-dasharray": [2, 1.5] }, ["get", "dashed"], { "line-cap": "round" });
  add("areas", "fill", { "fill-color": ["get", "colour"], "fill-opacity": 0.12 });
  add("areas:line", "line", { "line-color": ["get", "colour"], "line-width": 1.2 });
  add("heat", "heatmap", { "heatmap-weight": ["get", "w"], "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 10, 1, 16, 3],
    "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 10, 14, 13, 24, 16, 42], "heatmap-opacity": 0.85,
    "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "rgba(232,190,110,0)", 0.2, "rgba(232,190,110,0.6)", 0.45, "#d4952a", 0.75, "#c0612b", 1, "#a8382a"] });
  add("stations", "circle", { "circle-radius": ["case", ["get", "own"], 7, 4.5], "circle-color": ["case", ["get", "own"], "#a8382a", "#5c4a2c"],
    "circle-stroke-color": "#fbf7ef", "circle-stroke-width": 1.5, "circle-opacity": 0.9 });
  map.m.addLayer({ id: "stations:label", type: "symbol", source: "stations", minzoom: 12.5, layout: { ...labelFont, "text-field": ["get", "short"], "text-size": 11, "text-offset": [0, 1.1], "text-anchor": "top", "text-optional": true },
    paint: { "text-color": "#4d4234", "text-halo-color": "#fbf7ef", "text-halo-width": 1.4 } });
  add("points", "circle", { "circle-radius": ["get", "r"], "circle-color": ["get", "colour"], "circle-stroke-color": "#fbf7ef", "circle-stroke-width": 2 });
  map.m.addLayer({ id: "points:label", type: "symbol", source: "points", layout: { ...labelFont, "text-field": ["get", "label"], "text-size": 12, "text-offset": [0, 1.3], "text-anchor": "top", "text-optional": true },
    paint: { "text-color": "#241c13", "text-halo-color": "#fbf7ef", "text-halo-width": 1.5 } });
  map.loaded = true;
  for (const [name, data] of Object.entries(map.data)) { const src = map.m.getSource(name); if (src) src.setData(data); }
  map.m.on("click", "points", (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setDOMContent(h("div", {}, h("strong", { text: p.title || p.label }), p.detail ? h("div", { class: "muted", text: p.detail }) : null)).addTo(map.m);
  });
  map.m.on("click", "stations", (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ closeButton: false }).setLngLat(e.lngLat).setDOMContent(h("div", {}, h("strong", { text: p.name }), h("div", { class: "muted", text: p.detail }))).addTo(map.m);
  });
  map.m.on("mouseenter", "points", () => (map.m.getCanvas().style.cursor = "pointer"));
  map.m.on("mouseleave", "points", () => (map.m.getCanvas().style.cursor = map.picking ? "crosshair" : ""));
  map.m.on("click", (e) => {
    if (!map.picking) return;
    const cb = map.picking; map.picking = null; $("#map-banner").hidden = true; map.m.getCanvas().style.cursor = "";
    cb(e.lngLat.lat, e.lngLat.lng);
  });
  if (map.m.getSource("openmaptiles")) {
    map.m.addLayer({ id: "buildings-3d", type: "fill-extrusion", source: "openmaptiles", "source-layer": "building", minzoom: 13,
      filter: ["!=", ["get", "hide_3d"], true], layout: { visibility: "none" },
      paint: { "fill-extrusion-color": ["interpolate", ["linear"], ["coalesce", ["get", "render_height"], 6], 0, "#ebe2d0", 40, "#d6c6a8", 120, "#bca783"],
        "fill-extrusion-height": ["coalesce", ["get", "render_height"], 6], "fill-extrusion-base": ["coalesce", ["get", "render_min_height"], 0],
        "fill-extrusion-opacity": 0.9, "fill-extrusion-vertical-gradient": true } }, "iso");
  }
  document.querySelectorAll(".map-switch button[data-base]").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".map-switch button[data-base]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    map.m.setLayoutProperty("satellite", "visibility", b.dataset.base === "satellite" ? "visible" : "none");
  }));
  const t3 = $("#toggle-3d");
  t3.addEventListener("click", () => {
    const on = t3.getAttribute("aria-pressed") !== "true";
    t3.setAttribute("aria-pressed", String(on));
    if (map.m.getLayer("buildings-3d")) map.m.setLayoutProperty("buildings-3d", "visibility", on ? "visible" : "none");
    map.m.easeTo({ pitch: on ? 58 : 0, bearing: on ? -17 : 0, zoom: on ? Math.max(map.m.getZoom(), 15.2) : map.m.getZoom(), duration: 900 });
  });
}
function pickOnMap(message) {
  return new Promise((resolve) => {
    map.picking = (lat, lon) => resolve({ lat, lon });
    const banner = $("#map-banner"); banner.textContent = message; banner.hidden = false;
    map.m.getCanvas().style.cursor = "crosshair";
  });
}

function uploader(label, onFile) {
  const input = label.querySelector("input");
  input.addEventListener("change", () => { if (input.files[0]) onFile(input.files[0]); input.value = ""; });
  label.addEventListener("dragover", (e) => { e.preventDefault(); label.classList.add("over"); });
  label.addEventListener("dragleave", () => label.classList.remove("over"));
  label.addEventListener("drop", (e) => { e.preventDefault(); label.classList.remove("over"); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
}
const busy = (text, sub) => h("div", { class: "busy", role: "status" }, h("span", { class: "signal cycle", "aria-hidden": "true" }, h("i"), h("i"), h("i")),
  h("span", {}, text, sub ? h("small", { text: sub }) : null));
const problem = (text) => h("div", { class: "problem", role: "alert", text });
const section = (title, ...kids) => h("div", { class: "section" }, h("h2", { text: title }), ...kids);

function vehicleFacts(d) {
  if (d.model == null) return null;
  const sure = d.level === "class" || d.level === "family";
  const outside = d.level === "ood_make";
  const made = d.production ? `${d.production.from}–${d.production.to || "present"}` : null;
  const year = outside ? ["Unknown"] : d.years && d.years.length ? [d.years.join(" or ")]
    : d.year ? [sure ? String(d.year) : `${d.year} (closest match, not confirmed)`]
    : made ? [sure ? made : `${made} (closest match, not confirmed)`, `Years this model was made, from ${d.production.source.includes("wikipedia") ? "Wikipedia" : "Wikidata"}. The exact year cannot be told from the photo.`]
    : ["Unknown", "Photos in this catalogue are not labelled by model year, and no production years are recorded for this model."];
  const model = sure ? d.model : outside ? "Not one of the known models" : `${d.model} (closest match, not confirmed)`;
  const conf = sure ? pct(d.confidence) : d.level === "make" ? `${pct(d.confidence)} for the make, ${pct(d.candidate_confidence)} for this model`
    : outside ? `${pct(d.confidence)} for the make` : `${pct(d.candidate_confidence)} for this model`;
  return facts([["Make", d.make || "Unknown"], ["Model", model], ["Year", ...year], ["Confidence", conf]], { margin: "10px 0 8px" });
}

function facts(rows, style) {
  return h("dl", { class: "facts", style: style || {} }, rows.flatMap(([k, v, note]) => [h("dt", { text: k }),
    h("dd", {}, v, note ? h("div", { class: "muted", style: { fontSize: "12px" }, text: note }) : null)]));
}

function whereTaken(loc) {
  if (!loc) return { node: null, pins: [], heat: [] };
  const pins = [];
  const kids = [];
  const when = loc.taken_at ? [["Taken", loc.taken_at.replace("T", " "), loc.time_zone_recorded ? "Camera clock" : "Camera clock, time zone not recorded"]] : [];
  const cam = loc.camera ? [["Camera", loc.camera]] : [];
  if (loc.status === "gps") {
    const g = loc.gps;
    const rows = [["Source", "GPS recorded in the photo"], ["Coordinates", `${g.lat.toFixed(6)}, ${g.lon.toFixed(6)}`]];
    if (g.inside_bengaluru != null) rows.push(["Area", g.inside_bengaluru ? "Inside Bengaluru" : "Outside Bengaluru"]);
    if (g.accuracy_m != null) rows.push(["GPS accuracy", `${Math.round(g.accuracy_m)} m`, "As recorded by the camera"]);
    kids.push(facts([...rows, ...when, ...cam]),
      h("p", { class: "note", text: "Read from the photo's own metadata, not estimated. Metadata can be removed or edited after a photo is taken." }));
    pins.push({ lat: g.lat, lon: g.lon, extent: g.accuracy_m || 0, label: "Photo GPS", detail: "GPS recorded in the photo", best: true });
  } else if (loc.status === "clues") {
    kids.push(h("p", {}, h("strong", { text: "Not recorded in the photo. " }), "It has no GPS data. Text in it matches these named places in Bengaluru:"));
    kids.push(h("ul", { class: "rows" }, loc.clues.filter((c) => c.places).map((c) => h("li", { style: { gridTemplateColumns: "1fr" } },
      h("div", {}, h("div", { class: "title", text: `“${c.text}” matches ${c.name}` }),
        h("div", { class: "sub", text: `${c.places_in_city} place${c.places_in_city === 1 ? "" : "s"} in Bengaluru with this name: ` +
          c.places.map((p) => `${p.kinds.join(", ")} at ${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}${p.extent_m >= 50 ? `, spanning about ${p.extent_m} m` : ""}`).join("; ") }))))));
    loc.clues.filter((c) => c.places).forEach((c) => c.places.forEach((p) => pins.push({ lat: p.lat, lon: p.lon, label: c.name, detail: `Named in the photo: “${c.text}”` })));
    const common = loc.clues.filter((c) => !c.places);
    if (common.length) kids.push(h("p", { class: "note", text: `Also read, but shared by many places in the city: ${common.map((c) => `“${c.text}” (${c.places_in_city})`).join(", ")}.` }));
    if (when.length || cam.length) kids.push(facts([...when, ...cam], { marginTop: "10px" }));
    kids.push(h("p", { class: "note", text: "The heat on the map shows where these clues point. Each name read carries the same weight, shared equally by the places with that name, and clues pointing to the same area add up. They are clues, not a confirmed location or a measured probability: the match is on the name alone, and a sign can name a place some distance away." }));
  } else {
    kids.push(h("p", {}, h("strong", { text: "Not known. " }), loc.reason || ""));
    if (loc.texts && loc.texts.length) kids.push(h("p", { class: "note", text: `Text read in the photo: ${loc.texts.slice(0, 8).join(" · ")}` }));
    if (when.length || cam.length) kids.push(facts([...when, ...cam], { marginTop: "10px" }));
  }
  return { node: section("Where the photo was taken", ...kids), pins, heat: loc.heat || [] };
}

function verdictBlock(d, colour) {
  const [lvl, tone] = LEVEL[d.level] || [d.level, ""];
  const facts = vehicleFacts(d);
  return h("div", {},
    h("span", { class: `tag ${tone}`, text: lvl }),
    h("div", { class: "verdict", text: d.label }),
    facts,
    h("div", { class: "verdict-meta" },
      !facts && d.level !== "no_vehicle" ? h("span", { text: `Confidence ${pct(d.confidence)}` }) : null,
      colour ? h("span", { text: `Colour ${colour}` }) : null,
      !facts && d.make && d.level !== "class" ? h("span", { text: `Make ${d.make}` }) : null,
      d.catalogue_title ? h("span", { text: d.catalogue_title }) : null),
    d.level !== "no_vehicle" ? h("div", { class: "meter" }, h("i", { style: { width: pct(d.confidence) } })) : null,
    d.note ? h("p", { class: "note", text: d.note }) : null);
}

const identify = { file: null };
async function runIdentify(file) {
  identify.file = file;
  const out = $("#identify-out");
  out.replaceChildren(busy("Analysing the photo"));
  clearLayers("areas", "points", "heat");
  const fd = new FormData(); fd.append("file", file); fd.append("classify_anyway", $("#classify-anyway").checked ? "true" : "false");
  try {
    const r = await api("/api/predict", { method: "POST", form: fd });
    renderIdentify(r);
  } catch (e) { out.replaceChildren(problem(e.message)); }
}
function renderIdentify(r) {
  const out = $("#identify-out");
  const v = r.vehicles[0];
  const kids = [];
  if (!v) {
    kids.push(h("div", { class: "section" }, verdictBlock({ level: "no_vehicle", label: "No vehicle found in this photo" }),
      h("p", { class: "note", text: "Try a clearer photo, or enable classification without detection above." })));
  } else {
    kids.push(h("div", { class: "section" }, verdictBlock(v.decision)));
    kids.push(h("figure", { class: "figure", style: { marginTop: "16px" } }, h("img", { src: jpg(r.panels.annotated), alt: "Photo with detected vehicle" }),
      r.vehicles.length > 1 ? h("figcaption", { text: `${r.vehicles.length} vehicles detected; the most prominent is identified.` }) : null));
    kids.push(section("Closest matches", h("ol", { class: "ranked" }, v.decision.top5.map(([name, p]) => h("li", {}, h("span", { text: name }), h("span", { class: "val", text: pct(p) }))))));
    const others = Object.values(v.markets || {}).filter((m) => m.catalogue !== v.decision.catalogue);
    if (others.length) {
      kids.push(section("Other catalogue", h("table", { class: "table" },
        h("thead", {}, h("tr", {}, h("th", { text: "Catalogue" }), h("th", { text: "Answer" }), h("th", { class: "num", text: "Conf." }))),
        h("tbody", {}, others.map((m) => h("tr", {}, h("td", { text: m.catalogue_title }),
          h("td", {}, h("div", { text: m.label }), h("div", { class: "muted", style: { fontSize: "12px" }, text: (LEVEL[m.level] || [m.level])[0] })),
          h("td", { class: "num", text: pct(m.confidence) }))))),
        h("p", { class: "note", text: "Each catalogue has its own classifier, calibration and out-of-catalogue check. The answer shown above is the more specific and more confident of the two." })));
    }
    const figs = [["enhanced", "Quality-adjusted input"], ["grabcut", "Vehicle region (GrabCut)"], ["edges", "Edges (Canny)"], ["cam", "Where the model looked"],
      ["colour_histogram", "Colour profile (HSV)"], ["saliency_roi", "Classical region of interest"]].filter(([k]) => r.panels[k]);
    kids.push(section("How the decision was reached",
      r.enhancement.length ? h("p", { class: "note", text: `Image quality: ${r.quality.flags.join(", ") || "normal"}. Applied ${r.enhancement.join(", ")}.` }) : null,
      h("div", { class: "gallery", style: { marginTop: "10px" } }, figs.map(([k, cap]) => h("figure", { class: "figure" }, h("img", { src: jpg(r.panels[k]), alt: cap }), h("figcaption", { text: cap })))),
      v.geometry ? h("dl", { class: "facts", style: { marginTop: "14px" } }, h("dt", { text: "Viewpoint" }), h("dd", { text: v.geometry.viewpoint.replaceAll("_", " ") }),
        h("dt", { text: "Aspect ratio" }), h("dd", { text: v.geometry.aspect.toFixed(2) }), h("dt", { text: "Symmetry" }), h("dd", { text: v.geometry.symmetry.toFixed(2) })) : null));
    if (r.comparison) {
      const rows = [["Classical only", r.comparison.classical_only], ["Classical on detected region", r.comparison.classical_on_detector_crop],
        ["Deep network", r.comparison.deep], ["Hybrid (deployed)", r.comparison.hybrid]].filter(([, x]) => x);
      kids.push(section("Classical, deep and hybrid on this photo",
        h("table", { class: "table" }, h("thead", {}, h("tr", {}, h("th", { text: "Approach" }), h("th", { text: "Answer" }), h("th", { class: "num", text: "Conf." }))),
          h("tbody", {}, rows.map(([n, x]) => h("tr", {}, h("td", { text: n }), h("td", { text: x.label }), h("td", { class: "num", text: pct(x.confidence) }))))),
        h("p", { class: "note", text: r.comparison.final_catalogue
          ? `All four rows use the Stanford Cars catalogue (196 models). The answer at the top comes from the ${r.comparison.final_catalogue} catalogue, which gave the more specific and more confident answer for this photo.`
          : "All four rows use the Stanford Cars catalogue (196 models)." })));
    }
  }
  const where = whereTaken(r.location);
  if (where.node) kids.push(where.node);
  kids.push(h("p", { class: "disclaimer", text: `Processed in ${(r.timings_ms.total / 1000).toFixed(2)} s.` }));
  out.replaceChildren(...kids);
  showPins(where.pins, where.heat);
}
function showPins(pins, heat = []) {
  clearLayers("areas", "points", "heat");
  if (!pins.length && !heat.length) return;
  setLayer("areas", fc(pins.filter((p) => p.extent > 0).map((p) => ({ type: "Feature", properties: { colour: p.best ? PIN.best : PIN.other },
    geometry: { type: "Polygon", coordinates: [circle(p.lat, p.lon, p.extent)] } }))));
  const top = heat.reduce((m, x) => Math.max(m, x[2]), 0) || 1;
  setLayer("heat", fc(heat.map(([lat, lon, w]) => ({ type: "Feature", properties: { w: w / top }, geometry: { type: "Point", coordinates: [lon, lat] } }))));
  setLayer("points", fc(pins.map((p) => ({ type: "Feature", properties: { colour: p.best ? PIN.best : PIN.other, r: p.best ? 7 : 4, label: p.label, title: p.label, detail: p.detail },
    geometry: { type: "Point", coordinates: [p.lon, p.lat] } }))));
  fit([...pins.map((p) => [p.lon, p.lat]), ...heat.map(([lat, lon]) => [lon, lat])], 16);
  setKey(pins.some((p) => p.best) ? [["dot", PIN.best, "GPS recorded in the photo"]]
    : [["dot", "linear-gradient(90deg, #e8be6e, #a8382a)", "Where signboard clues point"], ["dot", PIN.other, "Place named in the photo"]]);
}

const inv = { id: null, data: null, poll: null };
async function loadCases(selectId) {
  const sel = $("#case-select");
  const list = await api("/api/cases");
  const admin = session.user && session.user.role === "division_admin";
  sel.replaceChildren(h("option", { value: "", text: list.length ? "Select a case" : "No cases yet" }), ...list.map((c) => h("option", { value: c.id,
    text: `${c.name}${c.target ? " · " + c.target : ""}${admin && c.station_name ? " · " + c.station_name : ""}${c.status === "closed" ? " (closed)" : ""}` })));
  if (selectId) sel.value = selectId;
  if (sel.value) openCase(sel.value); else renderCase(null);
}
$("#case-select").addEventListener("change", (e) => (e.target.value ? openCase(e.target.value) : renderCase(null)));
async function createCase(name, stationId) {
  const fd = new FormData(); fd.append("name", name);
  if (stationId) fd.append("station_id", stationId);
  const c = await api("/api/cases", { method: "POST", form: fd });
  await loadCases(c.id);
}
$("#case-new").addEventListener("click", async () => {
  const name = `Case ${new Date().toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`;
  if (!session.user || session.user.role !== "division_admin") { createCase(name).catch((e) => notice(e.message, 4000, true)); return; }
  const input = h("input", { type: "text", value: name, maxlength: 120 });
  const sel = h("select", {}, ...(session.region ? session.region.stations : []).map((x) => h("option", { value: x.id, text: x.name })));
  const go = h("button", { class: "button primary", type: "submit", text: "Open case" });
  const form = h("form", {}, h("label", { class: "field" }, "Case name", input), h("label", { class: "field" }, "Station responsible", sel), h("div", { class: "actions" }, go));
  form.addEventListener("submit", async (e) => { e.preventDefault(); try { await createCase(input.value.trim() || name, sel.value); $("#sheet").close(); } catch (x) { notice(x.message, 4000, true); } });
  $("#sheet-title").textContent = "Open a case"; $("#sheet-body").replaceChildren(form); $("#sheet").showModal();
});
async function openCase(id) {
  inv.id = id;
  try { inv.data = await api(`/api/cases/${id}`); renderCase(inv.data); drawCase(inv.data); } catch (e) { $("#case-out").replaceChildren(problem(e.message)); }
}
function renderCase(c) {
  const out = $("#case-out");
  if (!c) { out.replaceChildren(h("p", { class: "note", style: { marginTop: "16px" }, text: "Create a case to start. Add a photo of the vehicle you are looking for, then add camera footage." })); clearLayers("points", "routes", "iso", "areas", "gps", "heat"); setKey([]); return; }
  const target = c.target;
  const tgt = section("Vehicle of interest",
    target ? h("ul", { class: "rows" }, h("li", {}, h("img", { src: jpg(target.thumbnail), alt: "" }), h("div", {}, h("div", { class: "title", text: target.label }), h("div", { class: "sub", text: `${(LEVEL[target.decision.level] || [""])[0]} · ${pct(target.decision.confidence)}` })), h("span"))) : null,
    h("label", { class: "upload", style: { marginTop: "10px" } }, h("input", { type: "file", accept: "image/*,.heic,.heif", hidden: true }),
      h("span", { class: "upload-title", text: target ? "Replace the photo" : "Add a photo of the vehicle" }), h("span", { class: "upload-sub", text: "Used to search every camera feed in this case" })));
  uploader(tgt.querySelector(".upload"), async (file) => {
    const fd = new FormData(); fd.append("file", file);
    tgt.append(busy("Profiling the vehicle"));
    try { inv.data = await api(`/api/cases/${c.id}/target`, { method: "POST", form: fd }); renderCase(inv.data); drawCase(inv.data); } catch (e) { tgt.append(problem(e.message)); }
  });

  const cams = section("Camera feeds",
    c.cameras.length ? h("ul", { class: "rows" }, c.cameras.map((cam) => cameraRow(c, cam))) : h("p", { class: "note", text: "No footage yet." }),
    cameraForm(c));

  const sightings = c.target ? section("Where it was seen",
    c.sightings.length ? h("ul", { class: "rows" }, c.sightings.slice(0, 12).map((s) => h("li", { class: "clickable", onclick: () => openCamera(c, s.camera_id, s.t_video) },
      h("img", { src: jpg(s.thumbnail), alt: "" }),
      h("div", {}, h("div", { class: "title", text: s.camera_name }), h("div", { class: "sub", text: `${s.t_abs ? fmtTime(s.t_abs) : "at " + clock(s.t_video)} · ${s.label} · heading ${s.direction}` }),
        h("div", { class: "sub", text: `look ${s.appearance.toFixed(2)} · model ${pct(s.model_overlap, 0)} · colour ${s.colour === 1 ? "same" : s.colour > 0 ? "similar" : "different"}` })),
      h("span", { class: "score", text: pct(s.score, 0) }))))
      : h("p", { class: "note", text: c.cameras.some((x) => x.status === "done") ? "No matching vehicle in the analysed footage." : "Sightings appear here once footage is analysed." })) : null;

  const notes = h("label", { class: "field" }, "What else is known", h("textarea", { id: "case-notes", placeholder: "e.g. Last seen near Brigade Road around 9:30 pm, possibly heading towards Indiranagar via Old Airport Road. May have stopped for fuel." }),
    h("span", { class: "hint", text: "Places, times, directions and stops in the note are used to rank possible routes through areas without cameras." }));
  const reconstruct = h("button", { class: "button primary", type: "button", text: "Reconstruct route", onclick: () => runReason(c, reconstruct) });
  const routes = section("Route reconstruction", notes, h("div", { class: "actions" }, reconstruct), h("div", { id: "route-out" }));
  out.replaceChildren(tgt, cams, sightings, routes);
  if (c.routes) renderRoutes(c.routes);
}
const GEO_LABEL = { candidate: ["Candidate route", "good"], ambiguous: ["Two places fit", "caution"], not_located: ["Not located", "alert"], not_applicable: ["Cannot be matched", "alert"] };
function geoSummary(c, cam) {
  const g = cam.geolocation;
  if (!g) return null;
  const [label, kind] = GEO_LABEL[g.status] || [g.status, ""];
  return h("div", { class: "sub" }, h("span", { class: `tag ${kind}`, text: label }),
    g.confidence != null ? ` ${pct(g.confidence, 1)}${g.status === "candidate" && g.location && g.location.street ? " · " + g.location.street : ""}${g.signs && g.signs.conflicting && g.signs.conflicting.length ? " · signs disagree" : g.signs && g.signs.supporting && g.signs.supporting.length ? " · signs agree" : ""} · ` : " ",
    h("button", { class: "link", type: "button", text: "Evidence", onclick: (e) => { e.stopPropagation(); showGeolocation(c, cam); } }));
}
function caseStation(c) {
  return ((session.region && session.region.stations) || []).find((s) => s.id === c.station_id) || (session.user && session.user.station) || null;
}
function geolocateForm(c, cam) {
  const st = caseStation(c);
  const centre = { lat: st && st.lat != null ? st.lat : 12.9716, lon: st && st.lon != null ? st.lon : 77.5946 };
  const where = h("span", { class: "hint", text: st ? `Around ${st.name}` : "Around the city centre" });
  const fov = h("input", { type: "number", min: 20, max: 170, step: 1, placeholder: "e.g. 70", required: true });
  const radius = h("select", {}, ...[[1000, "1 km"], [1500, "1.5 km"], [2000, "2 km"], [3000, "3 km"]].map(([v, t]) => h("option", { value: v, text: t, selected: v === 1500 })));
  const show = () => { setLayer("areas", fc([{ type: "Feature", properties: { colour: "#a8382a" }, geometry: { type: "Polygon", coordinates: [circle(centre.lat, centre.lon, +radius.value)] } }])); fit(circle(centre.lat, centre.lon, +radius.value), 15); };
  radius.addEventListener("change", show);
  const pick = h("button", { class: "button small", type: "button", text: "Move the centre", onclick: async () => { $("#sheet").close(); const p = await pickOnMap("Click the centre of the area to search"); centre.lat = p.lat; centre.lon = p.lon; where.textContent = `${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}`; show(); $("#sheet").showModal(); } });
  const go = h("button", { class: "button primary", type: "submit", text: "Search this area" });
  const form = h("form", {},
    h("p", { class: "note", text: "For a moving camera, such as a dashcam or a phone in a vehicle. The camera's turns, stops and speed are read from the footage and compared with every road in the area, using several independent searches. A result is shown only when one place fits far better than any other. This usually takes one to five minutes." }),
    h("div", { class: "field" }, "Search area centre", h("div", { class: "actions", style: { marginTop: "5px" } }, pick, where)),
    h("label", { class: "field" }, "Camera field of view (horizontal, degrees)", fov, h("span", { class: "hint", text: "From the camera's specification sheet. Phone main cameras are usually 65 to 75 degrees, dashcams 100 to 140." })),
    h("label", { class: "field" }, "Search radius", radius),
    h("p", { class: "disclaimer", text: "A smaller area gives a faster and more reliable answer. The result is a lead to verify, not evidence on its own." }),
    h("div", { class: "actions" }, go));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    go.disabled = true; go.textContent = "Starting";
    const fd = new FormData(); fd.append("lat", centre.lat); fd.append("lon", centre.lon); fd.append("radius_m", radius.value); fd.append("hfov_deg", fov.value);
    try {
      const r = await api(`/api/cases/${c.id}/cameras/${cam.id}/geolocate`, { method: "POST", form: fd });
      $("#sheet").close(); notice("Searching the area. The result appears on the camera when it is ready.", 5000);
      await openCase(c.id); watchJob(c.id, r.job_id, cam.id);
    } catch (x) { notice(x.message, 5000, true); go.disabled = false; go.textContent = "Search this area"; }
  });
  $("#sheet-title").textContent = `Where was “${cam.name}” filmed?`; $("#sheet-body").replaceChildren(form); $("#sheet").showModal(); show();
}
function showGeolocation(c, cam) {
  const g = cam.geolocation;
  const [label, kind] = GEO_LABEL[g.status] || [g.status, ""];
  const kids = [h("p", {}, h("span", { class: `tag ${kind}`, text: label }), g.confidence != null ? ` Confidence ${pct(g.confidence, 1)}` : "")];
  kids.push(h("p", { class: "note", text: g.reason || "" }));
  const feats = [];
  if (g.search_area) feats.push({ type: "Feature", properties: { colour: "#5c4a2c" }, geometry: { type: "Polygon", coordinates: [circle(g.search_area.lat, g.search_area.lon, g.search_area.radius_m)] } });
  (g.alternatives || []).forEach((a) => feats.push({ type: "Feature", properties: { colour: "#d4952a" }, geometry: { type: "Polygon", coordinates: [circle(a.lat, a.lon, 60)] } }));
  if (g.location) feats.push({ type: "Feature", properties: { colour: "#a8382a" }, geometry: { type: "Polygon", coordinates: [circle(g.location.lat, g.location.lon, g.location.radius_m || 50)] } });
  setLayer("areas", fc(feats));
  const line = (g.route || []).map((p) => [p.lon, p.lat]);
  setLayer("routes", fc(line.length > 1 ? [{ type: "Feature", properties: { colour: "#a8382a", width: 4, opacity: 0.9, dashed: true }, geometry: { type: "LineString", coordinates: line } }] : []));
  if (line.length) fit(line, 16); else if (g.search_area) fit(circle(g.search_area.lat, g.search_area.lon, g.search_area.radius_m), 15);
  setKey([["sw", "#a8382a", "Estimated route and end point"], ["sw", "#d4952a", "Other places that fit"], ["sw", "#5c4a2c", "Search area"]]);
  if (g.log10_likelihood_ratio != null && g.status !== "not_applicable") {
    const lr = g.log10_likelihood_ratio;
    kids.push(h("p", { text: lr >= 6 ? `The best route explains the camera movement more than a million times better than the best alternative found elsewhere in the area (likelihood ratio 10^${lr.toFixed(1)}).`
      : `The best route explains the camera movement ${Math.round(10 ** lr).toLocaleString()} times better than the best alternative found elsewhere in the area.` }));
  }
  if (g.decoys && g.decoys.areas && g.decoys.areas.length) {
    const m = g.decoys.log10_margin;
    kids.push(h("p", { text: m == null ? `The same footage was also matched against ${g.decoys.areas.length} random areas of the city where it was not recorded.`
      : m >= 2 ? `Checked against ${g.decoys.areas.length} random areas of the city where the footage was not recorded: the route here fits ${m >= 6 ? "more than a million" : Math.round(10 ** m).toLocaleString()} times better than the best of them.`
      : `Checked against ${g.decoys.areas.length} random areas of the city where the footage was not recorded: the best of them fits ${m > 0 ? "only " + Math.round(10 ** m).toLocaleString() + " times worse than" : "as well as or better than"} the route here, so this area cannot be confirmed.` }));
  }
  if (g.search && g.search.agreeing_runs != null) kids.push(h("p", { class: "sub", text: `${g.search.agreeing_runs} of ${g.search.main_runs} independent searches ended at the same place.` }));
  if ((g.streets || []).length) kids.push(h("p", {}, h("strong", { text: "Roads on the estimated route: " }), g.streets.join(" → ")));
  if (g.signs) {
    const sup = g.signs.supporting || [], con = g.signs.conflicting || [];
    kids.push(h("h2", { style: { marginTop: "14px" }, text: "Signboards read in the footage" }));
    kids.push(h("p", { class: "sub", text: `${g.signs.read} pieces of text read. ${sup.length} ${sup.length === 1 ? "matches" : "match"} a named place within 100 m of the route, ${con.length} ${con.length === 1 ? "names" : "name"} a place far from it.` }));
    if (sup.length) kids.push(h("ul", { class: "rows" }, sup.map((s) => h("li", {}, h("span"), h("div", {}, h("div", { class: "title", text: `“${s.text}” at ${clock(s.t)}` }),
      h("div", { class: "sub", text: `${s.matched} (${(s.category || "").replace("=", ": ")}), ${s.distance_m} m from the route. ${s.same_name_in_city > 1 ? `${s.same_name_in_city} places in the city share this name.` : "The only place in the city with this name."}` })),
      h("span", { class: "score", text: "agrees" })))));
    if (con.length) kids.push(h("ul", { class: "rows" }, con.map((s) => h("li", {}, h("span"), h("div", {}, h("div", { class: "title", text: `“${s.text}” at ${clock(s.t)}` }),
      h("div", { class: "sub", text: `OpenStreetMap places ${s.matched} ${(s.distance_from_route_m / 1000).toFixed(1)} km from this route.` })),
      h("span", { class: "score", text: "disagrees" })))));
  }
  if ((g.turns || []).length) {
    kids.push(h("h2", { style: { marginTop: "14px" }, text: "Turns seen in the video and on the map" }));
    kids.push(h("ul", { class: "rows" }, g.turns.map((t) => h("li", {},
      t.frame_jpeg ? h("img", { src: jpg(t.frame_jpeg), alt: "" }) : h("span"),
      h("div", {}, h("div", { class: "title", text: `${clock(t.from_s)} to ${clock(t.to_s)}` }),
        h("div", { class: "sub", text: `Video ${t.video_deg > 0 ? "right" : "left"} ${Math.abs(t.video_deg).toFixed(0)}° · map ${t.map_deg > 0 ? "right" : "left"} ${Math.abs(t.map_deg).toFixed(0)}°` }),
        h("div", { class: "sub", text: [t.street_from, t.street_to].filter(Boolean).join(" → ") })),
      h("span", { class: "score", text: `${Math.abs(t.video_deg - t.map_deg).toFixed(0)}° apart` })))));
  }
  if (g.location) {
    const { lat, lon } = g.location;
    kids.push(h("p", { style: { marginTop: "12px" } }, "Check the place against street photos: ",
      h("a", { href: `https://kartaview.org/map/@${lat},${lon},18z`, target: "_blank", rel: "noopener noreferrer", text: "KartaView" }), " · ",
      h("a", { href: `https://www.mapillary.com/app/?lat=${lat}&lng=${lon}&z=18`, target: "_blank", rel: "noopener noreferrer", text: "Mapillary" })));
  }
  const m = g.camera_motion || {};
  const facts = [
    ["Footage used", `${clock(m.duration_s || 0)} (${clock(m.moving_s || 0)} moving, ${Math.round(m.total_turn_deg || 0)}° of turning)`],
    ["Search area", g.search_area ? `${(g.search_area.radius_m / 1000).toFixed(1)} km around ${g.search_area.lat.toFixed(4)}, ${g.search_area.lon.toFixed(4)}${g.search ? `, ${g.search.roads_km} km of roads` : ""}` : "–"],
    ["Simulations", g.search ? `${g.search.runs} independent searches of ${Number(g.search.particles_per_run).toLocaleString()} simulated vehicles each` : "–"],
    ["Camera", g.camera_model || "–"],
    ["Video SHA-256", g.video_sha256 || "–"],
    ["Map data", g.map_data ? `${g.map_data.osm_source || "OpenStreetMap"} (${(g.map_data.osm_date || "").slice(0, 10)})` : "OpenStreetMap"],
    ["Requested", `${g.requested_by || ""} ${g.requested_at ? fmtTime(g.requested_at) : ""}`],
  ];
  if (g.validation && g.validation.summary) facts.push(["Tested on", g.validation.summary]);
  kids.push(h("dl", { class: "facts" }, facts.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: v })])));
  kids.push(h("p", { class: "disclaimer", text: "An investigative lead, not evidence. Before acting on it, confirm at least two independent features of the footage (buildings, signboards, road layout) against street photos or a site visit." }));
  if (g.status === "candidate" && !g.accepted_by) {
    kids.push(h("div", { class: "actions" }, h("button", { class: "button primary", type: "button", text: "I have checked it: use this route for the camera", onclick: async () => {
      try { inv.data = await api(`/api/cases/${c.id}/cameras/${cam.id}/geolocation/accept`, { method: "POST" }); $("#sheet").close(); renderCase(inv.data); drawCase(inv.data); notice("Route saved to the camera as an estimate."); }
      catch (x) { notice(x.message, 5000, true); } } })));
  } else if (g.accepted_by) kids.push(h("p", { class: "sub", text: `Confirmed by ${g.accepted_by} on ${fmtTime(g.accepted_at)}.` }));
  $("#sheet-title").textContent = `Where “${cam.name}” was filmed`; $("#sheet-body").replaceChildren(...kids); $("#sheet").showModal();
}
function cameraRow(c, cam) {
  const status = cam.status === "done" ? `${cam.analysis.counts.total} vehicles · ${clock(cam.analysis.meta.analysed_s)} analysed` : cam.status === "error" ? `Failed: ${cam.error || "unknown error"}` : "Analysing";
  const where = cam.lat != null ? `${cam.lat.toFixed(4)}, ${cam.lon.toFixed(4)} (${cam.location_source})` : "Location not set";
  const sug = cam.suggested_location && cam.lat == null ? h("button", { class: "link", type: "button", text: `Use suggested location: ${cam.suggested_location.label}`, onclick: (e) => { e.stopPropagation(); updateCamera(c, cam, cam.suggested_location.lat, cam.suggested_location.lon); } }) : null;
  const setLoc = h("button", { class: "link", type: "button", text: cam.lat == null ? "Set on map" : "Move", onclick: async (e) => { e.stopPropagation(); const p = await pickOnMap(`Click the location of “${cam.name}”`); updateCamera(c, cam, p.lat, p.lon); } });
  return h("li", { class: cam.status === "done" ? "clickable" : "", onclick: () => cam.status === "done" && openCamera(c, cam.id, 0) },
    cam.analysis && cam.analysis.keyframe ? h("img", { src: jpg(cam.analysis.keyframe), alt: "" }) : h("span"),
    h("div", {}, h("div", { class: "title", text: cam.name }), h("div", { class: "sub", text: status }), h("div", { class: "sub", text: `${where}${cam.start_time ? " · starts " + fmtTime(cam.start_epoch) : ""}` }),
      cam.environment && cam.environment.texts.length ? h("div", { class: "sub", text: `Signs read: ${cam.environment.texts.slice(0, 5).join(" · ")}` }) : null,
      geoSummary(c, cam),
      h("div", { class: "actions" }, setLoc, sug, cam.status === "done" && !(cam.gps && cam.gps.length) ? h("button", { class: "link", type: "button", text: cam.geolocation ? "Search again" : "Find where this was filmed", onclick: (e) => { e.stopPropagation(); geolocateForm(c, cam); } }) : null), cam.status !== "done" && cam.status !== "error" ? h("div", { class: "progress" }, h("i", { id: `prog-${cam.id}`, style: { width: "4%" } })) : null),
    h("span", { class: "score", text: cam.analysis && cam.analysis.alerts.length ? `${cam.analysis.alerts.length} alert${cam.analysis.alerts.length > 1 ? "s" : ""}` : "" }));
}
async function updateCamera(c, cam, lat, lon) {
  const fd = new FormData(); fd.append("lat", lat); fd.append("lon", lon);
  inv.data = await api(`/api/cases/${c.id}/cameras/${cam.id}`, { method: "PATCH", form: fd }); renderCase(inv.data); drawCase(inv.data);
}
function cameraForm(c) {
  const state = { lat: null, lon: null };
  const where = h("span", { class: "hint", text: "Optional. Set it now or after upload." });
  const name = h("input", { type: "text", placeholder: "e.g. MG Road junction, north side", maxlength: 80 });
  const start = h("input", { type: "datetime-local", step: 1 });
  const dir = h("select", {}, h("option", { value: "", text: "Not set" }), h("option", { value: "1,0", text: "Left to right" }), h("option", { value: "-1,0", text: "Right to left" }),
    h("option", { value: "0,-1", text: "Away from the camera" }), h("option", { value: "0,1", text: "Towards the camera" }));
  const gps = h("input", { type: "file", accept: ".gpx,.csv,.nmea,.txt" });
  const video = h("input", { type: "file", accept: "video/*", required: true });
  const pick = h("button", { class: "button small", type: "button", text: "Pick on map", onclick: async () => { const p = await pickOnMap("Click where the camera is"); state.lat = p.lat; state.lon = p.lon; where.textContent = `${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}`; } });
  const submit = h("button", { class: "button primary", type: "submit", text: "Add footage" });
  const form = h("form", { style: { marginTop: "16px" } },
    h("label", { class: "field" }, "Video", video),
    h("label", { class: "field" }, "Camera name", name),
    h("div", { class: "field" }, "Camera location", h("div", { class: "actions", style: { marginTop: "5px" } }, pick, where)),
    h("div", { class: "split" }, h("label", { class: "field" }, "Recording started", start), h("label", { class: "field" }, "Permitted direction", dir)),
    h("label", { class: "field" }, "GPS track (dashcam, optional)", gps),
    h("div", { class: "actions" }, submit));
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!video.files[0]) { notice("Choose a video first."); return; }
    const fd = new FormData();
    fd.append("file", video.files[0]); if (name.value) fd.append("name", name.value);
    if (state.lat != null) { fd.append("lat", state.lat); fd.append("lon", state.lon); }
    if (start.value) fd.append("start_time", start.value); if (dir.value) fd.append("allowed_direction", dir.value);
    if (gps.files[0]) fd.append("gps", gps.files[0]);
    submit.disabled = true; submit.textContent = "Uploading";
    try {
      const r = await api(`/api/cases/${c.id}/cameras`, { method: "POST", form: fd });
      notice("Footage uploaded. Analysis has started.");
      await openCase(c.id); watchJob(c.id, r.job_id, r.camera_id);
    } catch (err) { notice(err.message); submit.disabled = false; submit.textContent = "Add footage"; }
  });
  return form;
}
function watchJob(cid, jobId, camId) {
  const tick = async () => {
    try {
      const j = await api(`/api/jobs/${jobId}`);
      const bar = document.getElementById(`prog-${camId}`); if (bar) bar.style.width = `${Math.max(4, j.progress * 100)}%`;
      if (j.status === "done" || j.status === "error") { if (inv.id === cid) openCase(cid); if (j.status === "error") notice(`Analysis failed: ${j.error}`); return; }
    } catch { return; }
    setTimeout(tick, 1500);
  };
  tick();
}
function drawCase(c) {
  const pts = [];
  c.cameras.filter((x) => x.lat != null).forEach((x) => pts.push({ type: "Feature", properties: { colour: PIN.cam, r: 6, label: x.name, title: x.name, detail: x.status === "done" ? `${x.analysis.counts.total} vehicles` : "Analysing" }, geometry: { type: "Point", coordinates: [x.lon, x.lat] } }));
  const seen = new Set();
  c.sightings.filter((s) => s.lat != null).forEach((s) => {
    if (seen.has(s.camera_id)) return; seen.add(s.camera_id);
    pts.push({ type: "Feature", properties: { colour: PIN.seen, r: 9, label: "", title: `Seen at ${s.camera_name}`, detail: `${s.t_abs ? fmtTime(s.t_abs) : clock(s.t_video)} · match ${pct(s.score, 0)}` }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } });
  });
  setLayer("points", fc(pts));
  const gps = c.cameras.filter((x) => x.route && x.route.matched).map((x) => ({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: x.route.geometry } }));
  setLayer("gps", fc(gps));
  clearLayers("areas");
  if (pts.length) fit(pts.map((f) => f.geometry.coordinates), 15);
  setKey([["dot", PIN.cam, "Camera"], ["dot", PIN.seen, "Vehicle seen"], ...(gps.length ? [["sw", "#5c4a2c", c.cameras.some((x) => x.route && x.route.estimated) ? "Camera route (GPS or estimated from motion)" : "Dashcam route"]] : [])]);
}
async function runReason(c, btn) {
  const out = $("#route-out");
  btn.disabled = true; out.replaceChildren(busy("Reconstructing possible routes"));
  const fd = new FormData(); fd.append("text", $("#case-notes").value || "");
  try { renderRoutes(await api(`/api/cases/${c.id}/reason`, { method: "POST", form: fd })); } catch (e) { out.replaceChildren(problem(e.message)); }
  btn.disabled = false;
}
function renderRoutes(r) {
  const out = $("#route-out");
  if (!out) return;
  const feats = [], kids = [], keys = [];
  const hints = r.hints || {};
  const understood = [...(hints.places || []).filter((p) => p.label).map((p) => `${p.role}: ${p.label}`), ...(hints.times || []).map((t) => `time: ${t.qualifier} ${t.text.replace(/^(after|before|around|about|by|at|since|till|until)\s+/i, "")}`),
    ...(hints.activities || []).map((a) => `stop: ${a.name}`), ...(hints.direction ? [`heading ${hints.direction.word}`] : [])];
  if (understood.length) kids.push(h("p", { class: "note", text: `Understood from the note: ${understood.join(" · ")}` }));
  (r.segments || []).forEach((seg) => {
    kids.push(h("h2", { style: { marginTop: "18px" }, text: `Between ${seg.from} and ${seg.to}` }));
    seg.routes.forEach((rt, i) => {
      const colour = ROUTE_COLOURS[i % ROUTE_COLOURS.length];
      feats.push({ type: "Feature", properties: { colour, width: i === 0 ? 5 : 3, opacity: 0.35 + 0.6 * rt.probability, dashed: false }, geometry: { type: "LineString", coordinates: rt.geometry } });
      kids.push(hypothesis(rt, colour, `${(rt.distance_m / 1000).toFixed(1)} km · ${timing(rt)}${seg.gap_s ? ` · ${(seg.gap_s / 60).toFixed(0)} min between sightings` : ""}`));
    });
  });
  if ((r.onward || []).length) {
    kids.push(h("h2", { style: { marginTop: "18px" }, text: "After the last sighting" }));
    r.onward.forEach((rt, i) => {
      const colour = ROUTE_COLOURS[(i + 1) % ROUTE_COLOURS.length];
      feats.push({ type: "Feature", properties: { colour, width: i === 0 ? 4 : 2.5, opacity: 0.3 + 0.65 * rt.probability, dashed: true }, geometry: { type: "LineString", coordinates: rt.geometry } });
      const arrive = rt.eta_range ? ` · arrives ${clockTime(rt.eta_range[0])}–${clockTime(rt.eta_range[1])}` : rt.eta_epoch ? " · arrives about " + fmtTime(rt.eta_epoch) : "";
      kids.push(hypothesis(rt, colour, `to ${rt.destination} · ${(rt.distance_m / 1000).toFixed(1)} km · ${timing(rt)}${arrive}`));
    });
  }
  setLayer("routes", fc(feats));
  const iso = (r.reachable && r.reachable.features) || [];
  const shades = ["#d4952a", "#c9a25c", "#e2c98f"];
  setLayer("iso", fc(iso.map((f, i) => ({ ...f, properties: { colour: shades[i % 3], opacity: 0.1 } }))));
  if (iso.length) keys.push(["sw", "#c9a25c", "Reachable in 5, 10, 15 min"]);
  if (!kids.length) kids.push(h("p", { class: "note", text: "Add footage with matched sightings, or describe what is known, to reconstruct a route." }));
  kids.push(h("p", { class: "disclaimer", text: (r.assumptions || []).join(" ") }));
  out.replaceChildren(...kids);
  setKey([["dot", PIN.cam, "Camera"], ["dot", PIN.seen, "Vehicle seen"], ["sw", ROUTE_COLOURS[0], "Route between sightings"], ["sw", ROUTE_COLOURS[1], "Possible onward route"], ...keys]);
  const all = feats.flatMap((f) => f.geometry.coordinates);
  if (all.length) fit(all, 15);
}
const clockTime = (epoch) => new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
function timing(rt) {
  const free = Math.max(1, Math.round((rt.freeflow_s || rt.duration_s) / 60));
  if (!rt.typical_s) return `${free} min free-flow`;
  const [a, , c] = rt.typical_s.map((x) => Math.max(1, Math.round(x / 60)));
  return `typically ${a}–${c} min (free-flow ${free} min)`;
}
function hypothesis(rt, colour, meta) {
  return h("div", { class: "hypothesis", onclick: () => fit(rt.geometry, 16) },
    h("div", { class: "h-head" }, h("span", { class: "h-title" }, h("span", { class: "h-swatch", style: { background: colour } }), (rt.roads || []).slice(0, 3).join(" → ") || "Unnamed roads"),
      h("span", { class: "h-prob", text: pct(rt.probability, 0) })),
    h("div", { class: "sub muted", style: { fontSize: "12px" }, text: meta }),
    (rt.reasons || []).length ? h("ul", { class: "h-reasons" }, rt.reasons.map((x) => h("li", { text: x }))) : null,
    rt.traffic_basis ? h("div", { class: "muted", style: { fontSize: "11.5px", marginTop: "4px" }, text: `Travel time: ${rt.traffic_basis}` }) : null);
}

function overlay(video, canvas, analysis, focusTrack) {
  const ctx = canvas.getContext("2d");
  const [fw, fh] = analysis.meta.frame_size;
  const draw = () => {
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio; canvas.height = rect.height * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const s = Math.min(rect.width / fw, rect.height / fh), ox = (rect.width - fw * s) / 2, oy = (rect.height - fh * s) / 2;
    const t = video.currentTime;
    for (const tr of analysis.tracks) {
      if (t < tr.first_t - 0.2 || t > tr.last_t + 0.2) continue;
      let box = null;
      for (const b of tr.boxes) { if (b[0] <= t + 0.07) box = b; else break; }
      if (!box || t - box[0] > 0.6) continue;
      const [, x1, y1, x2, y2] = box;
      const colour = tr.wrong_way ? "#ff6a4d" : focusTrack === tr.id ? "#e2ad2e" : "#f6efe2";
      ctx.strokeStyle = colour; ctx.lineWidth = focusTrack === tr.id ? 3 : 2;
      ctx.strokeRect(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s);
      const label = `${tr.id} · ${tr.label}${tr.wrong_way ? " · wrong way" : ""}`;
      ctx.font = "600 12px system-ui, sans-serif";
      const w = ctx.measureText(label).width + 10;
      ctx.fillStyle = "rgba(36,28,19,0.8)"; ctx.fillRect(ox + x1 * s, Math.max(0, oy + y1 * s - 20), w, 18);
      ctx.fillStyle = colour; ctx.fillText(label, ox + x1 * s + 5, Math.max(13, oy + y1 * s - 7));
    }
    if (!video.isConnected) return;
    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}
function vehicleTable(analysis, video) {
  return h("table", { class: "table" },
    h("thead", {}, h("tr", {}, h("th", { text: "#" }), h("th", { text: "Vehicle" }), h("th", { text: "Moving" }), h("th", { class: "num", text: "Seen" }))),
    h("tbody", {}, analysis.tracks.map((tr) => h("tr", { class: "clickable", onclick: () => { video.currentTime = tr.first_t; video.play(); } },
      h("td", { class: "mono", text: tr.id }),
      h("td", {}, h("div", { text: tr.label }), tr.decision ? h("div", { class: "muted", style: { fontSize: "12px" }, text: `${(LEVEL[tr.decision.level] || [""])[0]} · ${pct(tr.decision.confidence, 0)}` }) : h("div", { class: "muted", style: { fontSize: "12px" }, text: `${tr.class} · too small in the frame to name the make or model` })),
      h("td", {}, tr.wrong_way ? h("span", { class: "tag alert", text: "Wrong way" }) : h("span", { text: tr.direction.compass })),
      h("td", { class: "num", text: `${clock(tr.first_t)}–${clock(tr.last_t)}` })))));
}
function analysisView(analysis, src, startAt = 0) {
  const video = h("video", { src, controls: true, playsinline: true, preload: "metadata" });
  const canvas = h("canvas");
  video.addEventListener("loadedmetadata", () => { if (startAt) video.currentTime = startAt; });
  overlay(video, canvas, analysis);
  const c = analysis.counts;
  const stats = h("div", { class: "stats" }, Object.entries(c.by_class).map(([k, n]) => h("div", { class: "stat" }, h("div", { class: "n", text: n }), h("div", { class: "l", text: `${k}${n === 1 ? "" : "s"}` }))),
    h("div", { class: "stat" }, h("div", { class: "n", text: analysis.alerts.length }), h("div", { class: "l", text: "wrong-way alerts" })));
  return h("div", {}, h("div", { class: "player" }, video, canvas), section("Summary", stats), section("Vehicles", vehicleTable(analysis, video)));
}
function openCamera(c, camId, t) {
  const cam = c.cameras.find((x) => x.id === camId);
  if (!cam || !cam.analysis) return;
  $("#sheet-title").textContent = cam.name;
  $("#sheet-body").replaceChildren(analysisView(cam.analysis, `${API}/api/cases/${c.id}/cameras/${cam.id}/video`, t));
  $("#sheet").showModal();
}
$("#sheet-close").addEventListener("click", () => { $("#sheet").close(); $("#sheet-body").replaceChildren(); });
$("#sheet").addEventListener("close", () => $("#sheet-body").replaceChildren());

async function runVideo(file) {
  const out = $("#video-out");
  const url = URL.createObjectURL(file);
  out.replaceChildren(busy("Analysing the video. Longer clips take a few minutes."));
  const fd = new FormData(); fd.append("file", file);
  const dir = $("#video-direction").value; if (dir) fd.append("allowed_direction", dir);
  try {
    const r = await api("/api/video", { method: "POST", form: fd });
    out.replaceChildren(h("div", { class: "section" }, analysisView(r, url)));
  } catch (e) { out.replaceChildren(problem(e.message)); }
}

const KIND = { identify: "Identification", locate: "Photo located", video: "Video analysed", case_created: "Case opened", case_updated: "Case updated",
  case_deleted: "Case deleted", hunt_target: "Hunt target", camera_added: "Footage added", camera_analysed: "Footage analysed", hunt_sighting: "Possible sighting",
  route_inference: "Route reasoning", admin_event_deleted: "Record deleted" };
const when = (ts) => {
  const d = new Date(ts * 1000), now = new Date();
  const same = d.toDateString() === now.toDateString();
  return same ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : d.toLocaleDateString([], { day: "numeric", month: "short" });
};
async function loadActivity() {
  const out = $("#activity-out");
  out.replaceChildren(busy("Loading recent activity"));
  try {
    const r = await api("/api/activity?limit=40");
    if (!r.events.length) { out.replaceChildren(h("p", { class: "note", text: "Nothing recorded yet. Identifications, footage and cases from your station appear here." })); return; }
    out.replaceChildren(h("p", { class: "muted", style: { fontSize: "12px" }, text: `${r.total} record${r.total === 1 ? "" : "s"} in total` }),
      h("ul", { class: "feed" }, r.events.map((e) => h("li", {}, h("span", { class: "when", text: when(e.ts) }),
        h("span", {}, h("span", { class: "kind", text: KIND[e.kind] || e.kind }), h("div", { class: "what", text: e.summary }), h("div", { class: "muted", style: { fontSize: "12px" }, text: e.username }))))));
  } catch (e) { if (!(e instanceof AuthError)) out.replaceChildren(problem(e.message)); }
}

const consoleState = { mod: null };
async function showConsole(on) {
  $("#layout").hidden = on; $("#console").hidden = !on;
  if (!on) { map.m && map.m.resize(); return; }
  if (!consoleState.mod) consoleState.mod = await import("./admin.js?v=12");
  consoleState.mod.mount($("#console"), { api, h, busy, problem, notice, session, AuthError, KIND, LEVEL, pct, jpg, when });
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  const view = b.dataset.view;
  showConsole(view === "division").catch((e) => notice(e.message, 4000, true));
  if (view === "division") return;
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  clearLayers("points", "routes", "iso", "areas", "gps", "heat"); setKey([]);
  if (view === "investigate") loadCases(inv.id).catch((e) => !(e instanceof AuthError) && notice(e.message));
  if (view === "activity") loadActivity();
}));

function drawRegion() {
  const r = session.region;
  if (!r) return;
  if (r.boundary) setLayer("boundary", r.boundary);
  const ownId = session.user.station && session.user.station.id;
  setLayer("stations", fc(r.stations.filter((s) => s.lat != null).map((s) => ({ type: "Feature",
    properties: { own: s.id === ownId, name: s.name, short: s.name.replace(/ (Traffic )?Police Station$/i, ""),
      detail: `${({ law_and_order: "Law and order", traffic: "Traffic", cen: "CEN", women: "Women" })[s.kind] || "Police"} station${s.official_division ? " · " + s.official_division + " division" : ""}` },
    geometry: { type: "Point", coordinates: [s.lon, s.lat] } }))));
}

async function boot() {
  const entry = $("#entry");
  const me = await fetch(API + "/auth/me", { credentials: "same-origin" });
  if (!me.ok) { location.replace("/login"); return; }
  const info = await me.json();
  if (info.must_change) { location.replace("/login?change=1"); return; }
  session.user = info.user;
  const u = session.user;
  const isAdmin = u.role === "division_admin";
  $("#station-name").textContent = u.station ? u.station.name : u.display_name;
  $("#station-meta").replaceChildren(document.createTextNode(`${u.division.name} Division`), h("span", { class: "kn", text: ({ central: "ಕೇಂದ್ರ", north: "ಉತ್ತರ", south: "ದಕ್ಷಿಣ", east: "ಪೂರ್ವ", west: "ಪಶ್ಚಿಮ" })[u.division.id] || "" }),
    ...(isAdmin ? [h("span", { class: "role", text: "Division admin" })] : []));
  $("#who-name").textContent = u.station ? u.station.name : u.display_name;
  $("#who-id").textContent = `${u.username} · ${u.division.name} Division`;
  $("#tab-division").hidden = !isAdmin;
  $("#entry-name").textContent = u.station ? u.station.name : u.display_name;
  $("#entry-sub").textContent = `${u.division.name} Division` + (isAdmin ? " · Division administrator" : "");
  try { session.region = await api("/api/region"); } catch (e) { if (e instanceof AuthError) return; }
  health(); setInterval(health, 30000);
  drawRegion();
  initMap().catch((e) => notice(`Map failed to load: ${e.message}`, 5000, true));
  setTimeout(() => entry.classList.add("open"), 250);
  setTimeout(() => entry.classList.add("leaving"), 1250);
  setTimeout(() => { entry.hidden = true; entry.classList.remove("leaving"); }, 1650);
}
uploader($("#up-identify"), runIdentify);
uploader($("#up-video"), runVideo);
boot().catch((e) => { notice(e.message, 6000, true); $("#entry").hidden = true; });
