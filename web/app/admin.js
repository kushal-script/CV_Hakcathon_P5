const NS = "http://www.w3.org/2000/svg";
const KN_DIV = { central: "ಕೇಂದ್ರ ವಿಭಾಗ", north: "ಉತ್ತರ ವಿಭಾಗ", south: "ದಕ್ಷಿಣ ವಿಭಾಗ", east: "ಪೂರ್ವ ವಿಭಾಗ", west: "ಪಶ್ಚಿಮ ವಿಭಾಗ" };
const KINDS = { law_and_order: "Law and order", traffic: "Traffic", cen: "CEN", women: "Women", other: "Other" };
const LEVEL_NAMES = { class: "Model identified", family: "Model family", make: "Make only", abstain: "Not certain", ood_make: "Make, model outside catalogues", ood: "Outside catalogues" };
const SERIES = [
  { key: "identify", label: "Identifications", cls: "s1", colour: "var(--s1)" },
  { key: "video", label: "Footage analysed", cls: "s2", colour: "var(--s2)" },
  { key: "cases", label: "Case work", cls: "s3", colour: "var(--s3)" },
  { key: "other", label: "Other", cls: "s4", colour: "var(--s4)" },
];
let C = null;
const st = { tab: "overview", days: 30, root: null, body: null, stations: [], ev: { offset: 0, kind: "", station: "", q: "", days: 30 },
  cases: { status: "open", station: "" }, audit: [], resize: null };

const nf = new Intl.NumberFormat("en-IN");
const fmt = (n) => nf.format(Math.round(n || 0));
function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, v);
  for (const kid of kids.flat()) if (kid) el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return el;
}
const colPath = (x, y, w, hh, r = 4) => { r = Math.max(0, Math.min(r, w / 2, hh)); return `M${x},${y + hh}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hh}Z`; };
const barPath = (x, y, w, hh, r = 4) => { r = Math.max(0, Math.min(r, hh / 2, w)); return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hh - r}Q${x + w},${y + hh} ${x + w - r},${y + hh}H${x}Z`; };
function ticks(max, n = 4) {
  if (max <= 0) return [0, 1];
  const raw = max / n, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= raw) || 10 * mag;
  const out = [];
  for (let v = 0; v <= max + step * 0.001; v += step) out.push(v);
  if (out[out.length - 1] < max) out.push(out[out.length - 1] + step);
  return out;
}
const dayLabel = (d) => new Date(d + "T00:00:00").toLocaleDateString("en-IN", { day: "numeric", month: "short" });
const dateTime = (ts) => new Date(ts * 1000).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

function tipper(wrap) {
  const tip = C.h("div", { class: "tip", hidden: true });
  wrap.append(tip);
  return {
    show(x, y, title, rows) {
      tip.replaceChildren(C.h("div", { class: "t", text: title }), ...rows.map(([label, value, colour]) => C.h("div", { class: "r" },
        C.h("span", {}, colour ? C.h("i", { class: "key", style: { background: colour } }) : null, label), C.h("b", { text: value }))));
      tip.hidden = false;
      const w = wrap.clientWidth;
      tip.style.left = `${Math.max(70, Math.min(w - 70, x))}px`; tip.style.top = `${Math.max(40, y)}px`;
    },
    hide() { tip.hidden = true; },
  };
}

function card(title, sub, { wide = false, legend = null, draw, table }) {
  const body = C.h("div", { class: "chart" });
  const toggle = C.h("button", { class: "link", type: "button", text: "Table" });
  const el = C.h("section", { class: `card${wide ? " wide" : ""}` },
    C.h("div", { class: "card-head" }, C.h("div", {}, C.h("h3", { text: title }), sub ? C.h("p", { text: sub }) : null), table ? toggle : null),
    legend, body);
  let asTable = false;
  const render = () => { body.replaceChildren(); if (asTable) body.append(table()); else draw(body); };
  toggle.addEventListener("click", () => { asTable = !asTable; toggle.textContent = asTable ? "Chart" : "Table"; render(); });
  el.render = render;
  return el;
}
function simpleTable(headers, rows) {
  return C.h("div", { class: "tablewrap", style: { maxHeight: "320px" } }, C.h("table", { class: "dtable" },
    C.h("thead", {}, C.h("tr", {}, headers.map((x, i) => C.h("th", { class: i ? "num" : "", text: x })))),
    C.h("tbody", {}, rows.map((r) => C.h("tr", {}, r.map((v, i) => C.h("td", { class: i ? "num" : "", text: v })))))));
}
const empty = (text = "Nothing recorded in this period yet.") => C.h("div", { class: "empty", text });

function stackedColumns(box, days) {
  const W = Math.max(320, box.clientWidth || 640), H = 230, ml = 40, mr = 8, mt = 10, mb = 28;
  const totals = days.map((d) => SERIES.reduce((a, x) => a + d[x.key], 0));
  if (!totals.some((t) => t > 0)) return box.append(empty());
  const tk = ticks(Math.max(...totals)), top = tk[tk.length - 1];
  const iw = W - ml - mr, ih = H - mt - mb, band = iw / days.length, bw = Math.max(2, Math.min(24, band * 0.64));
  const y = (v) => mt + ih - (v / top) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Activity per day, stacked by type" });
  tk.forEach((v) => { svg.append(s("line", { class: "gridline", x1: ml, x2: W - mr, y1: y(v), y2: y(v) }), s("text", { x: ml - 8, y: y(v) + 4, "text-anchor": "end" }, fmt(v))); });
  svg.append(s("line", { class: "baseline", x1: ml, x2: W - mr, y1: y(0), y2: y(0) }));
  const every = Math.max(1, Math.ceil(days.length / 8));
  const tip = tipper(box);
  days.forEach((d, i) => {
    const cx = ml + band * i + band / 2, x = cx - bw / 2;
    let acc = 0;
    const parts = SERIES.filter((x2) => d[x2.key] > 0);
    parts.forEach((ser, j) => {
      const v = d[ser.key], y0 = y(acc), y1 = y(acc + v);
      const gap = j > 0 ? 2 : 0, hh = Math.max(0, y0 - y1 - gap);
      if (hh > 0) svg.append(s("path", { class: `mark ${ser.cls}`, d: j === parts.length - 1 ? colPath(x, y1, bw, hh) : `M${x},${y1}h${bw}v${hh}h${-bw}Z` }));
      acc += v;
    });
    if (i % every === 0) svg.append(s("text", { x: cx, y: H - 8, "text-anchor": "middle" }, dayLabel(d.day)));
    const hit = s("rect", { class: "hit", x: ml + band * i, y: mt, width: band, height: ih, tabindex: 0 });
    const show = () => tip.show(cx, y(totals[i]) - 4, `${dayLabel(d.day)} · ${fmt(totals[i])} total`, SERIES.map((x2) => [x2.label, fmt(d[x2.key]), x2.colour]));
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", tip.hide); hit.addEventListener("blur", tip.hide);
    svg.append(hit);
  });
  box.prepend(svg);
}

function hbars(box, rows, { label, value, sub, cls = "mag", fmtVal = fmt, max = null }) {
  if (!rows.length) return box.append(empty());
  const W = Math.max(300, box.clientWidth || 520), rowH = 30, lw = Math.min(210, W * 0.42), H = rows.length * rowH + 4, vmax = max ?? Math.max(...rows.map(value), 1);
  const iw = W - lw - 56;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  const tip = tipper(box);
  rows.forEach((r, i) => {
    const y = i * rowH + 6, v = value(r), w = Math.max(v > 0 ? 3 : 0, (v / vmax) * iw);
    const name = label(r);
    const t = s("text", { class: "lbl", x: 0, y: y + 12 }, name.length > 30 ? name.slice(0, 29) + "…" : name);
    svg.append(t);
    if (sub) svg.append(s("text", { class: "sub", x: 0, y: y + 24 }, sub(r)));
    if (w > 0) svg.append(s("path", { class: `mark ${cls}`, d: barPath(lw, y + 2, w, 14) }));
    svg.append(s("text", { class: "val", x: lw + w + 8, y: y + 13 }, fmtVal(v)));
    const hit = s("rect", { class: "hit", x: 0, y: y - 4, width: W, height: rowH, tabindex: 0 });
    const show = () => tip.show(lw + w / 2, y, name, [[sub ? sub(r) : "Count", fmtVal(v)]]);
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", tip.hide); hit.addEventListener("blur", tip.hide);
    svg.append(hit);
  });
  box.prepend(svg);
}

function histogram(box, bins) {
  const total = bins.reduce((a, b) => a + b.n, 0);
  if (!total) return box.append(empty());
  const W = Math.max(300, box.clientWidth || 520), H = 200, ml = 36, mr = 6, mt = 18, mb = 26;
  const tk = ticks(Math.max(...bins.map((b) => b.n))), top = tk[tk.length - 1];
  const iw = W - ml - mr, ih = H - mt - mb, band = iw / bins.length, bw = Math.min(24, band * 0.7);
  const y = (v) => mt + ih - (v / top) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Confidence of each answer" });
  tk.forEach((v) => svg.append(s("line", { class: "gridline", x1: ml, x2: W - mr, y1: y(v), y2: y(v) }), s("text", { x: ml - 8, y: y(v) + 4, "text-anchor": "end" }, fmt(v))));
  svg.append(s("line", { class: "baseline", x1: ml, x2: W - mr, y1: y(0), y2: y(0) }));
  const peak = bins.reduce((a, b, i) => (b.n > bins[a].n ? i : a), 0);
  const tip = tipper(box);
  bins.forEach((b, i) => {
    const cx = ml + band * i + band / 2;
    if (b.n > 0) svg.append(s("path", { class: "mark mag", d: colPath(cx - bw / 2, y(b.n), bw, y(0) - y(b.n)) }));
    if (i === peak) svg.append(s("text", { class: "val", x: cx, y: y(b.n) - 6, "text-anchor": "middle" }, fmt(b.n)));
    if (i % 2 === 0) svg.append(s("text", { x: ml + band * i, y: H - 8, "text-anchor": "middle" }, `${Math.round(b.from * 100)}%`));
    const hit = s("rect", { class: "hit", x: ml + band * i, y: mt, width: band, height: ih, tabindex: 0 });
    const show = () => tip.show(cx, y(b.n) - 4, `Confidence ${Math.round(b.from * 100)}–${Math.round(b.to * 100)}%`, [["Answers", fmt(b.n)], ["Share", `${((100 * b.n) / total).toFixed(1)}%`]]);
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", tip.hide); hit.addEventListener("blur", tip.hide);
    svg.append(hit);
  });
  svg.append(s("text", { x: W - mr, y: H - 8, "text-anchor": "end" }, "100%"));
  box.prepend(svg);
}

function split(box, parts) {
  const total = parts.reduce((a, p) => a + p.n, 0);
  if (!total) return box.append(empty());
  const W = Math.max(300, box.clientWidth || 520), H = 34;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  let x = 0;
  const tip = tipper(box);
  parts.forEach((p, i) => {
    const w = (p.n / total) * W - (i < parts.length - 1 ? 2 : 0);
    if (w > 0) {
      const path = s("path", { class: `mark ${p.cls}`, d: i === parts.length - 1 ? barPath(x, 6, w, 22) : `M${x},6h${w}v22h${-w}Z` });
      const show = () => tip.show(x + w / 2, 4, p.label, [["Vehicles", fmt(p.n)], ["Share", `${((100 * p.n) / total).toFixed(0)}%`]]);
      path.addEventListener("pointermove", show); path.addEventListener("pointerleave", tip.hide);
      svg.append(path);
    }
    x += (p.n / total) * W;
  });
  box.prepend(svg);
}

function cityMap(box, region, stations) {
  const feats = region && region.boundary ? (region.boundary.features || [region.boundary]) : [];
  const polys = [];
  feats.forEach((f) => { const g = f.geometry || f; if (g.type === "Polygon") polys.push(g.coordinates); if (g.type === "MultiPolygon") polys.push(...g.coordinates); });
  const pts = stations.filter((x) => x.lat != null);
  const all = polys.flatMap((p) => p[0]).concat(pts.map((p) => [p.lon, p.lat]));
  if (!all.length) return box.append(empty("No boundary or station coordinates available."));
  const lons = all.map((p) => p[0]), lats = all.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...lons), Math.max(...lons), Math.min(...lats), Math.max(...lats)];
  const k = Math.cos(((y0 + y1) / 2) * Math.PI / 180);
  const W = Math.max(300, box.clientWidth || 520), pad = 14, scale = Math.min((W - 2 * pad) / ((x1 - x0) * k), 420 / (y1 - y0));
  const H = (y1 - y0) * scale + 2 * pad;
  const px = (lon) => pad + (lon - x0) * k * scale + ((W - 2 * pad) - (x1 - x0) * k * scale) / 2, py = (lat) => pad + (y1 - lat) * scale;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Stations of the division on the outline of Bengaluru" });
  polys.forEach((poly) => svg.append(s("path", { class: "boundary", d: poly.map((ring) => "M" + ring.map(([lon, lat]) => `${px(lon).toFixed(1)},${py(lat).toFixed(1)}`).join("L") + "Z").join(" "), "fill-rule": "evenodd" })));
  const vmax = Math.max(1, ...pts.map((p) => p.events));
  const tip = tipper(box);
  [...pts].sort((a, b) => b.events - a.events).forEach((p) => {
    const cx = px(p.lon), cy = py(p.lat);
    const r = p.events ? 4 + Math.sqrt(p.events / vmax) * 15 : 3;
    svg.append(s("circle", { class: p.events ? "mark bubble" : "quiet", cx, cy, r }));
    const hit = s("circle", { class: "hit", cx, cy, r: Math.max(12, r), tabindex: 0 });
    const show = () => tip.show(cx, cy - r, p.name, [["Records", fmt(p.events)], ["Identifications", fmt(p.identifications)], ["Vehicles", fmt(p.vehicles)]]);
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", tip.hide); hit.addEventListener("blur", tip.hide);
    svg.append(hit);
  });
  box.prepend(svg);
}

function legend(items) {
  return C.h("div", { class: "legend" }, items.map((x) => C.h("span", {}, C.h("i", { style: { background: x.colour } }), x.label)));
}

async function overview() {
  const body = st.body;
  body.replaceChildren(filters({ range: true, exportLink: true }), C.busy("Gathering the division's records"));
  let d;
  try { d = await C.api(`/api/admin/overview?days=${st.days}`); } catch (e) { body.lastChild.replaceWith(C.problem(e.message)); return; }
  st.stations = d.by_station;
  const k = d.kpis;
  const kpi = (label, value, hint, of) => C.h("div", { class: "kpi" }, C.h("div", { class: "label", text: label }),
    C.h("div", { class: "value" }, fmt(value), of != null ? C.h("span", { class: "of", text: `of ${fmt(of)}` }) : null), hint ? C.h("div", { class: "hint", text: hint }) : null);
  const kpis = C.h("div", { class: "kpis" },
    kpi("Identifications", k.identifications, "photos analysed"),
    kpi("Vehicles analysed", k.vehicles, "in photos and footage"),
    kpi("Open cases", k.cases_open, `${fmt(k.cases_total)} cases in all`),
    kpi("Possible sightings", k.sightings, "across all cases"),
    kpi("Wrong-way alerts", k.wrong_way, "from footage"),
    kpi("Active stations", k.active_stations, "with records", k.stations));
  const levelRows = d.levels.map((x) => ({ ...x, name: LEVEL_NAMES[x.level] || x.level }));
  const cat = { india: "Indian market", us: "US market" };
  const cards = [
    card("Activity per day", `${d.range.days} days, every station in the division`, { wide: true, legend: legend(SERIES),
      draw: (b) => stackedColumns(b, d.daily), table: () => simpleTable(["Day", ...SERIES.map((x) => x.label)], d.daily.map((r) => [dayLabel(r.day), ...SERIES.map((x) => fmt(r[x.key]))])) }),
    card("Stations at a glance", "Circle area shows the number of records", { draw: (b) => cityMap(b, C.session.region, d.by_station),
      table: () => simpleTable(["Station", "Records", "Identifications", "Vehicles"], d.by_station.map((r) => [r.name, fmt(r.events), fmt(r.identifications), fmt(r.vehicles)])) }),
    card("Busiest stations", "Records in this period", { draw: (b) => hbars(b, d.by_station.filter((r) => r.events).slice(0, 12), { label: (r) => r.name.replace(/ Police Station$/i, ""), value: (r) => r.events, sub: (r) => KINDS[r.kind] || "Station" }),
      table: () => simpleTable(["Station", "Records"], d.by_station.map((r) => [r.name, fmt(r.events)])) }),
    card("Most identified models", "Answers given at model level", { draw: (b) => hbars(b, d.top_models, { label: (r) => r.label, value: (r) => r.n, sub: (r) => `${cat[r.catalogue] || "Catalogue"} · mean confidence ${C.pct(r.mean_confidence, 0)}` }),
      table: () => simpleTable(["Model", "Catalogue", "Count", "Mean confidence"], d.top_models.map((r) => [r.label, cat[r.catalogue] || "", fmt(r.n), C.pct(r.mean_confidence, 0)])) }),
    card("Makes seen", "Make of every vehicle answer, top ten", { draw: (b) => hbars(b, d.top_makes, { label: (r) => r.make, value: (r) => r.n }),
      table: () => simpleTable(["Make", "Count"], d.top_makes.map((r) => [r.make, fmt(r.n)])) }),
    card("How sure the system was", "Calibrated confidence of each answer", { draw: (b) => histogram(b, d.confidence_hist),
      table: () => simpleTable(["Confidence", "Answers"], d.confidence_hist.map((r) => [`${Math.round(r.from * 100)}–${Math.round(r.to * 100)}%`, fmt(r.n)])) }),
    card("Answer levels", "How specific each answer could safely be", { draw: (b) => hbars(b, levelRows, { label: (r) => r.name, value: (r) => r.n }),
      table: () => simpleTable(["Level", "Count"], levelRows.map((r) => [r.name, fmt(r.n)])) }),
    card("Catalogue used", "Which catalogue gave the shown answer", { legend: legend([{ label: "Indian market", colour: "var(--s1)" }, { label: "US market", colour: "var(--s4)" }]),
      draw: (b) => split(b, [{ label: "Indian market", cls: "s1", n: (d.catalogues.find((c) => c.catalogue === "india") || {}).n || 0 }, { label: "US market", cls: "s4", n: (d.catalogues.find((c) => c.catalogue === "us") || {}).n || 0 }]),
      table: () => simpleTable(["Catalogue", "Answers"], d.catalogues.map((r) => [cat[r.catalogue] || r.catalogue, fmt(r.n)])) }),
    card("Vehicles in footage", "Tracked vehicles by type", { draw: (b) => hbars(b, d.vehicle_classes, { label: (r) => r.vehicle_class, value: (r) => r.n }),
      table: () => simpleTable(["Type", "Count"], d.vehicle_classes.map((r) => [r.vehicle_class, fmt(r.n)])) }),
  ];
  body.lastChild.replaceWith(C.h("div", {}, kpis, C.h("div", { class: "grid" }, cards)));
  cards.forEach((c) => c.render());
  st.resize = () => cards.forEach((c) => c.render());
}

function filters({ range = false, exportLink = false, extra = [] }) {
  const row = C.h("div", { class: "filters" });
  if (range) {
    const seg = C.h("div", { class: "seg", role: "group", "aria-label": "Period" }, [[7, "7 days"], [30, "30 days"], [90, "90 days"], [365, "1 year"]].map(([d, label]) =>
      C.h("button", { type: "button", "aria-pressed": String(st.days === d), text: label, onclick: () => { st.days = d; st.ev.days = d; st.ev.offset = 0; show(st.tab); } })));
    row.append(seg);
  }
  extra.forEach((x) => row.append(x));
  if (exportLink) row.append(C.h("span", { class: "spacer" }), C.h("a", { class: "button small", href: `/api/admin/export.csv?days=${st.days}`, text: "Export CSV" }));
  return row;
}

function stationSelect(value, onchange, all = "All stations") {
  const sel = C.h("select", { "aria-label": "Station" }, C.h("option", { value: "", text: all }),
    ...st.stations.map((x) => C.h("option", { value: x.station_id, text: x.name })));
  sel.value = value || "";
  if (onchange) sel.addEventListener("change", () => onchange(sel.value));
  return sel;
}

async function ensureStations() {
  if (st.stations.length) return;
  const r = await C.api("/api/admin/stations?days=30");
  st.stations = r.stations;
}

async function events() {
  const body = st.body;
  await ensureStations().catch(() => {});
  const kinds = C.h("select", { "aria-label": "Type" }, C.h("option", { value: "", text: "All types" }), ...Object.entries(C.KIND).map(([k, v]) => C.h("option", { value: k, text: v })));
  kinds.value = st.ev.kind;
  kinds.addEventListener("change", () => { st.ev.kind = kinds.value; st.ev.offset = 0; loadEvents(); });
  const q = C.h("input", { type: "search", placeholder: "Search summaries and notes", value: st.ev.q, maxlength: 100 });
  let t = null;
  q.addEventListener("input", () => { clearTimeout(t); t = setTimeout(() => { st.ev.q = q.value.trim(); st.ev.offset = 0; loadEvents(); }, 300); });
  body.replaceChildren(filters({ range: true, exportLink: true, extra: [kinds, stationSelect(st.ev.station, (v) => { st.ev.station = v; st.ev.offset = 0; loadEvents(); }), q] }),
    C.h("div", { id: "ev-out" }));
  loadEvents();
}

async function loadEvents() {
  const out = document.getElementById("ev-out");
  if (!out) return;
  out.style.opacity = "0.6";
  const p = new URLSearchParams({ days: String(st.ev.days), limit: "50", offset: String(st.ev.offset) });
  if (st.ev.kind) p.set("kind", st.ev.kind);
  if (st.ev.station) p.set("station", st.ev.station);
  if (st.ev.q) p.set("q", st.ev.q);
  try {
    const r = await C.api(`/api/admin/events?${p}`);
    out.style.opacity = "";
    if (!r.events.length) { out.replaceChildren(empty("No records match these filters.")); return; }
    const table = C.h("table", { class: "dtable" },
      C.h("thead", {}, C.h("tr", {}, ["When", "Type", "Station", "Officer ID", "Summary", ""].map((x) => C.h("th", { text: x })))),
      C.h("tbody", {}, r.events.map((e) => C.h("tr", { class: "clickable", onclick: () => openEvent(e.id) },
        C.h("td", { class: "nowrap", text: dateTime(e.ts) }), C.h("td", { class: "nowrap", text: C.KIND[e.kind] || e.kind }),
        C.h("td", { text: e.station_name || "Division" }), C.h("td", { class: "nowrap mono", text: e.username }), C.h("td", { text: e.summary }),
        C.h("td", { class: "nowrap" }, e.reviewed ? C.h("span", { class: "pill ok", text: "Reviewed" }) : null, e.note ? C.h("span", { class: "pill", text: "Note" }) : null)))));
    const pager = C.h("div", { class: "pager" }, C.h("span", { text: `${fmt(st.ev.offset + 1)}–${fmt(st.ev.offset + r.events.length)} of ${fmt(r.total)}` }),
      C.h("button", { class: "button small", type: "button", text: "Newer", disabled: st.ev.offset === 0 ? true : null, onclick: () => { st.ev.offset = Math.max(0, st.ev.offset - 50); loadEvents(); } }),
      C.h("button", { class: "button small", type: "button", text: "Older", disabled: st.ev.offset + 50 >= r.total ? true : null, onclick: () => { st.ev.offset += 50; loadEvents(); } }));
    out.replaceChildren(C.h("div", { class: "tablewrap" }, table), pager);
  } catch (e) { out.style.opacity = ""; out.replaceChildren(C.problem(e.message)); }
}

async function openEvent(id) {
  const sheet = document.getElementById("sheet"), bodyEl = document.getElementById("sheet-body");
  document.getElementById("sheet-title").textContent = "Record";
  bodyEl.replaceChildren(C.busy("Loading the record"));
  sheet.showModal();
  let e;
  try { e = await C.api(`/api/admin/events/${id}`); } catch (x) { bodyEl.replaceChildren(C.problem(x.message)); return; }
  document.getElementById("sheet-title").textContent = `${C.KIND[e.kind] || e.kind} · ${dateTime(e.ts)}`;
  const note = C.h("textarea", { maxlength: 1000, placeholder: "Add a note for the division record" });
  note.value = e.note || "";
  const reviewed = C.h("input", { type: "checkbox" }); reviewed.checked = !!e.reviewed;
  const save = C.h("button", { class: "button primary", type: "button", text: "Save", onclick: async () => {
    try { await C.api(`/api/admin/events/${id}`, { method: "PATCH", json: { note: note.value, reviewed: reviewed.checked } }); C.notice("Record updated."); loadEvents(); } catch (x) { C.notice(x.message, 4000, true); }
  } });
  const del = C.h("button", { class: "button danger", type: "button", text: "Delete record", onclick: async () => {
    if (!confirm("Delete this record? It is hidden from every view and the deletion is logged.")) return;
    try { await C.api(`/api/admin/events/${id}`, { method: "DELETE" }); sheet.close(); C.notice("Record deleted."); loadEvents(); } catch (x) { C.notice(x.message, 4000, true); }
  } });
  const facts = C.h("dl", { class: "facts" },
    C.h("dt", { text: "Station" }), C.h("dd", { text: e.station_name || "Division" }),
    C.h("dt", { text: "Officer ID" }), C.h("dd", { text: e.username }),
    e.case_id ? [C.h("dt", { text: "Case" }), C.h("dd", { class: "mono", text: e.case_id })] : null,
    e.lat != null ? [C.h("dt", { text: "Location" }), C.h("dd", { text: `${e.lat.toFixed(5)}, ${e.lon.toFixed(5)}` })] : null,
    C.h("dt", { text: "Summary" }), C.h("dd", { text: e.summary }));
  const dets = (e.detections || []).length ? simpleTable(["Vehicle", "Level", "Confidence"], e.detections.map((d) => [d.label || d.vehicle_class || "Vehicle", LEVEL_NAMES[d.level] || d.level || "", d.confidence != null ? C.pct(d.confidence, 0) : ""])) : null;
  bodyEl.replaceChildren(C.h("div", { class: "detail" },
    e.thumb ? C.h("img", { src: C.jpg(e.thumb), alt: "Image recorded with this entry" }) : C.h("div", { class: "empty", text: "No image stored" }),
    C.h("div", {}, facts, dets ? C.h("div", { style: { marginTop: "14px" } }, dets) : null,
      C.h("label", { class: "field", style: { marginTop: "16px" } }, "Note", note),
      C.h("label", { class: "toggle" }, reviewed, C.h("span", { text: "Reviewed by the division" })),
      C.h("div", { class: "actions", style: { marginTop: "16px" } }, save, del))));
}

async function cases() {
  const body = st.body;
  await ensureStations().catch(() => {});
  const seg = C.h("div", { class: "seg", role: "group", "aria-label": "Status" }, [["open", "Open"], ["closed", "Closed"], ["", "All"]].map(([v, label]) =>
    C.h("button", { type: "button", "aria-pressed": String(st.cases.status === v), text: label, onclick: () => { st.cases.status = v; cases(); } })));
  const name = C.h("input", { type: "text", placeholder: "New case name", maxlength: 120 });
  const station = stationSelect("", null, "Assign to station");
  const create = C.h("button", { class: "button primary small", type: "button", text: "Open case", onclick: async () => {
    if (!name.value.trim() || !station.value) { C.notice("Give the case a name and choose a station.", 3500, true); return; }
    try { await C.api("/api/admin/cases", { method: "POST", json: { name: name.value.trim(), station_id: station.value } }); C.notice("Case opened."); cases(); } catch (x) { C.notice(x.message, 4000, true); }
  } });
  body.replaceChildren(C.h("div", { class: "filters" }, seg, stationSelect(st.cases.station, (v) => { st.cases.station = v; cases(); }), C.h("span", { class: "spacer" }), name, station, create),
    C.h("div", { id: "cases-out" }, C.busy("Loading cases")));
  const p = new URLSearchParams();
  if (st.cases.status) p.set("status", st.cases.status);
  if (st.cases.station) p.set("station", st.cases.station);
  try {
    const list = await C.api(`/api/admin/cases?${p}`);
    const out = document.getElementById("cases-out");
    if (!list.length) { out.replaceChildren(empty("No cases with these filters.")); return; }
    out.replaceChildren(C.h("div", { class: "tablewrap" }, C.h("table", { class: "dtable" },
      C.h("thead", {}, C.h("tr", {}, ["Case", "Station", "Status", "Vehicle of interest", "Footage", "Sightings", "Updated", ""].map((x, i) => C.h("th", { class: i === 4 || i === 5 ? "num" : "", text: x })))),
      C.h("tbody", {}, list.map((c) => {
        const reassign = stationSelect(c.station_id, async (v) => {
          if (!v || v === c.station_id) return;
          try { await C.api(`/api/cases/${c.id}`, { method: "PATCH", json: { station_id: v } }); C.notice("Case reassigned."); cases(); } catch (x) { C.notice(x.message, 4000, true); }
        }, "Reassign");
        return C.h("tr", {},
          C.h("td", {}, C.h("strong", { text: c.name })), C.h("td", {}, reassign),
          C.h("td", {}, C.h("span", { class: `pill ${c.status === "open" ? "warn" : "ok"}`, text: c.status === "open" ? "Open" : "Closed" })),
          C.h("td", { text: c.target_label || "Not set" }), C.h("td", { class: "num", text: fmt(c.cameras) }), C.h("td", { class: "num", text: fmt(c.sightings) }),
          C.h("td", { class: "nowrap", text: dateTime(c.updated_at) }),
          C.h("td", {}, C.h("div", { class: "acts" },
            C.h("button", { class: "button small", type: "button", text: c.status === "open" ? "Close" : "Reopen", onclick: async () => {
              try { await C.api(`/api/cases/${c.id}`, { method: "PATCH", json: { status: c.status === "open" ? "closed" : "open" } }); cases(); } catch (x) { C.notice(x.message, 4000, true); } } }),
            C.h("button", { class: "button small danger", type: "button", text: "Delete", onclick: async () => {
              if (!confirm(`Delete case “${c.name}”? It is hidden from every view; the files are kept for audit.`)) return;
              try { await C.api(`/api/admin/cases/${c.id}`, { method: "DELETE" }); C.notice("Case deleted."); cases(); } catch (x) { C.notice(x.message, 4000, true); } } }))));
      })))));
  } catch (e) { document.getElementById("cases-out").replaceChildren(C.problem(e.message)); }
}

function accountStatus(a) {
  if (a.disabled) return ["Disabled", "bad"];
  if (a.locked) return ["Locked", "bad"];
  if (a.must_change) return ["Awaiting first sign-in", "warn"];
  return ["Active", "ok"];
}

async function accounts() {
  const body = st.body;
  body.replaceChildren(C.busy("Loading stations and accounts"));
  try {
    const [acc, stat] = await Promise.all([C.api("/auth/admin/accounts"), C.api("/api/admin/stations?days=30")]);
    st.stations = stat.stations;
    const activity = Object.fromEntries(stat.stations.map((x) => [x.station_id, x]));
    const rows = acc.accounts.filter((a) => a.role === "station");
    const act = (a, action, label, cls = "") => C.h("button", { class: `button small ${cls}`, type: "button", text: label, onclick: () => accountAction(a, action) });
    body.replaceChildren(C.h("p", { class: "note", style: { margin: "0 0 14px" }, text: `${rows.length} station accounts in the ${C.session.user.division.name} Division. Resetting a password signs the station out and issues a one-time password that must be changed at the next sign-in.` }),
      C.h("div", { class: "tablewrap" }, C.h("table", { class: "dtable" },
        C.h("thead", {}, C.h("tr", {}, ["Station", "Type", "Account", "Status", "Last sign-in", "Records (30 days)", ""].map((x, i) => C.h("th", { class: i === 5 ? "num" : "", text: x })))),
        C.h("tbody", {}, rows.map((a) => {
          const [label, tone] = accountStatus(a);
          const ac = activity[a.station_id] || {};
          return C.h("tr", {}, C.h("td", {}, C.h("strong", { text: a.station_name || a.display_name })), C.h("td", { text: KINDS[a.station_kind] || "" }),
            C.h("td", { class: "mono nowrap", text: a.username }), C.h("td", {}, C.h("span", { class: `pill ${tone}`, text: label })),
            C.h("td", { class: "nowrap", text: a.last_login_at ? dateTime(a.last_login_at) : "Never" }), C.h("td", { class: "num", text: fmt(ac.events || 0) }),
            C.h("td", {}, C.h("div", { class: "acts" }, act(a, "reset", "Reset password"), a.locked ? act(a, "unlock", "Unlock") : null,
              a.active_sessions ? act(a, "signout", "Sign out") : null, a.disabled ? act(a, "enable", "Enable") : act(a, "disable", "Disable", "danger"))));
        })))));
  } catch (e) { body.replaceChildren(C.problem(e.message)); }
}

async function accountAction(a, action) {
  const names = { reset: "reset the password of", disable: "disable", enable: "enable", unlock: "unlock", signout: "sign out every session of" };
  if (!confirm(`Do you want to ${names[action]} ${a.station_name || a.username}?`)) return;
  try {
    const r = await C.api(`/auth/admin/accounts/${a.id}/${action}`, { method: "POST" });
    if (action === "reset" && r.temporary_password) {
      const sheet = document.getElementById("sheet");
      document.getElementById("sheet-title").textContent = `Temporary password for ${a.station_name || a.username}`;
      const secret = C.h("div", { class: "secret", text: r.temporary_password });
      document.getElementById("sheet-body").replaceChildren(
        C.h("p", { class: "note", text: "Give this to the station through a trusted channel. It is shown only once, and the station must replace it at the next sign-in." }),
        C.h("p", { class: "note" }, "Account ID: ", C.h("strong", { class: "mono", text: a.username })), secret,
        C.h("div", { class: "actions" }, C.h("button", { class: "button", type: "button", text: "Copy", onclick: async () => { try { await navigator.clipboard.writeText(r.temporary_password); C.notice("Copied."); } catch { C.notice("Select the text and copy it.", 3000, true); } } })));
      sheet.showModal();
    } else C.notice("Account updated.");
    accounts();
  } catch (e) { C.notice(e.message, 4000, true); }
}

async function audit(more = false) {
  const body = st.body;
  if (!more) { st.audit = []; body.replaceChildren(C.busy("Loading the audit log")); }
  const before = more && st.audit.length ? st.audit[st.audit.length - 1].id : "";
  try {
    const r = await C.api(`/auth/admin/audit?limit=100${before ? `&before=${before}` : ""}`);
    st.audit.push(...r.audit);
    const OUT = { ok: ["Success", "ok"], bad_password: ["Wrong password", "bad"], unknown_user: ["Unknown ID", "bad"], locked: ["Locked", "bad"], locked_now: ["Locked now", "bad"],
      disabled: ["Disabled account", "bad"], outside_bengaluru: ["Outside Bengaluru", "bad"], no_location: ["No location", "warn"], bad_current: ["Wrong current password", "bad"] };
    body.replaceChildren(C.h("p", { class: "note", style: { margin: "0 0 14px" }, text: "Security events for accounts in this division: sign-ins, failures, password changes and account actions." }),
      C.h("div", { class: "tablewrap" }, C.h("table", { class: "dtable" },
        C.h("thead", {}, C.h("tr", {}, ["When", "Account", "Action", "Target", "Outcome", "Network", "Detail"].map((x) => C.h("th", { text: x })))),
        C.h("tbody", {}, st.audit.map((a) => {
          const [label, tone] = OUT[a.outcome] || [a.outcome, ""];
          return C.h("tr", {}, C.h("td", { class: "nowrap", text: dateTime(a.ts) }), C.h("td", { class: "mono", text: a.actor || "" }), C.h("td", { text: a.action.replace(/_/g, " ") }),
            C.h("td", { class: "mono", text: a.target || "" }), C.h("td", {}, C.h("span", { class: `pill ${tone}`, text: label })),
            C.h("td", { class: "nowrap mono", text: `${a.ip || ""}${a.country ? " · " + a.country : ""}` }), C.h("td", { class: "muted", text: a.detail || "" }));
        })))),
      ...(r.audit.length === 100 ? [C.h("div", { class: "pager" }, C.h("button", { class: "button small", type: "button", text: "Load older", onclick: () => audit(true) }))] : []));
  } catch (e) { body.replaceChildren(C.problem(e.message)); }
}

function hourlyChart(box, prior, journeys) {
  const hours = Array.from({ length: 24 }, (_, h) => prior[String(h)] || null);
  const pts = journeys.map((j) => ({ hour: new Date(j.depart_ts * 1000).getHours(), ratio: j.observed_s / j.freeflow_s, j }))
    .filter((x) => x.ratio > 0 && x.ratio < 8);
  if (!hours.some(Boolean) && !pts.length) return box.append(empty("No travel-time data yet."));
  const W = Math.max(320, box.clientWidth || 640), H = 250, ml = 40, mr = 10, mt = 12, mb = 28;
  const top = ticks(Math.max(1, ...hours.filter(Boolean).map((x) => x.p90_ratio), ...pts.map((x) => x.ratio)));
  const ymax = top[top.length - 1];
  const iw = W - ml - mr, ih = H - mt - mb, x = (h) => ml + (h + 0.5) * (iw / 24), y = (v) => mt + ih - (v / ymax) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Travel time as a multiple of free-flow time, by hour of day" });
  top.forEach((v) => svg.append(s("line", { class: "gridline", x1: ml, x2: W - mr, y1: y(v), y2: y(v) }), s("text", { x: ml - 8, y: y(v) + 4, "text-anchor": "end" }, `${v}×`)));
  svg.append(s("line", { class: "baseline", x1: ml, x2: W - mr, y1: y(1), y2: y(1) }));
  const valid = hours.map((v, h) => [v, h]).filter(([v]) => v);
  if (valid.length) {
    const band = valid.map(([v, h]) => `${x(h)},${y(v.p90_ratio)}`).concat([...valid].reverse().map(([v, h]) => `${x(h)},${y(v.p10_ratio)}`));
    svg.append(s("polygon", { class: "band", points: band.join(" ") }));
    svg.append(s("polyline", { points: valid.map(([v, h]) => `${x(h)},${y(v.median_ratio)}`).join(" "), class: "median" }));
    const peak = valid.reduce((a, b) => (b[0].median_ratio > a[0].median_ratio ? b : a));
    svg.append(s("text", { class: "val", x: x(peak[1]), y: y(peak[0].median_ratio) - 8, "text-anchor": "middle" }, `${peak[0].median_ratio.toFixed(2)}× at ${String(peak[1]).padStart(2, "0")}:00`));
  }
  pts.forEach((p) => svg.append(s("circle", { cx: x(p.hour) + ((p.j.id % 7) - 3) * 1.5, cy: y(p.ratio), r: 4, class: "jdot" })));
  [0, 3, 6, 9, 12, 15, 18, 21].forEach((h) => svg.append(s("text", { x: x(h), y: H - 8, "text-anchor": "middle" }, `${String(h).padStart(2, "0")}:00`)));
  const tip = tipper(box);
  for (let h = 0; h < 24; h++) {
    const v = hours[h], local = pts.filter((p) => p.hour === h);
    const hit = s("rect", { class: "hit", x: ml + h * (iw / 24), y: mt, width: iw / 24, height: ih, tabindex: 0 });
    const show = () => tip.show(x(h), y(v ? v.p90_ratio : 1) - 4, `${String(h).padStart(2, "0")}:00 to ${String(h + 1).padStart(2, "0")}:00`, [
      ...(v ? [["Typical (median)", `${v.median_ratio.toFixed(2)}×`, "var(--mag)"], ["10–90% of trips", `${v.p10_ratio.toFixed(1)}–${v.p90_ratio.toFixed(1)}×`]] : []),
      ["Journeys observed here", fmt(local.length), "var(--s1)"]]);
    hit.addEventListener("pointermove", show); hit.addEventListener("focus", show);
    hit.addEventListener("pointerleave", tip.hide); hit.addEventListener("blur", tip.hide);
    svg.append(hit);
  }
  box.prepend(svg);
}

async function traffic() {
  const body = st.body;
  body.replaceChildren(C.busy("Loading travel-time data"));
  let d;
  try { d = await C.api("/api/admin/traffic?limit=200"); } catch (e) { body.replaceChildren(C.problem(e.message)); return; }
  const prior = d.prior, local = d.local, ho = prior && prior.holdout;
  const pctv = (v) => `${(100 * v).toFixed(0)}%`;
  const mins = (s2) => `${(s2 / 60).toFixed(1)} min`;
  const kpi = (label, value, hint) => C.h("div", { class: "kpi" }, C.h("div", { class: "label", text: label }), C.h("div", { class: "value", text: value }), hint ? C.h("div", { class: "hint", text: hint }) : null);
  const sources = Object.entries(local.by_source || {}).map(([k, n]) => `${n} ${k}`).join(", ");
  const kpis = C.h("div", { class: "kpis" },
    kpi("Free-flow error", ho ? pctv(ho.free_flow.median_abs_pct_error) : "n/a", ho ? `median, ${mins(ho.free_flow.mae_s)} mean` : ""),
    kpi("Calibrated error", ho ? pctv(ho.calibrated.median_abs_pct_error) : "n/a", ho ? `median, ${mins(ho.calibrated.mae_s)} mean` : ""),
    kpi("80% range holds", ho ? pctv(ho.interval_80_coverage) : "n/a", ho ? `of ${fmt(ho.test_rows)} held-out trips` : ""),
    kpi("Journeys observed here", fmt(local.n), sources || "none yet"),
    kpi("Local correction", local.n ? `${local.factor.toFixed(2)}×` : "none yet", local.n ? "applied on top of 2019" : "starts with the first journey"),
    kpi("Local check", local.holdout ? `${pctv(local.holdout.corrected_mdape)}` : "waiting", local.holdout ? `vs ${pctv(local.holdout.prior_mdape)} with 2019 only` : `after ${d.min_holdout} journeys`));
  const chart = card("Bengaluru slowdown by hour of day", "Travel time as a multiple of the free-flow estimate", {
    wide: true, legend: legend([{ label: "Uber Movement 2019 weekdays (median, 10–90% band)", colour: "var(--mag)" }, { label: "Journeys observed here", colour: "var(--s1)" }]),
    draw: (b) => hourlyChart(b, (prior && prior.hourly) || {}, d.journeys),
    table: () => simpleTable(["Hour", "Median", "10%", "90%", "Trips"], Object.entries((prior && prior.hourly) || {}).map(([h2, v]) => [`${h2.padStart(2, "0")}:00`, `${v.median_ratio.toFixed(2)}×`, `${v.p10_ratio.toFixed(2)}×`, `${v.p90_ratio.toFixed(2)}×`, fmt(v.n)])) });
  const file = C.h("input", { type: "file", accept: ".csv,text/csv" });
  const status = C.h("span", { class: "muted", style: { fontSize: "12.5px" } });
  const importBtn = C.h("button", { class: "button primary small", type: "button", text: "Import journeys", onclick: async () => {
    if (!file.files[0]) { C.notice("Choose a CSV file first.", 3000, true); return; }
    const fd = new FormData(); fd.append("file", file.files[0]);
    try {
      const r = await C.api("/api/admin/traffic/import", { method: "POST", form: fd });
      const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
      const skipped = r.skipped.length ? `, ${plural(r.skipped.length, "row")} skipped (${r.skipped[0].replace(/\.$/, "")}${r.skipped.length > 1 ? " and more" : ""})` : "";
      status.textContent = `Routing ${plural(r.rows, "journey")}${skipped}`;
      const poll = async () => {
        const j = await C.api(`/api/jobs/${r.job_id}`);
        if (j.status === "done") { C.notice(`Imported ${plural(j.added, "journey")}${j.unrouted ? `, ${j.unrouted} could not be routed` : ""}${skipped}.`, 8000); traffic(); return; }
        if (j.status === "error") { status.textContent = `Import failed: ${j.error}`; return; }
        status.textContent = `Routing ${plural(r.rows, "journey")}: ${Math.round(100 * j.progress)}%${skipped}`;
        setTimeout(poll, 2000);
      };
      poll();
    } catch (x) { C.notice(x.message, 5000, true); }
  } });
  const refit = C.h("button", { class: "button small", type: "button", text: "Refit now", onclick: async () => { try { await C.api("/api/admin/traffic/refit", { method: "POST" }); C.notice("Model refitted."); traffic(); } catch (x) { C.notice(x.message, 4000, true); } } });
  const importCard = C.h("section", { class: "card" },
    C.h("div", { class: "card-head" }, C.h("div", {}, C.h("h3", { text: "Add observed journeys" }), C.h("p", { text: "Journey times from any source: number-plate cameras, patrol logs, GPS fleets" }))),
    C.h("p", { class: "note", text: "CSV columns: from_lat, from_lon, to_lat, to_lon, depart (local time, for example 2026-11-06T18:05), seconds. Both ends must be in Bengaluru. Each row is routed for its free-flow time, then the model refits." }),
    C.h("div", { class: "actions", style: { marginTop: "12px" } }, file, importBtn, refit), status);
  const how = C.h("section", { class: "card" },
    C.h("div", { class: "card-head" }, C.h("div", {}, C.h("h3", { text: "How the loop works" }))),
    C.h("ol", { class: "note", style: { paddingLeft: "18px", margin: 0 } },
      C.h("li", { text: "Starting point: real 2019 Bengaluru trip times between wards, by hour of day, compared with the free-flow estimate for the same trip." }),
      C.h("li", { text: "Every time the same vehicle is seen at two cameras, or a dashcam with GPS is uploaded, the actual travel time is stored as a journey." }),
      C.h("li", { text: "Journeys shift the 2019 profile towards today's traffic. Few journeys move it a little; many move it a lot." }),
      C.h("li", { text: `Once ${d.min_holdout} journeys exist, the newest fifth are held out and the error with and without the local correction is reported here.` })),
    prior ? C.h("p", { class: "disclaimer", text: `${prior.source.attribution}. Licence: ${prior.source.licence}.` }) : null);
  const rows = d.journeys.map((j) => C.h("tr", {},
    C.h("td", { class: "nowrap", text: dateTime(j.depart_ts) }), C.h("td", { text: j.source }), C.h("td", { text: j.note || `${j.from_lat.toFixed(3)}, ${j.from_lon.toFixed(3)} to ${j.to_lat.toFixed(3)}, ${j.to_lon.toFixed(3)}` }),
    C.h("td", { class: "num", text: mins(j.observed_s) }), C.h("td", { class: "num", text: mins(j.freeflow_s) }), C.h("td", { class: "num", text: `${(j.observed_s / j.freeflow_s).toFixed(2)}×` }),
    C.h("td", {}, C.h("div", { class: "acts" }, C.h("button", { class: "button small danger", type: "button", text: "Delete", onclick: async () => {
      if (!confirm("Delete this journey? The model refits without it.")) return;
      try { await C.api(`/api/admin/traffic/journeys/${j.id}`, { method: "DELETE" }); C.notice("Journey deleted."); traffic(); } catch (x) { C.notice(x.message, 4000, true); }
    } })))));
  const table = d.journeys.length ? C.h("div", { class: "tablewrap" }, C.h("table", { class: "dtable" },
    C.h("thead", {}, C.h("tr", {}, ["Departed", "Source", "Journey", "Took", "Free-flow", "Ratio", ""].map((x2, i) => C.h("th", { class: i >= 3 && i <= 5 ? "num" : "", text: x2 })))),
    C.h("tbody", {}, rows))) : empty("No journeys observed in this division yet.");
  body.replaceChildren(kpis, C.h("div", { class: "grid" }, chart, importCard, how), C.h("h3", { style: { font: "600 17px var(--display)", margin: "26px 0 10px" }, text: "Journeys in this division" }), table);
  chart.render();
  st.resize = () => chart.render();
}

const TABS = [["overview", "Overview"], ["events", "Records"], ["cases", "Cases"], ["traffic", "Traffic"], ["accounts", "Stations and accounts"], ["audit", "Audit log"]];
function show(tab) {
  st.tab = tab;
  st.resize = null;
  st.root.querySelectorAll(".subtabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === tab)));
  ({ overview, events, cases, traffic, accounts, audit })[tab]();
}

export function mount(root, ctx) {
  C = ctx;
  st.root = root;
  if (!root.dataset.ready) {
    const div = C.session.user.division;
    st.body = C.h("div", {});
    root.replaceChildren(C.h("div", { class: "cx" },
      C.h("div", { class: "cx-head" }, C.h("div", {}, C.h("h1", {}, `${div.name} Division`, C.h("span", { class: "kn", text: KN_DIV[div.id] || "" })),
        C.h("p", { text: "Every record, case and account of the stations in your division." }))),
      C.h("nav", { class: "subtabs", role: "tablist" }, TABS.map(([k, label]) => C.h("button", { type: "button", role: "tab", "data-tab": k, "aria-selected": "false", text: label, onclick: () => show(k) }))),
      st.body));
    root.dataset.ready = "1";
    let t = null;
    window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => st.resize && !root.hidden && st.resize(), 200); });
  }
  show(st.tab);
}
