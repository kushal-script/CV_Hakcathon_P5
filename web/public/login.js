const $ = (s) => document.querySelector(s);
const art = document.querySelector(".skyline");
const params = new URLSearchParams(location.search);
const next = (() => {
  try {
    const u = new URL(params.get("next") || "/", location.origin);
    return u.origin === location.origin && !u.pathname.startsWith("/login") ? u.pathname + u.search + u.hash : "/";
  } catch {
    return "/";
  }
})();
let policy = { device_location_required: false };
let account = null;
const area = () => policy.boundary || "Bengaluru";

if (art) {
  art.querySelectorAll(".draw").forEach((el) => el.setAttribute("pathLength", "1"));
  requestAnimationFrame(() => art.classList.add("ready"));
}

function setLocation(state, text) {
  $("#check-location").className = "check " + ({ ok: "ok", wait: "wait", bad: "bad" }[state] || "");
  $("#location-text").textContent = text;
}

function busy(on, text = "Checking credentials", kn = "ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ") {
  $("#busy").hidden = !on;
  $("#busy-text").textContent = text;
  $("#busy-kn").textContent = kn;
  art?.classList.toggle("cycle", on);
  art?.classList.toggle("moving", on);
  $("#submit").disabled = on;
  $("#change-submit").disabled = on;
}

function fail(message, target = "#error") {
  $(target).textContent = message;
  const card = $("#card");
  card.classList.remove("shake");
  void card.offsetWidth;
  card.classList.add("shake");
  art?.classList.add("stop");
  setTimeout(() => art?.classList.remove("stop"), 1400);
}

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-Requested-With": "vahana" },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) throw Object.assign(new Error(data.detail || `Request failed (${res.status}).`), { status: res.status });
  return data;
}

function position() {
  return new Promise((resolve) => {
    if (!("geolocation" in navigator)) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude, accuracy: p.coords.accuracy }),
      () => resolve(null),
      { enableHighAccuracy: false, timeout: 12000, maximumAge: 120000 },
    );
  });
}

function welcome(user) {
  art?.classList.remove("cycle", "stop");
  art?.classList.add("go");
  const station = user.station ? user.station.name : user.display_name;
  $("#welcome-name").textContent = station;
  $("#welcome-sub").textContent = `${user.division.name} Division` + (user.role === "division_admin" ? " · Division administrator" : "");
  const barrier = $("#barrier");
  barrier.hidden = false;
  setTimeout(() => barrier.classList.add("open"), 280);
  setTimeout(() => location.replace(next), 1500);
}

function showChange(user, current = "") {
  account = user;
  $("#login-form").hidden = true;
  $("#change-form").hidden = false;
  $("#signin-title").textContent = "Set a new password";
  $("#change-form [name=username]").value = user?.username || "";
  $("#current").value = current;
  (current ? $("#new") : $("#current")).focus();
  checkRules();
}

function rules() {
  const pw = $("#new").value;
  const id = (account?.username || $("#username").value || "").toLowerCase();
  const classes = [/[a-z]/, /[A-Z]/, /[0-9]/, /[^A-Za-z0-9]/].filter((r) => r.test(pw)).length;
  return {
    len: pw.length >= 12,
    mix: classes >= 3 || pw.length >= 16,
    id: pw.length > 0 && (!id || !pw.toLowerCase().includes(id)),
    same: pw.length > 0 && pw === $("#repeat").value,
  };
}

function checkRules() {
  const r = rules();
  document.querySelectorAll("#rules li").forEach((li) => li.classList.toggle("met", r[li.dataset.rule]));
  return Object.values(r).every(Boolean);
}

$("#reveal").addEventListener("click", (e) => {
  const input = $("#password");
  const show = input.type === "password";
  input.type = show ? "text" : "password";
  e.currentTarget.textContent = show ? "Hide" : "Show";
  e.currentTarget.setAttribute("aria-pressed", String(show));
});

["keydown", "keyup"].forEach((t) => $("#password").addEventListener(t, (e) => {
  if (e.getModifierState) $("#caps").hidden = !e.getModifierState("CapsLock");
}));

["input", "change"].forEach((t) => {
  $("#new").addEventListener(t, checkRules);
  $("#repeat").addEventListener(t, checkRules);
});

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#error").textContent = "";
  const username = $("#username").value.trim();
  const password = $("#password").value;
  if (!username || !password) return fail("Enter your station or division ID and password.");
  let where = null;
  if (policy.device_location_required) {
    busy(true, `Confirming you are in ${area()}`, "ಸ್ಥಳ ಪರಿಶೀಲನೆ");
    setLocation("wait", "finding your location");
    where = await position();
    if (!where) {
      busy(false);
      setLocation("bad", "not shared");
      return fail(`Allow location access in your browser so we can confirm you are signing in from within ${area()}.`);
    }
    setLocation("ok", "shared, checked against the sign-in area");
  }
  busy(true);
  try {
    const r = await post("/auth/login", { username, password, location: where });
    busy(false);
    if (r.must_change) return showChange(r.user, password);
    welcome(r.user);
  } catch (err) {
    busy(false);
    if (err.status === 403 && /only allowed from within/.test(err.message)) setLocation("bad", `outside ${area()}`);
    fail(err.message);
  }
});

$("#change-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#change-error").textContent = "";
  if (!checkRules()) return fail("The new password does not meet every rule yet.", "#change-error");
  busy(true, "Saving your new password", "ಉಳಿಸಲಾಗುತ್ತಿದೆ");
  try {
    const r = await post("/auth/password", { current: $("#current").value, new: $("#new").value });
    busy(false);
    welcome(r.user);
  } catch (err) {
    busy(false);
    fail(err.message, "#change-error");
  }
});

fetch("/auth/policy", { credentials: "same-origin" })
  .then((r) => (r.ok ? r.json() : null))
  .then((p) => {
    if (!p) return;
    policy = p;
    if (!p.device_location_required) setLocation("ok", "not required in this deployment");
    else setLocation("", `checked when you sign in, ${area()} only`);
  })
  .catch(() => {});

if (params.get("change") === "1") {
  fetch("/auth/me", { credentials: "same-origin" })
    .then((r) => (r.ok ? r.json() : null))
    .then((me) => { if (me?.must_change) showChange(me.user); else if (me) location.replace(next); })
    .catch(() => {});
} else if (params.get("ended") === "1") {
  $("#error").textContent = "Your session ended. Sign in again to continue.";
}
$("#username").focus();
