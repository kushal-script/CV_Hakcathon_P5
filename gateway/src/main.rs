use std::{
    collections::HashMap,
    env,
    net::{IpAddr, SocketAddr},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use argon2::{password_hash::PasswordHash, password_hash::SaltString, Argon2, PasswordHasher, PasswordVerifier};
use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header, HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{any, get, post},
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use futures_util::TryStreamExt;
use hmac::{Hmac, Mac};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tower_http::{limit::RequestBodyLimitLayer, services::ServeDir};
use tracing::{info, warn};

type HmacSha256 = Hmac<Sha256>;
const COOKIE: &str = "vd_session";
const HOP: [&str; 8] = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"];

#[derive(Deserialize)]
struct UsersFile {
    #[serde(default)]
    users: Vec<User>,
    #[serde(default)]
    api_tokens: Vec<ApiToken>,
}

#[derive(Deserialize, Clone)]
struct User {
    username: String,
    password_hash: String,
}

#[derive(Deserialize, Clone)]
struct ApiToken {
    name: String,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
struct Claims {
    u: String,
    exp: u64,
    n: String,
}

struct Limiter {
    rate: f64,
    burst: f64,
    buckets: Mutex<HashMap<IpAddr, (f64, Instant)>>,
}

impl Limiter {
    fn new(per_minute: f64, burst: f64) -> Self {
        Self { rate: per_minute / 60.0, burst, buckets: Mutex::new(HashMap::new()) }
    }

    fn allow(&self, ip: IpAddr) -> bool {
        let mut map = self.buckets.lock().expect("limiter lock");
        let now = Instant::now();
        if map.len() > 50_000 {
            map.retain(|_, (_, t)| now.duration_since(*t) < Duration::from_secs(600));
        }
        let (tokens, last) = map.entry(ip).or_insert((self.burst, now));
        *tokens = (*tokens + now.duration_since(*last).as_secs_f64() * self.rate).min(self.burst);
        *last = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

struct AppState {
    upstream: String,
    client: reqwest::Client,
    secret: Vec<u8>,
    users: HashMap<String, User>,
    tokens: Vec<ApiToken>,
    dummy_hash: String,
    secure_cookie: bool,
    session_ttl: u64,
    trust_proxy: bool,
    api_limit: Limiter,
    login_limit: Limiter,
    auth_required: bool,
}

type Shared = Arc<AppState>;

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn sign(secret: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(secret).expect("hmac key");
    mac.update(payload);
    mac.finalize().into_bytes().to_vec()
}

fn issue(state: &AppState, user: &str) -> String {
    let mut nonce = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut nonce);
    let claims = Claims { u: user.to_string(), exp: now() + state.session_ttl, n: B64.encode(nonce) };
    let body = B64.encode(serde_json::to_vec(&claims).expect("claims"));
    let sig = B64.encode(sign(&state.secret, body.as_bytes()));
    format!("{body}.{sig}")
}

fn verify(state: &AppState, token: &str) -> Option<String> {
    let (body, sig) = token.split_once('.')?;
    let expected = sign(&state.secret, body.as_bytes());
    let given = B64.decode(sig).ok()?;
    if expected.ct_eq(&given).unwrap_u8() != 1 {
        return None;
    }
    let claims: Claims = serde_json::from_slice(&B64.decode(body).ok()?).ok()?;
    (claims.exp > now() && state.users.contains_key(&claims.u)).then_some(claims.u)
}

fn cookie_value(headers: &HeaderMap) -> Option<String> {
    headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';')).find_map(|kv| {
        let (k, v) = kv.trim().split_once('=')?;
        (k == COOKIE).then(|| v.to_string())
    })
}

fn identity(state: &AppState, headers: &HeaderMap) -> Option<String> {
    if let Some(user) = cookie_value(headers).and_then(|t| verify(state, &t)) {
        return Some(user);
    }
    let bearer = headers.get(header::AUTHORIZATION)?.to_str().ok()?.strip_prefix("Bearer ")?;
    let digest = hex(&Sha256::digest(bearer.as_bytes()));
    state.tokens.iter().find(|t| t.sha256.as_bytes().ct_eq(digest.as_bytes()).unwrap_u8() == 1).map(|t| format!("token:{}", t.name))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn client_ip(state: &AppState, addr: SocketAddr, headers: &HeaderMap) -> IpAddr {
    if state.trust_proxy {
        for name in ["cf-connecting-ip", "x-real-ip"] {
            if let Some(ip) = headers.get(name).and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok()) {
                return ip;
            }
        }
        if let Some(ip) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()).and_then(|v| v.split(',').next()).and_then(|v| v.trim().parse().ok()) {
            return ip;
        }
    }
    addr.ip()
}

fn session_cookie(state: &AppState, value: &str, max_age: u64) -> HeaderValue {
    let secure = if state.secure_cookie { "; Secure" } else { "" };
    HeaderValue::from_str(&format!("{COOKIE}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}")).expect("cookie")
}

fn error(status: StatusCode, detail: &str) -> Response {
    (status, Json(serde_json::json!({ "detail": detail }))).into_response()
}

#[derive(Deserialize)]
struct Login {
    username: String,
    password: String,
}

async fn login(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(body): Json<Login>) -> Response {
    let ip = client_ip(&state, addr, &headers);
    if !state.login_limit.allow(ip) {
        warn!(%ip, "login rate limited");
        return error(StatusCode::TOO_MANY_REQUESTS, "Too many attempts.");
    }
    if body.username.len() > 64 || body.password.len() > 256 {
        return error(StatusCode::UNAUTHORIZED, "Incorrect username or password.");
    }
    let user = state.users.get(&body.username);
    let hash = user.map(|u| u.password_hash.as_str()).unwrap_or(state.dummy_hash.as_str());
    let password = body.password.clone();
    let hash = hash.to_string();
    let ok = tokio::task::spawn_blocking(move || PasswordHash::new(&hash).map(|h| Argon2::default().verify_password(password.as_bytes(), &h).is_ok()).unwrap_or(false))
        .await
        .unwrap_or(false);
    if !(ok && user.is_some()) {
        warn!(%ip, user = %body.username, "login failed");
        return error(StatusCode::UNAUTHORIZED, "Incorrect username or password.");
    }
    info!(%ip, user = %body.username, "login");
    let token = issue(&state, &body.username);
    let mut res = Json(serde_json::json!({ "user": body.username })).into_response();
    res.headers_mut().insert(header::SET_COOKIE, session_cookie(&state, &token, state.session_ttl));
    res
}

async fn logout(State(state): State<Shared>) -> Response {
    let mut res = StatusCode::NO_CONTENT.into_response();
    res.headers_mut().insert(header::SET_COOKIE, session_cookie(&state, "", 0));
    res
}

async fn me(State(state): State<Shared>, headers: HeaderMap) -> Response {
    match identity(&state, &headers) {
        Some(user) => Json(serde_json::json!({ "user": user })).into_response(),
        None if !state.auth_required => Json(serde_json::json!({ "user": null })).into_response(),
        None => error(StatusCode::UNAUTHORIZED, "Not signed in."),
    }
}

async fn proxy(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, req: Request) -> Response {
    let ip = client_ip(&state, addr, req.headers());
    let path = req.uri().path().to_string();
    let public = path == "/api/health";
    if !state.api_limit.allow(ip) {
        return error(StatusCode::TOO_MANY_REQUESTS, "Too many requests. Slow down and try again.");
    }
    let user = identity(&state, req.headers());
    if !public && state.auth_required && user.is_none() {
        return error(StatusCode::UNAUTHORIZED, "Sign in to continue.");
    }
    if !matches!(*req.method(), Method::GET | Method::POST | Method::PATCH | Method::HEAD) {
        return error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed.");
    }
    if path.contains("..") || path.contains('\\') {
        return error(StatusCode::BAD_REQUEST, "Invalid path.");
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let url = format!("{}{}{}", state.upstream, path, query);
    let method = reqwest::Method::from_bytes(req.method().as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
    let mut out = state.client.request(method, &url).header("x-request-id", &request_id);
    for name in ["content-type", "content-length", "accept", "range"] {
        if let Some(v) = req.headers().get(name) {
            out = out.header(name, v.as_bytes());
        }
    }
    if let Some(u) = &user {
        out = out.header("x-authenticated-user", u.as_str());
    }
    let started = Instant::now();
    let stream = req.into_body().into_data_stream().map_err(std::io::Error::other);
    let upstream = match out.body(reqwest::Body::wrap_stream(stream)).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(%request_id, error = %e, "upstream unavailable");
            return error(StatusCode::BAD_GATEWAY, "The analysis service is unavailable. Try again shortly.");
        }
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    info!(%request_id, %ip, user = user.as_deref().unwrap_or("-"), %path, status = status.as_u16(), ms = started.elapsed().as_millis() as u64, "proxied");
    let mut builder = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if HOP.contains(&name.as_str()) || name.as_str() == "server" {
            continue;
        }
        if let (Ok(n), Ok(v)) = (HeaderName::from_bytes(name.as_str().as_bytes()), HeaderValue::from_bytes(value.as_bytes())) {
            builder = builder.header(n, v);
        }
    }
    builder = builder.header("x-request-id", request_id).header(header::CACHE_CONTROL, "no-store");
    builder.body(Body::from_stream(upstream.bytes_stream())).unwrap_or_else(|_| error(StatusCode::BAD_GATEWAY, "Bad upstream response."))
}

async fn security_headers(State(state): State<Shared>, req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    let csp = "default-src 'self'; script-src 'self' https://unpkg.com; style-src 'self' https://unpkg.com; \
               img-src 'self' data: blob: https://tiles.openfreemap.org https://tiles.maps.eox.at; \
               connect-src 'self' https://tiles.openfreemap.org https://tiles.maps.eox.at; worker-src blob:; child-src blob:; \
               media-src 'self' blob:; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(csp));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    h.insert(HeaderName::from_static("permissions-policy"), HeaderValue::from_static("camera=(), microphone=(), geolocation=(), payment=()"));
    h.insert(HeaderName::from_static("cross-origin-opener-policy"), HeaderValue::from_static("same-origin"));
    h.insert(HeaderName::from_static("cross-origin-resource-policy"), HeaderValue::from_static("same-origin"));
    if !h.contains_key(header::CACHE_CONTROL) {
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    }
    if state.secure_cookie {
        h.insert(header::STRICT_TRANSPORT_SECURITY, HeaderValue::from_static("max-age=31536000; includeSubDomains"));
    }
    res
}

fn load_users(path: &str) -> UsersFile {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| panic!("cannot parse users file {path}: {e}")),
        Err(_) => UsersFile { users: vec![], api_tokens: vec![] },
    }
}

fn session_secret() -> Vec<u8> {
    match env::var("P5_SESSION_SECRET") {
        Ok(s) if s.len() >= 32 => s.into_bytes(),
        Ok(_) => panic!("P5_SESSION_SECRET must be at least 32 characters"),
        Err(_) => {
            warn!("P5_SESSION_SECRET not set: using a random secret, sessions end on restart");
            let mut key = vec![0u8; 32];
            rand::thread_rng().fill_bytes(&mut key);
            key
        }
    }
}

async fn hash_password() {
    use tokio::io::AsyncBufReadExt;
    let mut line = String::new();
    tokio::io::BufReader::new(tokio::io::stdin()).read_line(&mut line).await.expect("read stdin");
    let password = line.trim_end_matches(['\r', '\n']);
    if password.len() < 12 {
        eprintln!("Use a password of at least 12 characters.");
        std::process::exit(2);
    }
    let salt = SaltString::generate(&mut argon2::password_hash::rand_core::OsRng);
    println!("{}", Argon2::default().hash_password(password.as_bytes(), &salt).expect("hash").to_string());
}

#[tokio::main]
async fn main() {
    if env::args().nth(1).as_deref() == Some("hash-password") {
        hash_password().await;
        return;
    }
    tracing_subscriber::fmt().with_env_filter(env_or("RUST_LOG", "info")).json().init();
    let users = load_users(&env_or("P5_USERS_FILE", "/run/secrets/users.json"));
    let auth_required = env_or("P5_AUTH", "required") != "disabled";
    if auth_required && users.users.is_empty() && users.api_tokens.is_empty() {
        panic!("authentication is required but no users or API tokens are configured");
    }
    let salt = SaltString::generate(&mut argon2::password_hash::rand_core::OsRng);
    let dummy_hash = Argon2::default().hash_password(b"not-a-real-password", &salt).expect("dummy hash").to_string();
    let max_body = env_or("P5_MAX_BODY_MB", "300").parse::<usize>().unwrap_or(300) * 1024 * 1024;
    let state = Arc::new(AppState {
        upstream: env_or("P5_UPSTREAM", "http://127.0.0.1:8001").trim_end_matches('/').to_string(),
        client: reqwest::Client::builder().timeout(Duration::from_secs(900)).build().expect("client"),
        secret: session_secret(),
        users: users.users.into_iter().map(|u| (u.username.clone(), u)).collect(),
        tokens: users.api_tokens,
        dummy_hash,
        secure_cookie: env_or("P5_COOKIE_SECURE", "true") == "true",
        session_ttl: env_or("P5_SESSION_HOURS", "12").parse::<u64>().unwrap_or(12) * 3600,
        trust_proxy: env_or("P5_TRUST_PROXY", "false") == "true",
        api_limit: Limiter::new(env_or("P5_API_PER_MINUTE", "120").parse().unwrap_or(120.0), 40.0),
        login_limit: Limiter::new(env_or("P5_LOGIN_PER_MINUTE", "5").parse().unwrap_or(5.0), 5.0),
        auth_required,
    });
    let web = PathBuf::from(env_or("P5_WEB_DIR", "web"));
    let api = Router::new().route("/api/{*rest}", any(proxy)).layer(RequestBodyLimitLayer::new(max_body));
    let auth = Router::new()
        .route("/auth/login", post(login))
        .route("/auth/logout", post(logout))
        .route("/auth/me", get(me))
        .layer(RequestBodyLimitLayer::new(16 * 1024));
    let app = Router::new()
        .merge(api)
        .merge(auth)
        .fallback_service(ServeDir::new(web).append_index_html_on_directories(true))
        .layer(middleware::from_fn_with_state(state.clone(), security_headers))
        .with_state(state.clone());
    let addr: SocketAddr = env_or("P5_LISTEN", "0.0.0.0:8080").parse().expect("P5_LISTEN");
    info!(%addr, upstream = %state.upstream, auth_required, "gateway listening");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async { let _ = tokio::signal::ctrl_c().await; })
        .await
        .expect("server");
}
