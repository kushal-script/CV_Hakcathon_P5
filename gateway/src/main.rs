mod admin;
mod auth;
mod cli;
mod db;
mod geo;
mod proxy;
mod util;
mod web;

use std::{env, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use axum::{
    extract::{Request, State},
    http::{header, HeaderName, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{any, get, post},
    Router,
};
use rand::RngCore;
use tower_http::{limit::RequestBodyLimitLayer, services::ServeFile};
use tracing::{info, warn};

use util::{country, env_or, wants_html, Limiter};

pub struct AppState {
    pub db: db::Db,
    pub upstream: String,
    pub client: reqwest::Client,
    pub internal_secret: Vec<u8>,
    pub dummy_hash: String,
    pub secure_cookie: bool,
    pub idle_ttl: i64,
    pub absolute_ttl: i64,
    pub trust_proxy: bool,
    pub api_limit: Limiter,
    pub login_limit: Limiter,
    pub geofence: Option<geo::Geofence>,
    pub device_required: bool,
    pub require_password_change: bool,
    pub geofence_margin_m: f64,
    pub allowed_countries: Vec<String>,
    pub lockout_after: i64,
    pub lockout_secs: i64,
    pub lockout_account_after: i64,
    pub login_failures: std::sync::Mutex<std::collections::HashMap<(String, std::net::IpAddr), (i64, i64)>>,
    pub web_public: PathBuf,
    pub web_app: PathBuf,
}

pub type Shared = Arc<AppState>;

async fn security_headers(State(state): State<Shared>, req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    let csp = "default-src 'self'; script-src 'self' https://unpkg.com/maplibre-gl@4.7.1/; style-src 'self' https://unpkg.com/maplibre-gl@4.7.1/; \
               img-src 'self' data: blob: https://tiles.openfreemap.org https://tiles.maps.eox.at; \
               connect-src 'self' https://tiles.openfreemap.org https://tiles.maps.eox.at; worker-src blob:; child-src blob:; \
               media-src 'self' blob:; font-src 'self' https://tiles.openfreemap.org; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'";
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(csp));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    h.insert(HeaderName::from_static("permissions-policy"), HeaderValue::from_static("camera=(), microphone=(), geolocation=(self), payment=()"));
    h.insert(HeaderName::from_static("cross-origin-opener-policy"), HeaderValue::from_static("same-origin"));
    h.insert(HeaderName::from_static("cross-origin-resource-policy"), HeaderValue::from_static("same-origin"));
    let cache = h.get(header::CACHE_CONTROL).and_then(|v| v.to_str().ok()).unwrap_or("no-cache").to_string();
    if !cache.contains("no-transform") {
        if let Ok(v) = HeaderValue::from_str(&format!("{cache}, no-transform")) {
            h.insert(header::CACHE_CONTROL, v);
        }
    }
    if state.secure_cookie {
        h.insert(header::STRICT_TRANSPORT_SECURITY, HeaderValue::from_static("max-age=31536000; includeSubDomains"));
    }
    res
}

async fn https_only(State(state): State<Shared>, req: Request, next: Next) -> Response {
    if state.trust_proxy && state.secure_cookie {
        let proto = req.headers().get("x-forwarded-proto").and_then(|v| v.to_str().ok()).unwrap_or("");
        let host = req.headers().get(header::HOST).and_then(|v| v.to_str().ok()).unwrap_or("");
        if proto.eq_ignore_ascii_case("http") && !host.is_empty() && host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':')) {
            let path = req.uri().path_and_query().map(|p| p.as_str()).unwrap_or("/");
            if let Ok(location) = HeaderValue::from_str(&format!("https://{host}{path}")) {
                return (StatusCode::PERMANENT_REDIRECT, [(header::LOCATION, location)]).into_response();
            }
        }
    }
    next.run(req).await
}

async fn country_gate(State(state): State<Shared>, req: Request, next: Next) -> Response {
    if state.trust_proxy && !state.allowed_countries.is_empty() {
        if let Some(cc) = country(req.headers()) {
            if !state.allowed_countries.iter().any(|c| c == &cc) {
                warn!(country = %cc, path = %req.uri().path(), "blocked by country policy");
                if wants_html(req.headers()) {
                    return (StatusCode::FORBIDDEN, Html(BLOCKED_PAGE)).into_response();
                }
                return util::error(StatusCode::FORBIDDEN, "This service is only available to police stations in Bengaluru.");
            }
        }
    }
    next.run(req).await
}

const BLOCKED_PAGE: &str = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>Not available</title></head><body style=\"font:16px/1.5 system-ui,sans-serif;background:#f5efe2;color:#2a2219;display:grid;place-items:center;height:100vh;margin:0\">\
<main style=\"max-width:32rem;padding:24px\"><h1 style=\"font-size:20px\">Not available in your region</h1>\
<p>This system is restricted to police stations in Bengaluru.</p></main></body></html>";

fn internal_secret() -> Vec<u8> {
    match env::var("P5_INTERNAL_SECRET") {
        Ok(s) if s.len() >= 32 => s.into_bytes(),
        Ok(_) => panic!("P5_INTERNAL_SECRET must be at least 32 characters"),
        Err(_) => {
            warn!("P5_INTERNAL_SECRET not set: generating one, the analysis service will reject identities it cannot verify");
            let mut key = vec![0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut key);
            key
        }
    }
}

fn usage() -> ! {
    eprintln!("usage:\n  vahana-gateway                      run the gateway\n  vahana-gateway provision --stations <json> --credentials <new csv> [--db <path>]\n  vahana-gateway reset-password <username> [--db <path>]\n  vahana-gateway demo-accounts --file <json> [--db <path>]   (one \"username password\" pair per line on stdin)");
    std::process::exit(2)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("provision") => {
            if let Err(e) = cli::provision(&args) {
                eprintln!("provision failed: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("reset-password") => {
            if let Err(e) = cli::reset_password(&args) {
                eprintln!("reset failed: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some("demo-accounts") => {
            if let Err(e) = cli::demo_accounts(&args) {
                eprintln!("demo accounts failed: {e}");
                std::process::exit(1);
            }
            return;
        }
        Some(_) => usage(),
        None => {}
    }
    tracing_subscriber::fmt().with_env_filter(env_or("RUST_LOG", "info")).json().init();
    let db = db::Db::open(&PathBuf::from(env_or("P5_DB", "/app/state/gateway.db")), 4).expect("open gateway database");
    let users: i64 = db.with(|c| c.query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))).unwrap_or(0);
    if users == 0 {
        warn!("no accounts yet: run `vahana-gateway provision` to create station and division accounts");
    }
    let boundary = env_or("P5_BOUNDARY", "assets/bengaluru/boundary.geojson");
    let geofence = match geo::Geofence::load(&boundary) {
        Ok(g) => Some(g),
        Err(e) => {
            warn!(error = %e, "boundary not loaded: device location checks are off");
            None
        }
    };
    let dummy_hash = auth::hash_password("not-a-real-password-for-timing");
    let max_body = env_or("P5_MAX_BODY_MB", "300").parse::<usize>().unwrap_or(300) * 1024 * 1024;
    let web_root = PathBuf::from(env_or("P5_WEB_DIR", "web"));
    let state = Arc::new(AppState {
        db,
        upstream: env_or("P5_UPSTREAM", "http://127.0.0.1:8001").trim_end_matches('/').to_string(),
        client: reqwest::Client::builder().timeout(Duration::from_secs(900)).build().expect("client"),
        internal_secret: internal_secret(),
        dummy_hash,
        secure_cookie: env_or("P5_COOKIE_SECURE", "true") == "true",
        idle_ttl: env_or("P5_IDLE_MINUTES", "60").parse::<i64>().unwrap_or(60) * 60,
        absolute_ttl: env_or("P5_SESSION_HOURS", "12").parse::<i64>().unwrap_or(12) * 3600,
        trust_proxy: env_or("P5_TRUST_PROXY", "false") == "true",
        api_limit: {
            let per_minute: f64 = env_or("P5_API_PER_MINUTE", "240").parse().unwrap_or(240.0);
            Limiter::new(per_minute, (per_minute / 4.0).max(60.0))
        },
        login_limit: {
            let per_minute: f64 = env_or("P5_LOGIN_PER_MINUTE", "6").parse().unwrap_or(6.0);
            Limiter::new(per_minute, per_minute.max(1.0))
        },
        device_required: env_or("P5_DEVICE_GEOFENCE", "required") == "required",
        require_password_change: env_or("P5_REQUIRE_PASSWORD_CHANGE", "false") == "true",
        geofence,
        geofence_margin_m: env_or("P5_GEOFENCE_MARGIN_M", "2000").parse().unwrap_or(2000.0),
        allowed_countries: env_or("P5_ALLOWED_COUNTRIES", "IN").split(',').map(|c| c.trim().to_ascii_uppercase()).filter(|c| !c.is_empty()).collect(),
        lockout_after: env_or("P5_LOCKOUT_AFTER", "5").parse().unwrap_or(5),
        lockout_secs: env_or("P5_LOCKOUT_MINUTES", "15").parse::<i64>().unwrap_or(15) * 60,
        lockout_account_after: env_or("P5_LOCKOUT_ACCOUNT_AFTER", "50").parse().unwrap_or(50),
        login_failures: std::sync::Mutex::new(std::collections::HashMap::new()),
        web_public: web_root.join("public"),
        web_app: web_root.join("app"),
    });
    let api = Router::new().route("/api/{*rest}", any(proxy::proxy)).layer(RequestBodyLimitLayer::new(max_body));
    let auth_routes = Router::new()
        .route("/auth/login", post(auth::login))
        .route("/auth/logout", post(auth::logout))
        .route("/auth/password", post(auth::change_password))
        .route("/auth/me", get(auth::me))
        .route("/auth/policy", get(auth::policy))
        .route("/auth/admin/accounts", get(admin::accounts))
        .route("/auth/admin/accounts/{id}/{action}", post(admin::account_action))
        .route("/auth/admin/audit", get(admin::audit_log))
        .layer(RequestBodyLimitLayer::new(16 * 1024));
    let app = Router::new()
        .merge(api)
        .merge(auth_routes)
        .route("/login", get(web::login_page))
        .route("/public/{*path}", get(web::public_files))
        .route_service("/favicon.svg", ServeFile::new(state.web_public.join("favicon.svg")))
        .fallback(web::app_files)
        .layer(middleware::from_fn_with_state(state.clone(), country_gate))
        .layer(middleware::from_fn_with_state(state.clone(), https_only))
        .layer(middleware::from_fn_with_state(state.clone(), security_headers))
        .with_state(state.clone());
    let addr: SocketAddr = env_or("P5_LISTEN", "0.0.0.0:8080").parse().expect("P5_LISTEN");
    info!(%addr, upstream = %state.upstream, accounts = users, device_geofence = state.device_required && state.geofence.is_some(),
          countries = ?state.allowed_countries, require_password_change = state.require_password_change, "gateway listening");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .expect("server");
}
