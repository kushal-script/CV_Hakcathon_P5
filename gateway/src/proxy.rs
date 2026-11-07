use std::{net::SocketAddr, time::Instant};

use axum::{
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{header, HeaderName, HeaderValue, Method, StatusCode},
    response::Response,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use futures_util::TryStreamExt;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use tracing::{info, warn};

use crate::{
    auth::{current, Session},
    util::{client_ip, error, hex, now, same_origin_request},
    AppState, Shared,
};

pub const READ_ONLY: &str = "Demo division accounts can see every division but cannot change anything.";

const HOP: [&str; 8] = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"];

pub fn identity_headers(state: &AppState, s: &Session) -> (String, String) {
    let payload = serde_json::json!({
        "uid": s.user.id,
        "u": s.user.username,
        "r": s.user.role,
        "s": s.user.station.as_ref().map(|st| st.id.clone()),
        "d": s.user.division.id,
        "n": s.user.display_name,
        "a": s.user.all_divisions,
        "ro": s.user.demo && s.user.role == "division_admin",
        "ts": now(),
    });
    let encoded = B64.encode(serde_json::to_vec(&payload).expect("identity json"));
    let mut mac = Hmac::<Sha256>::new_from_slice(&state.internal_secret).expect("hmac key");
    mac.update(encoded.as_bytes());
    (encoded, hex(&mac.finalize().into_bytes()))
}

pub async fn proxy(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, req: Request) -> Response {
    let ip = client_ip(state.trust_proxy, addr, req.headers());
    let path = req.uri().path().to_string();
    if !state.api_limit.allow(ip) {
        return error(StatusCode::TOO_MANY_REQUESTS, "Too many requests. Slow down and try again.");
    }
    if path.contains("..") || path.contains('\\') {
        return error(StatusCode::BAD_REQUEST, "Invalid path.");
    }
    let method = req.method().clone();
    if !matches!(method, Method::GET | Method::POST | Method::PATCH | Method::DELETE | Method::HEAD) {
        return error(StatusCode::METHOD_NOT_ALLOWED, "Method not allowed.");
    }
    let public = path == "/api/health";
    let session = current(&state, req.headers()).await;
    if !public {
        match &session {
            None => return error(StatusCode::UNAUTHORIZED, "Sign in to continue."),
            Some(s) if s.restricted => return error(StatusCode::FORBIDDEN, "Change your password first."),
            _ => {}
        }
        if !matches!(method, Method::GET | Method::HEAD) && !same_origin_request(req.headers()) {
            return error(StatusCode::FORBIDDEN, "Request blocked.");
        }
        if let Some(s) = &session {
            let analysis = matches!(path.as_str(), "/api/predict" | "/api/locate" | "/api/video");
            if s.user.demo && s.user.role == "division_admin" && !matches!(method, Method::GET | Method::HEAD) && !analysis {
                return error(StatusCode::FORBIDDEN, READ_ONLY);
            }
        }
    }
    let request_id = uuid::Uuid::new_v4().to_string();
    let query = req.uri().query().map(|q| format!("?{q}")).unwrap_or_default();
    let url = format!("{}{}{}", state.upstream, path, query);
    let out_method = reqwest::Method::from_bytes(method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
    let mut out = state.client.request(out_method, &url).header("x-request-id", &request_id);
    for name in ["content-type", "content-length", "accept", "range"] {
        if let Some(v) = req.headers().get(name) {
            out = out.header(name, v.as_bytes());
        }
    }
    if let Some(s) = &session {
        let (identity, signature) = identity_headers(&state, s);
        out = out.header("x-p5-identity", identity).header("x-p5-identity-sig", signature);
    }
    let user = session.as_ref().map(|s| s.user.username.clone()).unwrap_or_else(|| "-".into());
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
    info!(%request_id, %ip, %user, %path, status = status.as_u16(), ms = started.elapsed().as_millis() as u64, "proxied");
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
