use std::{
    collections::HashMap,
    env,
    net::{IpAddr, SocketAddr},
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64, Engine};
use rand::{rngs::OsRng, Rng, RngCore};
use sha2::{Digest, Sha256};

pub fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(value: &str) -> String {
    hex(&Sha256::digest(value.as_bytes()))
}

pub fn random_token() -> String {
    let mut bytes = [0u8; 32];
    OsRng.fill_bytes(&mut bytes);
    B64.encode(bytes)
}

pub fn temp_password() -> String {
    const ALPHABET: &[u8] = b"abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
    loop {
        let groups: Vec<String> = (0..3)
            .map(|_| (0..4).map(|_| ALPHABET[OsRng.gen_range(0..ALPHABET.len())] as char).collect())
            .collect();
        let pw = groups.join("-");
        let classes = [pw.chars().any(|c| c.is_ascii_lowercase()), pw.chars().any(|c| c.is_ascii_uppercase()), pw.chars().any(|c| c.is_ascii_digit())];
        if classes.iter().all(|c| *c) {
            return pw;
        }
    }
}

pub fn error(status: StatusCode, detail: &str) -> Response {
    (status, Json(serde_json::json!({ "detail": detail }))).into_response()
}

pub struct Limiter {
    rate: f64,
    burst: f64,
    buckets: Mutex<HashMap<IpAddr, (f64, Instant)>>,
}

impl Limiter {
    pub fn new(per_minute: f64, burst: f64) -> Self {
        Self { rate: per_minute / 60.0, burst, buckets: Mutex::new(HashMap::new()) }
    }

    pub fn allow(&self, ip: IpAddr) -> bool {
        let mut map = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if map.len() > 50_000 {
            map.retain(|_, (_, t)| now.duration_since(*t) < Duration::from_secs(600));
        }
        let (tokens, last) = map.entry(source_key(ip)).or_insert((self.burst, now));
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

pub fn source_key(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => {
                let s = v6.segments();
                IpAddr::V6(std::net::Ipv6Addr::new(s[0], s[1], s[2], s[3], 0, 0, 0, 0))
            }
        },
        v4 => v4,
    }
}

pub fn client_ip(trust_proxy: bool, addr: SocketAddr, headers: &HeaderMap) -> IpAddr {
    if trust_proxy {
        for name in ["cf-connecting-ip", "x-real-ip"] {
            if let Some(ip) = headers.get(name).and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse().ok()) {
                return ip;
            }
        }
        if let Some(ip) = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
            .and_then(|v| v.trim().parse().ok())
        {
            return ip;
        }
    }
    addr.ip()
}

pub fn country(headers: &HeaderMap) -> Option<String> {
    headers.get("cf-ipcountry").and_then(|v| v.to_str().ok()).map(|v| v.trim().to_ascii_uppercase()).filter(|v| !v.is_empty())
}

pub fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    headers.get_all(header::COOKIE).iter().filter_map(|v| v.to_str().ok()).flat_map(|v| v.split(';')).find_map(|kv| {
        let (k, v) = kv.trim().split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

pub fn same_origin_request(headers: &HeaderMap) -> bool {
    let marked = headers.get("x-requested-with").and_then(|v| v.to_str().ok()) == Some("vahana");
    let site_ok = match headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
        Some(site) => site == "same-origin" || site == "none",
        None => true,
    };
    marked && site_ok
}

pub fn wants_html(headers: &HeaderMap) -> bool {
    headers.get(header::ACCEPT).and_then(|v| v.to_str().ok()).map(|v| v.contains("text/html")).unwrap_or(false)
}

pub fn user_agent(headers: &HeaderMap) -> String {
    headers.get(header::USER_AGENT).and_then(|v| v.to_str().ok()).unwrap_or("").chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_passwords_are_strong_and_distinct() {
        let a = temp_password();
        let b = temp_password();
        assert_eq!(a.len(), 14);
        assert_ne!(a, b);
        assert!(a.chars().any(|c| c.is_ascii_digit()) && a.chars().any(|c| c.is_ascii_uppercase()));
    }

    #[test]
    fn cross_site_requests_are_rejected() {
        let mut h = HeaderMap::new();
        assert!(!same_origin_request(&h));
        h.insert("x-requested-with", "vahana".parse().unwrap());
        assert!(same_origin_request(&h));
        h.insert("sec-fetch-site", "cross-site".parse().unwrap());
        assert!(!same_origin_request(&h));
    }
}
