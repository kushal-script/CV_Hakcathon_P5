use std::net::SocketAddr;

use argon2::{password_hash::PasswordHash, password_hash::SaltString, Argon2, PasswordHasher, PasswordVerifier};
use axum::{
    extract::{ConnectInfo, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{
    db::audit,
    util::{client_ip, cookie_value, country, error, now, random_token, same_origin_request, sha256_hex, source_key, user_agent},
    AppState, Shared,
};

pub const COOKIE: &str = "vd_session";
const FAILED: &str = "Sign-in failed. Check your ID and password. After 5 failed attempts from one network, sign-in for that ID is paused there for 15 minutes.";

fn reserve_attempt(state: &AppState, key: &(String, std::net::IpAddr), t: i64) -> bool {
    let mut map = state.login_failures.lock().unwrap_or_else(|e| e.into_inner());
    if map.len() > 50_000 {
        map.retain(|_, (_, until)| *until > t);
    }
    let entry = map.entry(key.clone()).or_insert((0, 0));
    if entry.1 > t {
        return false;
    }
    if entry.1 != 0 {
        *entry = (0, 0);
    }
    if entry.0 >= state.lockout_after {
        *entry = (0, t + state.lockout_secs);
        return false;
    }
    entry.0 += 1;
    true
}

fn settle_attempt(state: &AppState, key: &(String, std::net::IpAddr), ok: bool, t: i64) {
    let mut map = state.login_failures.lock().unwrap_or_else(|e| e.into_inner());
    if ok {
        map.remove(key);
    } else if let Some(entry) = map.get_mut(key) {
        if entry.0 >= state.lockout_after {
            *entry = (0, t + state.lockout_secs);
        }
    }
}

#[derive(Clone, Serialize)]
pub struct StationView {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub lat: Option<f64>,
    pub lon: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct DivisionView {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Serialize)]
pub struct UserView {
    pub id: i64,
    pub username: String,
    pub role: String,
    pub display_name: String,
    pub station: Option<StationView>,
    pub division: DivisionView,
    pub must_change: bool,
}

#[derive(Clone)]
pub struct Session {
    pub token_hash: String,
    pub user: UserView,
    pub restricted: bool,
    pub expires_at: i64,
}

const USER_SQL: &str = "SELECT u.id, u.username, u.role, u.display_name, u.must_change, u.division_id, d.name,
       st.id, st.name, st.kind, st.lat, st.lon
FROM users u JOIN divisions d ON d.id = u.division_id LEFT JOIN stations st ON st.id = u.station_id";

fn user_from_row(r: &rusqlite::Row, offset: usize) -> rusqlite::Result<UserView> {
    let station_id: Option<String> = r.get(offset + 7)?;
    let station = match station_id {
        Some(id) => Some(StationView { id, name: r.get(offset + 8)?, kind: r.get(offset + 9)?, lat: r.get(offset + 10)?, lon: r.get(offset + 11)? }),
        None => None,
    };
    Ok(UserView {
        id: r.get(offset)?,
        username: r.get(offset + 1)?,
        role: r.get(offset + 2)?,
        display_name: r.get(offset + 3)?,
        must_change: r.get::<_, i64>(offset + 4)? != 0,
        division: DivisionView { id: r.get(offset + 5)?, name: r.get(offset + 6)? },
        station,
    })
}

pub fn load_user(c: &Connection, id: i64) -> rusqlite::Result<Option<UserView>> {
    c.query_row(&format!("{USER_SQL} WHERE u.id = ?1"), [id], |r| user_from_row(r, 0)).optional()
}

pub async fn current(state: &AppState, headers: &HeaderMap) -> Option<Session> {
    let token = cookie_value(headers, COOKIE)?;
    if token.is_empty() || token.len() > 100 {
        return None;
    }
    let hash = sha256_hex(&token);
    let idle = state.idle_ttl;
    let t = now();
    state
        .db
        .run(move |c| {
            let row = c
                .query_row(
                    "SELECT s.user_id, s.expires_at, s.last_seen, s.restricted, u.disabled FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?1",
                    [&hash],
                    |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, i64>(4)?)),
                )
                .optional()?;
            let Some((user_id, expires_at, last_seen, restricted, disabled)) = row else { return Ok(None) };
            if expires_at <= t || t - last_seen > idle || disabled != 0 {
                c.execute("DELETE FROM sessions WHERE token_hash = ?1", [&hash])?;
                return Ok(None);
            }
            if t - last_seen > 60 {
                c.execute("UPDATE sessions SET last_seen = ?1 WHERE token_hash = ?2", params![t, hash])?;
            }
            let Some(user) = load_user(c, user_id)? else { return Ok(None) };
            Ok(Some(Session { token_hash: hash, user, restricted: restricted != 0, expires_at }))
        })
        .await
        .ok()
        .flatten()
}

pub fn session_cookie(state: &AppState, value: &str, max_age: i64) -> HeaderValue {
    let secure = if state.secure_cookie { "; Secure" } else { "" };
    HeaderValue::from_str(&format!("{COOKIE}={value}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}{secure}")).expect("cookie")
}

pub fn hash_password(password: &str) -> String {
    let salt = SaltString::generate(&mut argon2::password_hash::rand_core::OsRng);
    Argon2::default().hash_password(password.as_bytes(), &salt).expect("argon2 hash").to_string()
}

fn verify_password(password: &str, hash: &str) -> bool {
    PasswordHash::new(hash).map(|h| Argon2::default().verify_password(password.as_bytes(), &h).is_ok()).unwrap_or(false)
}

pub fn password_problem(password: &str, username: &str) -> Option<&'static str> {
    let len = password.chars().count();
    if len < 12 {
        return Some("Use at least 12 characters.");
    }
    if len > 128 {
        return Some("Use at most 128 characters.");
    }
    if !username.is_empty() && password.to_lowercase().contains(&username.to_lowercase()) {
        return Some("The password must not contain the user ID.");
    }
    let classes = [
        password.chars().any(|c| c.is_lowercase()),
        password.chars().any(|c| c.is_uppercase()),
        password.chars().any(|c| c.is_ascii_digit()),
        password.chars().any(|c| !c.is_alphanumeric()),
    ]
    .iter()
    .filter(|b| **b)
    .count();
    if classes < 3 && len < 16 {
        return Some("Mix at least three of lower case, upper case, digits and symbols, or use 16 or more characters.");
    }
    None
}

#[derive(Deserialize, Clone, Copy)]
pub struct Location {
    lat: f64,
    lon: f64,
    #[serde(default)]
    accuracy: Option<f64>,
}

#[derive(Deserialize)]
pub struct LoginBody {
    username: String,
    password: String,
    #[serde(default)]
    location: Option<Location>,
}

struct Account {
    id: i64,
    hash: String,
    disabled: bool,
    locked_until: i64,
    must_change: bool,
    division_id: String,
    username: String,
}

pub async fn login(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(body): Json<LoginBody>) -> Response {
    let ip = client_ip(state.trust_proxy, addr, &headers).to_string();
    let cc = country(&headers);
    if !same_origin_request(&headers) {
        return error(StatusCode::FORBIDDEN, "Request blocked.");
    }
    if !state.login_limit.allow(client_ip(state.trust_proxy, addr, &headers)) {
        warn!(%ip, "login rate limited");
        return error(StatusCode::TOO_MANY_REQUESTS, "Too many attempts from this network. Wait a minute and try again.");
    }
    let username = body.username.trim().to_string();
    if username.is_empty() || username.len() > 64 || body.password.is_empty() || body.password.len() > 256 {
        return error(StatusCode::UNAUTHORIZED, FAILED);
    }
    let source = (username.to_ascii_lowercase(), source_key(client_ip(state.trust_proxy, addr, &headers)));
    if !reserve_attempt(&state, &source, now()) {
        warn!(%ip, user = %username, "sign-in paused for this ID from this network");
        let (ip2, cc2, name) = (ip.clone(), cc.clone(), username.clone());
        let _ = state.db.run(move |c| audit(c, None, Some(&name), None, "login", None, &ip2, cc2.as_deref(), "locked_source", None)).await;
        return error(StatusCode::UNAUTHORIZED, FAILED);
    }
    let lookup = username.clone();
    let account = state
        .db
        .run(move |c| {
            c.query_row(
                "SELECT id, password_hash, disabled, failed, locked_until, must_change, division_id, username FROM users WHERE username = ?1",
                [&lookup],
                |r| {
                    Ok(Account {
                        id: r.get(0)?,
                        hash: r.get(1)?,
                        disabled: r.get::<_, i64>(2)? != 0,
                        locked_until: r.get(4)?,
                        must_change: r.get::<_, i64>(5)? != 0,
                        division_id: r.get(6)?,
                        username: r.get(7)?,
                    })
                },
            )
            .optional()
        })
        .await
        .ok()
        .flatten();
    let hash = account.as_ref().map(|a| a.hash.clone()).unwrap_or_else(|| state.dummy_hash.clone());
    let password = body.password.clone();
    let ok = tokio::task::spawn_blocking(move || verify_password(&password, &hash)).await.unwrap_or(false);
    let t = now();
    settle_attempt(&state, &source, ok, t);
    let Some(acc) = account else {
        let (ip2, cc2, name) = (ip.clone(), cc.clone(), username.clone());
        let _ = state.db.run(move |c| audit(c, None, Some(&name), None, "login", None, &ip2, cc2.as_deref(), "unknown_user", None)).await;
        return error(StatusCode::UNAUTHORIZED, FAILED);
    };
    if acc.locked_until > t {
        let (ip2, cc2) = (ip.clone(), cc.clone());
        let (id, name, div) = (acc.id, acc.username.clone(), acc.division_id.clone());
        let _ = state.db.run(move |c| audit(c, Some(id), Some(&name), Some(&div), "login", None, &ip2, cc2.as_deref(), "locked", None)).await;
        return error(StatusCode::UNAUTHORIZED, FAILED);
    }
    if !ok {
        let (ip2, cc2) = (ip.clone(), cc.clone());
        let (id, name, div) = (acc.id, acc.username.clone(), acc.division_id.clone());
        let (limit, lock_for) = (state.lockout_account_after, state.lockout_secs);
        let _ = state
            .db
            .run(move |c| {
                let failed: i64 = c.query_row("UPDATE users SET failed = failed + 1 WHERE id = ?1 RETURNING failed", [id], |r| r.get(0))?;
                let locked = failed >= limit;
                if locked {
                    c.execute("UPDATE users SET failed = 0, locked_until = ?1 WHERE id = ?2", params![t + lock_for, id])?;
                }
                audit(c, Some(id), Some(&name), Some(&div), "login", None, &ip2, cc2.as_deref(), if locked { "locked_now" } else { "bad_password" }, None)
            })
            .await;
        return error(StatusCode::UNAUTHORIZED, FAILED);
    }
    if acc.disabled {
        let (ip2, cc2) = (ip.clone(), cc.clone());
        let (id, name, div) = (acc.id, acc.username.clone(), acc.division_id.clone());
        let _ = state.db.run(move |c| audit(c, Some(id), Some(&name), Some(&div), "login", None, &ip2, cc2.as_deref(), "disabled", None)).await;
        return error(StatusCode::UNAUTHORIZED, FAILED);
    }
    let mut where_detail = None;
    if state.device_required {
        if let Some(fence) = &state.geofence {
            let verdict = match body.location {
                Some(l) => {
                    where_detail = Some(format!("{:.3},{:.3} ±{:.0} m", l.lat, l.lon, l.accuracy.unwrap_or(-1.0)));
                    fence.admits(l.lon, l.lat, state.geofence_margin_m)
                }
                None => false,
            };
            if !verdict {
                let outcome = if body.location.is_some() { "outside_area" } else { "no_location" };
                let msg = if body.location.is_some() {
                    format!("Sign-in is only allowed from within {}. Your device reports a location outside it.", fence.name)
                } else {
                    format!("Allow location access so we can confirm you are signing in from within {}.", fence.name)
                };
                let (ip2, cc2, det) = (ip.clone(), cc.clone(), where_detail.clone());
                let (id, name, div) = (acc.id, acc.username.clone(), acc.division_id.clone());
                let _ = state.db.run(move |c| audit(c, Some(id), Some(&name), Some(&div), "login", None, &ip2, cc2.as_deref(), outcome, det.as_deref())).await;
                return error(StatusCode::FORBIDDEN, &msg);
            }
        }
    }
    let token = random_token();
    let token_hash = sha256_hex(&token);
    let ua = user_agent(&headers);
    let expires = t + state.absolute_ttl;
    let (ip2, cc2) = (ip.clone(), cc.clone());
    let (id, name, div, restricted) = (acc.id, acc.username.clone(), acc.division_id.clone(), acc.must_change && state.require_password_change);
    let user = state
        .db
        .run(move |c| {
            c.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, last_seen, expires_at, ip, user_agent, restricted) VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7)",
                params![token_hash, id, t, expires, ip2, ua, restricted as i64],
            )?;
            c.execute("UPDATE users SET failed = 0, locked_until = 0, last_login_at = ?1 WHERE id = ?2", params![t, id])?;
            c.execute("DELETE FROM sessions WHERE expires_at <= ?1", [t])?;
            audit(c, Some(id), Some(&name), Some(&div), "login", None, &ip2, cc2.as_deref(), "ok", where_detail.as_deref())?;
            load_user(c, id)
        })
        .await;
    let Ok(Some(user)) = user else { return error(StatusCode::INTERNAL_SERVER_ERROR, "Could not start a session.") };
    let mut res = Json(serde_json::json!({ "user": user, "must_change": restricted })).into_response();
    res.headers_mut().insert(header::SET_COOKIE, session_cookie(&state, &token, state.absolute_ttl));
    res
}

#[derive(Deserialize)]
pub struct PasswordBody {
    current: String,
    new: String,
}

pub async fn change_password(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap, Json(body): Json<PasswordBody>) -> Response {
    if !same_origin_request(&headers) {
        return error(StatusCode::FORBIDDEN, "Request blocked.");
    }
    let Some(session) = current(&state, &headers).await else { return error(StatusCode::UNAUTHORIZED, "Sign in to continue.") };
    if !state.login_limit.allow(client_ip(state.trust_proxy, addr, &headers)) {
        return error(StatusCode::TOO_MANY_REQUESTS, "Too many attempts. Wait a minute and try again.");
    }
    let ip = client_ip(state.trust_proxy, addr, &headers).to_string();
    let cc = country(&headers);
    let uid = session.user.id;
    let stored: Option<String> = state.db.run(move |c| c.query_row("SELECT password_hash FROM users WHERE id = ?1", [uid], |r| r.get(0)).optional()).await.ok().flatten();
    let Some(stored) = stored else { return error(StatusCode::UNAUTHORIZED, "Sign in to continue.") };
    let current_pw = body.current.clone();
    let ok = tokio::task::spawn_blocking(move || verify_password(&current_pw, &stored)).await.unwrap_or(false);
    if !ok {
        let (name, div, ip2, cc2) = (session.user.username.clone(), session.user.division.id.clone(), ip.clone(), cc.clone());
        let _ = state.db.run(move |c| audit(c, Some(uid), Some(&name), Some(&div), "password_change", None, &ip2, cc2.as_deref(), "bad_current", None)).await;
        return error(StatusCode::BAD_REQUEST, "The current password is incorrect.");
    }
    if body.new == body.current {
        return error(StatusCode::BAD_REQUEST, "Choose a password different from the current one.");
    }
    if let Some(problem) = password_problem(&body.new, &session.user.username) {
        return error(StatusCode::BAD_REQUEST, problem);
    }
    let new_pw = body.new.clone();
    let new_hash = tokio::task::spawn_blocking(move || hash_password(&new_pw)).await.unwrap_or_default();
    if new_hash.is_empty() {
        return error(StatusCode::INTERNAL_SERVER_ERROR, "Could not update the password.");
    }
    let t = now();
    let (name, div, keep) = (session.user.username.clone(), session.user.division.id.clone(), session.token_hash.clone());
    let user = state
        .db
        .run(move |c| {
            c.execute("UPDATE users SET password_hash = ?1, must_change = 0, password_changed_at = ?2 WHERE id = ?3", params![new_hash, t, uid])?;
            c.execute("DELETE FROM sessions WHERE user_id = ?1 AND token_hash <> ?2", params![uid, keep])?;
            c.execute("UPDATE sessions SET restricted = 0 WHERE token_hash = ?1", [&keep])?;
            audit(c, Some(uid), Some(&name), Some(&div), "password_change", None, &ip, cc.as_deref(), "ok", None)?;
            load_user(c, uid)
        })
        .await;
    match user {
        Ok(Some(user)) => Json(serde_json::json!({ "user": user })).into_response(),
        _ => error(StatusCode::INTERNAL_SERVER_ERROR, "Could not update the password."),
    }
}

pub async fn logout(State(state): State<Shared>, ConnectInfo(addr): ConnectInfo<SocketAddr>, headers: HeaderMap) -> Response {
    if !same_origin_request(&headers) {
        return error(StatusCode::FORBIDDEN, "Request blocked.");
    }
    if let Some(s) = current(&state, &headers).await {
        let ip = client_ip(state.trust_proxy, addr, &headers).to_string();
        let cc = country(&headers);
        let _ = state
            .db
            .run(move |c| {
                c.execute("DELETE FROM sessions WHERE token_hash = ?1", [&s.token_hash])?;
                audit(c, Some(s.user.id), Some(&s.user.username), Some(&s.user.division.id), "logout", None, &ip, cc.as_deref(), "ok", None)
            })
            .await;
    }
    let mut res = StatusCode::NO_CONTENT.into_response();
    res.headers_mut().insert(header::SET_COOKIE, session_cookie(&state, "", 0));
    res
}

pub async fn me(State(state): State<Shared>, headers: HeaderMap) -> Response {
    match current(&state, &headers).await {
        Some(s) => Json(serde_json::json!({ "user": s.user, "must_change": s.restricted, "expires_at": s.expires_at,
                                            "idle_timeout_s": state.idle_ttl }))
        .into_response(),
        None => error(StatusCode::UNAUTHORIZED, "Sign in to continue."),
    }
}

pub async fn policy(State(state): State<Shared>) -> Response {
    Json(serde_json::json!({
        "device_location_required": state.device_required && state.geofence.is_some(),
        "boundary": state.geofence.as_ref().map(|g| g.name.clone()),
        "lockout_after": state.lockout_after,
        "lockout_minutes": state.lockout_secs / 60,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_policy() {
        assert!(password_problem("short1A!", "ps-x").is_some());
        assert!(password_problem("ps-cubbon-park-2026", "ps-cubbon-park").is_some());
        assert!(password_problem("alllowercaseletters", "u").is_none());
        assert!(password_problem("Lowercase1234", "u").is_none());
        assert!(password_problem("lowercaseonly", "u").is_some());
    }

    #[test]
    fn hashes_verify() {
        let h = hash_password("Correct-Horse-9");
        assert!(verify_password("Correct-Horse-9", &h));
        assert!(!verify_password("correct-horse-9", &h));
    }
}
