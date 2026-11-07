use std::{collections::HashMap, net::SocketAddr};

use axum::{
    extract::{ConnectInfo, Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use rusqlite::{params, OptionalExtension};
use serde::Serialize;

use crate::{
    auth::{current, hash_password, Session},
    db::audit,
    util::{client_ip, country, error, now, same_origin_request, temp_password},
    Shared,
};

async fn require_admin(state: &Shared, headers: &HeaderMap, mutating: bool) -> Result<Session, Response> {
    if mutating && !same_origin_request(headers) {
        return Err(error(StatusCode::FORBIDDEN, "Request blocked."));
    }
    match current(state, headers).await {
        Some(s) if !s.restricted && s.user.role == "division_admin" => Ok(s),
        Some(s) if s.restricted => Err(error(StatusCode::FORBIDDEN, "Change your password first.")),
        Some(_) => Err(error(StatusCode::FORBIDDEN, "Division administrators only.")),
        None => Err(error(StatusCode::UNAUTHORIZED, "Sign in to continue.")),
    }
}

#[derive(Serialize)]
struct Account {
    id: i64,
    username: String,
    display_name: String,
    role: String,
    station_id: Option<String>,
    station_name: Option<String>,
    station_kind: Option<String>,
    disabled: bool,
    must_change: bool,
    demo: bool,
    locked: bool,
    failed: i64,
    last_login_at: Option<i64>,
    password_changed_at: Option<i64>,
    active_sessions: i64,
}

pub async fn accounts(State(state): State<Shared>, headers: HeaderMap) -> Response {
    let admin = match require_admin(&state, &headers, false).await {
        Ok(s) => s,
        Err(r) => return r,
    };
    let div = admin.user.division.id.clone();
    let all = admin.user.all_divisions;
    let t = now();
    let rows = state
        .db
        .run(move |c| {
            let mut stmt = c.prepare(
                "SELECT u.id, u.username, u.display_name, u.role, u.station_id, st.name, st.kind, u.disabled, u.must_change, u.locked_until, u.failed,
                        u.last_login_at, u.password_changed_at,
                        (SELECT COUNT(*) FROM sessions s WHERE s.user_id = u.id AND s.expires_at > ?2), u.demo
                 FROM users u LEFT JOIN stations st ON st.id = u.station_id
                 WHERE (?3 OR u.division_id = ?1) ORDER BY u.role DESC, st.kind, st.name",
            )?;
            let rows = stmt
                .query_map(params![div, t, all], |r| {
                    Ok(Account {
                        id: r.get(0)?,
                        username: r.get(1)?,
                        display_name: r.get(2)?,
                        role: r.get(3)?,
                        station_id: r.get(4)?,
                        station_name: r.get(5)?,
                        station_kind: r.get(6)?,
                        disabled: r.get::<_, i64>(7)? != 0,
                        must_change: r.get::<_, i64>(8)? != 0,
                        demo: r.get::<_, i64>(14)? != 0,
                        locked: r.get::<_, i64>(9)? > t,
                        failed: r.get(10)?,
                        last_login_at: r.get(11)?,
                        password_changed_at: r.get(12)?,
                        active_sessions: r.get(13)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .await;
    match rows {
        Ok(rows) => Json(serde_json::json!({ "accounts": rows })).into_response(),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "Could not load accounts."),
    }
}

pub async fn account_action(
    State(state): State<Shared>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    Path((id, action)): Path<(i64, String)>,
    headers: HeaderMap,
) -> Response {
    let admin = match require_admin(&state, &headers, true).await {
        Ok(s) => s,
        Err(r) => return r,
    };
    if !["reset", "disable", "enable", "unlock", "signout"].contains(&action.as_str()) {
        return error(StatusCode::NOT_FOUND, "Unknown action.");
    }
    if admin.user.demo {
        return error(StatusCode::FORBIDDEN, crate::proxy::READ_ONLY);
    }
    let div = admin.user.division.id.clone();
    let target: Option<(String, String, bool)> = state
        .db
        .run(move |c| {
            c.query_row("SELECT username, role, demo FROM users WHERE id = ?1 AND division_id = ?2", params![id, div], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? != 0))
            })
            .optional()
        })
        .await
        .ok()
        .flatten();
    let Some((username, role, demo)) = target else { return error(StatusCode::NOT_FOUND, "Account not found in your division.") };
    if role != "station" {
        return error(StatusCode::FORBIDDEN, "Only station accounts can be managed here.");
    }
    if demo {
        return error(StatusCode::FORBIDDEN, "Demo accounts are managed from the server console so that their published passwords keep working.");
    }
    let temp = if action == "reset" { Some(temp_password()) } else { None };
    let new_hash = match &temp {
        Some(pw) => {
            let pw = pw.clone();
            Some(tokio::task::spawn_blocking(move || hash_password(&pw)).await.unwrap_or_default())
        }
        None => None,
    };
    let ip = client_ip(state.trust_proxy, addr, &headers).to_string();
    let cc = country(&headers);
    let (aid, aname, adiv, act, uname) = (admin.user.id, admin.user.username.clone(), admin.user.division.id.clone(), action.clone(), username.clone());
    let t = now();
    let done = state
        .db
        .run(move |c| {
            let tx = c.unchecked_transaction()?;
            match act.as_str() {
                "reset" => {
                    tx.execute(
                        "UPDATE users SET password_hash = ?1, must_change = 1, failed = 0, locked_until = 0, password_changed_at = ?2 WHERE id = ?3",
                        params![new_hash.unwrap_or_default(), t, id],
                    )?;
                    tx.execute("DELETE FROM sessions WHERE user_id = ?1", [id])?;
                }
                "disable" => {
                    tx.execute("UPDATE users SET disabled = 1 WHERE id = ?1", [id])?;
                    tx.execute("DELETE FROM sessions WHERE user_id = ?1", [id])?;
                }
                "enable" => {
                    tx.execute("UPDATE users SET disabled = 0 WHERE id = ?1", [id])?;
                }
                "unlock" => {
                    tx.execute("UPDATE users SET failed = 0, locked_until = 0 WHERE id = ?1", [id])?;
                }
                _ => {
                    tx.execute("DELETE FROM sessions WHERE user_id = ?1", [id])?;
                }
            }
            audit(&tx, Some(aid), Some(&aname), Some(&adiv), &format!("account_{act}"), Some(&uname), &ip, cc.as_deref(), "ok", None)?;
            tx.commit()
        })
        .await;
    match done {
        Ok(()) => Json(serde_json::json!({ "ok": true, "username": username, "temporary_password": temp })).into_response(),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "Could not update the account."),
    }
}

#[derive(Serialize)]
struct AuditRow {
    id: i64,
    ts: i64,
    actor: Option<String>,
    action: String,
    target: Option<String>,
    ip: Option<String>,
    country: Option<String>,
    outcome: String,
    detail: Option<String>,
}

pub async fn audit_log(State(state): State<Shared>, headers: HeaderMap, Query(q): Query<HashMap<String, String>>) -> Response {
    let admin = match require_admin(&state, &headers, false).await {
        Ok(s) => s,
        Err(r) => return r,
    };
    let limit: i64 = q.get("limit").and_then(|v| v.parse().ok()).unwrap_or(100).clamp(1, 500);
    let before: i64 = q.get("before").and_then(|v| v.parse().ok()).unwrap_or(i64::MAX);
    let div = admin.user.division.id.clone();
    let (all, hide_ip) = (admin.user.all_divisions, admin.user.demo);
    let rows = state
        .db
        .run(move |c| {
            let mut stmt = c.prepare(
                "SELECT id, ts, actor, action, target, ip, country, outcome, detail FROM audit WHERE (?4 OR division_id = ?1) AND id < ?2 ORDER BY id DESC LIMIT ?3",
            )?;
            let rows = stmt
                .query_map(params![div, before, limit, all], |r| {
                    Ok(AuditRow {
                        id: r.get(0)?,
                        ts: r.get(1)?,
                        actor: r.get(2)?,
                        action: r.get(3)?,
                        target: r.get(4)?,
                        ip: if hide_ip { None } else { r.get(5)? },
                        country: r.get(6)?,
                        outcome: r.get(7)?,
                        detail: r.get(8)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(rows)
        })
        .await;
    match rows {
        Ok(rows) => Json(serde_json::json!({ "audit": rows })).into_response(),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "Could not load the audit log."),
    }
}
