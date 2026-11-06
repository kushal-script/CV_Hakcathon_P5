use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
};

use rusqlite::{params, OptionalExtension};
use serde::Deserialize;

use crate::{auth::hash_password, db::Db, util::{now, temp_password}};

#[derive(Deserialize)]
struct StationsFile {
    divisions: Vec<DivisionIn>,
    stations: Vec<StationIn>,
}

#[derive(Deserialize)]
struct DivisionIn {
    id: String,
    name: String,
}

#[derive(Deserialize)]
struct StationIn {
    id: String,
    name: String,
    kind: String,
    division: String,
    #[serde(default)]
    official_division: Option<String>,
    #[serde(default)]
    lat: Option<f64>,
    #[serde(default)]
    lon: Option<f64>,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn valid_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 64 && id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

fn csv_field(v: &str) -> String {
    if v.contains([',', '"', '\n']) {
        format!("\"{}\"", v.replace('"', "\"\""))
    } else {
        v.to_string()
    }
}

pub fn provision(args: &[String]) -> Result<(), String> {
    let db_path = PathBuf::from(arg(args, "--db").unwrap_or_else(|| crate::util::env_or("P5_DB", "/app/state/gateway.db")));
    let stations_path = arg(args, "--stations").ok_or("--stations <police_stations.json> is required")?;
    let creds_path = arg(args, "--credentials").ok_or("--credentials <new csv file> is required")?;
    let text = std::fs::read_to_string(&stations_path).map_err(|e| format!("{stations_path}: {e}"))?;
    let data: StationsFile = serde_json::from_str(&text).map_err(|e| format!("{stations_path}: {e}"))?;
    let divisions: Vec<&str> = data.divisions.iter().map(|d| d.id.as_str()).collect();
    for s in &data.stations {
        if !valid_id(&s.id) {
            return Err(format!("station id {:?} is not a lowercase slug", s.id));
        }
        if !divisions.contains(&s.division.as_str()) {
            return Err(format!("station {} has unknown division {}", s.id, s.division));
        }
    }
    let db = Db::open(&db_path, 1).map_err(|e| e.to_string())?;
    let t = now();
    let mut created: Vec<(String, String, String, String, String)> = Vec::new();
    let mut synced = 0usize;
    db.with(|c| {
        let tx = c.unchecked_transaction()?;
        for d in &data.divisions {
            tx.execute(
                "INSERT INTO divisions (id, name) VALUES (?1, ?2) ON CONFLICT(id) DO UPDATE SET name = excluded.name",
                params![d.id, d.name],
            )?;
        }
        for s in &data.stations {
            tx.execute(
                "INSERT INTO stations (id, name, kind, division_id, official_division, lat, lon) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(id) DO UPDATE SET name = excluded.name, kind = excluded.kind, division_id = excluded.division_id,
                   official_division = excluded.official_division, lat = excluded.lat, lon = excluded.lon",
                params![s.id, s.name, s.kind, s.division, s.official_division, s.lat, s.lon],
            )?;
            let exists: Option<i64> = tx.query_row("SELECT id FROM users WHERE station_id = ?1", [&s.id], |r| r.get(0)).optional()?;
            match exists {
                Some(uid) => {
                    tx.execute("UPDATE users SET division_id = ?1, display_name = ?2 WHERE id = ?3", params![s.division, s.name, uid])?;
                    synced += 1;
                }
                None => {
                    let pw = temp_password();
                    tx.execute(
                        "INSERT INTO users (username, password_hash, role, station_id, division_id, display_name, must_change, created_at)
                         VALUES (?1, ?2, 'station', ?3, ?4, ?5, 1, ?6)",
                        params![s.id, hash_password(&pw), s.id, s.division, s.name, t],
                    )?;
                    created.push((s.id.clone(), pw, "station".into(), s.name.clone(), s.division.clone()));
                }
            }
        }
        for d in &data.divisions {
            let username = format!("admin-{}", d.id);
            let exists: Option<i64> = tx.query_row("SELECT id FROM users WHERE username = ?1", [&username], |r| r.get(0)).optional()?;
            if exists.is_none() {
                let pw = temp_password();
                let display = format!("{} Division Administrator", d.name);
                tx.execute(
                    "INSERT INTO users (username, password_hash, role, station_id, division_id, display_name, must_change, created_at)
                     VALUES (?1, ?2, 'division_admin', NULL, ?3, ?4, 1, ?5)",
                    params![username, hash_password(&pw), d.id, display, t],
                )?;
                created.push((username, pw, "division_admin".into(), display, d.id.clone()));
            }
        }
        tx.commit()
    })
    .map_err(|e| e.to_string())?;
    if !created.is_empty() {
        write_credentials(Path::new(&creds_path), &created)?;
    }
    println!(
        "divisions: {}, stations: {}, new accounts: {} (written to {}), existing station accounts synced: {}",
        data.divisions.len(),
        data.stations.len(),
        created.len(),
        if created.is_empty() { "nothing" } else { creds_path.as_str() },
        synced
    );
    Ok(())
}

fn write_credentials(path: &Path, rows: &[(String, String, String, String, String)]) -> Result<(), String> {
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path).map_err(|e| format!("{}: {e} (refusing to overwrite an existing credentials file)", path.display()))?;
    writeln!(f, "username,temporary_password,role,name,division").map_err(|e| e.to_string())?;
    for (u, p, r, n, d) in rows {
        writeln!(f, "{},{},{},{},{}", csv_field(u), csv_field(p), csv_field(r), csv_field(n), csv_field(d)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn reset_password(args: &[String]) -> Result<(), String> {
    let db_path = PathBuf::from(arg(args, "--db").unwrap_or_else(|| crate::util::env_or("P5_DB", "/app/state/gateway.db")));
    let username = args.iter().skip(2).find(|a| !a.starts_with("--") && Some(a.as_str()) != arg(args, "--db").as_deref()).ok_or("username is required")?;
    let db = Db::open(&db_path, 1).map_err(|e| e.to_string())?;
    let pw = temp_password();
    let hash = hash_password(&pw);
    let n = db
        .with(|c| {
            let n = c.execute(
                "UPDATE users SET password_hash = ?1, must_change = 1, failed = 0, locked_until = 0, disabled = 0 WHERE username = ?2",
                params![hash, username],
            )?;
            c.execute("DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE username = ?1)", [username])?;
            crate::db::audit(c, None, Some("operator"), None, "account_reset_cli", Some(username), "local", None, if n > 0 { "ok" } else { "unknown_user" }, None)?;
            Ok(n)
        })
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("no account named {username}"));
    }
    println!("{pw}");
    Ok(())
}
