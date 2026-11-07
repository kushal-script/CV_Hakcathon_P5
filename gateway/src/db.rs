use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use rusqlite::{params, Connection};

const SCHEMA_V1: &str = "
CREATE TABLE divisions (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL
);
CREATE TABLE stations (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL,
  division_id TEXT NOT NULL REFERENCES divisions(id),
  official_division TEXT,
  lat REAL,
  lon REAL
);
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('station', 'division_admin')),
  station_id TEXT REFERENCES stations(id),
  division_id TEXT NOT NULL REFERENCES divisions(id),
  display_name TEXT NOT NULL,
  disabled INTEGER NOT NULL DEFAULT 0,
  must_change INTEGER NOT NULL DEFAULT 1,
  failed INTEGER NOT NULL DEFAULT 0,
  locked_until INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  password_changed_at INTEGER,
  last_login_at INTEGER,
  CHECK ((role = 'station') = (station_id IS NOT NULL))
);
CREATE INDEX users_division ON users(division_id);
CREATE TABLE sessions (
  token_hash TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  ip TEXT,
  user_agent TEXT,
  restricted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX sessions_user ON sessions(user_id);
CREATE TABLE audit (
  id INTEGER PRIMARY KEY,
  ts INTEGER NOT NULL,
  actor_id INTEGER,
  actor TEXT,
  division_id TEXT,
  action TEXT NOT NULL,
  target TEXT,
  ip TEXT,
  country TEXT,
  outcome TEXT NOT NULL,
  detail TEXT
);
CREATE INDEX audit_division ON audit(division_id, id);
";

#[derive(Clone)]
pub struct Db(Arc<Inner>);

struct Inner {
    conns: Vec<Mutex<Connection>>,
    next: AtomicUsize,
}

impl Db {
    pub fn open(path: &Path, size: usize) -> rusqlite::Result<Db> {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut conns = Vec::new();
        for i in 0..size.max(1) {
            let c = Connection::open(path)?;
            c.busy_timeout(Duration::from_secs(5))?;
            c.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))?;
            c.execute_batch("PRAGMA foreign_keys = ON; PRAGMA synchronous = NORMAL;")?;
            if i == 0 {
                migrate(&c)?;
            }
            conns.push(Mutex::new(c));
        }
        Ok(Db(Arc::new(Inner { conns, next: AtomicUsize::new(0) })))
    }

    pub fn with<T>(&self, f: impl FnOnce(&Connection) -> rusqlite::Result<T>) -> rusqlite::Result<T> {
        let i = self.0.next.fetch_add(1, Ordering::Relaxed) % self.0.conns.len();
        let guard = self.0.conns[i].lock().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    pub async fn run<T, F>(&self, f: F) -> Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> rusqlite::Result<T> + Send + 'static,
    {
        let db = self.clone();
        tokio::task::spawn_blocking(move || db.with(f)).await.map_err(|e| e.to_string())?.map_err(|e| e.to_string())
    }
}

fn migrate(c: &Connection) -> rusqlite::Result<()> {
    let version: i64 = c.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version < 1 {
        let tx = c.unchecked_transaction()?;
        tx.execute_batch(SCHEMA_V1)?;
        tx.execute_batch("PRAGMA user_version = 1")?;
        tx.commit()?;
    }
    if version < 2 {
        let tx = c.unchecked_transaction()?;
        tx.execute_batch(
            "ALTER TABLE users ADD COLUMN demo INTEGER NOT NULL DEFAULT 0;
             ALTER TABLE users ADD COLUMN all_divisions INTEGER NOT NULL DEFAULT 0;
             PRAGMA user_version = 2",
        )?;
        tx.commit()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn audit(
    c: &Connection,
    actor_id: Option<i64>,
    actor: Option<&str>,
    division_id: Option<&str>,
    action: &str,
    target: Option<&str>,
    ip: &str,
    country: Option<&str>,
    outcome: &str,
    detail: Option<&str>,
) -> rusqlite::Result<()> {
    c.execute(
        "INSERT INTO audit (ts, actor_id, actor, division_id, action, target, ip, country, outcome, detail) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![crate::util::now(), actor_id, actor, division_id, action, target, ip, country, outcome, detail],
    )?;
    Ok(())
}
