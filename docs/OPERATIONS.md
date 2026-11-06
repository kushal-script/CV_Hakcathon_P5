# Operating Vahana Drishti

How the console is deployed, how accounts work, and what to do when something needs changing. Commands assume a Linux host with Docker and the container named `vahana`.

## What runs where

| Piece | Where | Notes |
|---|---|---|
| Container `vahana` | Host VM, bound to `127.0.0.1:8080` | Rust gateway on 8080 in front of the Python service on 127.0.0.1:8001 inside the container |
| `/app/state` | Host folder mounted into the container | `gateway.db`: divisions, stations, accounts, sessions, audit log |
| `/app/data` | Host folder mounted into the container | `vahana.db` (records and case index) and `cases/` (case files and uploaded footage) |
| HTTPS | A Cloudflare tunnel service on the host | Forwards the public address to `127.0.0.1:8080`. The container is never exposed directly |

The container runs as a non-root user on a read-only filesystem with every Linux capability dropped. Only the two mounted folders and a temporary `/tmp` are writable.

## First deployment

```bash
docker build -t vahana-drishti .
docker run -d --name vahana --restart unless-stopped -p 127.0.0.1:8080:8080 \
  -e P5_COOKIE_SECURE=true -e P5_TRUST_PROXY=true -e P5_DEVICE_GEOFENCE=required -e P5_ALLOWED_COUNTRIES=IN \
  -v "$PWD/state:/app/state" -v "$PWD/data:/app/data" \
  --read-only --tmpfs /tmp:rw,size=1g --cap-drop ALL --security-opt no-new-privileges:true vahana-drishti
docker exec vahana vahana-gateway provision --stations /app/assets/bengaluru/police_stations.json --credentials /app/state/credentials.csv
```

`provision` creates the five divisions, one account per police station (its ID is the station ID, for example `ps-cubbon-park` for the law-and-order station or `tps-cubbon-park` for the traffic station) and one administrator per division (`admin-central`, `admin-north`, `admin-south`, `admin-east`, `admin-west`). It writes every one-time password to a new file readable only by its owner and refuses to overwrite an existing file. Copy that file to a safe place, hand each station its own line through a trusted channel, and delete the copy on the server.

Every account must choose its own password at the first sign-in.

## Accounts

| Task | How |
|---|---|
| A station forgot its password | Its division administrator opens Division, then Stations and accounts, then Reset password. The one-time password is shown once |
| An administrator forgot their password | On the host: `docker exec vahana vahana-gateway reset-password admin-central` prints a new one-time password |
| A station's account must be stopped | Division administrator: Disable. All of its sessions end at once |
| An account is locked after 5 wrong passwords | It unlocks by itself after 15 minutes, or the division administrator can Unlock it |
| New stations were added to the register | Rebuild `assets/bengaluru/police_stations.json`, rebuild the image, then run `provision` again with a new credentials file name. Existing accounts are kept and only new stations get accounts |

The audit log (Division, then Audit log) shows every sign-in, failure, password change and account action for the division, with the network address, country and the device location given at sign-in.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `P5_DEVICE_GEOFENCE` | `required` | `required`: sign-in needs a device location inside the Bengaluru boundary (with `P5_GEOFENCE_MARGIN_M`, default 2000). `off`: no location check |
| `P5_ALLOWED_COUNTRIES` | `IN` | Countries allowed by Cloudflare's country header when `P5_TRUST_PROXY=true`. Empty turns it off |
| `P5_TRUST_PROXY` | `false` | Set `true` behind the tunnel so the real client address and country are used |
| `P5_COOKIE_SECURE` | `true` | Keep `true` whenever the site is served over HTTPS |
| `P5_IDLE_MINUTES`, `P5_SESSION_HOURS` | 60, 12 | Session idle and absolute lifetimes |
| `P5_LOCKOUT_AFTER`, `P5_LOCKOUT_MINUTES` | 5, 15 | Account lockout |
| `P5_LOGIN_PER_MINUTE`, `P5_API_PER_MINUTE` | 6, 240 | Per-address rate limits |

For a demonstration outside Bengaluru, recreate the container with `-e P5_DEVICE_GEOFENCE=off`. Everything else stays restricted.

## Updating the software

```bash
docker build -t vahana-drishti .
docker rm -f vahana
docker run ...   # the same command as above; the mounted folders keep all accounts and records
```

Sessions are stored in `gateway.db`, so signed-in users stay signed in across an update.

## Backups

Both databases use SQLite in WAL mode. Copy them consistently with the SQLite backup command (the `sqlite3` package) while the container runs:

```bash
sqlite3 state/gateway.db ".backup state-backup.db"
sqlite3 data/vahana.db ".backup data-backup.db"
```

Also copy `data/cases/` for case files and footage.

## The public address

The tunnel service prints its address to its log on start. A quick tunnel gets a new address whenever it restarts. For a permanent address, create a named tunnel in a Cloudflare account with a domain and point it at `http://127.0.0.1:8080`.
