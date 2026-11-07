# Operating Vahana Drishti

How the console is deployed, how accounts work, and what to do when something needs changing. Commands assume a Linux host with Docker and the container named `vahana`.

## What runs where

| Piece | Where | Notes |
|---|---|---|
| Container `vahana` | Host VM, bound to `127.0.0.1:8080` | Rust gateway on 8080 in front of the Python service on 127.0.0.1:8001 inside the container |
| `/app/state` | Host folder mounted into the container | `gateway.db`: divisions, stations, accounts, sessions, audit log |
| `/app/data` | Host folder mounted into the container | `vahana.db` (records and case index) and `cases/` (case files and uploaded footage) |
| HTTPS | Cloudflare tunnel services on the host | Forward the public address to `127.0.0.1:8080`. The container is never exposed directly and the host opens no web ports |
| Backups | `vahana-backup.timer` on the host | Every 6 hours into `/home/ubuntu/backups`, see Backups |
| Watchdog | `vahana-watchdog.timer` on the host | Every minute. Restarts the container when its health check keeps failing and checks every public address |

The container runs as a non-root user on a read-only filesystem with every Linux capability dropped. Only the two mounted folders and a temporary `/tmp` are writable.

## First deployment

```bash
docker build -t vahana-drishti .
docker run -d --name vahana --restart unless-stopped -p 127.0.0.1:8080:8080 \
  -e P5_COOKIE_SECURE=true -e P5_TRUST_PROXY=true -e P5_DEVICE_GEOFENCE=required -e P5_ALLOWED_COUNTRIES=IN \
  -e P5_MAX_VIDEO_MB=90 -e P5_MAX_BODY_MB=95 -e P5_BOUNDARY=/app/assets/bengaluru/signin_area.geojson \
  -e P5_LOGIN_PER_MINUTE=30 -e P5_API_PER_MINUTE=1200 \
  -v "$PWD/state:/app/state" -v "$PWD/data:/app/data" \
  --read-only --tmpfs /tmp:rw,size=1g --cap-drop ALL --security-opt no-new-privileges:true --memory 24g --pids-limit 512 \
  --log-opt max-size=20m --log-opt max-file=5 vahana-drishti
docker exec vahana vahana-gateway provision --stations /app/assets/bengaluru/police_stations.json --credentials /app/state/credentials.csv
sudo install -m 644 deploy/host/*.service deploy/host/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vahana-backup.timer vahana-watchdog.timer
```

The upload limits stay under the 100 MB request limit of Cloudflare's free plan, so an oversized video gets the console's own message instead of an error page from Cloudflare. Container logs are rotated at 5 files of 20 MB.

`provision` creates the five divisions, one account per police station (its ID is the station ID, for example `ps-cubbon-park` for the law-and-order station or `tps-cubbon-park` for the traffic station) and one administrator per division (`admin-central`, `admin-north`, `admin-south`, `admin-east`, `admin-west`). It writes every one-time password to a new file readable only by its owner and refuses to overwrite an existing file. Copy that file to a safe place, hand each station its own line through a trusted channel, and delete the copy on the server.

### Demo accounts

`assets/bengaluru/demo_accounts.json` describes two demo stations (Demo Police Station 1 near MG Road, Central division, and Demo Police Station 2 in Koramangala, South division) and two demo administrators who see every division. Create them, or set new passwords, with one `username password` pair per line on stdin, so the passwords never appear in the shell history or the process list:

```bash
docker exec -i vahana vahana-gateway demo-accounts --file /app/assets/bengaluru/demo_accounts.json < demo_passwords.txt
```

Demo accounts cannot change their own password, administrators cannot reset or disable them from the console, and failed sign-ins pause only the network they come from, never the whole account. The demo administrators can read every division, including the audit log without IP addresses, and can identify photos, but every other change is refused. The published passwords are listed in the README.

An account signs in with its one-time password and can change it under Account. To make every account choose its own password at the first sign-in, set `P5_REQUIRE_PASSWORD_CHANGE=true`.

## Accounts

| Task | How |
|---|---|
| A station forgot its password | Its division administrator opens Division, then Stations and accounts, then Reset password. The one-time password is shown once |
| An administrator forgot their password | On the host: `docker exec vahana vahana-gateway reset-password admin-central` prints a new one-time password |
| A station's account must be stopped | Division administrator: Disable. All of its sessions end at once |
| Sign-in is refused after 5 wrong passwords | It is paused only from that network and resumes by itself after 15 minutes. After 50 failures from all networks together the account locks, and the division administrator can Unlock it |
| New stations were added to the register | Rebuild `assets/bengaluru/police_stations.json`, rebuild the image, then run `provision` again with a new credentials file name. Existing accounts are kept and only new stations get accounts |

The audit log (Division, then Audit log) shows every sign-in, failure, password change and account action for the division, with the network address, country and the device location given at sign-in.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `P5_DEVICE_GEOFENCE` | `required` | `required`: sign-in needs a device location inside the sign-in area (with `P5_GEOFENCE_MARGIN_M`, default 2000). `off`: no location check |
| `P5_BOUNDARY` | `assets/bengaluru/boundary.geojson` | The sign-in area. The live server uses `assets/bengaluru/signin_area.geojson`, Karnataka from OpenStreetMap, rebuilt with `python3 scripts/build_signin_area.py Karnataka`. The console's map, stations and camera checks always use the Bengaluru boundary in `P5_REGION_DIR` |
| `P5_ALLOWED_COUNTRIES` | `IN` | Countries allowed by Cloudflare's country header when `P5_TRUST_PROXY=true`. Empty turns it off |
| `P5_TRUST_PROXY` | `false` | Set `true` behind the tunnel so the real client address and country are used |
| `P5_COOKIE_SECURE` | `true` | Keep `true` whenever the site is served over HTTPS |
| `P5_REQUIRE_PASSWORD_CHANGE` | `false` | `true`: an account still on its one-time password must choose a new password before it can use the console. `false`: the one-time password signs in directly |
| `P5_IDLE_MINUTES`, `P5_SESSION_HOURS` | 60, 12 | Session idle and absolute lifetimes |
| `P5_LOCKOUT_AFTER`, `P5_LOCKOUT_MINUTES`, `P5_LOCKOUT_ACCOUNT_AFTER` | 5, 15, 50 | Failures before sign-in for an ID is paused from one network, the pause length, and failures from all networks before the account locks |
| `P5_LOGIN_PER_MINUTE`, `P5_API_PER_MINUTE` | 6, 240 | Per-address rate limits. A burst of up to the sign-in limit, or a quarter of the API limit (at least 60), is allowed at once. The live server uses 30 and 1200 because evaluators in one room share one public address |
| `P5_MAX_UPLOAD_MB` | 15 | Largest photo |
| `P5_MAX_VIDEO_MB`, `P5_MAX_BODY_MB` | 300, 300 | Largest video, and largest request the gateway accepts. Use 90 and 95 behind Cloudflare's free plan |

To allow sign-in from anywhere in India, recreate the container with `-e P5_DEVICE_GEOFENCE=off`. Everything else stays restricted.

## Travel times

Drive times between sightings come from the Bengaluru profile in `assets/bengaluru/traffic_prior.json` (E20), corrected by journeys observed in the city. The model refits by itself after every case update, dashcam upload, import or deletion. Division administrators see it under Division, then Traffic.

Journeys from any other source (number-plate camera pairs, patrol logs, fleet GPS) can be imported as a CSV file of at most 2 MB and 2,000 rows:

```csv
from_lat,from_lon,to_lat,to_lon,depart,seconds,note
12.9763,77.5929,12.9352,77.6245,2026-11-04T18:05:00,1500,Cubbon Park to Koramangala
```

| Column | Meaning |
|---|---|
| `from_lat`, `from_lon`, `to_lat`, `to_lon` | Start and end in decimal degrees. Both must be inside the Bengaluru boundary |
| `depart` | Departure time in ISO 8601. Without a time zone it is read as IST |
| `seconds` | Measured travel time, between 1 second and 6 hours |
| `note` | Optional label shown in the journeys table |

Rows that fail a check are skipped, and the console says which rows and why. Each accepted row is routed with OSRM for its free-flow time, one per second, then the model refits. Journeys with a travel time below half or above eight times the free-flow time are kept in the table but not used. An imported or observed journey can be deleted from the journeys table, and the model refits without it.

To rebuild the 2019 profile from the source data, run `python3 scripts/build_traffic_prior.py` and rebuild the image.

## Locating footage

ExifTool is installed in the image, so GPS tracks recorded inside dashcam or phone videos are read when the footage is uploaded. For footage without GPS, an investigator can open a camera in a case and choose Find where this was filmed. They set a search area of up to 3 km and the camera's horizontal field of view.

* **Server load:** the job uses up to `P5_GEOLOC_WORKERS` processes, by default the number of CPU cores minus one, capped at 6. One job runs at a time. These times were measured inside the production container on the 8 vCPU server, with the 122 s KITTI test video at 640 × 192 px:
  * **Searching central Bengaluru** (1.5 km area) took 29 s, including 3 s to read the camera motion. The footage was filmed in Karlsruhe, so the search declined it.
  * **Extra steps for a candidate route:** the four decoy areas took 31 s and reading signboards from 62 frames took 11 s. A two-minute clip that gives a candidate route in Bengaluru therefore takes about 70 s.
  * **Karlsruhe, where the footage was filmed** (2 km area): the whole job took 47 s and gave a candidate route 31 m from the true position.
  * **Memory:** the container peaked at 3.1 GB, against its 24 GB limit.
  * **Not measured:** higher-resolution video takes longer to decode.
* **Data files:** the road graph (`assets/bengaluru/road_graph.npz`) and the place names (`assets/bengaluru/places.json.gz`) are built from the Geofabrik southern India extract. Rebuild them with `python3 scripts/build_road_graph.py` and `python3 scripts/build_place_index.py` (both need `pip install osmium shapely`), then rebuild the image. The graph records the date of its OpenStreetMap data, and every result shows it.
* **Records:** each result stores the video's SHA-256 hash, the search settings and who requested it. It is written to the case and the activity log. Using a candidate route for a camera is a separate, logged action by the investigator.

## Updating the software

```bash
docker build -t vahana-drishti .
docker rm -f vahana
docker run ...   # the same command as above; the mounted folders keep all accounts and records
```

Sessions are stored in `gateway.db`, so signed-in users stay signed in across an update.

Cloudflare tells browsers to keep scripts and stylesheets for 4 hours (its default Browser Cache TTL), whatever the gateway sends. When a `.js` or `.css` file changes, raise the `?v=` number on every reference in `web/public/login.html`, `web/app/index.html` and the `admin.js` import in `web/app/app.js`, so browsers fetch the new files at once. The pages themselves are sent with `no-store` and are never kept. Setting Browser Cache TTL to Respect Existing Headers (Caching, then Configuration in the Cloudflare dashboard) removes the need.

## Backups

`vahana-backup.timer` runs `deploy/host/vahana-backup.sh` every 6 hours (00:00, 06:00, 12:00 and 18:00 UTC). Each run writes `/home/ubuntu/backups/<UTC time>/` with:

* `gateway.db` (accounts, sessions, audit log) and `vahana.db` (records and cases). They are copied with SQLite's online backup while the container runs, then checked with `pragma integrity_check`. A failed check fails the run, and the failure shows in `journalctl -u vahana-backup`.
* `cases/` with the case files and footage. Files that did not change are hard links to the previous backup, so they take no extra space.

The newest 28 backups (7 days) are kept. Run one now with `sudo systemctl start vahana-backup`. The backups sit on the same disk as the data, so copy them off the server from time to time:

```bash
rsync -a host:/home/ubuntu/backups/ ./vahana-backups/
```

To restore a backup:

```bash
B=/home/ubuntu/backups/<UTC time>
sudo docker stop vahana
cp "$B/gateway.db" /home/ubuntu/deploy/state/gateway.db && rm -f /home/ubuntu/deploy/state/gateway.db-wal /home/ubuntu/deploy/state/gateway.db-shm
cp "$B/vahana.db" /home/ubuntu/deploy/data/vahana.db && rm -f /home/ubuntu/deploy/data/vahana.db-wal /home/ubuntu/deploy/data/vahana.db-shm
rsync -a --delete "$B/cases/" /home/ubuntu/deploy/data/cases/
sudo docker start vahana
```

## Watchdog

`vahana-watchdog.timer` runs `deploy/host/vahana-watchdog.sh` every minute. After 3 failed health checks in a row on `127.0.0.1:8080` it restarts the container. It also requests `/api/health` on every public address and records the current quick tunnel address in `/home/ubuntu/deploy/public_url`. Read its findings with `journalctl -u vahana-watchdog`.

## Known limits

Found in the end-to-end and security tests of 7 November 2026 and left as they are for now:

* **Sign-in location:** the browser reports the device location, so it can be faked. It is a convenience check. The controls that matter are the accounts, sessions, lockouts and the audit log.
* **One-time passwords:** `P5_REQUIRE_PASSWORD_CHANGE` is off for the evaluation, so a one-time password stays valid until it is changed under Account.
* **Public services:** GPS tracks go to the OSRM demo router, signboard text from video key frames and investigator notes go to Photon, and travel-time areas go to Valhalla. For real casework, host these yourself and point `P5_OSRM_URL`, `P5_PHOTON_URL` and `P5_VALHALLA_URL` at them.
* **Load:** video analysis runs inside the request, and uploads are held in the container's 1 GB in-memory `/tmp`, so several large uploads at once slow everyone down. A job queue and per-user limits are the next step.
* **Small vehicles:** make and model are named only for vehicles that cover at least 0.4% of the frame. Distant or rooftop footage is counted and tracked without names, and the console says so.
* **Locating moving footage:** it needs the camera's field of view and declines when the motion fits several places. On a real phone recording of a drive along Nandidurga Road (Wikimedia Commons, CC0) it declined.

## The public address

The console is served at `https://www.vahanadrishti.in` and `https://vahanadrishti.in` through the Cloudflare named tunnel `vahana-drishti` in the team's Cloudflare account. `vahana-named-tunnel.service` runs it with `/home/ubuntu/deploy/cloudflared/config.yml`, and the tunnel's credentials are in `/home/ubuntu/.cloudflared/`, readable only by `ubuntu`. To set it up on a new host, as the `ubuntu` user:

```bash
cloudflared tunnel login                        # prints a link: open it and approve the domain in Cloudflare
cloudflared tunnel create vahana-drishti        # writes ~/.cloudflared/<tunnel id>.json
cloudflared tunnel route dns vahana-drishti <host>
mkdir -p /home/ubuntu/deploy/cloudflared
cat > /home/ubuntu/deploy/cloudflared/config.yml <<EOF
tunnel: <tunnel id>
credentials-file: /home/ubuntu/.cloudflared/<tunnel id>.json
ingress:
  - hostname: <host>
    service: http://127.0.0.1:8080
  - service: http_status:404
EOF
sudo install -m 644 deploy/host/vahana-named-tunnel.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vahana-named-tunnel
```

If Cloudflare imported an address (A or AAAA) record for the host from the previous DNS provider, delete it in the dashboard first: `route dns --overwrite-dns` only replaces CNAME records.

Behind the tunnel (`P5_TRUST_PROXY=true` with `P5_COOKIE_SECURE=true`) the gateway answers a plain `http://` request with a permanent redirect to the same address over HTTPS, because sign-in cookies are only sent over HTTPS.

`vahana-tunnel.service` also runs a quick tunnel as a fallback. Its address is random, changes whenever the service restarts, and some Indian networks refuse to resolve `trycloudflare.com` names, so give people the permanent address. The watchdog writes the current quick tunnel address to `/home/ubuntu/deploy/public_url`.
