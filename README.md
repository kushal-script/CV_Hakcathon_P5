# P5: Intelligent Fine-Grained Vehicle Recognition and Analysis System

Computer Vision Hackathon 2026, Problem Statement P5, Team T55 (1RUA24CSE0216, 1RUA24CSE0228, 1RUA24CSE0229, 1RUA24CSE0275, 1RUA24CSE0244). The system detects a vehicle in an uploaded image, identifies its make, model and year among the 196 Stanford Cars classes, and gives a calibrated confidence with every answer. Classical computer vision (quality analysis, quality-gated enhancement, GrabCut ROI refinement, Canny edges and contours, colour/texture/geometry features) works alongside deep learning (a fine-tuned YOLO26s detector and a fine-tuned TResNet-L classifier). A decision policy answers at the model, family or make level, flags cars outside the catalogue, and reports "no vehicle" when there is none.

## Results (Stanford Cars test, n = 8,041)

| Metric | Value |
|---|---|
| **Top-1 / Top-5, deployed system** | **94.69% / 98.94%** (make level 98.37%) |
| Classical only / classical on detector crop | 2.96% / 14.74% |
| Deep only, full image (no detection) | 91.88% |
| Detection mean IoU / IoU ≥ 0.5 | 0.949 / 99.03% |
| mAP@0.5 / mAP@0.5:0.95 | 0.976 / 0.926 |
| No-vehicle images correctly rejected | 99 / 100 |
| Calibration error (ECE) | 10.4% → 1.1% after temperature scaling |
| Classifier size, INT8 | 52.9 MB (3.9× smaller, −0.37 points on val) |
| Full pipeline latency, 2 CPU threads (Apple M3, INT8) | p50 0.59 s, p95 1.10 s |
| Robustness, Gaussian noise s1 / speckle s3 / contrast s5 | 81.9 → 90.9 / 78.9 → 88.3 / 83.9 → 91.2 with the classical enhancement |

Full tables, ablations and failure cases: [eval_P5_T55.md](eval_P5_T55.md). Per-experiment records: [experiments/INDEX.md](experiments/INDEX.md).

## Beyond the brief: an investigation console for traffic police

The recognition system is the core of **Vahana Drishti**, a web console designed around the needs of the Bengaluru Traffic Police. Everything runs on open data and open-source maps (no Google services).

| Feature | What it does | Where |
|---|---|---|
| Identify | Upload a photo: vehicle, make, model and year with calibrated confidence, every classical and deep intermediate step, and the classical, deep and hybrid answers side by side | `src/p5/pipeline.py` |
| Locate a photo | Exact coordinates from EXIF GPS when present, otherwise reads signboards and road names (RapidOCR) and geocodes them inside Bengaluru with OpenStreetMap (Photon), clustering agreeing candidates | `src/p5/geo/` |
| Video analysis | Detects and tracks every vehicle (Kalman filter with two-stage Hungarian matching), names make and model per track, colour, direction of travel and wrong-way alerts. Signboards and shop names read from key frames describe the surroundings and suggest where the camera is | `src/p5/video/` |
| Route from video | Dashcam GPS (GPX, CSV, NMEA or the MP4 location atom) is snapped to the road network with OSRM map matching and drawn on the map | `src/p5/geo/route.py` |
| Vehicle hunt | Upload a photo of the wanted car and the feeds of several cameras: every track is scored with a calibrated match probability and the sightings are placed on the map in time order | `src/p5/investigate/hunt.py`, E19 |
| Blind-spot reasoning | Free-text notes from officers ("seen near Brigade Road heading to Indiranagar, may stop for fuel, avoided Silk Board") are parsed into places, roles, times and activities. Candidate routes between sightings and onward are ranked by consistency with the notes, with travel-time feasibility and cameras that did not see the car. Reachable areas are drawn as isochrones | `src/p5/investigate/hints.py`, `routes.py` |
| Map | MapLibre with OpenFreeMap streets, Sentinel-2 cloudless satellite imagery and extruded 3D buildings from OpenStreetMap | `web/` |

Every tunable value is in a config file, and [docs/PARAMETERS.md](docs/PARAMETERS.md) states whether it was fitted on validation data, is an operational constant, or is a documented assumption.

### Access model

* **One account per police station**, created from the station register in `assets/bengaluru/police_stations.json`. Stations sit in five divisions: Central, North, South, East and West.
* **One administrator per division.** An administrator sees every record, case and account of the stations in the division. The admin can correct or delete records, open, close, reassign and delete cases, reset station passwords, disable or unlock accounts, read the security audit log, and see charts of everything the system predicted. A station sees only its own records and cases.
* **Bengaluru only.** Sign-in requires the device to report a location inside the Bengaluru boundary (with a 2 km margin). The gateway refuses requests that Cloudflare marks as coming from outside India. Camera locations must fall inside the city, and the map cannot leave it.
* **Nothing is visible before sign-in.** The gateway serves only the sign-in page and its assets to anonymous visitors. The console's HTML and scripts, and every API call, require a valid session.

### Security

* A Rust gateway (`gateway/`, axum) is the only public entry point. The Python service listens on loopback only and accepts a request only with an identity that the gateway signed with HMAC-SHA256 seconds earlier.
* Accounts, sessions and the audit log live in SQLite (the C library is compiled into the gateway, WAL mode). Passwords are Argon2id hashes. Every account starts with a one-time password that must be changed at the first sign-in.
* Sessions are random 256-bit tokens stored only as SHA-256 hashes, so they can be revoked server side. They expire after 60 idle minutes or 12 hours. Changing a password or a reset by an administrator signs out every other session.
* An account locks for 15 minutes after 5 failed attempts. Sign-in is also rate limited per IP. Unknown IDs take as long to reject as wrong passwords.
* Every state-changing request needs a same-origin marker header and is refused when the browser reports a cross-site origin. Cookies are HttpOnly, Secure and SameSite=Strict.
* A strict Content-Security-Policy, `X-Frame-Options: DENY`, `nosniff`, `no-referrer`, a Permissions-Policy, COOP/CORP and HSTS.
* Uploads are checked by content (image decoding, video file signatures) and size. The frontend builds the DOM with `textContent` only.
* The container runs as a non-root user on a read-only filesystem with all Linux capabilities dropped.

## Documentation

| Document | Contents |
|---|---|
| [eval_P5_T55.md](eval_P5_T55.md) | Comparative evaluation in the required metric order, ablation and failure-case analysis |
| [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) | What was built and every deviation from the plan |
| [docs/PARAMETERS.md](docs/PARAMETERS.md) | Every configurable value and where it comes from |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Deployment, accounts, settings, updates and backups |
| [assets/bengaluru/SOURCES.md](assets/bengaluru/SOURCES.md) | Where the police station register and the city boundary come from |
| [docs/FINAL_APPROACH.md](docs/FINAL_APPROACH.md) | Execution plan: requirement coverage, components, experiments, timeline, risks, viva prep |
| [docs/architecture.md](docs/architecture.md) | Architecture diagrams, module responsibilities and interfaces |
| [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) | Literature survey (178 references) |
| [docs/research/](docs/research/), [docs/pipeline/](docs/pipeline/) | Research reports and the synthesis, critique and revision of the approach |
| [experiments/](experiments/) | One timestamped folder per experiment with config, logs, metrics and write-up |

## Repository layout

```
src/p5/
  classical/       quality analysis, enhancement, saliency ROI, GrabCut refinement, geometry, features, SVM, colour naming
  detection/       ONNX detector, box selection, YOLO training
  classification/  crops, dataset, model factory (timm and I-JEPA), training, export, ONNX classifier, CAM
  confidence/      temperature scaling, selective thresholds, decision policy
  hybrid/          late fusion
  eval/            classification, detection, robustness, statistics
  video/           vehicle tracker and video analysis
  geo/             EXIF and OCR geolocation, geocoding, GPS tracks, map matching and routing
  investigate/     cases, vehicle hunt, note parsing and route inference
  catalogue.py     several classifiers (US and Indian market) behind one decision
  pipeline.py      end-to-end image inference used by the API
app/server.py      FastAPI service (loopback only behind the gateway)
gateway/           Rust gateway: auth, rate limits, security headers, static files, proxy
web/               web console (HTML, CSS, JavaScript, MapLibre)
configs/           app, investigation and hunt settings
deploy/            container entrypoint
scripts/           data preparation, training launchers, calibration, evaluation, packaging
experiments/       experiment records
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-train.txt fastapi "uvicorn[standard]" python-multipart pillow-heif
python3 scripts/build_assets.py        # validates the dataset, writes taxonomy and the val split
python3 scripts/build_cache.py         # GT crops and downscaled full images
```

Run the API locally once `models/` is populated (see Reproduction):

```bash
PYTHONPATH=src .venv/bin/uvicorn app.server:app --port 8000
curl -F "file=@car.jpg" http://localhost:8000/api/predict
```

Docker (gateway and API in one image):

```bash
docker build -t vahana-drishti .
docker run -d --name vahana -p 127.0.0.1:8080:8080 -v "$PWD/state:/app/state" -v "$PWD/cases:/app/data" --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges vahana-drishti
docker exec vahana vahana-gateway provision --stations /app/assets/bengaluru/police_stations.json --credentials /app/state/credentials.csv
```

`provision` creates the divisions, the stations, one account per station and one administrator per division, and writes their one-time passwords to a new file readable only by its owner. Rerunning it adds new stations and leaves existing accounts alone. `vahana-gateway reset-password <id>` issues a new one-time password from the server console.

Serve it behind HTTPS and set `P5_TRUST_PROXY=true` when a reverse proxy or tunnel sets the client IP and country headers. Other settings: `P5_DEVICE_GEOFENCE` (`required` or `off`), `P5_ALLOWED_COUNTRIES` (default `IN`), `P5_IDLE_MINUTES`, `P5_SESSION_HOURS`, `P5_LOCKOUT_AFTER`, `P5_LOCKOUT_MINUTES`.

## Reproduction

GPU jobs are plain scripts launched on a JarvisLabs A100 (`scripts/remote/*.sh`); everything they produce is synced to `runs/remote/` (git-ignored).

| Step | Command |
|---|---|
| Classifier (E05) | `bash scripts/remote/train_classifier.sh e05_tresnet_l_368 --backbone tresnet_l.miil_in1k_448 --size 368 --epochs 40` |
| Detector (E03, E06) | `bash scripts/remote/train_detector.sh --epochs 30 --batch 32` |
| Ablations (E09, E10) | `bash scripts/remote/ablations.sh` |
| Logits, INT8, robustness | `bash scripts/remote/predict_logits.sh`, `bash scripts/remote/quantize.sh`, `bash scripts/remote/robustness.sh` |
| I-JEPA (E17) | `bash scripts/remote/jepa.sh` |
| Classical baseline (E08) | `python3 scripts/train_classical.py` |
| Calibration (E13) | `python3 scripts/calibrate.py --logits <INT8 val logits> --out runs/e13_calibration` |
| Package models | `python3 scripts/package_models.py --classifier <export_int8> --detector <detector.onnx> --classical <classical_global.npz> --thresholds <thresholds.yaml>` |
| Test evaluation (E15) | `python3 scripts/eval_test.py ...` (the only script that reads test labels; calls are logged) |
| Evaluation file | `python3 scripts/make_tables.py --team T55 --usns 1RUA24CSE0216,1RUA24CSE0228,1RUA24CSE0229,1RUA24CSE0275,1RUA24CSE0244` |

## Dataset

Stanford Cars (Krause et al., 2013): 16,185 images, 196 classes, 8,144 train and 8,041 test, one bounding box per image. The official host is offline; we use the torchvision-layout Kaggle mirror [`rickyyyyyyy/torchvision-stanford-cars`](https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars), which includes test labels. Extract to `data/stanford_cars/`. The `data/` folder is git-ignored.

## Limitations

- Stanford Cars covers mostly US-market cars up to 2012; newer models and many regional makes are outside the catalogue and can only receive a make-level or "not a known model" answer.
- Low light at high severity, strong blur and car regions under ~60 px remain weak points (see the robustness table).
- The out-of-catalogue threshold was set on in-distribution validation data only.
- The system identifies vehicle models only; it does not read number plates or identify people.
