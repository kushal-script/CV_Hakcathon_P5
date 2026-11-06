# Parameters and where each value comes from

Every tunable value lives in a config file, not in code: `configs/app.yaml` (image pipeline), `models/thresholds.yaml` and `models/india/thresholds.yaml` (decision policy, written by calibration), `configs/hunt_model.json` (vehicle hunt, written by calibration) and `configs/investigation.yaml` (video, tracking, geocoding, routes). Service URLs can be overridden with `P5_PHOTON_URL`, `P5_OSRM_URL`, `P5_VALHALLA_URL` and `P5_USER_AGENT`.

Each value below is one of three kinds:

* **Fitted**: produced by a script from validation data. Test data is never used. Rerun the script to reproduce it.
* **Operational**: a resource budget, an external service rule or a geographic fact. It is not a model choice.
* **Assumption**: a hand-set prior with no labelled data available to fit it. These are listed openly, and the investigation UI shows the reasoning they drive next to every route hypothesis.

## Decision policy (`models/thresholds.yaml`, `models/india/thresholds.yaml`)

| Value | Kind | Source |
|---|---|---|
| `t_d` (temperature) | Fitted | `scripts/calibrate.py`: NLL minimisation on validation logits (E13 for Stanford, E18 for India) |
| `tau_hi`, `tau_fam`, `tau_make` | Fitted | Lowest confidence at which validation accuracy reaches 97% (model, family) or 95% (make). Bootstrap CIs are in each `calibration.json` |
| `tau_ood` | Fitted | 5th percentile of the validation max-logit, i.e. 95% of in-catalogue cars pass. No out-of-catalogue set was used, so OOD rejection is not tuned |
| `tau_lo = 0.10`, `tau_det = 0.25` | Fitted | E06 on validation: 99.14% recall and 0 of 100 COCO no-car dev images accepted. Held-out check in E15: 99 of 100 COCO no-car test images rejected |
| `beta = 0` | Fitted | E12 fusion rule rejected classical fusion (+0.12 points, p = 1.0) |

## Image pipeline (`configs/app.yaml`)

| Value | Kind | Source |
|---|---|---|
| `max_side: 1600`, `threads: 2`, `max_extra_vehicles: 2` | Operational | CPU latency budget on the host |
| `flip_tta: false` | Fitted | E05: flip TTA gains +0.37 points on validation (3 of 815 images) for twice the classifier time |
| `fusion: false` | Fitted | See `beta` |
| `india: true`, `prefer: india` | Operational | Deployment for Bengaluru: the Indian catalogue wins ties between equally specific, equally confident answers |

## Vehicle hunt (`configs/hunt_model.json`)

The match score is a logistic model, P(same vehicle), over three features: appearance cosine (global-average-pooled classifier feature map), Bhattacharyya overlap of the calibrated class distributions, and colour agreement (1 same name, 0.5 neighbouring names, 0 different).

| Value | Kind | Source |
|---|---|---|
| `weights`, `bias` | Fitted | `scripts/calibrate_hunt.py`. 815 Stanford validation cars. Positives pair a clean crop with a simulated second-camera view of the same car (box jitter, flip, 20–60% downscale, gamma 0.6–1.6, per-channel gain ±15%, blur, JPEG quality 25–70). Negatives pair it with views of other cars, one third of them the same model. Fitted on half of the cars, evaluated on the other half |
| `threshold` | Fitted | 95th percentile of negative scores on the fitting half, i.e. a 5% false-positive operating point. On the held-out half: TPR 89.7% at FPR 5.1%. Same model but another vehicle is the hard case (22% FPR), other models 0% |
| `colour_groups` (investigation.yaml) | Assumption | Neighbouring colour names under camera colour casts. Measured: the colour name agrees for only 67% of same-vehicle pairs, and removing colour leaves held-out AUC unchanged (0.981) |

## Video analysis (`investigation.yaml: video`)

| Value | Kind | Source |
|---|---|---|
| `target_fps: 8`, `max_side: 1280`, `max_seconds: 180` | Operational | CPU budget: one 3-minute clip analyses in a few minutes on 8 vCPUs |
| `detection_conf: 0.15`, `nms_iou: 0.5` | Assumption | Low confidence so the tracker's second association stage can recover weak boxes (ByteTrack design) |
| `nested_box_fraction: 0.7`, `min_track_detections: 4`, `min_crop_area_fraction: 0.004` | Assumption | Set while debugging duplicate and flickering tracks on test clips. Not tuned on labelled tracks |
| `crop_padding: 0.10` | Fitted | Same padding the classifier was trained with |
| `best_crops: 3` | Operational | Classifier calls per track |
| `wrong_way_cos: -0.3`, `min_travel_fraction: 0.08` | Assumption | "Moving against the allowed direction by more than about 107°, after travelling at least 8% of the frame diagonal" |

## Tracker (`investigation.yaml: tracker`)

| Value | Kind | Source |
|---|---|---|
| `high_score: 0.45`, `low_score: 0.15`, `match_iou: 0.25` | Assumption | Two-stage association in the style of ByteTrack. No labelled traffic-camera tracking set was available to tune them |
| `max_age_seconds: 2.0`, `min_hits: 3`, `duplicate_iou: 0.5` | Assumption | Track lifetime and confirmation rules |

## Geolocation (`investigation.yaml: geocoding`)

| Value | Kind | Source |
|---|---|---|
| `bbox`, `centre` | Operational | Geographic extent of Bengaluru used to bias and filter geocoder results |
| `min_request_interval_s: 0.35` | Operational | Polite rate for the public Photon instance |
| `min_similarity: 0.55` | Assumption | Fuzzy match ratio between OCR text and a geocoder result name before a candidate is kept |
| `cluster_radius_m: 400` | Assumption | Candidates closer than this are treated as one place |

## Blind-spot route inference (`investigation.yaml: routes`)

There is no ground-truth dataset of routes taken through camera blind spots in Bengaluru, so these values cannot be fitted. They are explicit, documented assumptions. Each hypothesis in the UI lists the reasons that changed its score.

| Value | Meaning |
|---|---|
| `near_place_m: 300` | A route "passes" a place named in the note if it comes within 300 m |
| `via_bonus: 1.6`, `avoid_penalty: 0.2` | Multipliers for passing a place the note says the vehicle went through or avoided |
| `camera_on_route_m: 40`, `camera_time_margin_s: 120` | A camera lies on a route if within 40 m. It "should have seen" the vehicle within ±2 minutes of the expected passing time |
| `missed_camera_penalty: 0.35` | Multiplier when a camera on the route did not record the vehicle |
| `infeasible_penalty: 0.05` | Floor on the multiplier from the probability that a route fits the gap between sightings (see Travel times below). A route that almost certainly cannot fit keeps 5% of its score instead of vanishing, so a wrong clock on one camera cannot erase the true route |
| `infeasible_tolerance: 0.25` | Used only when no travel-time profile is installed: a route needing 25% (plus 60 s) more than its free-flow time is treated as infeasible |
| `slack_decay_s: 1800`, `slower_route_decay_s: 300` | Exponential preference for routes that use the time gap and for routes not slower than the fastest alternative. Slack is measured against the typical drive time for that hour, not the free-flow time |
| `activity_stop_weight: 0.6` | Weight for a destination inferred from an activity in the note (for example "fuel") |
| `max_targets`, `max_hypotheses`, `isochrone_minutes` | Display limits |

## Travel times (`assets/bengaluru/traffic_prior.json`, `src/p5/geo/traffic.py`)

Drive times start from OSRM's free-flow estimate and are scaled to Bengaluru traffic. E20 has the method and the held-out numbers.

| Value | Kind | Source |
|---|---|---|
| Hourly slowdown cells (median, 10th and 90th percentile of observed / free-flow) | Fitted | `scripts/build_traffic_prior.py` on Uber Movement Bangalore 2019 Q3 weekday travel times between 198 wards, against OSRM free-flow times between the ward centroids. 630,003 trips fit, 155,667 trips on held-out ward pairs evaluate: median error 52.5% → 15.2%, and 80.0% of trips fall inside the stated 10 to 90% range |
| Distance bands 3 to 6, 6 to 12 and 12 km and over | Assumption | Coarse enough that every hour has thousands of trips per band. A missing cell falls back to the nearest band at the same hour |
| Minimum distance 3 km, ratio range 0.5 to 8 | Assumption | Short ward pairs are dominated by travel inside the wards, and ratios outside the range are routing or data errors. The same ratio range filters observed journeys |
| Weekend trips | Assumption | Use the weekday profile, because only the weekday hourly file is used. The route explanation says so |
| Shrinkage of 10 pseudo-journeys | Assumption | Local correction = Σ residuals / (n + 10): ten observed journeys move the profile halfway to their mean, so the first few cannot overrule 630,003 trips |
| Local check after 20 journeys, newest 20% held out | Assumption | Smallest count at which a held-out comparison (2019 profile with and without the correction) has more than a handful of trips. It is time ordered so the check always predicts later journeys from earlier ones |
| Dashcam windows: 1 km minimum, 3 km target, 900 s maximum, 15 per track | Assumption | Long enough for the timing to be meaningful, short enough that one long stop does not dominate, and capped so one long video cannot outweigh many cases |
| Import: 2,000 rows, 1 s to 6 h, both ends inside Bengaluru, 1 s between routing calls | Operational | File size limit, plausibility checks and a polite rate for the public OSRM server |

## Indian-market dataset (`scripts/build_india_dataset.py`)

| Value | Kind | Source |
|---|---|---|
| Candidate models | Data | Wikidata car models made in, originating from, or made by a manufacturer based in India, plus models linked from the Wikipedia articles on the Indian automotive industry and on each manufacturer's Indian arm |
| `--per-class 300`, `--width 960` | Operational | Listing budget: up to 300 photos listed per model, as links to 960 px thumbnails, a standard Commons size |
| `--take 150`, `--fetch-width 500`, `--store-side 800` | Operational | Download budget under the Commons per-address rate limit (HTTP 429): at most 150 photos per model, fetched as 500 px thumbnails. Some photos were fetched at 960 px before the fetch size was lowered, and stored photos are at most 800 px |
| `--clean-side 500` | Assumption | Every photo is scaled to at most 500 px before detection and cropping, so the download size cannot identify a class |
| `--min-det 0.5`, `--min-box-frac 0.18` | Assumption | The photo must show one dominant car: a COCO car or truck detection at ≥ 0.5 covering ≥ 18% of the frame, and no second vehicle at least half its size |
| `--min-car-share 0` | Assumption | Off. A rule that dropped classes the detector mostly labels as trucks also removed boxy cars such as the Mahindra Scorpio and Thar, and the Wikidata type already limits candidates to cars |
| `--min-images 15`, `--holdout 0.15`, `--min-holdout 2` | Assumption | Passed for the shipped catalogue (the script defaults are 40 and 6). Smallest class kept, and photographer-disjoint 70/15/15 splits with at least 2 validation and 2 test images per class. Lower than the defaults so that models with few free photos stay in the catalogue, at the cost of test accuracy for those models resting on few images (E18) |
| Holdout cap `2 × --holdout` | Assumption | Validation and test each take at most 30% of a class, so one large photographer group cannot fill a holdout split and push the class out |
| `--merge-overlap 0.5` | Assumption | A class is merged into another when at least half of its listed photos also appear under the other (sub-category) |

## Access control (gateway environment)

| Value | Kind | Source |
|---|---|---|
| `P5_IDLE_MINUTES = 60`, `P5_SESSION_HOURS = 12` | Operational | A shift-length session that ends after an hour without use |
| `P5_LOCKOUT_AFTER = 5`, `P5_LOCKOUT_MINUTES = 15` | Operational | Common account-lockout practice. Sign-in is also limited to 6 attempts per minute per IP (`P5_LOGIN_PER_MINUTE`) |
| `P5_API_PER_MINUTE = 240` | Operational | Per-IP limit on API calls, above the rate a person can click |
| Password rule | Operational | At least 12 characters, three character classes or 16 or more characters, not containing the user ID. One-time passwords are 3 groups of 4 characters from a 55-symbol alphabet (about 69 bits) |
| `P5_DEVICE_GEOFENCE = required`, `P5_GEOFENCE_MARGIN_M = 2000` | Operational | Sign-in only from a device location inside the Bengaluru boundary in `assets/bengaluru/boundary.geojson`, with 2 km of slack for location error |
| `P5_ALLOWED_COUNTRIES = IN` | Operational | Applied to Cloudflare's country header when `P5_TRUST_PROXY=true` |
| Five divisions | Data | The owner's requirement (Central, North, South, East, West). How official divisions map onto them is recorded in `assets/bengaluru/police_stations.json` |
