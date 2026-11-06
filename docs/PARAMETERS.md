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
| `tau_lo = 0.10`, `tau_det = 0.25` | Assumption, checked | Planned defaults for "weak" and "confident" detections. Held-out check (E15): 99 of 100 COCO no-car test images are rejected |
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
| `infeasible_tolerance: 0.25`, `infeasible_penalty: 0.05` | A route needing 25% (plus 60 s) more time than the gap between sightings is treated as infeasible |
| `slack_decay_s: 1800`, `slower_route_decay_s: 300` | Exponential preference for routes that use the time gap and for routes not slower than the fastest alternative |
| `activity_stop_weight: 0.6` | Weight for a destination inferred from an activity in the note (for example "fuel") |
| `max_targets`, `max_hypotheses`, `isochrone_minutes` | Display limits |

## Indian-market dataset (`scripts/build_india_dataset.py`)

| Value | Kind | Source |
|---|---|---|
| Candidate models | Data | Wikidata car models made in, originating from, or made by a manufacturer based in India, plus models linked from the Wikipedia articles on the Indian automotive industry and on each manufacturer's Indian arm |
| `--per-class 300`, `--width 960` | Operational | Download budget under the Wikimedia rate limits. 960 px is a standard thumbnail size |
| `--min-det 0.5`, `--min-box-frac 0.18` | Assumption | The photo must show one dominant car: a COCO car or truck detection at ≥ 0.5 covering ≥ 18% of the frame, and no second vehicle at least half its size |
| `--min-car-share 0.5` | Assumption | Drops classes that the detector mostly sees as trucks (commercial vehicles) |
| `--min-images 40`, `--holdout 0.15`, `--min-holdout 6` | Assumption | Smallest class kept, and photographer-disjoint 70/15/15 splits with at least 6 validation and test images per class |
| `--merge-overlap 0.5` | Assumption | A class is merged into another when at least half of its listed photos also appear under the other (sub-category) |
