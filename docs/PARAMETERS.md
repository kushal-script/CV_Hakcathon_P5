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
| `quality: dark 0.30, low_contrast 0.12, cast 0.12, noisy 0.035, blurry 60` | Assumption | Hand-set limits on mean luma, RMS contrast, colour cast ratio, noise sigma (Immerkaer) and Laplacian variance, measured on a copy at most 512 px wide. They were not fitted: E12 compared enhancement variants with these flags held fixed. Blur is only reported, never corrected |
| `enhancement: clip 2.0, tile 8, nlm_scale 0.8, noisy 0.035, gamma_target 0.45` | Assumption | Hand-set CLAHE clip limit (×0.75 for dark images) and tile grid, Non-Local Means strength (h = noise sigma × 255 × 0.8, kept between 3 and 30), the noise level that triggers denoising after gamma, and the mean brightness that auto-gamma aims for (gamma kept between 0.3 and 1.0). Whether the enhanced or the original photo is used is decided per photo by the detector's confidence (E12, E14) |

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
| `target_fps: 8`, `max_side: 1280`, `max_seconds: 180` | Operational | CPU budget: one 3-minute clip analyses in a few minutes. On the 4 vCPU server a 15 s dashcam clip at 640 × 360 px took 14 s, about the same as on 8 vCPUs |
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

## Where a photo was taken (`src/p5/geo/where.py`)

| Value | Kind | Source |
|---|---|---|
| GPS position, accuracy, camera time and time zone, camera model | Measured | Read from the photo's EXIF data, never estimated. Shown only when the photo carries them |
| OCR text kept at a recognition score of 0.6 or more | Operational | The same rule as signboards read from video |
| Words on one row are joined into a line when they overlap vertically by half a text height and are less than 1.5 text heights apart | Operational | OCR often returns the words of one sign separately. Without joining, the word "BRIGADE" of a "BRIGADE ROAD" sign matched an unrelated place named "Brigade" |
| A name matches only when a run of up to 6 words in a line equals an OpenStreetMap name exactly after lower-casing, and is not a generic word such as "road" | Operational | Exact matches only, so nothing is invented from a near miss |
| Entries with the same name within 218 m of each other form one place | Fitted on the map data | 99th percentile of the distance from each named road piece to the nearest piece with the same name, over 87,759 pieces in the Bengaluru index (median 11 m). One road is stored as many pieces, so "Brigade Road" has 78 entries but is one place |
| Heat map: every name read carries a weight of 1, split equally between the distinct places with that name and then over each place's OpenStreetMap entries. Weights of several names add up. With GPS the map shows the point and the accuracy the camera recorded, and with no clue there is no heat | Rule | Spreads only the evidence actually read in the photo. It is not a fitted probability, so the console calls it clues |
| A name shared by more than 3 distinct places is reported with its count only | Operational | The same limit as the video signboard check |
| Year for the Indian catalogue: the years the model was made, shown as a range | Data | Its photos are not labelled by model year, so the year cannot be read from a photo. `scripts/build_production_years.py` reads the production field of each model's English Wikipedia infobox (252 of 282 models), else the start and end years on Wikidata (2 models). The other 28 show the year as unknown. Stored in `assets/india/production_years.json` and the deployed catalogue |

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

## Locating moving-camera footage (`src/p5/geo/videoloc/`)

A dashcam or a phone in a vehicle is located by matching its turns, stops and speed against the road network. E21 has the method and the evaluation. Values marked fitted come from three KITTI training drives (2011_09_30 drives 0018, 0020 and 0027). Values chosen on development clips used the KITTI odometry drives and never the fresh test drives.

| Value | Kind | Source |
|---|---|---|
| Road network: drivable OpenStreetMap ways inside the city boundary plus 0.02°, without parking aisles, driveways and private roads | Operational | `scripts/build_road_graph.py` from the Geofabrik southern India extract. The asset records the extract date |
| Frames at 5 per second, 640 px wide, 600 corners on an 8 × 8 grid, forward-backward tracking within 1 px, RANSAC 1 px | Operational | Standard settings for pyramidal Lucas-Kanade tracking and the five-point essential matrix in OpenCV |
| Standstill when the median image motion is below 0.031 image widths per second | Fitted | Geometric middle between the 90th percentile for stopped frames (2.4 px) and the 10th percentile for moving frames (6.5 px) on the training drives. Stopped frames had produced turn errors of up to 177° |
| Rotation check: fall back to a homography when the rotation implies more image motion than was measured | Physics | A rotation cannot move image points further than they actually moved |
| Heading noise 4° | Assumption | The training drives give 2.2 to 2.6° and their likelihood is highest at 1.5 to 3°. 4° leaves room for less precise OpenStreetMap geometry in Bengaluru |
| Heading outliers 5% | Development | 1% was the training-drive maximum likelihood value, but 5% fixed two of three confidently wrong development clips |
| Heading drift 1° per √s | Fitted | Maximum likelihood on the training drives |
| Turn shape: heading blends over 7 m either side of a junction | Fitted | Maximum likelihood on the training drives |
| Evidence correlation time 1.34 s | Fitted | Integrated autocorrelation of heading residuals on the training drives. Each frame counts as 0.2 / 1.34 of an independent observation, so correlated frames are not counted many times |
| Evidence ramp over the first 30 s | Development | Early frames count gradually so that the true place is not dropped before the first turns are seen |
| Initial camera to road angle, standard deviation 45° | Assumption | The camera may start mid-turn. One training drive does, and a tight prior lost the true place within 2 s |
| Speed noise 0.224 per √s | Fitted | Distance error of the road-surface speed reading over 5, 10 and 30 s windows on the training drives |
| Camera scale: unknown camera height, log-uniform between 0.8 and 3 | Assumption | Covers car and bus dashcams |
| Field of view: required from the investigator, with ±10% lens tolerance | Assumption | The focal length cannot be recovered from forward motion: the epipolar fit was flat from 150 to 900 px. Without the field of view an evaluation run was confidently wrong. Development clips with the field of view stated 5° wrong stay correct |
| U-turns turn clockwise | Geographic fact | Traffic in India keeps to the left |
| 1 simulated vehicle per metre of road, between 150,000 and 400,000 | Operational | Fewer per metre lost the true place in testing. The cap bounds memory on the server |
| 4 standard and 2 exact-mass searches, 2 challenge searches outside 150 m of the best start, 4 decoy areas with 3 searches each | Operational | Independent searches cover search failures. The challenge measures uniqueness inside the area, and the decoys measure fit against random places in the city |
| Candidate route only at 99% or above, against both the best alternative in the area and the best of 4 decoy areas at least 5 km away | Development | The only confidently wrong development answers had likelihood ratios below 100. The same 99:1 rule is used for the challenge and for the decoys. Decoys were added after footage from outside the area produced a confident route in 18 of 40 controls. With them it is 7 of 40, which is why every candidate needs the investigator's confirmation |
| Signboards: OCR every 2 s on up to 90 frames, OCR confidence 0.6, a name must keep 6 letters after common words are removed; support within 100 m of the route, contradiction when every place with that name (at most 3 in the city) is over 300 m away | Assumption | Signs are legible from a vehicle for roughly a block. The checks are shown to the investigator and do not change the status |
| Search radius 300 m to 3 km, at least 10 s of camera movement | Operational | Larger areas need more simulated vehicles than the server can hold |

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
| `P5_LOCKOUT_AFTER = 5`, `P5_LOCKOUT_MINUTES = 15`, `P5_LOCKOUT_ACCOUNT_AFTER = 50` | Operational | 5 failures pause sign-in for that ID from that network (IPv4 address or IPv6 /64) only, so lockouts cannot be used to shut officers out. 50 failures from all networks together lock the account, which stops guessing spread over many networks. Common lockout practice, adjusted after a security review found that a single shared counter let anyone lock every account |
| `P5_API_PER_MINUTE = 240` | Operational | Per-IP limit on API calls, above the rate a person can click |
| Password rule | Operational | At least 12 characters, three character classes or 16 or more characters, not containing the user ID. One-time passwords are 3 groups of 4 characters from a 55-symbol alphabet (about 69 bits) |
| `P5_DEVICE_GEOFENCE = required`, `P5_GEOFENCE_MARGIN_M = 2000` | Operational | Sign-in only from a device location inside the sign-in area, with 2 km of slack for location error. The live server's area is Karnataka (`assets/bengaluru/signin_area.geojson`, OpenStreetMap relation 2019939, simplified by 0.005°, about 0.5 km, well inside the slack). Without `P5_BOUNDARY` it is the Bengaluru boundary |
| `P5_LOGIN_PER_MINUTE = 30`, `P5_API_PER_MINUTE = 1200` on the live server | Operational | Per-address limits with a burst of the sign-in limit, or a quarter of the API limit (at least 60). Raised from 6 and 240 because evaluators in one room share one public address. Account lockout after 5 wrong passwords still stops guessing |
| `P5_ALLOWED_COUNTRIES = IN` | Operational | Applied to Cloudflare's country header when `P5_TRUST_PROXY=true` |
| Five divisions | Data | The owner's requirement (Central, North, South, East, West). How official divisions map onto them is recorded in `assets/bengaluru/police_stations.json` |
