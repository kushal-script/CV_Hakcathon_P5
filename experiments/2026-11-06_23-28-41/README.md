# E21: locating moving-camera footage by matching its motion to the Bengaluru road network

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_23-28-41` |
| Type | evaluation |
| Author | Kushal |
| Started | 2026-11-06 23:28 IST |
| Git commit | `6e0a727` (code added after it) |
| Status | done. Built into the service as candidate routes that the investigator confirms |
| Compute | JarvisLabs 16 vCPU CPU VM for the KITTI downloads and the evaluations (about 3 hours), local CPU for development |

## Question
Can the place where footage was filmed be found from the footage itself, with no GPS, without Google services, without hardcoded places, and reliably enough for police use?

## What was researched first
Three research reports looked at every published family of methods:

* **Matching frames against street-level photos** (MegaLoc, BoQ, ALIKED and LightGlue). These are the strongest methods, but they need a reference collection. KartaView has photos within 25 m of 15 to 71% of the road length in the three Bengaluru areas checked, mostly from 2020. Mapillary needs an access token.
* **Matching a frame to OpenStreetMap with a neural network** (OrienterNet and successors). These need a starting position within about 100 m, transfer poorly to new cities and are mostly licensed non-commercially.
* **Whole-image geolocation models** (GeoCLIP, PIGEON). These reach city level at best and cannot be audited.

The approach that needs no reference photos follows Brubaker et al. (CVPR 2013, TPAMI 2016). The camera's own motion is matched against the road graph with a probabilistic filter. It is the client's "many simulations" idea: every particle is a simulated vehicle driving the real roads.

## Method
* **Camera motion** (`src/p5/geo/videoloc/motion.py`):
  * Frames are taken at 5 per second at 640 px. Features are tracked with Lucas-Kanade and a forward-backward check.
  * Each frame pair gives a five-point essential matrix (RANSAC), which gives the turn angle.
  * Frames whose median motion is under 0.031 image widths per second count as standstill.
  * When a rotation implies more image motion than was measured, a homography is used instead.
  * Speed is read from features on the road surface below the horizon, up to an unknown scale.
* **Road graph:** OpenStreetMap drivable ways from the Geofabrik extract of 4 Nov 2026. That is 405,925 edges and 780,055 directed edges (one-way streets respected) covering 53,814 km of directed road.
* **Particle filter** (`filter.py`):
  * Each particle holds a directed edge, a position on it, the next edge, a speed, a camera scale and a lens factor, and runs a Kalman filter for the heading offset.
  * Particles advance by the measured speed and choose turns at junctions uniformly. Through a junction, heading blends over 7 m either side.
  * Heading residuals use a Gaussian with 5% outliers. Each frame counts as dt / 1.34 s of evidence, and evidence ramps up over the first 30 s.
* **Search** (`search.py`):
  * The road inside the search area gets 1 particle per metre, from 150,000 to 400,000.
  * Six independent searches run: 4 with standard resampling and 2 with exact-mass resampling, which keeps competing places alive.
  * Two challenge searches then exclude a 150 m circle around the best route's start and look for the best alternative.
  * Four decoy searches run in random areas of the city at least 5 km away.
* **Decision:** a candidate route needs at least 0.99 probability within 50 m. Its likelihood ratio must be at least 99 against both the best alternative in the area and the best decoy.
* **Signboards** (`signs.py`): OCR text is compared with 144,027 named OpenStreetMap places. A name within 100 m of the route supports it. A name whose every match in the city (at most 3) is over 300 m away contradicts it.

## Data
* **KITTI raw** (Geiger et al., CC BY-NC-SA 3.0): only the grey left camera and the GPS/IMU files were fetched, with HTTP range requests on the zip archives.
  * **Training drives** (noise settings only): 2011_09_30 drives 0018, 0020 and 0027.
  * **Development drives** (threshold, outlier share, ramp): the training drives plus 2011_09_30 drives 0028, 0033 and 0034, and 2011_10_03 drives 0027 and 0034. These are odometry sequences 00, 02, 05, 06, 07, 08, 09 and 10.
  * **Fresh drives** (never used for any setting): 2011_09_26 drives 0022, 0036, 0101 and 0117, 2011_09_29 drive 0071, and 2011_10_03 drives 0042 and 0047.
* **Karlsruhe road graph:** the Geofabrik Karlsruhe extract, 14 × 14 km around the drives.
* **Simulated Bengaluru drives:**
  * Shortest-path routes run between random places on the Bengaluru graph, with KITTI speed profiles.
  * Heading follows the road with an 8 to 20 m turning arc.
  * Turn and speed errors are copied frame by frame from the real training videos.
* **Search areas:** a 2 km circle in Karlsruhe (or 1.5 km in Bengaluru) whose centre is offset at random by up to 1 km (or 750 m), so the answer is never simply the centre.

## What did not work, in order
1. Heading compared at every frame as independent evidence was overconfident. A symmetric KITTI drive came out at 85 to 95% for a place 870 m from the truth. Counting evidence by its measured correlation time fixed the over-counting.
2. A camera that starts in the middle of a turn lost its true place within 2 s, because the prior assumed the camera faced along the road. The initial angle now has a 45° standard deviation.
3. Protecting the top places from resampling spread the particles too thin. Exact-mass resampling is used in only 2 of the 6 searches.
4. With the lens unknown, a held-out video was confidently placed 1.2 km from the truth. The epipolar fit was flat for focal lengths from 150 to 900 px, so the focal length cannot be estimated from forward motion. The field of view is now required.
5. **Footage from outside the search area.**
   * KITTI clips searched in random Bengaluru areas produced a route at 99% or above in 18 of 40 cases. Bengaluru drives searched 4 to 8 km away did so in 9 of 23. In a road network this dense, some route often fits a minute of turns.
   * An absolute fit score did not separate these cases from true matches. Its distributions overlapped.
   * Decoy searches cut the KITTI case to 7 of 40 and the Bengaluru case to 3 of 20, and kept 27 of 30 true KITTI matches.
   * The remaining risk is why results are shown as candidate routes for the investigator to confirm, with the signboard checks and street-photo links next to them.

## Results
Correct means the candidate route ends within 50 m of the true position at the end of the clip.

When the footage is from the search area:

| Set | Clips | Candidate routes | Correct | Notes |
|---|---|---|---|---|
| KITTI development, all checks | 40 | 27 | 27 (95% CI 87.5 to 100%) | Median error 4.3 m, largest 39.5 m. Answered 11 of 20 one-minute clips, 12 of 15 two-minute clips and 4 of 5 four-minute clips |
| KITTI development, without the decoy check | 40 | 30 | 30 | |
| KITTI fresh drives, without the decoy check | 13 | 1 | 1 | Short, slow or highway clips with few turns, so it declined almost all of them |
| Field of view stated about 5° wrong, camera height unknown, without the decoy check | 20 | 13 | 13 | |
| Simulated Bengaluru drives, without the decoy check | 30 | 15 | 15 | |
| Simulated Bengaluru drives, all checks | 20 | 10 | 10 (95% CI 72 to 100%) | New routes, not the 30 above |

When the footage is not from the search area:

| Set | Clips | Wrong candidate without decoys | Wrong candidate with decoys |
|---|---|---|---|
| KITTI clips searched in random Bengaluru areas | 40 | 18 | 7 (17.5%) |
| Simulated Bengaluru drives searched 4 to 8 km away | 20 | 9 | 3 (15%) |

A full KITTI video end to end (122 s, file decoded by the service, calibrated lens) became a candidate route 32 m from the truth. Its likelihood ratio was 10^65, all 6 searches agreed, and it took 47 s on 6 cores. Run through the service, the same video searched in central Bengaluru was declined.

## Conclusion
Matching camera motion to the road network answers one question precisely: if this footage was recorded inside this area, which route did it take? It cannot by itself show that the footage was recorded inside the area. The service therefore reports candidate routes with:
* the evidence (turn-by-turn agreement, likelihood ratios, the decoy comparison, and signboards that agree or disagree);
* the video's SHA-256 hash and the map data date;
* links to street photos.

The investigator confirms the route with independent features before it is used for a camera, and that confirmation is logged.

## How to reproduce
```bash
python3 scripts/build_road_graph.py
python3 scripts/build_place_index.py
python3 scripts/videoloc/kitti_fetch.py data/kitti 2011_09_30_drive_0018 2011_09_30_drive_0020 2011_09_30_drive_0027
python3 scripts/videoloc/kitti_motion.py data/kitti 2011_09_30_drive_0018 2 runs/videoloc/2011_09_30_drive_0018_e2_f1.jsonl 1
python3 scripts/videoloc/evaluate_clips.py <graph.npz> <config.json>
python3 scripts/videoloc/simulate_bengaluru.py assets/bengaluru/road_graph.npz <config.json>
pytest tests/test_videoloc.py
```
`config.yaml` lists the clip lists and settings of every run. `metrics.json` holds the numbers above.
