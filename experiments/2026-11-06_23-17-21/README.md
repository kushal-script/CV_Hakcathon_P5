# E20: Bengaluru travel-time calibration from Uber Movement 2019 and observed journeys

| Field | Value |
|---|---|
| Experiment ID | `2026-11-06_23-17-21` |
| Type | calibration |
| Author | Kushal |
| Started | 2026-11-06 23:17 IST |
| Git commit | `152eea5` |
| Status | done (2019 profile), local check waiting for 20 observed journeys |
| Compute | Local CPU, under a minute after the downloads, plus 16 table requests to the public OSRM server |

## Hypothesis
Route reasoning compared the time between two sightings with OSRM's free-flow drive time. In Bengaluru that estimate is far too short, so nearly every real journey looked as if it included a stop, and routes that were truly impossible were hard to tell from slow ones. Measured trip times by hour of day should give drive-time ranges that hold for real trips. Journeys the system observes later should then move the 2019 profile towards current traffic without letting a handful of them overrule it.

## Setup
- **Travel times:** Uber Movement, Bangalore, 2019 Q3, weekdays, hourly aggregate between 198 wards (`geometric_mean_travel_time`). Uber Movement has been discontinued, so the archived copy published with the QGIS tutorials is used.
- **Free-flow estimate:** OSRM `table` (driving) between area-weighted ward centroids. It is the same engine the investigation uses for routes, so the ratio corrects the engine's own bias as well as congestion.
- **Filters:** ward pairs at least 3 km apart by road, observed to free-flow ratio between 0.5 and 8, and same-ward rows dropped. 785,670 rows kept and 32,592 dropped.
- **Split:** 20% of unordered ward pairs are held out by a hash of the pair, so both directions and all 24 hours of a pair fall on the same side. 630,003 rows fit the model and 155,667 evaluate it.
- **Model:** for each hour of day and road distance band (3 to 6 km, 6 to 12 km, 12 km and over), the median, 10th and 90th percentile of log(observed / free-flow). A route's typical time is its free-flow time scaled by the median, its range uses the percentiles, and the chance that it fits a gap between sightings is a lognormal whose spread comes from the 10 to 90% range.
- **Local correction:** each observed journey gives a residual, log(observed / free-flow) minus the cell median. The correction is the shrunken mean Σr / (n + 10), so ten journeys move the profile halfway to their own mean. Journeys come from consecutive sightings of the same vehicle at two cameras, from 1 to 3 km windows of dashcam GPS tracks and from CSV imports. Ratios outside 0.5 to 8 are ignored. Once 20 journeys exist, the newest fifth by departure time is held out and the error with and without the correction is shown in the console.

## How to reproduce
```bash
python3 scripts/build_traffic_prior.py
pytest tests/test_traffic.py
```
Downloads go to `data/traffic_raw/`, the OSRM table is cached in `data/traffic_raw/osrm_freeflow.npz`, and the profile is written to `assets/bengaluru/traffic_prior.json`.

## Results
Split used for these numbers: **held out 20% of ward pairs (155,667 trips)**

| Metric | Free-flow time | Hourly profile |
|---|---|---|
| Median absolute error | 52.5% | 15.2% |
| Mean absolute error | 1,176 s (19.6 min) | 403 s (6.7 min) |
| Trips inside the stated 10 to 90% range | n/a | 80.0% |

Slowdown against free-flow time by hour of day (fitting rows, all bands):

| Hour | Median | 10% | 90% |
|---|---|---|---|
| 03:00 | 1.26× | 0.89× | 1.74× |
| 09:00 | 2.64× | 1.77× | 3.45× |
| 13:00 | 2.29× | 1.54× | 2.94× |
| 18:00 (peak) | 3.05× | 2.04× | 3.88× |
| 22:00 | 1.80× | 1.30× | 2.21× |

All 24 hours are in `metrics.json` and in the console (Division, then Traffic).

**Case check on the dev system:** sightings at MG Road at 18:00 and Indiranagar 100 Feet Road at 18:25. The fastest route takes 6.3 min free-flow, so the old rule reported "19 min unaccounted (possible stop)". With the profile the typical time is 16.9 min (range 8.4 to 23.0), the hypothesis reads "8 min more than the typical 17 min", and the journey was stored and the model refitted (one journey, correction 1.04×).

**Local correction:** not measured yet. The deployed system has no observed journeys. Unit tests check the shrinkage arithmetic and the held-out report on synthetic journeys. They say nothing about real accuracy.

## Analysis
- Free-flow time is too short at every hour, by 1.2 to 1.4× even at 03:00. OSRM's car speeds are optimistic, and a trip between two wards includes driving inside each ward. Fitting against the same engine the investigation uses corrects both effects.
- The evening peak at 18:00 (3.05×) is the slowest hour, followed by the morning peak at 09:00 (2.64×), with a midday trough of 2.29× at 13:00.
- The 10 to 90% range holds 80.0% of held-out trips. The ranges are calibrated on unseen ward pairs, not only accurate on average, which is what makes the fit probability usable for ruling routes out.
- Distance matters less than hour: at 18:00 the median is 2.70× for 3 to 6 km, 3.12× for 6 to 12 km and 3.07× beyond 12 km.

## Decision
Ship the profile as `assets/bengaluru/traffic_prior.json`. Each route hypothesis is multiplied by the probability that it fits the gap between sightings (never below `routes.infeasible_penalty`, 0.05). Slack is measured against the typical time instead of the free-flow time, and onward routes show an arrival window. Observed journeys are stored and the model refits after every case update, dashcam upload, import or deletion.

## Limitations
- The data is from 2019 and covers weekdays only. Weekend trips use the weekday profile, and the console says so. Metro lines and flyovers opened since then change traffic, and only the local correction tracks that.
- Uber trips are app-hailed cars between ward centroids. Two-wheelers, autos, buses and goods vehicles move differently.
- The local correction is one city-wide factor. Per-hour or per-area corrections would need many more journeys and would overfit the first few.
- A range for a ward pair is wider than the spread on one specific road, so the range for a single route is conservative.
- Uber Movement data is licensed CC BY-NC and is attributed in the console and the README. The source has been discontinued, so the profile cannot be refreshed from it.
