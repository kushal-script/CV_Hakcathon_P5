# Vahana Drishti: design document

Team T55, Computer Vision Hackathon 2026, Problem Statement P5. Live at **https://vahanadrishti.in**.

This document describes the system as it is built and deployed. The plan written before implementation is `docs/FINAL_APPROACH.md`, the deviations from it are in `docs/IMPLEMENTATION_NOTES.md`, the architecture diagram is `docs/architecture.md`, every tunable value and its source is in `docs/PARAMETERS.md`, and operation is in `docs/OPERATIONS.md`. Measured results are in `eval_P5_T55.md`.

## 1. Problem and goals

P5 asks for a system that detects a vehicle in an uploaded image, identifies its make, model and year among the 196 Stanford Cars classes, gives a calibrated confidence, combines classical computer vision with deep learning, and reports accuracy, localisation, efficiency, robustness, an ablation study and a failure analysis.

We built that recognition system and then grew it into **Vahana Drishti**, a console for the Bengaluru Traffic Police that identifies vehicles in photos and video, finds a target vehicle across camera footage, reconstructs routes, and locates footage on the map.

Three principles shape every part of it:

* **No answer without evidence.** Each result shows why it was given: detection boxes, activation maps, classical panels, the closest alternatives, the clues behind a location.
* **Say so when the evidence is not there.** The decision policy backs off to the make, to "not sure" or to "outside the catalogue" instead of naming a wrong model. Locations come only from GPS in the file or exact place names on signs, and otherwise read "not known". Footage location shows a candidate route only when it beats every alternative and random decoy areas by 99 to 1.
* **No hidden constants.** Every threshold is fitted on training or validation data, measured from the data it describes, or marked as an operational choice in `docs/PARAMETERS.md`. Test labels are read once, by `scripts/eval_test.py`, and every read is logged.

## 2. Requirements and where they are met

| P5 requirement | Design | Measured in `eval_P5_T55.md` |
|---|---|---|
| Detect the vehicle | Fine-tuned YOLO26s detector for photos, and a COCO detector for video | §3.1 IoU: mean 0.949, 99.03% at IoU ≥ 0.5. mAP@0.5 0.976, mAP@0.5:0.95 0.926 |
| Identify make, model and year | Fine-tuned TResNet-L classifier, INT8, on the detector crop | §3.1 Top-1 94.69%, Top-5 98.94%, make 98.37% |
| Calibrated confidence | Temperature scaling, a decision policy with model, family, make, not-sure and outside-catalogue levels, thresholds fitted on validation | §3.1 ECE 10.4% to 1.1% |
| Classical computer vision | Quality analysis, quality-gated enhancement, GrabCut region refinement, Canny edges and contours, colour, texture and geometry features with an SVM, compared with the deep model. Score fusion was tested (E12) and left off | §2.1, §2.3 and §3.1 |
| Robustness | Quality problems corrected, keeping the original photo unless the correction makes the detection more confident | §3.1 noise, speckle and contrast corruptions |
| Efficiency | ONNX INT8 classifier of 52.9 MB, CPU inference | §3.1 p50 0.59 s, p95 1.10 s end to end on 2 threads of an Apple M3 |
| Ablation and failure analysis | One change at a time with significance tests, and the 430 wrong test images grouped by cause | §3.3 and §4 |

## 3. System overview

A browser talks only to Cloudflare, which carries the traffic through an outbound tunnel to a JarvisLabs CPU machine in Chennai. There a Docker container runs two processes: a Rust gateway, which is the only thing that accepts requests, and a Python service on the loopback interface, which runs the models. The gateway authenticates every request and passes an HMAC-signed identity to the service. Accounts and the audit log are in one SQLite database, records and cases in another, and case files and footage on the server's disk. The diagram with every component is in `docs/architecture.md` §0.

## 4. Recognition design

1. **Input checks.** Images are decoded by content, limited in size and pixel count. HEIC photos are accepted.
2. **Quality analysis and enhancement.** Darkness, low contrast, colour cast, noise and blur are measured. The flagged problems are corrected, and the detector then runs on both the corrected and the original photo. The original is kept unless the corrected one gives a more confident detection.
3. **Detection.** The most prominent vehicle is selected. Several vehicles are reported, and a photo with none says so.
4. **Classification in two catalogues.** The US catalogue covers the 196 Stanford Cars models with their years. The Indian catalogue covers 282 models sold or made in India, built from free-licensed Wikimedia Commons photos (Top-1 84.32%, and when it names a model it is right 97.5% of the time). Each catalogue has its own calibration and out-of-catalogue check. The more specific, then more confident, answer is shown, with the other beside it.
5. **Classical and deep together.** The classical branch decides how the photo is cleaned before detection, and supplies the region, edge, colour and feature panels and the comparison rows. Adding the SVM's scores to the deep scores (late fusion) was tested in E12, gained 0.12 points on validation with p = 1.00, and is switched off (`fusion: false`, `beta = 0`).
6. **Decision.** Make, model and year are shown as separate fields. When the model is uncertain, the closest match is shown and marked as not confirmed. The year is exact for the US catalogue, a pair of years when two model years cannot be told apart, and for the Indian catalogue, whose photos carry no model year, the years the model was made from Wikipedia or Wikidata (253 of 282 models). Otherwise it reads unknown.
7. **Explanation.** Results show the annotated photo, the quality-adjusted input, the GrabCut region, Canny edges, the class activation map, the colour profile and the classical, deep and hybrid answers side by side.

## 5. Console design

| Area | What it does |
|---|---|
| Identify | Photo to vehicle, make, model, year and confidence, with the explanation panels and where the photo was taken. The GPS position the camera recorded is shown as a point with its accuracy circle. Without GPS, signboard text is matched by exact name against 144,027 OpenStreetMap places in Bengaluru and drawn as a heat map, where each name's weight is shared by the places that carry it. With neither, it says the location is not known |
| Video | Every vehicle tracked with its colour, direction and a wrong-way check. Make and model are named for vehicles that cover at least 0.4% of the frame |
| Investigate | Cases with a target photo and camera footage. Sightings are scored with a calibrated match model (E19). Routes between sightings use hourly Bengaluru drive times corrected by observed journeys (E20). Moving footage without GPS is matched to the road network by its turns and speed (E21) and shown as a candidate route that the investigator confirms |
| Division | For each division administrator: activity overview, every record and case of its stations, station accounts, the security audit log, observed journeys and a spreadsheet export |

## 6. Security design

* One account per police station and one administrator per division. A station sees only its own records and cases, and an administrator only those of the division. Both rules are enforced in the service and checked by tests.
* Passwords are Argon2id hashes. Sessions are random 256-bit tokens kept only as hashes, expire after 60 idle minutes or 12 hours, and use HttpOnly, Secure, SameSite=Strict cookies.
* After 5 failed attempts from one network, sign-in for that ID pauses from that network for 15 minutes. After 50 failures from all networks together the account locks. Attempts are counted before the password check, so parallel guesses cannot pass the limit. Unknown IDs, disabled accounts and wrong passwords get the same answer.
* Sign-in is limited to devices that report a location in Karnataka and to connections from India. Everything else, including the console's own scripts, needs a signed-in session.
* Requests that change data need a same-site origin. The content security policy allows scripts only from the site and the pinned map library, which loads with an integrity hash. HTTP is redirected to HTTPS, and HSTS is set.
* The container runs as a non-root user on a read-only filesystem with every capability dropped. Uploads are checked by content and size, and GPS tracks that declare entities are refused.
* The design was tested end to end on 7 November 2026: 58 attack checks and 7 lockout checks passed, and an independent code review's findings were fixed. The remaining known limits are listed in `docs/OPERATIONS.md`.

## 7. Deployment and operations

The service is one Docker image with the models baked in. It runs on a JarvisLabs CPU machine with 8 cores and 32 GB of memory. A Cloudflare named tunnel serves `vahanadrishti.in` without opening any port on the machine. Systemd starts everything on boot. A timer backs up both databases and the case files every 6 hours, with integrity checks and 7 days kept. A watchdog restarts the container when its health check keeps failing. A scheduled GitHub job resumes the machine if the provider pauses it. Uploads are capped below Cloudflare's 100 MB request limit, so an oversized file gets the console's own message.

## 8. Data

Stanford Cars (8,144 training and 8,041 test images) for the main catalogue. Wikimedia Commons photos selected through Wikidata and Wikipedia for the Indian catalogue, split so that no photographer appears in two splits. OpenStreetMap through Geofabrik for the Bengaluru road graph and place names. Uber Movement's 2019 Bengaluru travel times for the traffic prior. KITTI raw for the footage location evaluation. Sources and licences are listed in `NOTICE.md`, the README and `assets/bengaluru/SOURCES.md`.

## 9. Key decisions and trade-offs

| Decision | Why | Cost |
|---|---|---|
| Back off instead of guessing | A wrong model name misleads an investigation more than "make only" | Fewer full answers on hard photos |
| Two catalogues | Stanford Cars has few cars seen on Indian roads | A second model to train, calibrate and serve |
| Location from evidence only | GPS and exact place names can be checked by an officer. A learned guess cannot | Many photos honestly get "not known" |
| Footage location as a confirmed candidate | Motion alone cannot prove the footage is from the search area | An extra step for the investigator |
| Rust gateway in front of Python | One small, strict entry point for authentication and limits | Two processes in one container |
| SQLite | One server, simple consistent backups | Not built for many servers |
| CPU and INT8 | Runs on an affordable CPU machine | Slower than a GPU for long videos |

## 10. Limitations

The failure analysis is in `eval_P5_T55.md` §4. Operational limits are in `docs/OPERATIONS.md` under Known limits: the sign-in location is reported by the browser, some case data goes to public routing and geocoding services, video analysis runs inside the request, small or distant vehicles are not named, and footage location needs the camera's field of view and declines when the motion fits several places.
