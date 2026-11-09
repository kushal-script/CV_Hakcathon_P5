# Notices

**Vahana Drishti**, the P5 vehicle recognition and investigation console.
Copyright (C) 2026 Team T55 (1RUA24CSE0216, 1RUA24CSE0228, 1RUA24CSE0229, 1RUA24CSE0275, 1RUA24CSE0244).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details. The full text is in [LICENSE](LICENSE).

Vahana Drishti is a prototype built for the Computer Vision Hackathon 2026. It is not an official Government of Karnataka service.

## Why the AGPL

The detector is trained with Ultralytics YOLO, which is itself released under the AGPL v3. The AGPL also closes the gap that a hosted service would otherwise leave open: whoever runs a changed copy of this console for other people has to give those people its source code.

In short, and without replacing the licence text:

* You may use, study, change and share the code, also commercially, as long as these notices stay and the same freedoms are passed on.
* Anyone who distributes the program or a changed version must make the complete source code available under the same licence.
* Anyone who runs a changed version as a network service must offer its users the complete source code of that version (section 13).
* There is no warranty.

## What the licence covers

The source code, configuration, tests, web console, artwork and documentation written for this project. Contributions are accepted under the same licence, as described in [CONTRIBUTING.md](CONTRIBUTING.md).

## Third-party data in this repository

These files keep their own licences. The licence above does not change them.

| Files | Source | Terms |
|---|---|---|
| `assets/bengaluru/road_graph.npz`, `places.json.gz`, `boundary.geojson`, `signin_area.geojson`, and the station coordinates in `police_stations.json` | OpenStreetMap, through Geofabrik, the Overpass API and Nominatim | Open Database Licence 1.0, © OpenStreetMap contributors |
| `assets/bengaluru/police_stations.json`: names, divisions and phone numbers | Bengaluru City Police, Bengaluru Traffic Police and Karnataka State Police websites, listed in [SOURCES.md](assets/bengaluru/SOURCES.md) | Facts published on official pages, credited to them |
| `assets/bengaluru/traffic_prior.json` | Uber Movement, Bangalore wards, 2019 Q3, © 2019 Uber Technologies, Inc. | Creative Commons Attribution Non-Commercial, under the Uber Movement data terms |
| `assets/india/`: model list, production years, photo credits in `split.csv` | Wikidata, English Wikipedia and Wikimedia Commons | Wikidata CC0, Wikipedia CC BY-SA 4.0. The licence and author of every photo are listed in `split.csv` |
| `assets/autorickshaw/sources.csv`: credits for the auto-rickshaw photos | Wikimedia Commons | Each photo keeps its own free licence (CC BY, CC BY-SA, CC0, public domain, GFDL or Free Art Licence), listed with its author and page |
| `assets/taxonomy/`, `assets/splits/` | Stanford Cars class names and image IDs, COCO image IDs | Names and IDs only. No images |

## Not in this repository

* **Images.** Stanford Cars (Krause et al., 2013) is downloaded separately, the Indian catalogue photos are fetched again from Wikimedia Commons by `scripts/build_india_dataset.py`, the auto-rickshaw photos by `scripts/build_autorickshaw_set.py`, and KITTI raw recordings, which are licensed CC BY-NC-SA 3.0, are fetched for evaluation only by `scripts/videoloc/kitti_fetch.py`. Each keeps the terms of its provider.
* **Model weights.** `models/` is not committed. The weights are fine-tuned from pretrained models and trained on the datasets above, so anyone who shares them must follow those terms as well.

## Services used while the console runs

The console sends requests to these services. Their own terms and usage policies apply.

| Service | Used for |
|---|---|
| OpenFreeMap (OpenMapTiles, © OpenStreetMap contributors) | Street map tiles and style |
| Sentinel-2 cloudless 2021 by EOX IT Services GmbH, containing modified Copernicus Sentinel data, under the terms at https://s2maps.eu | Satellite imagery |
| MapLibre GL JS 4.7.1 from unpkg, BSD 3-Clause | Map rendering in the browser |
| Photon (photon.komoot.io), OSRM (router.project-osrm.org), Valhalla (valhalla1.openstreetmap.de) | Geocoding, routing and map matching, reachable areas |
| KartaView and Mapillary | Links to street photos for the investigator |

## Software dependencies

Python packages (`requirements-app.txt`, `requirements-train.txt`) and Rust crates (`gateway/Cargo.toml`) keep their own licences. Ultralytics YOLO, used to train the detector, is AGPL v3.
