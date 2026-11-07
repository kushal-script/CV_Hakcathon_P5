# Bengaluru police stations: sources

`police_stations.json` and `boundary.geojson` are built by `scripts/build_police_stations.py` from pages fetched on 6 November 2026 (15:40 to 15:52 UTC). Raw responses are cached in `data/police_raw/` (not committed). Rebuild with `.venv/bin/python scripts/build_police_stations.py`, or add `--offline` to rebuild from the cache only.

## What came from where

| Data | Source |
|---|---|
| Law-and-order, CEN and women police stations, their division, phone numbers | Bengaluru City Police, Law & Order: https://bcp.karnataka.gov.in/24/law-&-order-/en |
| Cyber Crime Police Station | Bengaluru City Police, Cyber Crime Police Station: https://bcp.karnataka.gov.in/10/cyber-crime-police-station-/en |
| Traffic police stations, traffic division, phone numbers | Bengaluru Traffic Police station locator: https://btp.karnataka.gov.in/btp_map_locations/en, checked against https://bcp.karnataka.gov.in/26/traffic-/en |
| Division structure | https://bcp.karnataka.gov.in/6/overview/en and https://bcp.karnataka.gov.in/41/-senior-officers-list-/en |
| Alternative official names | Karnataka State Police (KSP) station locator for Bengaluru City (the "Police Stations Locator" link in the BCP menu) |
| Coordinates | OpenStreetMap `amenity=police` via the Overpass API (105), Nominatim (1), BTP locator (20), KSP locator (23) |
| Boundary | OpenStreetMap relation 2020589, Bengaluru Urban district, via the Overpass API, plus a 3 km area around each station outside it |

Wikipedia was not used; the official pages state the structure. bcp.gov.in and btp.gov.in did not respond, so the karnataka.gov.in sites were used. Their servers send an incomplete certificate chain; the script adds the missing GlobalSign intermediate, pinned by its SHA-256, rather than turning verification off.

## Official structure on 6 November 2026

* Law and order: 11 divisions, each under a DCP: Central, East, West, North, South, South East, North East, Whitefield, South West, North West, Electronic City. The officers list places Central, West, North, South, North West and South West in the West zone and the other five in the East zone.
* Traffic: 4 divisions under DCPs: Traffic East, Traffic West, Traffic North, Traffic South.

## Counts

* 184 stations: 116 law and order, 52 traffic, 8 CEN, 7 women, 1 other (the city Cyber Crime Police Station).
* Five divisions: central 14, north 35, south 51, east 50, west 34.
* Positions (`coord_source`): 105 osm, 1 geocoded, 20 official_btp, 23 official_ksp, 35 missing.

## How records were matched

* Rows of the official lists are linked when their normalised names agree, when they share a phone number that appears only once in each list, or through a short table of abbreviations that one official list uses and another spells out (V V Puram, K G Nagar, KS Layout, J.P. Nagar, C K Acchukattu, International Airport and Devanahalli Airport as BIAL, Govindrajpura as Govindapura).
* OSM: 99 by name, 1 by an OSM alternative name (City Market), 1 by phone number (HAL Old Airport traffic), 2 by a near spelling within 500 m of an official locator position (J.J. Nagar, Gangammanagudi), 2 by a feature with only a generic name ("Police Station", "Women's Police Station") within 150 m of an official locator position (International Airport, Basavanagudi Women). Where two OSM features share a name far apart, the one within 10 m of the BTP position was kept (Jeevan Bheemanagar and Vijayanagar traffic).
* Geocoded: Subramanyapura PS, from a bus stop named "Subramanyapura Police Station", 90 m from the KSP locator position.
* Official locator positions (`coordinate_rule` in the JSON) are used only for stations with neither of the above: the BTP position first, then the KSP one. A position is accepted only if all of these hold:
  * inside the district polygon, it lies within 8 km of the centroid of OSM-located stations in the same official division, or of the five-way division when the official division has fewer than 3 (only Traffic North, which has 2);
  * outside the district polygon, at any distance from it, it lies within 10 km of an OSM-located station that is itself outside the district (cluster check; today only International Airport PS);
  * it is not within 50 m of another station's position of the same kind in the same locator, which catches copied values;
  * for a traffic or women station, it lies within 3 km of the OSM position of the law-and-order station of the same place.
* Each accepted or rejected position, with its distances, is written in that station's `coord_note`. Many KSP values cannot be used at all (minutes written as decimals, swapped, zero, or outside the area).
* Phone numbers come from the BCP and BTP pages; 8 digit numbers are written with the Bengaluru code 080.

## Divisions

The rule is `division_rule` in the JSON; it uses every station with a position, of any source. The centroids and all distances are in `division_centroids` and `division_mapping`.

* South East to south (5.04 km; central 6.22)
* North East to east (13.78 km; north 14.02)
* Whitefield to east (9.29 km; central 13.00)
* South West to south (6.78 km; west 7.87)
* North West to north (7.56 km; west 10.69)
* Electronic City to south (11.92 km; central 14.60)
* Traffic East, West, North and South keep their names.
* The Cyber Crime Police Station has no division on the official site; it takes central from the nearest station that has one (High Ground PS, 0.70 km) and is marked `division_inferred`.

North East is close: east wins by 0.24 km over north, from 10 of its 13 stations with positions, and adding or removing a position there can flip it. Its 13 stations are the only ones such a flip moves.

## Boundary

The first feature of `boundary.geojson` is the Bengaluru Urban district polygon, simplified from 6,177 to 2,000 vertices (Douglas-Peucker, about 5.4 m). No OSM boundary exists for the Commissionerate. Every station whose position lies outside the district gets its own feature: a 64 vertex polygon of 3 km radius around it. Three stations qualify:

* International Airport PS (OSM position);
* Devanahalli Airport Traffic PS (KSP position, 0.66 km outside the district, 0.45 km from International Airport PS);
* Devanahalli PS (KSP position, 5.28 km outside the district, 5.19 km from International Airport PS).

`inside_boundary` is computed against the district plus these areas, and every station with a position is inside.

## Known gaps and doubts

* 35 stations have no position:
  * no usable position anywhere (OSM, Nominatim and both locators give nothing usable): 10 law and order (Halasuru Gate, Govindrajpura, Bandepalya, Parappana Agrahara, Rajanakunte, Hanumantha Nagara, Puttenahalli, Bommanahalli, Govindarajanagar, Avalahalli), all 8 CEN, 5 women (Shivajinagar, North, South East, West, Whitefield);
  * KSP gives the same position for both, so neither is used: Annapurneshwari Nagar, Vidyaranyapura;
  * failed the 8 km check: North East Women PS, and Kengeri, Chikkajala, Chikkabanavara, V.V. Puram, K S Layout and Thalagattapura traffic;
  * failed the same place check: Sheshadripuram, Sanjaynagar and Yelahanka traffic.
* The 8 km check rejects some positions that agree with the same place, because a few divisions spread far from the centroid of their OSM-located stations: Kengeri traffic is 0.01 km from Kengeri PS, Thalagattapura traffic 0.04 km from Talaghattapura PS, K S Layout traffic 2.05 km from KS Layout PS, Yelahanka traffic (KSP) 0.04 km from Yelahanka PS.
* The Devanahalli positions rest on the KSP locator alone and are accepted only through the cluster check. The BTP position of Devanahalli Airport Traffic PS was rejected by the same place check (3.83 km from International Airport PS).
* No addresses: the only official addresses (KSP locator) are cut off at 45 characters at the source.
* Not included (listed in `not_included`): names that only the KSP locator lists (Central Women PS, East Women PS, South Women PS, chikkabanavara PS, Malleshwaram Traffic PS) and units that are not on the BCP or BTP station lists (CCB and three DCRE police stations).
* The BCP traffic page uses older names for four stations (Frazer Town, Indiranagar, Airport, Devanahalli; matched to the BTP names by identical phone numbers) and omits HSR Layout and J P Nagar traffic, so names and traffic divisions follow the BTP locator.
* Five women stations and all CEN stations have no place name on the BCP page; they are named after their division, as the KSP locator names them.
* "Govindrajpura" (BCP spelling) is linked to "Govindapura PS" (KSP) only through the abbreviation table.
* The North division table on the BCP page skips row numbers 14 to 19, and the Kannada version of the page still shows 8 divisions; the English page and the Overview agree on 11.
* Fields beyond the requested schema: `aliases`, `coord_note`, `inside_boundary`, `division_note` (inferred division only) and, at the top level, `coordinate_rule`, `division_centroids`, `official_structure`, `not_included`.

## Sign-in area

`signin_area.geojson` is the area a device must be in to sign in. It holds the Karnataka (relation 2019939) and Maharashtra (relation 1950884) state boundaries from OpenStreetMap contributors (ODbL), retrieved through Nominatim on 7 November 2026 and simplified by 0.005° (about 0.5 km), well inside the gateway's 2 km margin. Rebuild with `python3 scripts/build_signin_area.py Karnataka Maharashtra`. The console's own map, stations and camera checks still use `boundary.geojson`.
