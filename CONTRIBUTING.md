# Contributing to Vahana Drishti

Thank you for helping. This guide is for everyone who works on the repository: team members, reviewers and outside contributors. It explains the principles the system is built on, how to set up, and what a change needs before it is merged.

## The four principles

Every change is reviewed against these. They are why officers can trust what the console shows.

1. **No answer without evidence.** Each result shows why it was given: detection boxes, activation maps, the classical panels, the closest alternatives, the clues behind a location.
2. **Say "not known" instead of guessing.** When the evidence is not there, the system backs off to the make, to "not sure", to "outside the catalogue" or to "location not known". A feature that cannot tell it is wrong does not ship.
3. **No hidden constants.** Every tunable value lives in `configs/` or in the calibrated files under `models/`, and [docs/PARAMETERS.md](docs/PARAMETERS.md) states whether it was fitted on validation data, measured from the data it describes, or is an operational choice.
4. **Test labels are read once.** Tuning uses the validation split only. Test labels are read only by `scripts/eval_test.py` and `scripts/eval_india.py`, and every read is logged in `experiments/test_log.csv`.

## Setting up

You need Python 3.10 or newer, a stable Rust toolchain for the gateway, and Docker if you want to run the full image.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt -r requirements-app.txt pytest httpx
```

The [README](README.md#quick-start) explains how to build the dataset caches, run the API on its own and run the gateway and API together in Docker. The packaged models are not in the repository. See [Reproduction](README.md#reproduction) for how they are trained and packaged.

## Running the tests

```bash
.venv/bin/python -m pytest -q
(cd gateway && cargo test)
```

`tests/test_architecture.py` checks that every `p5.*` module named in [docs/architecture.md](docs/architecture.md) exists, so renaming or adding a module means updating the diagram in the same change.

## Making a change

1. Branch from `main` with a short descriptive name.
2. Keep one topic per pull request, small enough to review in one sitting.
3. Update the documentation in the same pull request:
   * `docs/PARAMETERS.md` for every new or changed tunable value, with its origin;
   * `docs/architecture.md` for new or renamed modules and interfaces;
   * `docs/OPERATIONS.md` for settings, deployment and known limits;
   * the README feature table for anything an officer will see;
   * `assets/bengaluru/SOURCES.md` and `NOTICE.md` for new data.
4. Open the pull request and fill in its checklist.

## Code style

* **Code explains itself.** Choose clear names and small functions. Add a comment only where the reason for something is not visible in the code. No boilerplate docstrings.
* **Python** follows the surrounding code: `pathlib` for paths, settings from `configs/`, errors that the API turns into clear messages instead of stack traces.
* **Rust** in `gateway/` builds without warnings and keeps every security check on the server side.
* **Web console** (`web/`):
  * build the page with `textContent` and the `h()` helper in `web/app/app.js`, never with `innerHTML`;
  * no inline scripts or styles, because the content security policy forbids them. A new outside origin needs a pinned version, an integrity hash and an update to the policy in `gateway/src/main.rs`;
  * use the colour tokens in `web/public/theme.css`. The palette is khaki, ink, turmeric, red and green, with no blue;
  * check every screen in light and dark mode and at phone width;
  * raise the `?v=` asset version in `web/public/login.html`, `web/app/index.html` and the imports in `web/app/app.js` whenever a script or stylesheet changes, because Cloudflare keeps assets for 4 hours.

## Experiments and evaluation

* Start every experiment with `python3 scripts/new_experiment.py --title "E22: what is tested" --type deep --author yourname`. It creates a timestamped folder from `experiments/_template` and adds a row to `experiments/INDEX.md`. Record the config, metrics and a short write-up, including runs that did not work.
* Compare with significance tests where the ablations do, and report the numbers you measured, not the numbers you hoped for.
* A new test run needs a reason and a new row in `experiments/test_log.csv`. Never tune anything after looking at test results.

## Commit messages

* Start with one summary line in plain English that says what changes for the user, for example `Show a heat map of where a photo's signboard clues point on the Bengaluru map`.
* When more is needed, add a blank line and then bullet points that start with `*`.
* Do not wrap lines. The only line breaks are between bullets and between paragraphs.
* Avoid hyphens and semicolons unless they are part of a name or of code.
* Do not add trailers.
* Make few, meaningful commits: one per finished change. Squash work in progress before it is merged.

## Data and privacy

* Add only data whose licence allows it in a public repository, and record its source, retrieval date and licence in the relevant `SOURCES.md` and in `NOTICE.md`.
* Datasets, photos and model weights stay out of git (`data/`, `cache/`, `runs/` and `models/` are ignored). Add a script that downloads and builds them instead, as the existing `scripts/build_*.py` do.
* Never commit photos, footage or case data from real investigations.

## Security

* Never commit credentials, `.env` files, the `state/` folder or a `credentials.csv` from `vahana-gateway provision`. One-time passwords are handed to stations outside the repository.
* Report a vulnerability privately through **Security → Report a vulnerability** on this repository, not in a public issue. Include the steps to reproduce it and what an attacker could reach. Known limits that are already documented are listed in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Deployment

Only maintainers deploy to https://vahanadrishti.in. The steps, settings, backups and recovery are in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Licence of contributions

By contributing you confirm that you have the right to submit the work, and you agree that it is released under the GNU Affero General Public License v3 or later, like the rest of the project. See [NOTICE.md](NOTICE.md).

## Conduct

Be respectful, patient and specific. Critique code and results, not people. Harassment and discrimination are not accepted, and maintainers may remove such comments and block those who post them.
