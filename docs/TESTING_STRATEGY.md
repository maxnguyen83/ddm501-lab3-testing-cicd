# Testing Strategy: Movie Rating Prediction API

This document explains how we test the movie rating system, what each layer of
tests can and cannot catch, why the thresholds are where they are, and which
real bugs the tests found. All numbers come from our own runs (seeded training,
`random_state=42`, MovieLens 100K, Python 3.12 on macOS and Python 3.10 on
Linux unless stated otherwise).

## 1. System under test

| Component | File | Risk we care about |
|---|---|---|
| Training | `scripts/train_model.py` | Bad data in, silently worse model out; non-reproducible runs |
| Model wrapper | `app/model.py` | Wrong post-processing (range, rounding), wrong ID handling |
| Request/response contract | `app/schemas.py` | Invalid or oversized input reaching the model |
| HTTP service | `app/main.py` | Wrong status codes, leaking internals, starting without a model |
| Container | `Dockerfile` | Image that builds but cannot serve, or reports healthy when it is not |

The model is a Surprise SVD (100 factors, 20 epochs) trained on all 100,000
MovieLens 100K ratings. It always returns a number: unknown users or movies fall
back to the global mean plus whatever bias is known.

## 2. The ML testing pyramid applied

```
                 ┌──────────────────────┐
                 │  System / container  │  CI docker job, CD smoke tests
                 ├──────────────────────┤
                 │   Model validation   │  5-fold CV thresholds (scripts/validate_model.py)
                 ├──────────────────────┤
                 │   Model behaviour    │  31 tests  tests/model/
                 ├──────────────────────┤
                 │     Data quality     │  49 tests  tests/data/
                 ├──────────────────────┤
                 │     Integration      │  53 tests  tests/integration/
                 ├──────────────────────┤
                 │         Unit         │  80 tests  tests/unit/
                 └──────────────────────┘
```

Measured on the final code (213 tests, about 2 s for the whole suite once the model is trained):

| Layer | Tests | Runtime | Coverage of `app/` if run alone | Needs trained model? | Needs dataset? |
|---|---:|---:|---:|---|---|
| Unit | 80 | 0.37 s | 80% | Some tests (others use a stub) | No |
| Integration | 53 | 0.36 s | 87% | Yes | No |
| Data | 49 | 0.66 s | 55% | One lineage test | Yes |
| Model behaviour | 31 | 0.40 s | 66% | Yes | No (uses the model's trainset) |
| **All** | **213** | **~2 s** | **100% lines and branches** | | |

The suite is fast enough that every layer runs on every push. The pyramid shape
is about *what* is tested at each level, not about saving minutes.

### 2.1 Unit tests (`tests/unit/`)

**What they test.**
- `test_schemas.py`: every Pydantic rule on its own: required fields, empty and
  whitespace-only IDs, the 50-character ID bound (accepted at 50, rejected at 51
  and at 10,000), the 100-item batch bound, int IDs (Pydantic v2 rejects them,
  it does not coerce), rating bounds on the response, and that single and batch
  inputs follow identical rules.
- `test_model.py`: the wrapper. Tests that need real predictions use the trained
  model. Tests of the wrapper's *own* logic load a tiny `StubAlgo` from a
  temporary pickle, so rounding (3.14159 -> 3.14), clipping (7.3 -> 5.0,
  -2.0 -> 1.0), whitespace stripping, call order in `predict_batch`, and the
  "model not loaded" / missing file / corrupt pickle paths are tested
  deterministically, independent of what SVD happens to predict.
- `test_utils.py`: configuration parsing from environment variables, the
  `get_model()` / `reset_model()` singleton, and that the package version, API
  version and `pyproject.toml` version agree.

**What they catch.** Contract regressions (someone loosens a bound), broken
post-processing, error paths that are hard to trigger through the API.

**What they cannot catch.** Whether the model is any good, whether the parts
work together (routing, lifespan, serialisation), or anything about the data.

### 2.2 Integration tests (`tests/integration/test_api.py`)

**What they test.** The real FastAPI app through `TestClient`, with the
lifespan handler loading the real model. Covered: response structure and types,
`/predict` equals the wrapper's own prediction, batch equals single, batch order
is preserved, whitespace in IDs is normalised the same way on both endpoints,
404 for unknown routes, 405 for wrong methods (GET `/predict`, POST `/health`,
GET `/predict/batch`), 422 for missing fields, empty body, invalid JSON, int IDs,
form-encoded bodies and oversized IDs (single and inside a batch), the batch size
bounds (100 accepted, 101 and 0 rejected), 503 when the model is missing,
generic 500 without leaking exception text, and startup with a missing model
(service still starts and reports `unhealthy`).

**What they catch.** Wiring problems: wrong status codes, the model not being
loaded at startup, validation not applied on one endpoint, error handlers
leaking internals.

**What they cannot catch.** Network, process and container behaviour (ports,
uvicorn workers, the Docker health check), concurrency and load.

### 2.3 Data tests (`tests/data/test_data_quality.py`)

**What they test.**
- The starter's rule tests on the small `sample_ratings` fixture.
- A data contract on the *real* training data, loaded with the same Surprise
  loader that `train_model.py` uses: schema and types, exact published volume
  (100,000 ratings, 943 users, 1,682 movies), no nulls, whole stars 1-5 only, no
  duplicate (user, movie) pairs, every user has at least 20 ratings, timestamps
  inside the collection window (1997-09-20 to 1998-04-22), mean/std bands,
  star-share bands, matrix density, and that the served model's trainset has
  exactly the validated counts (lineage).
- Fixture integrity: the `actual_rating` values that model tests rely on really
  are in `u.data`, and the "unknown" IDs really are unknown.
- **Negative tests.** The rule checker (`find_quality_issues`) and the
  distribution checker (`distribution_issues`) are fed deliberately corrupted
  records (rating 6, 0, -1, None, "4", True, NaN; empty, blank and int IDs;
  missing field; duplicate pair; constant ratings; all 1-star). Each corruption
  must be reported. This proves the data checks can fail; a check that never
  fails is not a test.

**What they catch.** Truncated or partial downloads, a changed file format, a
scale change (e.g. 0-10 ratings), duplicated rows, a constant or collapsed
rating column, and stale fixtures.

**What they cannot catch.** Whether production traffic looks like the training
data (no drift monitoring here), label quality (a user's 1-star might be a
mistake), or bias in who rates what (popular movies get far more ratings: 338
movies have 100+ ratings, 141 have exactly one).

### 2.4 Model behaviour tests (`tests/model/test_model_behavior.py`)

CheckList-style tests on the trained artefact. Expected directions are derived
from the ratings stored inside the model's own trainset, so the tests need no
extra download and always match the model being tested.

| Type | Test | Measured on our model |
|---|---|---|
| Invariance | Same input twice / five times gives the same output | identical |
| Invariance | Batch order, batch vs single, a pair alone vs inside 50 others | identical |
| Invariance | Whitespace around IDs (`" 196"`, `"\t196\n"`) | identical |
| Invariance | Reloading the pickle (new container) | identical |
| Directional | Per user: mean prediction on their 5-star movies > on their 1-star movies | 715 / 715 users, smallest gap 0.40, median 1.63 stars |
| Directional | 10 best-rated popular movies > 10 worst-rated popular movies, per user | 943 / 943 users |
| Directional | 50 most generous users > 50 harshest users, same movie | smallest gap 1.19 stars over the 10 movies checked |
| Directional | Cold-start user still prefers acclaimed movies | 4.49 vs 2.76 |
| Directional | Known 1-star pairs predicted lower than known 3-star pairs | 2.30 vs 3.71 |
| MFT | Known users, unknown users/movies, unknown both (= global mean 3.53) | all in [1, 5] |
| MFT | Known user is personalised (differs from cold-start) | 3.58 vs 4.09 |
| Performance | MAE on a fixed sample of 2,000 training ratings | 0.528 |
| Performance | Share of sampled errors > 2 stars | 0.25% |
| Performance | 1,000 single predictions | about 2 ms |
| Robustness | `"0196"` is not user `"196"` (IDs are opaque strings) | cold-start value |
| Robustness | Int IDs raise `TypeError` instead of silently cold-starting | raises |

**What they catch.** A model that lost personalisation (collapsed to biases or a
constant), a broken training run (we checked: an under-trained model fails four
of these tests, see section 3), non-determinism, and serialisation problems.

**What they cannot catch.** Generalisation to *unseen* ratings: these tests run
on the training data the model has already seen. Held-out quality is the job of
the model validation gate below. They also say nothing about fairness, diversity
or popularity bias in recommendations.

### 2.5 Model validation gate (`scripts/validate_model.py`)

`train_model.py` writes `models/metrics.json` with 5-fold cross-validation
metrics for the shipped configuration and for a `BaselineOnly` (bias-only)
model on the **same seeded folds**. `validate_model.py` fails the job if:

- CV RMSE > 0.95,
- CV MAE > 0.75,
- SVD does not beat `BaselineOnly` RMSE (if it does not, the simpler model should ship),
- the saved artefact does not load through the API wrapper or predicts outside [1, 5].

It also writes the result table to the GitHub job summary.

### 2.6 System / container tests

The CI `docker` job builds the image with the model the tests just used, waits
for the Docker `HEALTHCHECK` to report `healthy`, then calls `/health` and
`/predict` over HTTP. The CD pipeline repeats this on the release candidate and
again on the image pulled back from GHCR by digest. There is no load test; see
section 6.

## 3. Thresholds and why

### Model validation thresholds

We trained deliberately broken variants on the same folds to see where a
threshold must sit to catch them:

| Configuration (5-fold CV, seed 42) | RMSE | MAE | Gate result |
|---|---:|---:|---|
| **Shipped SVD** (100 factors, 20 epochs, lr 0.005, reg 0.02) | **0.9350** (std 0.0036) | **0.7372** | pass |
| BaselineOnly (biases only) | 0.9436 | 0.7481 | used as the bar to beat |
| SVD, 5 epochs (under-trained) | 0.9581 | 0.7601 | fails all three metric checks |
| SVD, lr 0.0005 (learning rate /10) | 0.9820 | 0.7841 | fails |
| SVD, reg 1.0 (over-regularised) | 0.9871 | 0.8019 | fails |
| SVD, lr 0.05 (learning rate x10) | 0.9877 | 0.7751 | fails |
| SVD, 1 epoch | 1.0097 | 0.8149 | fails |
| Global mean | 1.1257 | | fails |
| NormalPredictor (random) | 1.5243 | 1.2245 | fails |

- **RMSE <= 0.95 and MAE <= 0.75** sit about 4 CV standard deviations above
  the shipped model, so normal noise does not trip them, while every broken
  variant we tried fails. A tighter bar (e.g. 0.94) would also reject
  BaselineOnly, but would leave only 1.4 standard deviations of headroom.
- **Beat BaselineOnly** covers the gap the absolute bar leaves: BaselineOnly
  passes 0.95, so the relative check is what stops us from shipping a model whose
  latent factors add nothing.

### Behaviour and performance test thresholds

| Threshold | Value | Measured | Reason |
|---|---|---|---|
| Known pair error | < 2.5 stars | max 1.83 | See finding F9: the starter's 1.5 fails for pair (166, 346) with every seed |
| Loved > hated movies, share of users | >= 99% | 100% | 1% slack for borderline users; 5-epoch model drops to 97.8% and fails |
| Acclaimed > panned movies, share of users | >= 99% | 100% | same idea |
| Generous vs harsh users gap | >= 0.5 stars | min 1.19 | half a star is a clear, visible difference |
| Cold-start acclaimed vs panned gap | >= 1.0 star | 1.74 | item biases alone should separate them |
| Sample MAE (2,000 training ratings) | < 0.70 | 0.528 | training fit must not be worse than held-out MAE (0.737); 5-epoch model scores 0.712 and fails |
| Share of errors > 2 stars | <= 2% | 0.25% | 5-epoch model scores 2.55% and fails |
| Known pair error (extreme) | < 3.0 stars | 1.83 | starter's bound; kept |
| Latency, 1,000 predictions | < 0.5 s | ~2 ms | 250x headroom so a slow shared CI runner does not cause flaky failures, but an accidental O(n) lookup per call would |
| Coverage | >= 80% | 100% lines and branches | course minimum; enforced by `--cov-fail-under=80` in CI and `fail_under = 80` in `pyproject.toml` |

To check that these thresholds discriminate, we retrained with `n_epochs=5`:
`validate_model.py` failed RMSE, MAE and the BaselineOnly check, and four
behaviour tests failed (known-pair error 2.58, loved > hated for 699/715 users,
sample MAE 0.712, 2.55% large errors).

### Data thresholds

| Check | Band | Measured |
|---|---|---|
| Mean rating | 3.0-4.0 | 3.53 |
| Std of ratings | 0.9-1.3 | 1.13 |
| Each star value's share | 3%-50% | 6.1% (1 star) to 34.2% (4 stars) |
| Matrix density | 1%-10% | 6.3% |
| Ratings per user | >= 20 | min 20, median 65 |

The bands are wide on purpose: they should trip on a broken file (wrong scale,
collapsed column), not on a normal refresh of the data.

## 4. Where each test runs

| When | What runs | Fails the change if |
|---|---|---|
| `git commit` (pre-commit) | trailing whitespace, EOF, YAML/TOML/JSON syntax, large files, merge markers, private keys, black, isort, flake8, mypy | any hook fails |
| `git push` (pre-push hook) | unit tests | a unit test fails |
| Every push / PR (`ci.yml`) | lint, mypy, train, **all** tests on Python 3.10 and 3.12 with coverage >= 80%, Docker build + health + predict | any job fails |
| Model-related changes, weekly, manual (`model-validation.yml`) | data tests **before** training, train, CV thresholds, behaviour tests | any gate fails |
| Version tag `v*` (`cd.yml`) | all tests + CV thresholds again, smoke test candidate image, push to GHCR, smoke test the pulled image, GitHub Release | any step fails, nothing is published |

## 5. Findings: bugs the tests uncovered

All of these were reproduced on the original starter code before fixing. Run
against the original `app/` (plus the two bound constants the tests import), the
new suite fails 17 of 213 tests.

| # | Finding | Evidence (before) | Fix | Regression test |
|---|---|---|---|---|
| F1 | The starter's `TestClient(app)` fixture never ran the startup event, so the model was never loaded and the provided test `test_predict_valid_request_returns_200` failed | `assert 503 == 200`; 88 passed, 1 failed, 72% coverage | Fixture uses `with TestClient(app)`; app moved from deprecated `@app.on_event` to a `lifespan` handler | every integration test; `TestStartup` |
| F2 | Batch items skipped the whitespace check and stripping that single requests have | batch `"   "` returned 200; batch `" 196"` was looked up verbatim, treated as a new user: 4.09 instead of 3.58 from `/predict` | Shared `_UserMovieIds` base class for `PredictionRequest` and `PredictionItem` | `test_batch_whitespace_id_returns_422`, `test_batch_strips_whitespace_like_single`, `TestPredictionItem` |
| F3 | The model wrapper accepted `None`, `""` and int IDs and silently returned the cold-start rating | `predict(196, 242)` = 3.53 (global mean), `predict(None, "242")` = 4.09 | `_clean_id()` raises `TypeError` / `ValueError` and strips whitespace | `test_predict_with_none_user_id`, `test_integer_ids_are_rejected_not_cold_started` |
| F4 | 500 responses returned `str(exception)` to the client | body `{"detail": "internal detail: /srv/models/svd_model.pkl corrupted"}` | Generic `"Prediction failed"`; details only in the server log | `test_predict_internal_error_returns_generic_500` (single and batch) |
| F5 | `train_model.py` crashed in CI: `Dataset.load_builtin` asks for confirmation with `input()` | `EOFError: EOF when reading a line` with stdin closed | `prompt=False` | CI and model-validation train step |
| F6 | Training was not reproducible (unseeded SVD and CV folds) | different model and metrics each run | `random_state=42` for SVD and `KFold`; two runs now give a byte-identical pickle (same SHA-1) | thresholds in section 3 are stable |
| F7 | The image could not be built, and its health check could never pass | `scikit-surprise==1.1.3` is source-only: `gcc failed: No such file or directory` in `python:3.10-slim`; the `curl` health check reports `/bin/sh: curl: not found` | Pin `scikit-surprise==1.1.5` (wheels for cp310-cp314); Python health check that also requires `model_loaded == true` | CI `docker` job waits for `healthy` |
| F8 | `/health` returns 200 even with no model, so an HTTP-status health check marks a model-less container healthy | container with `MODEL_PATH` pointing nowhere: status-only probe reported `healthy` | Docker `HEALTHCHECK` reads `model_loaded` from the body; the same container is now `unhealthy` | CI `docker` job; manual check in README |
| F9 | The starter's suggested behaviour expectations were wrong for its own fixture: per-pair error < 1.5 and MAE < 1.0 over the 5 known pairs | pair (166, 346) error 1.78-2.29 across 20 training seeds (fails 20/20); 5-pair MAE 1.03 | Known-pair bound 2.5; average error measured on a fixed 2,000-rating sample | `test_predictions_are_reasonable`, `test_average_error_acceptable` |
| F10 | The starter `.gitignore` entry `data/` also matched `tests/data/`, so the data tests were never committed: CI would silently skip them and `pytest tests/data/` in the model-validation workflow would collect nothing | `git ls-files` on a fresh repo listed 31 files, no `tests/data/` | Anchored to the root: `/data/` | `git check-ignore tests/data/test_data_quality.py` returns nothing; 33 files tracked |
| F11 | Smaller issues | mypy error on `model: MovieRatingModel = None`; Pydantic `protected_namespaces` warning for `model_version` / `model_loaded` on every import | `Optional[...]`; `ConfigDict(protected_namespaces=())` | mypy in CI; `test_schemas_import_without_warnings` |

Installing `scikit-surprise==1.1.3` with current pip also fails outside Docker
(its `setup.py` fetches numpy through an isolated build without pip, and the
package imports `pkg_resources`, which setuptools 81+ no longer ships). This is
why the pin was raised to 1.1.5 rather than adding a compiler to the image.

## 6. Known gaps and limitations

- **No production monitoring.** Data drift, prediction drift and live error
  rates are out of scope for this lab (they belong to a monitoring setup).
- **Behaviour tests use training data.** They prove the model learned the
  expected structure, not that it generalises; generalisation is covered only by
  the CV gate.
- **Cold start is weak by design.** Every unknown user gets the same answer per
  movie; `"0196"` and `"196"` are different users. We keep IDs as opaque strings
  because normalising them could merge two real IDs.
- **No load or concurrency tests.** The service is single-process uvicorn; a
  locust or k6 run would be the next step before real traffic.
- **CORS** allows any origin with credentials. There is no authentication or
  cookie in this API, so we noted it rather than changed it.
- **Thresholds are tied to MovieLens 100K.** A different dataset needs the
  experiment in section 3 repeated, not just the numbers copied.
- **Fairness / popularity bias** is not measured.

## 7. How to run

```bash
python scripts/train_model.py              # also downloads MovieLens 100K once
pytest tests/ --cov=app --cov-fail-under=80
pytest tests/unit/ -v                      # one layer
python -m scripts.validate_model           # CV threshold gate
pre-commit run --all-files                 # hooks
```
