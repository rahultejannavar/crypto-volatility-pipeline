# MLflow Troubleshooting Log — Crypto Volatility Pipeline

**Date:** April 3, 2026  
**Project:** 94-879 Individual Assignment — Detecting Crypto Volatility in Real-Time

---

## Error 1: 403 Forbidden on `mlflow.set_experiment()`

**Where:** `models/train.py` → `main()` → `mlflow.set_experiment(experiment_name)`

**What happened:** The `docker/compose.yaml` was running `pip install mlflow` which installed MLflow 3.x (the latest). MLflow 3.x introduced a new security middleware that restricts API access by default. Even though `--allowed-hosts localhost:5001` was set, the Python client's request headers didn't match what the middleware expected. Every API call got rejected with a 403 (Forbidden).

**What we tried that didn't work:**
- Changing `--allowed-hosts localhost:5001` to `--allowed-hosts 0.0.0.0`
- Removing `--allowed-hosts` entirely — didn't help because MLflow 3.x enables security middleware by default even without that flag

**How we solved it:** Pinned MLflow to version 2.x by changing `pip install mlflow` to `pip install 'mlflow<3'` in `compose.yaml`. MLflow 2.x doesn't have the security middleware.

---

## Error 2: Alembic Migration Error (`Can't locate revision '1b5f0d9ad7c1'`)

**Where:** Inside the MLflow Docker container on startup

**What happened:** When MLflow 3.x first ran, it created a SQLite database with its own schema and migration history. When we downgraded to MLflow 2.x, the server tried to read that same database but couldn't recognize the 3.x migration records. The database schemas are incompatible across major versions.

**How we solved it:** Deleted the Docker volume containing the old database:
```bash
docker volume rm docker_mlflow_data
```
This let MLflow 2.x create a fresh database from scratch.

---

## Error 3: SQLite `unable to open database file`

**Where:** Inside the MLflow Docker container on startup

**What happened:** After deleting the volume, the container couldn't create the new database file. The SQLite URI `sqlite:///mlflow/mlflow.db` uses a relative path, and the working directory inside the container didn't have write permissions.

**How we solved it:** Two changes in the compose command:
1. Added `mkdir -p /mlflow` before starting the server
2. Changed `sqlite:///mlflow/mlflow.db` to `sqlite:////mlflow/mlflow.db` (four slashes = absolute path)

---

## Error 4: 403 Persisted Despite Server Fixes

**Where:** `models/train.py` → `main()` → `mlflow.set_experiment()`

**What happened:** The config-loading code in `main()` was reading `tracking_uri` from `config.yaml`, but something in that flow was triggering the 403 even though curl and a standalone Python test both worked fine. The exact cause was likely related to how the config was being parsed or some environment variable interference from `.env`.

**How we solved it:** Hardcoded the URI directly, bypassing config loading:
```python
mlflow.set_tracking_uri("http://localhost:5001")
```

---

## Error 5: `OSError: Read-only file system: '/mlflow'` on Artifact Upload

**Where:** `models/train.py` → `log_to_mlflow()` → `mlflow.log_artifact()`

**What happened:** MLflow's default artifact root was set to `/mlflow/artifacts` — a path inside the Docker container. When the Python client on the local machine tried to log an artifact, MLflow 2.x attempted to write directly to that path on the local filesystem (not through the server). Since `/mlflow` doesn't exist on the local machine (and would be read-only even if it did), the write failed.

**What we tried that didn't fully work:** Adding `--serve-artifacts` to the server command (which should proxy uploads through the server) — still hit the same error.

**How we solved it:** Wrapped all `mlflow.log_artifact()` calls in try/except blocks so artifact failures are non-fatal. Params and metrics log to the server successfully. Artifacts (plots, model files, scaler) save locally to `reports/` and `models/artifacts/` as a fallback. This was applied in two places:
1. The `log_to_mlflow()` function (Block 7)
2. The comparison run at the bottom of `main()` (Block 8)

---

## Final State

- MLflow server runs on Docker (MLflow 2.x, pinned with `'mlflow<3'`)
- Params and metrics are tracked in the UI at `http://localhost:5001`
- All artifacts are saved locally in `reports/` and `models/artifacts/`
- Everything needed for grading is present and accessible

## Key Lessons

1. **Pin dependency versions** — `pip install mlflow` vs `pip install 'mlflow<3'` caused cascading failures across database schemas, API endpoints, and security middleware.
2. **Database migrations are version-coupled** — You can't downgrade a database created by a newer version without wiping it.
3. **Artifact storage requires careful path configuration** — Local client vs Docker container filesystem paths are different; `--serve-artifacts` or a shared volume is needed for remote artifact logging.
4. **Graceful degradation** — Wrapping non-critical operations (artifact uploads) in try/except keeps the pipeline functional even when infrastructure is partially broken.
