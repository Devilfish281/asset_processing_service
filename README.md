### To run the project, follow these steps:

poetry run asset-processing-service

## How to run tests from the command line

Think of `--tests 1,4` like checking boxes.

- You type numbers in the terminal.
- The program reads them.
- It turns on specific “test switches” (flags) based on which numbers you gave it.
- Then it creates only those test jobs.

Example mapping:

- `1` → create a normal job
- `2` → create a “max attempts” job
- `3` → create an “in progress” job with a recent heartbeat
- `4` → create a “stuck” job with an old heartbeat

And we keep the ids of the jobs we created, so we can delete exactly those jobs at shutdown.

---

## How to run from the command line (Windows)

From your project root (where `pyproject.toml` is), run:

```bash
poetry run python -m asset_processing_service.main --tests 1
```

Or multiple tests:

```bash
poetry run python -m asset_processing_service.main --tests 1,3,4
```

No `--tests` means “normal run” (unless `TESTING_FLAG=true` in your `.env`).

---

---

## How you run pytest

- Run all tests:
  - `poetry run pytest`

- Run just the smoke test:
  - `poetry run pytest -k smoke`

---
