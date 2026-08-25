# Make SQL true — SensorGuard SQLite backend

Hand this to Claude Code from `projects/sensorguard-ml`.

## Why

SQL is the single most-requested skill in the data-science internship postings and it
is currently absent from every project. It is also the cheapest gap to close: this
project already loads a 10,000-row table and filters it in pandas, which is a SQL query
written in the wrong language.

**This is not resume decoration.** After this ships, "SQL" and "SQLite" go on the
resume because they are used in a public repo with tests, and you can answer "where did
you use SQL?" with a file path.

Scope is deliberately small — one afternoon, not a database course.

---

You are working in `sensorguard-ml`. Read `README.md`, `docs/results.md`, and
`src/sensorguard/` before changing anything. `pytest -q tests` must stay green.

## What to build

### 1. `src/sensorguard/database.py`

Load the UCI AI4I CSV into a local SQLite file, then read training data back out with
real SQL instead of pandas filtering.

- `build(csv_path, db_path) -> Path` — create the database and load the table.
  Define an explicit schema with typed columns and a primary key on the record id;
  do **not** use `pandas.to_sql` with inferred types. Writing the `CREATE TABLE` by
  hand is the part worth learning and the part worth talking about.
- `query(db_path, sql, params=()) -> list[dict]` — thin `sqlite3` wrapper using
  parameterised queries, `row_factory = sqlite3.Row`.
- `load_features(db_path) -> tuple[np.ndarray, np.ndarray]` — replaces the current
  CSV read. The column selection that drops identifiers and the five target-derived
  failure flags must happen **in the SELECT statement**, not in Python afterwards.
  That is the leakage-prevention story told in SQL.

Add an index on the target column and note in a comment why (the queries in step 2
group by it).

### 2. `src/sensorguard/queries.py` — the EDA the JD asks for

Named queries returning summary statistics, each a module-level constant so they are
readable and testable:

- `FAILURE_RATE_BY_TYPE` — failure count and rate grouped by product quality type
  (L/M/H), using `GROUP BY` and a `CAST(... AS REAL)` ratio
- `TORQUE_PERCENTILES` — approximate deciles via `NTILE(10) OVER (ORDER BY torque)`
- `FAILURES_BY_TOOL_WEAR_BUCKET` — `CASE WHEN` bucketing on tool wear, with counts
- `CORRELATED_EXTREMES` — rows in the top decile of both torque and tool wear, using
  a CTE

Use window functions and a CTE at least once each. They are what distinguishes "I can
SELECT" from "I know SQL" in an interview.

Expose `resume-radar`-style CLI access: `sensorguard eda --db path` prints each
query's result as a small table.

### 3. Wire it in

`load_features` becomes the default data path for training. Keep the CSV loader as a
fallback so nothing breaks if the database is absent, and have `train` build the
database automatically on first run.

**The model results must not change.** Same seed, same splits, same metrics —
0.701 F1, 0.974 ROC-AUC. If they move, the SQL is selecting different rows and that is
a bug, not an improvement. Assert this in a test.

### 4. Tests

- schema is created with the expected columns and types
- `load_features` returns arrays identical to the existing CSV path (`np.allclose`)
- the identifier and five leak columns are absent from the SELECT output — the
  regression that matters
- each named query runs and returns non-empty, correctly-shaped rows
- `query()` is parameterised: a value containing `'; DROP TABLE` is treated as data
- database is rebuilt idempotently — running `build` twice does not duplicate rows

Use a temporary database in tests; never write into `data/`.

### 5. Docs

- `README.md`: a "Data access" section showing the SQLite path and one example query
- `docs/results.md`: state that the metrics are unchanged and why that was the goal

## Definition of done

- `pytest -q tests` green, new tests included
- `sensorguard eda` prints four query results against the real dataset
- Training from SQLite reproduces the existing metrics exactly
- No new runtime dependency — `sqlite3` is in the standard library

## Then, and only then

Add to the resume skills line: `SQL, SQLite`. And to the SensorGuard entry, one bullet:

> Loaded the 10,000-row dataset into **SQLite** with an explicit schema and selected
> training features in SQL, excluding identifiers and five target-derived flags at the
> query level; window-function and CTE queries summarise failure rates by product type
> and tool-wear bucket.

Before it goes on the resume, be able to say out loud: what a CTE is, why the leak
columns are excluded in the SELECT rather than after it, and what `NTILE(10)` returns.
That is the same gate every other project on the resume had to pass.
