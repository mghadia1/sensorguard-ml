# Make the CUDA benchmark claim survive statistics

Hand this to Claude Code from `projects/sensorguard-ml`.

## Why

The August 4 Colab T4 run reports a **2.05x median CPU-over-CUDA speedup**. The direction
is consistent — median 2.05x, mean 1.97x, 2.29x with the first run dropped as warm-up — but
it is **not statistically established**:

```
CPU  runs: 0.368  0.806  0.855  1.105  1.580   (spread 4.29x, stdev 0.444)
CUDA runs: 0.413  0.416  0.418  0.419  0.731   (spread 1.77x, stdev 0.141)

exact permutation test on the median difference, 252 splits:
    one-sided p = 0.0595
```

The fastest CPU run beat the slowest CUDA run, so the ranges overlap. With n=5 vs 5 the
smallest achievable p is 1/252 = 0.004, so this is **not** a power ceiling — five repeats
simply is not enough to separate a noisy shared Colab CPU from the GPU.

The variance result, by contrast, is solid and does not depend on a median holding up.

Read `README.md`, `docs/results.md`, `src/sensorguard/gpu_benchmark.py`,
`src/sensorguard/evidence.py`, and `notebooks/sensorguard_cuda_colab.ipynb` before changing
anything. `pytest -q tests` must stay green.

---

## Who does what — read this first

**No coding agent can run this benchmark.** The notebook executes in Mayank's browser on
Google's GPU; there is no agent with access to that runtime. That is fine, because only one
step actually needs a GPU:

| Phase | Who | Needs a GPU? |
|---|---|---|
| **1.** Steps 1–4 and 6 — code, verifier, tests | the agent, locally | **No** |
| **2.** Run the notebook, download `report.json` | Mayank, in Colab | Yes |
| **3.** Step 5 — fill the bracketed numbers, commit evidence | the agent | No |

The permutation test, the verifier, the disagreement counter and every test run on CPU
against the existing 5-run report. **Do all of phase 1 before Mayank opens Colab** — a
second GPU run costs nothing, but a second run that discovers a bug in the reporting code
wastes the trip.

The known-answer test in step 6 is what makes phase 1 checkable without a GPU: it pins the
implementation to `p = 0.0595` on the numbers already in this document.

At the end of phase 1, stop and report. Do not invent, estimate, or carry over numbers for
step 5 — those come from phase 2 and nowhere else. If a placeholder is needed to keep the
docs compiling, write `TODO(phase-3)` so it is impossible to mistake for a result.

---

## 1. Raise the repeat count

- `src/sensorguard/cli.py:58` — `--repeats` default `3` → `15`.
- `notebooks/sensorguard_cuda_colab.ipynb` — the `sensorguard gpu-benchmark` cell currently
  passes `--repeats 5`. Change to `--repeats 15`.
- `src/sensorguard/gpu_benchmark.py:72` — `repeats: int = 3` → `15`.

Add an explicit **discarded warm-up fit** per device before the timed loop, recorded in the
report as `warmup_seconds` and excluded from every statistic. The first CUDA run (0.731s)
is 75% slower than the other four; that is context creation, not compute, and it currently
contaminates the median.

## 2. Compute significance inside the benchmark, do not assert it

Add to `gpu_benchmark.py`:

```python
def permutation_p_value(cpu: list[float], cuda: list[float], *, alternative="cuda_faster"):
    """Exact one-sided permutation test on the median difference when the split
    count is tractable, otherwise a seeded Monte Carlo approximation.

    Returns (p_value, exact: bool, n_permutations: int).
    """
```

Use the exact enumeration while `C(n_cpu+n_cuda, n_cpu) <= 200_000`, otherwise 100,000
seeded random permutations (`random_state` from the existing CLI flag, so the p-value is
reproducible).

Extend the emitted `timing_seconds` block with:

```
cpu_fit_stdev, cuda_fit_stdev,
cpu_fit_spread_ratio, cuda_fit_spread_ratio,     # max/min
speedup_p_value, speedup_test_exact, speedup_test_permutations,
warmup_seconds: {"cpu": ..., "cuda": ...}
```

Keep `cpu_over_cuda_speedup` — it is not wrong, it is just incomplete.

## 3. Teach the evidence verifier the new fields

`verify_cuda_evidence` already recomputes both medians and the speedup from the raw runs and
raises if the published values disagree. Extend it in the same style:

- recompute stdev and spread ratio from the raw runs and check the published values
- **recompute the p-value from the raw runs** and check it matches
- require `len(cpu_fit_runs) == len(cuda_fit_runs) >= 10`, with a clear error naming the
  actual counts if not
- assert `warmup_seconds` values are absent from both run lists

A verifier that only checks the numbers it is handed is a checksum. Recomputing the p-value
is what makes it evidence.

## 4. Fix the parity framing — this is the substantive change

`validation_parity` reports `maximum_absolute_probability_difference: 0.2762`. That is not
parity. The frozen decision threshold is **0.66**, so a row at 0.55 on CPU can be 0.83 on
CUDA — opposite sides of the boundary. CPU `hist` and CUDA `hist` sketch quantiles
differently: these are **two different models**, not one model on two devices.

- Rename the block `validation_agreement` (keep `validation_parity` as a deprecated alias
  for one release so existing evidence files still verify).
- Add `disagreement_at_threshold`: the count and fraction of validation rows where
  `(cpu_prob >= t) != (cuda_prob >= t)` at the frozen threshold. This is the number that
  actually matters and it is currently not computed anywhere.
- **Run the threshold sweep on the CUDA probabilities** using the same validation-only
  protocol as the CPU sweep, and report `cuda_selected_threshold` beside the CPU one. If
  they differ, that is the finding — say so plainly.

Do not touch the official test split. This is all validation-only.

## 5. Rewrite the two claims

`docs/results.md` and `README.md` currently lead with 2.05x. Replace with the measured
position — fill the bracketed values from the new run, do not restate the old ones:

> CUDA was [X]x faster than CPU by median fit time over [N] timed repeats per device
> (warm-up excluded). An exact permutation test on the median difference gives p = [P].
> CPU fit times varied [A]x across repeats against CUDA's [B]x, so the more robust
> finding is consistency rather than raw speed. At the frozen 0.66 threshold the CPU and
> CUDA models disagreed on [K] of [M] validation rows; the maximum absolute probability
> difference was [D]. These are two different models, not one model on two devices —
> XGBoost's CPU and CUDA `hist` implementations sketch quantiles differently.

If the new p-value is still above 0.05, **say that**. "Not established at n=15" is a real
result and a better interview answer than a number that does not survive a follow-up
question. Do not raise n until the p-value cooperates.

## 6. Tests

- `permutation_p_value` returns exactly 1.0 for identical inputs, and its minimum attainable
  value for perfectly separated inputs
- known-answer test: the five-vs-five run above reproduces **p = 0.0595**, pinning the
  implementation against the numbers in this document
- exact and Monte Carlo paths agree to within 0.01 on a case small enough for both
- `verify_cuda_evidence` rejects a report whose published p-value is edited by hand
- `verify_cuda_evidence` rejects a report with fewer than 10 runs per device
- `disagreement_at_threshold` is 0 when both probability arrays are identical

## Definition of done

- `pytest -q tests` green, new tests included
- The notebook runs end to end on a free Colab T4 and writes a report carrying the new fields
- `sensorguard verify-evidence` passes on the new report and **fails** on the old 5-run one
  with a message naming the run count
- README and `docs/results.md` state the p-value and the disagreement count
- The old `docs/evidence/cuda-colab-t4-report.json` is kept, not overwritten — it is the
  underpowered run, and the two files side by side are the story

## Do not

- Do not delete the old evidence file or edit its numbers.
- Do not evaluate the official test split anywhere in this work.
- Do not add "GPU acceleration" or "CUDA" to the résumé skills line on the strength of this.
  Running one benchmark notebook is not the same as having built on CUDA, and the gate is
  unchanged: it goes on the résumé when it can be explained unaided, not when it passes.
