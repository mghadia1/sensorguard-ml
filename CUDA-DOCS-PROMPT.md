# Phase 3 — correct the CUDA claim in the docs

Hand this to Codex or Claude Code from `projects/sensorguard-ml`.
This is the follow-up to `CUDA-STATS-PROMPT.md`, phases 1 and 2 of which are complete.

## What happened

The 15-repeat rerun with a discarded warm-up fit **reversed the headline result**. The
published 2.05x CUDA speedup was warm-up contamination: the first CPU fit took 1.307 s
against a steady-state median of 0.274 s, and in the old 5-run design that first fit sat
inside the sample and inflated the CPU median to 0.855 s.

`README.md` and `docs/results.md` still claim 2.05x. **They are now wrong and must be
corrected.**

## Verified numbers — use these exactly

Every value below was recomputed from the raw run arrays in the new report and matched the
published fields exactly. Do not recompute and do not round differently.

| Quantity | Value |
|---|---|
| Timed repeats per device | 15 (warm-up excluded) |
| Discarded warm-up fit | CPU 1.3072 s · CUDA 0.7949 s |
| Median fit time | CPU **0.2742 s** · CUDA **0.4248 s** |
| `cpu_over_cuda_speedup` | **0.6456** — i.e. **CPU was 1.55x faster** |
| Permutation p, "CUDA faster" | **0.9999** (100,000 permutations, not exact) |
| Permutation p, "CPU faster" | **0.0003** |
| Standard deviation | CPU **0.0928** · CUDA **0.0340** |
| Spread ratio (max/min) | CPU **2.25x** · CUDA **1.25x** |
| Runs beating the other device | 13 of 15 CPU runs were faster than every CUDA run |
| Disagreement at threshold 0.66 | **8 of 2000** validation rows (0.4%) |
| Sweep-selected threshold | CPU **0.84** · CUDA **0.76** |
| Max absolute probability difference | **0.2762** |

The validation agreement metrics (AP 0.7769458889549244, ROC-AUC 0.9660211910851297, and
the CUDA counterparts) are **byte-identical to the August 4 run**. That is a reproducibility
signal under `random_state=42`, not a copy-paste error — say so rather than treating it as
suspicious.

## 1. Commit the new evidence

Save the new report as `docs/evidence/cuda-colab-t4-report-n15.json`. Add its
`source_file_sha256` the same way the existing file records it.

**Keep `cuda-colab-t4-report.json` exactly as it is.** Do not edit or delete it. The two
files side by side are the point: one underpowered run and one that corrected it.

Add `docs/evidence/README.md` (or a section in `docs/results.md`) stating plainly which file
supersedes which and why.

## 2. Handle the verifier conflict

`CUDA-STATS-PROMPT.md` step 3 asked the verifier to reject reports with fewer than 10 runs
per device. The old file has 5, so a blanket check will now fail CI on an artifact we are
deliberately keeping.

Resolve it explicitly — do not weaken the check:

- `verify_cuda_evidence` keeps requiring >= 10 runs **for current evidence**.
- Add a `superseded_by` field to the old report naming the n15 file, and have the verifier
  return a `superseded` status for such files instead of raising.
- A file with fewer than 10 runs and **no** `superseded_by` field must still be a hard error.

That way the old run stays in the repo, is machine-readably marked as retired, and cannot be
cited by accident.

## 3. Rewrite `docs/results.md`

Replace the "Verified CUDA comparison — August 4, 2026" section. Keep the old section as a
clearly labelled subsection titled something like "Superseded: the underpowered August 4
run", with a one-line note on why it was wrong. Then state the corrected result:

> **Corrected CUDA comparison — 15 repeats, warm-up excluded.** With a discarded warm-up fit
> and 15 timed repeats per device, the CPU was faster than the Tesla T4: median 0.2742 s
> against 0.4248 s, a CPU-over-CUDA ratio of 0.6456. A permutation test over 100,000
> resamples gives p = 0.0003 for the CPU being faster; 13 of 15 CPU runs beat every CUDA run.
>
> The earlier 2.05x CUDA speedup was an artifact of warm-up cost. The first CPU fit took
> 1.3072 s against a steady-state median of 0.2742 s, and the original five-repeat design
> included it, inflating the CPU median to 0.8554 s. Excluding the warm-up fit removed the
> effect entirely and reversed its direction.
>
> On 6,000 rows and 500 trees there is not enough work per boosting round to amortise host-to-
> device transfer and kernel launch overhead. The notebook's original interpretation string
> anticipated this.
>
> **What survived the reversal:** CUDA remained the more consistent device — standard
> deviation 0.0340 against 0.0928, spread 1.25x against 2.25x. A finding that holds through a
> reversal of the headline is the more trustworthy of the two.

Add a short subsection on model divergence:

> **CPU and CUDA are two different models.** XGBoost's `hist` implementations sketch
> quantiles differently per device. At the frozen 0.66 threshold the two disagree on 8 of
> 2000 validation rows (0.4%), with a maximum absolute probability difference of 0.2762. Run
> independently, the validation sweep selects 0.84 on CPU and 0.76 on CUDA — the tuned
> threshold does not transfer between devices.

## 4. Rewrite the README section

`README.md` lines ~57–73. The current text says "a 2.05x speedup for this run." Replace with
a three-or-four-line summary of the corrected result and a pointer to `docs/results.md`.
Keep the existing sentence that the benchmark does not establish a universal result — it was
correct then and it is correct now.

Also check `PROJECT_STATUS.md` and `projects/GAME_PLAN.md` for any surviving "2.05x" or
"CUDA speedup" wording and correct it. Grep the whole repo for `2.05` and `speedup` before
declaring done.

## 5. Resolve the threshold discrepancy — do not paper over it

`outputs/xgboost-comparison/comparison-report.md` states the frozen XGBoost threshold is
**0.66**. The new sweeps select **0.84** (CPU) and **0.76** (CUDA). These cannot all be the
same procedure on the same data.

Find out which is true. Likely candidates: the sweeps optimise different objectives (F1 vs
average precision), or they run over different splits, or the sweep grid differs.

**Report the finding; do not silently change 0.66.** If the frozen threshold was selected by
a different and still-valid rule, document both and say which governs the reported test
metrics. If it was a genuine inconsistency, say that plainly and state what the corrected
test F1 becomes. The 0.7368 test F1 currently on record depends on this answer.

## 6. Tests and done criteria

- A test asserting the n15 report verifies clean, and the n5 report returns `superseded`
- A test asserting an n5 report **without** `superseded_by` still raises
- `pytest -q tests` green
- `grep -rn "2.05" --exclude-dir=.venv --exclude-dir=.git .` returns only historical
  references inside the explicitly-labelled superseded section
- Both evidence files present; neither modified except for the added `superseded_by`

## Do not

- Do not delete or edit the numbers in the original evidence file.
- Do not evaluate the official test split anywhere in this work.
- Do not add CUDA, GPU, or XGBoost to the résumé or portfolio site as part of this task.
  That gate is Track A and is unchanged.
- Do not describe the reversal as a mistake to be minimised. It is the most defensible thing
  in the project: a measurement that was corrected by a better experimental design, with both
  runs kept on record.
