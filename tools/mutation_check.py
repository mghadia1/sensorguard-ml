#!/usr/bin/env python3
"""Fault injection: does this test suite actually catch bugs?

"31 tests pass" says nothing on its own. Thirty-one assertions that never fail
also pass. The claim worth making is *"31 tests, and every one of N deliberately
injected faults is caught"* — which requires injecting the faults and counting.

SensorGuard's README makes three claims that rest on arithmetic rather than on
prose: that identifier and failure-mode columns are kept out of the features
(the leakage guard), that the 60/20/20 splits are stratified and deterministic,
and that the reported precision/recall/AP describe what the model actually did.
Each mutation below breaks exactly one of those, in the way a real mistake would
— a leaked column, a dropped `stratify=`, a ranking metric accidentally fed
hard predictions instead of probabilities.

Two guards make the count mean something:

  * a control run with no mutation must PASS. If the suite fails for unrelated
    reasons, every mutation would look "caught" and the score would be a lie.
  * each mutation's target text must appear EXACTLY once in the file. If the
    source moves on and a pattern stops matching, the harness errors instead of
    silently skipping the fault and reporting a smaller, flattering total.

Bytecode is disabled in the child runs. A stale .pyc can keep executing code
that is no longer on disk, which makes a suite report on a file it is not
actually testing. That is not hypothetical here: this portfolio shipped for
weeks with pytest importing an installed snapshot instead of `src/`.

    python tools/mutation_check.py            # human-readable table
    python tools/mutation_check.py --json     # machine-readable, for the README
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Mutation:
    identifier: str
    module: str
    original: str
    mutated: str
    fault: str


MUTATIONS: tuple[Mutation, ...] = (
    # --- leakage guard: the project's central claim ---------------------------
    Mutation(
        "leak-failure-mode-column-into-features",
        "data.py",
        '    "Tool wear [min]",\n)',
        '    "Tool wear [min]",\n    "TWF",\n)',
        "a failure-mode flag joins the features, leaking the answer into the model",
    ),
    Mutation(
        "leak-identifier-column-into-features",
        "data.py",
        '    "Tool wear [min]",\n)',
        '    "Tool wear [min]",\n    "UDI",\n)',
        "a row identifier becomes a feature, which is leakage disguised as a number",
    ),
    Mutation(
        "drop-failure-modes-from-expected-schema",
        "data.py",
        "EXPECTED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,) + FAILURE_MODE_COLUMNS",
        "EXPECTED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + (TARGET_COLUMN,)",
        "schema validation stops requiring the failure-mode columns it exists to exclude",
    ),
    # --- split protocol: stratified, deterministic, 60/20/20 ------------------
    Mutation(
        "first-split-not-stratified",
        "data.py",
        "        stratify=frame[TARGET_COLUMN],\n",
        "",
        "train/holdout split stops preserving the class balance of a rare-failure label",
    ),
    Mutation(
        "second-split-not-stratified",
        "data.py",
        "        stratify=temporary[TARGET_COLUMN],\n",
        "",
        "validation/test split stops preserving the class balance",
    ),
    Mutation(
        "split-proportions-wrong",
        "data.py",
        "        test_size=0.40,",
        "        test_size=0.30,",
        "the documented 60/20/20 split silently becomes 70/15/15",
    ),
    Mutation(
        "validation-test-split-uneven",
        "data.py",
        "        test_size=0.50,",
        "        test_size=0.70,",
        "validation and test stop being the same size, so they are no longer comparable",
    ),
    Mutation(
        "splits-nondeterministic",
        "data.py",
        "    random_state: int = 42,",
        "    random_state: int | None = None,",
        "splits stop being reproducible, so no reported number can be re-derived",
    ),
    # --- metric arithmetic: what the reported numbers mean --------------------
    Mutation(
        "precision-reports-recall",
        "modeling.py",
        '        "precision": float(precision_score(label_values, predictions, zero_division=0)),',
        '        "precision": float(recall_score(label_values, predictions, zero_division=0)),',
        "precision and recall are swapped, which flatters a rare-positive model",
    ),
    Mutation(
        "average-precision-fed-hard-predictions",
        "modeling.py",
        '        "average_precision": float(average_precision_score(label_values, probability_values)),',
        '        "average_precision": float(average_precision_score(label_values, predictions)),',
        "a ranking metric is computed on thresholded labels instead of probabilities",
    ),
    Mutation(
        "roc-auc-fed-hard-predictions",
        "modeling.py",
        '        "roc_auc": float(roc_auc_score(label_values, probability_values)),',
        '        "roc_auc": float(roc_auc_score(label_values, predictions)),',
        "ROC AUC collapses to a single operating point but is still reported as AUC",
    ),
    Mutation(
        "threshold-boundary-flipped",
        "modeling.py",
        "    predictions = (probability_values >= threshold).astype(np.int64)",
        "    predictions = (probability_values > threshold).astype(np.int64)",
        "the decision boundary excludes the threshold it is documented to include",
    ),
    Mutation(
        "threshold-range-check-removed",
        "modeling.py",
        "    if not 0.0 <= threshold <= 1.0:",
        "    if not -1.0 <= threshold <= 2.0:",
        "a threshold outside [0, 1] is accepted, producing all-zero or all-one predictions",
    ),
    Mutation(
        "threshold-selection-ignored",
        "modeling.py",
        "            best_threshold = float(threshold)",
        "            best_threshold = 0.5",
        "the F1 sweep runs, then throws its answer away and returns the default",
    ),
    Mutation(
        "class-weight-inverted",
        "modeling.py",
        "    scale_pos_weight = negative_count / positive_count",
        "    scale_pos_weight = positive_count / negative_count",
        "imbalance weighting is inverted, down-weighting the rare class it should lift",
    ),
)


def _copy_project(destination: Path) -> None:
    """Copy just enough of the project to run its suite."""
    # docs/evidence holds committed CUDA reports the suite verifies by hash.
    for item in ("src", "tests", "pyproject.toml", "docs"):
        source = PROJECT_ROOT / item
        target = destination / item
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "*.pyc")
            )
        else:
            shutil.copy2(source, target)


def _run_suite(project: Path) -> tuple[bool, str]:
    """Run the suite inside `project`. Returns (passed, last line of output)."""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # The copied tests/ directory contains the test that invokes this harness.
    # Without both of these the child would run it, spawning its own children,
    # forever. The ignore handles it; the env var is the backstop if the file is
    # ever renamed.
    environment["SENSORGUARD_MUTATION_CHILD"] = "1"
    completed = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
            "--ignore=tests/test_mutation_coverage.py",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = (completed.stdout + completed.stderr).strip().splitlines()
    summary = output[-1] if output else "(no output)"
    return completed.returncode == 0, summary


def _apply(project: Path, mutation: Mutation) -> None:
    path = project / "src" / "sensorguard" / mutation.module
    text = path.read_text()
    occurrences = text.count(mutation.original)
    if occurrences != 1:
        raise SystemExit(
            f"mutation {mutation.identifier!r}: its target appears {occurrences} times in "
            f"{mutation.module} (expected exactly 1). The source moved; update the mutation "
            f"rather than letting the harness skip a fault it can no longer inject."
        )
    path.write_text(text.replace(mutation.original, mutation.mutated))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    arguments = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="sensorguard-mutation-") as scratch:
        control = Path(scratch) / "control"
        control.mkdir()
        _copy_project(control)
        control_passed, control_summary = _run_suite(control)
        if not control_passed:
            raise SystemExit(
                "control run FAILED before any mutation was applied "
                f"({control_summary}). Every mutation would look 'caught'. Fix the suite first."
            )

        for index, mutation in enumerate(MUTATIONS):
            workspace = Path(scratch) / f"mutant-{index:02d}"
            workspace.mkdir()
            _copy_project(workspace)
            _apply(workspace, mutation)
            passed, summary = _run_suite(workspace)
            results.append(
                {
                    "id": mutation.identifier,
                    "module": mutation.module,
                    "fault": mutation.fault,
                    "caught": not passed,
                    "suite": summary,
                }
            )

    caught = sum(1 for row in results if row["caught"])
    report = {
        "control": control_summary,
        "mutations": len(results),
        "caught": caught,
        "results": results,
    }

    if arguments.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"control (no mutation): {control_summary}\n")
        for row in results:
            mark = "caught " if row["caught"] else "MISSED "
            print(f"  {mark} {row['id']:<40} {row['suite']}")
        print(f"\n{caught}/{len(results)} injected faults caught")
        if caught != len(results):
            print("\nA missed fault is a hole in the suite, not a rounding error.")
            for row in results:
                if not row["caught"]:
                    print(f"  - {row['id']}: {row['fault']}")

    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
