"""The suite is only worth its count if it fails when the code is wrong.

`tools/mutation_check.py` injects deliberate faults into the arithmetic that
SensorGuard's claims rest on — the leakage guard, the stratified split, and the
reported metrics — one at a time, and records whether the suite notices. This
test makes that a gate rather than a script someone remembers to run: if a
future change makes the tests looser, a fault goes uncaught here and CI says so.

The first run of this harness caught 4 of 15. The other eleven are the reason
`ClaimArithmeticTests` exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS = PROJECT_ROOT / "tools" / "mutation_check.py"


class MutationCoverageTests(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("SENSORGUARD_MUTATION_CHILD") == "1",
        "running inside the harness's own child suite; nesting would not terminate",
    )
    def test_every_injected_fault_is_caught(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(HARNESS), "--json"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertTrue(completed.stdout.strip(), f"harness produced no report: {completed.stderr}")
        report = json.loads(completed.stdout)

        # A control run that already fails would make every mutation look caught.
        self.assertIn("passed", report["control"])
        self.assertNotIn("failed", report["control"])

        missed = [row["id"] for row in report["results"] if not row["caught"]]
        self.assertEqual(
            missed,
            [],
            f"{len(missed)} injected fault(s) went unnoticed by the suite: {missed}",
        )
        self.assertEqual(report["caught"], report["mutations"])
        self.assertGreaterEqual(report["mutations"], 15)


if __name__ == "__main__":
    unittest.main()
