"""Regression checks for frozen Git state and compatible matrix result reuse."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_matrix as RM


class TestFreezeCheck(unittest.TestCase):
    def test_head_comparison_and_dirty_filtering(self):
        """Successful Git reads preserve the expected HEAD and dirty-file rules."""
        status = (" M run_matrix.py\n?? matrix_out/\n?? __pycache__/\n"
                  " M engine_v2.py\n")
        for expected, matches in (("frozen", True), ("other", False)):
            with self.subTest(expected=expected), patch.object(
                    RM.subprocess, "check_output", side_effect=["frozen\n", status]):
                check = RM.freeze_check(expected, "matrix_out")
            self.assertTrue(check["verified"])
            self.assertEqual(check["git_revision"], "frozen")
            self.assertIs(check["matches_expected"], matches)
            self.assertEqual(check["other_dirty_files"], [" M engine_v2.py"])

    def test_failed_head_or_status_remains_unverified(self):
        """Neither a failed HEAD read nor a failed status read can verify Git."""
        for error in (OSError("missing Git"), subprocess.CalledProcessError(1, "git"),
                      subprocess.TimeoutExpired("git", 10)):
            for reads in ([error], ["frozen\n", error]):
                with self.subTest(reads=reads), patch.object(
                        RM.subprocess, "check_output", side_effect=reads):
                    check = RM.freeze_check("frozen", "matrix_out")
                self.assertFalse(check["verified"])
                self.assertIsNone(check["matches_expected"])


class TestMatrixRunner(unittest.TestCase):
    def setUp(self):
        """Use real row files and exports while isolating expensive engine runs."""
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.out = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.plan = [{"block": "M1", "run_id": "M1_test", "label": "test",
                      "factors": {}, "inp": "synthetic input"}]
        self.stack.enter_context(patch.object(RM, "build_plan", return_value=self.plan))
        self.check = self.stack.enter_context(patch.object(
            RM, "freeze_check", return_value={"verified": True, "matches_expected": True,
                                             "other_dirty_files": []}))
        self.run = self.stack.enter_context(patch.object(RM.E, "run"))
        self.stack.enter_context(patch.object(RM, "result_row", side_effect=lambda res, item, wall: {
            "run_id": item["run_id"], "block": item["block"], "inputs_sha256": "test-hash",
            "N": 20_000, "converged": False, "convergence_reasons": ["test cap"],
            "P_labor_return_below_wage": 0.5}))
        self.export = self.stack.enter_context(patch.object(RM.DL, "export_manifest", return_value=b"{}"))
        self.heatmap = self.stack.enter_context(patch.object(RM, "heatmap"))
        self.stack.enter_context(patch.object(RM.E, "source_identity", return_value={}))
        self.stdout = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def invoke(self, *args):
        """Execute the CLI against this test's isolated output directory."""
        RM.main(["--out", str(self.out), *args])

    def test_execution_requires_verified_matching_clean_git_unless_forced(self):
        """Unknown Git state, a different HEAD, and dirty files stop execution."""
        cases = ({"verified": False, "matches_expected": None, "other_dirty_files": None},
                 {"verified": False, "matches_expected": True, "other_dirty_files": []},
                 {"verified": True, "matches_expected": None, "other_dirty_files": []},
                 {"verified": True, "matches_expected": False, "other_dirty_files": []},
                 {"verified": True, "matches_expected": True, "other_dirty_files": [" M engine_v2.py"]})
        for check in cases:
            with self.subTest(check=check):
                self.check.return_value = check
                self.run.reset_mock()
                with self.assertRaisesRegex(SystemExit, "Frozen-code check failed"):
                    self.invoke()
                self.run.assert_not_called()
                self.invoke("--force")
                self.run.assert_called_once()

    def test_matching_resume_and_missing_manifest(self):
        """A compatible row resumes only when its manifest also exists."""
        self.invoke("--quick", "--seed", "7")
        self.run.assert_called_once_with("synthetic input", seed=7, n_fixed=20_000)
        row = json.loads((self.out / "rows/M1_test.json").read_text())
        inventory = json.loads((self.out / "matrix_inventory.json").read_text())
        self.assertTrue(row["run_settings"]["quick_mode"])
        self.assertEqual(row["run_settings"]["seed"], 7)
        self.assertEqual(row["run_settings"]["data_freeze"], inventory["data_freeze"])
        self.run.reset_mock()
        self.invoke("--quick", "--seed", "7", "--resume")
        self.run.assert_not_called()
        (self.out / "manifests/M1_test.json").unlink()
        self.invoke("--quick", "--seed", "7", "--resume")
        self.run.assert_called_once()

    def test_changed_run_settings_recompute_cached_row(self):
        """Every CLI setting and the data freeze participate in resume compatibility."""
        self.invoke()
        row_path = self.out / "rows/M1_test.json"
        original = row_path.read_text()
        for args in (("--quick",), ("--seed", "7"), ("--n-cap", "50000"),
                     ("--spot", "9000"), ("--spot-low", "7000"), ("--spot-high", "11000")):
            with self.subTest(args=args):
                row_path.write_text(original)
                self.run.reset_mock()
                self.invoke("--resume", *args)
                self.run.assert_called_once()
        for name, value in (("DATA_FREEZE", "2026-10-20"), ("W_OPP", 100.),
                            ("F_PERT", (0.5, 0.6, 0.7)), ("FOOD", (100., 200., 300.))):
            with self.subTest(name=name), patch.object(RM, name, value):
                row_path.write_text(original)
                self.run.reset_mock()
                self.invoke("--resume")
                self.run.assert_called_once()
        legacy = json.loads(original)
        del legacy["run_settings"]
        row_path.write_text(json.dumps(legacy))
        self.run.reset_mock()
        self.invoke("--resume")
        self.run.assert_called_once()

    def test_aggregation_excludes_incompatible_rows(self):
        """A cached row replaced after resume is excluded from CSV, heatmap and inventory."""
        self.invoke()
        row_path = self.out / "rows/M1_test.json"
        original = json.loads(row_path.read_text())
        self.plan.append({**self.plan[0], "block": "M2", "run_id": "M2_test"})
        settings = original["run_settings"]
        variants = []
        for key in ("quick_mode", "seed", "n_cap", "matrix_version"):
            changed = copy.deepcopy(settings)
            changed[key] = "different"
            variants.append(changed)
        for key in settings["data_freeze"]:
            changed = copy.deepcopy(settings)
            changed["data_freeze"][key] = "different"
            variants.append(changed)
        for changed in [None, *variants]:
            with self.subTest(settings=changed):
                row_path.write_text(json.dumps(original))
                (self.out / "rows/M2_test.json").unlink(missing_ok=True)

                def replace_cached_row(res, extras):
                    """Replace the skipped row while the next run exports its manifest."""
                    stale = {**original, "run_settings": changed}
                    row_path.write_text(json.dumps(stale))
                    return b"{}"

                self.export.side_effect = replace_cached_row
                self.invoke("--resume")
                df = pd.read_csv(self.out / "matrix_results.csv")
                self.assertEqual(df["run_id"].tolist(), ["M2_test"])
                self.assertEqual(self.heatmap.call_args.args[0]["run_id"].tolist(), ["M2_test"])
                inventory = json.loads((self.out / "matrix_inventory.json").read_text())
                self.assertEqual(inventory["n_runs_with_results"], 1)
                self.assertEqual([r["run_id"] for r in inventory["runs"]], ["M2_test"])

    def test_unknown_blocks_rejected_before_execution_or_listing(self):
        """Typos fail for both an unknown-only selection and a mixed selection."""
        for blocks in ("M99", "M1,M99"):
            for args in ((), ("--list",)):
                with self.subTest(blocks=blocks, args=args), self.assertRaises(SystemExit) as exc:
                    self.invoke("--blocks", blocks, *args)
                self.assertEqual(exc.exception.code, 2)
        self.assertIn("unknown block IDs: M99", self.stderr.getvalue())
        self.run.assert_not_called()
        self.heatmap.assert_not_called()

    def test_valid_block_selection_preserves_order_and_whitespace(self):
        """Valid subsets and listing retain the original plan order."""
        self.plan.append({**self.plan[0], "block": "M2", "run_id": "M2_test"})
        self.invoke("--blocks", " M2, M1, M2, ", "--list")
        output = self.stdout.getvalue()
        self.assertLess(output.index("M1_test"), output.index("M2_test"))
        self.assertIn("2 runs", output)
        self.run.assert_not_called()
        self.invoke("--blocks", " M2, ")
        self.assertEqual(self.heatmap.call_args.args[0]["run_id"].tolist(), ["M2_test"])


if __name__ == "__main__":
    unittest.main()
