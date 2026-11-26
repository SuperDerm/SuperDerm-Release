"""End-to-end tests of ``python3 -m superderm_eval`` in a subprocess."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import helpers

F = helpers.FIXTURES


def run_cli(*args, hash_seed="0"):
    env = dict(os.environ, PYTHONHASHSEED=hash_seed)
    proc = subprocess.run(
        [sys.executable, "-m", "superderm_eval", *map(str, args)],
        cwd=helpers.EVALUATOR_DIR,
        env=env,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_json(*args):
    code, out, err = run_cli(*args)
    return code, json.loads(out.decode("ascii"))


SCORE_ARGS = ("score", "--predictions", F / "predictions.jsonl", "--reference", F / "reference.csv", "--config", F / "config.json")


class CliTest(unittest.TestCase):
    def test_validate_valid_file_against_case_list_and_reference(self):
        for flag, path in (("--cases", F / "cases.txt"), ("--reference", F / "reference.csv")):
            with self.subTest(flag=flag):
                code, out = run_json("validate", "--predictions", F / "predictions.jsonl", flag, path)
                self.assertEqual(code, 0)
                self.assertEqual((out["ok"], out["n_cases"]), (True, 5))

    def test_validate_invalid_file_exits_1_with_row_errors(self):
        code, out = run_json("validate", "--predictions", F / "predictions_invalid.jsonl", "--cases", F / "cases.txt")
        self.assertEqual(code, 1)
        self.assertEqual((out["ok"], out["input"], out["error_count"]), (False, "predictions", 9))
        duplicate = out["errors"][0]
        self.assertEqual((duplicate["code"], duplicate["line"], duplicate["positions"]), ("duplicate_prediction", 1, [1, 2]))

    def test_validate_missing_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.jsonl"
            lines = (F / "predictions.jsonl").read_bytes().splitlines(keepends=True)
            path.write_bytes(b"".join(lines[:-1]))
            code, out = run_json("validate", "--predictions", path, "--cases", F / "cases.txt")
        self.assertEqual(code, 1)
        self.assertEqual(out["errors"], [
            {
                "case_ids": ["s_demo_005"],
                "code": "missing_case_ids",
                "line": None,
                "message": "1 expected case ID(s) have no row: s_demo_005",
            }
        ])

    def test_score_dev_fixture(self):
        code, out = run_json(*SCORE_ARGS)
        self.assertEqual(code, 0)
        values = {m["label"]: m["value"] for m in out["secondary"]}
        self.assertEqual((values["acc@1"], values["acc@3"]), (0.4, 0.6))
        self.assertEqual(out["primary"]["value"], 1.0)
        self.assertNotIn("per_case", out)

    def test_score_per_case_flag(self):
        code, out = run_json(*SCORE_ARGS, "--per-case")
        self.assertEqual(code, 0)
        self.assertEqual([c["case_id"] for c in out["per_case"]], [f"s_demo_00{i}" for i in range(1, 6)])

    def test_score_invalid_predictions_exits_1_without_scores(self):
        code, out = run_json(
            "score", "--predictions", F / "predictions_invalid.jsonl", "--reference", F / "reference.csv", "--config", F / "config.json"
        )
        self.assertEqual(code, 1)
        self.assertEqual(out["input"], "predictions")
        self.assertNotIn("primary", out)

    def test_config_vocabulary_mismatch_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            config = json.loads((F / "config.json").read_text(encoding="utf-8"))
            config["vocabulary"]["version"] = "1.1.0"
            path.write_text(json.dumps(config), encoding="utf-8")
            code, out = run_json(
                "score", "--predictions", F / "predictions.jsonl", "--reference", F / "reference.csv", "--config", path
            )
        self.assertEqual(code, 2)
        self.assertEqual((out["input"], out["errors"][0]["code"]), ("config", "vocabulary_mismatch"))

    def test_vocab_override_and_vocab_check(self):
        code, out = run_json("vocab-check")
        self.assertEqual(code, 0)
        self.assertEqual((out["n_terms"], out["ambiguous"]), (214, []))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            path.write_text(json.dumps(helpers.tiny_vocab_dict()), encoding="utf-8")
            code, out = run_json("vocab-check", "--vocab", path)
        self.assertEqual(code, 0)
        self.assertEqual([a["normalized"] for a in out["ambiguous"]], ["pox"])

    def test_usage_and_unreadable_file_errors_are_json_with_exit_2(self):
        code, out = run_json("score", "--predictions", F / "predictions.jsonl")
        self.assertEqual((code, out["input"], out["errors"][0]["code"]), (2, "arguments", "usage"))
        code, out = run_json("validate", "--predictions", F / "nope.jsonl", "--cases", F / "cases.txt")
        self.assertEqual((code, out["input"], out["errors"][0]["code"]), (2, "predictions", "unreadable_file"))

    def test_package_and_project_versions_agree(self):
        from superderm_eval import __version__

        pyproject = (helpers.EVALUATOR_DIR / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(f'\nversion = "{__version__}"\n', pyproject)
        code, out, _ = run_cli("--version")
        self.assertEqual((code, out.strip()), (0, f"superderm-eval {__version__}".encode()))

    def test_output_is_byte_identical_across_processes(self):
        outputs = {run_cli(*SCORE_ARGS, "--per-case", hash_seed=seed)[1] for seed in ("1", "2", "3")}
        self.assertEqual(len(outputs), 1)
        self.assertTrue(next(iter(outputs)).endswith(b"}\n"))


if __name__ == "__main__":
    unittest.main()
