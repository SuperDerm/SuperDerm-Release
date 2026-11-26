import hashlib
import json
import unittest
from fractions import Fraction

import helpers
from superderm_eval import __version__
from superderm_eval.score import (
    dumps,
    hit_rank,
    parse_config,
    round_value,
    score_files,
    score_predictions,
)
from superderm_eval.vocab import InputError, SubmissionError

F = helpers.FIXTURES


def dev_result(per_case=False):
    return score_files(F / "predictions.jsonl", F / "reference.csv", F / "config.json", per_case=per_case)


def by_label(result):
    return {m["label"]: m for m in [result["primary"], *result["secondary"]]}


class AcceptanceFixtureTest(unittest.TestCase):
    """The five-case dev fixture from the project specification."""

    def setUp(self):
        self.result = dev_result(per_case=True)
        self.cases = {c["case_id"]: c for c in self.result["per_case"]}

    def test_case_level_hits(self):
        expected = {
            "s_demo_001": (1, 1, 1),  # rank-one hit: Mpox
            "s_demo_002": (3, 0, 1),  # rank-three hit: Smallpox
            "s_demo_003": (1, 1, 1),  # synonym hit at rank one: "insect bite" -> DX-111
            "s_demo_004": (None, 0, 0),  # unknown term only
            "s_demo_005": (None, 0, 0),  # explicit abstention
        }
        for case_id, (rank, hit1, hit3) in expected.items():
            with self.subTest(case_id=case_id):
                case = self.cases[case_id]
                self.assertEqual((case["hit_rank"], case["hit@1"], case["hit@3"]), (rank, hit1, hit3))
        self.assertEqual(self.cases["s_demo_003"]["predictions"][0]["term_id"], "DX-111")
        self.assertEqual(self.cases["s_demo_004"]["predictions"][0]["status"], "unknown")
        self.assertEqual(self.cases["s_demo_005"]["predictions"], [])

    def test_overall_accuracy_keeps_abstention_in_denominator(self):
        metrics = by_label(self.result)
        self.assertEqual(metrics["acc@1"]["value"], round_value(Fraction(2, 5)))
        self.assertEqual(metrics["acc@3"]["value"], round_value(Fraction(3, 5)))
        self.assertEqual((metrics["acc@1"]["value"], metrics["acc@3"]["value"]), (0.4, 0.6))
        self.assertEqual(metrics["acc@1"]["n_cases"], 5)
        self.assertEqual(metrics["acc@3"]["n_cases"], 5)
        self.assertEqual(self.result["n_cases"], 5)

    def test_primary_and_cohort_metrics(self):
        primary = self.result["primary"]
        self.assertEqual(primary["metric"], "macro_recall@3")
        self.assertEqual((primary["value"], primary["n_cases"], primary["n_diagnoses"]), (1.0, 2, 2))
        self.assertIsNone(primary["reason"])
        metrics = by_label(self.result)
        mimic1, mimic3 = metrics["acc@1[cohort=common_mimic]"], metrics["acc@3[cohort=common_mimic]"]
        self.assertEqual((mimic1["value"], mimic1["n_cases"]), (0.333333, 3))
        self.assertEqual((mimic3["value"], mimic3["n_cases"]), (0.333333, 3))

    def test_result_identifies_everything_needed_to_reproduce_it(self):
        result = self.result
        self.assertEqual(result["scorer_version"], __version__)
        self.assertEqual((result["challenge_id"], result["challenge_version"]), ("rare-and-mimics", "dev-1"))
        self.assertEqual((result["vocabulary"]["id"], result["vocabulary"]["version"]), ("superderm-diagnoses", "1.0.0"))
        self.assertEqual(result["vocabulary"]["synonyms_status"], "draft")
        self.assertEqual(len(result["vocabulary"]["content_sha256"]), 64)
        expected_sha = hashlib.sha256((F / "predictions.jsonl").read_bytes()).hexdigest()
        self.assertEqual(result["predictions_sha256"], expected_sha)
        self.assertEqual([m["label"] for m in result["secondary"]], ["acc@1", "acc@3", "acc@1[cohort=common_mimic]", "acc@3[cohort=common_mimic]"])

    def test_per_case_details_only_on_request(self):
        self.assertNotIn("per_case", dev_result())
        self.assertIn("per_case", self.result)


class MetricTest(unittest.TestCase):
    def score(self, ref_rows, prediction_rows, primary=None, secondary=None, vocab=None):
        vocab = vocab or helpers.real_vocab()
        ref = helpers.reference(ref_rows, vocab)
        cfg = helpers.config(primary, secondary, vocab=vocab) if vocab is helpers.real_vocab() else parse_config(
            helpers.config_dict(primary, secondary, vocabulary={"id": vocab.id, "version": vocab.version}), vocab
        )
        data = helpers.jsonl(*({"case_id": c, "predictions": p} for c, p in prediction_rows))
        return score_predictions(data, ref, vocab, cfg)

    def test_macro_recall_averages_per_diagnosis_not_per_case(self):
        ref_rows = [
            ("s1", "DX-070", "x"),
            ("s2", "DX-070", "x"),
            ("s3", "DX-070", "x"),
            ("s4", "DX-074", "x"),
            ("s5", "DX-111", "x"),  # outside the subset: excluded from both
        ]
        predictions = [
            ("s1", ["Mpox"]),
            ("s2", ["Varicella", "Monkeypox"]),
            ("s3", ["Varicella", "Herpes Zoster", "mpox"]),
            ("s4", ["Varicella"]),  # the single smallpox case is missed
            ("s5", ["Insect Bite"]),
        ]
        subset = ["DX-070", "DX-074"]
        result = self.score(
            ref_rows,
            predictions,
            primary={"metric": "macro_recall@3", "diagnoses": subset},
            secondary=[{"metric": "acc@3", "diagnoses": subset}],
        )
        macro, micro = result["primary"], result["secondary"][0]
        self.assertEqual(macro["value"], 0.5)  # (3/3 + 0/1) / 2
        self.assertEqual(micro["value"], 0.75)  # 3/4
        self.assertEqual((macro["n_cases"], macro["n_diagnoses"], macro["n_diagnoses_in_set"]), (4, 2, 2))
        self.assertEqual((micro["n_cases"], micro["n_diagnoses"]), (4, 2))

    def test_macro_recall_is_null_when_no_diagnosis_of_the_set_is_represented(self):
        result = self.score(
            [("s1", "DX-070", "x")],
            [("s1", ["Mpox"])],
            primary={"metric": "macro_recall@3", "diagnoses": ["DX-003", "DX-074"]},
        )
        primary = result["primary"]
        self.assertIsNone(primary["value"])
        self.assertEqual((primary["n_cases"], primary["n_diagnoses"]), (0, 0))
        self.assertIn("none of the 2 diagnoses", primary["reason"])

    def test_unrepresented_diagnoses_do_not_enter_the_average(self):
        result = self.score(
            [("s1", "DX-070", "x"), ("s2", "DX-070", "x")],
            [("s1", ["Mpox"]), ("s2", [])],
            primary={"metric": "macro_recall@1", "diagnoses": ["DX-070", "DX-074", "DX-003"]},
        )
        self.assertEqual((result["primary"]["value"], result["primary"]["n_diagnoses"]), (0.5, 1))
        self.assertEqual(result["primary"]["n_diagnoses_in_set"], 3)

    def test_cohort_restriction_and_empty_cohort(self):
        result = self.score(
            [("s1", "DX-070", "rare"), ("s2", "DX-076", "mimic"), ("s3", "DX-067", "mimic")],
            [("s1", []), ("s2", ["Chickenpox"]), ("s3", ["Varicella"])],
            secondary=[{"metric": "acc@1", "cohort": "mimic"}, {"metric": "acc@1", "cohort": "absent"}],
        )
        mimic, absent = result["secondary"]
        self.assertEqual((mimic["value"], mimic["n_cases"]), (0.5, 2))
        self.assertIsNone(absent["value"])
        self.assertEqual(absent["n_cases"], 0)
        self.assertIn("'absent'", absent["reason"])

    def test_rare_and_high_consequence_selector(self):
        vocab = helpers.real_vocab()
        cfg = parse_config(
            helpers.config_dict(primary={"metric": "macro_recall@3", "diagnoses": "rare_and_high_consequence"}), vocab
        )
        self.assertEqual(cfg.primary.diagnoses, frozenset(vocab.rare_and_high_consequence()))
        self.assertEqual(cfg.primary.label, "macro_recall@3[diagnoses=rare_and_high_consequence]")
        ref = helpers.reference([("s1", "DX-070", "x"), ("s2", "DX-076", "x")])
        data = helpers.jsonl({"case_id": "s1", "predictions": []}, {"case_id": "s2", "predictions": ["Varicella"]})
        primary = score_predictions(data, ref, vocab, cfg)["primary"]
        self.assertEqual(primary["diagnoses"], "rare_and_high_consequence")
        self.assertEqual(primary["n_diagnoses_in_set"], 70)
        # Only the mpox case counts (varicella is not rare); it abstained.
        self.assertEqual((primary["value"], primary["n_cases"], primary["n_diagnoses"]), (0.0, 1, 1))
        # On the dev fixture, the mpox and smallpox cases are the rare and high-consequence ones.
        dev = score_predictions(
            (F / "predictions.jsonl").read_bytes(),
            helpers.parse_reference((F / "reference.csv").read_bytes(), vocab),
            vocab,
            cfg,
        )["primary"]
        self.assertEqual((dev["value"], dev["n_cases"], dev["n_diagnoses"]), (1.0, 2, 2))

    def test_ambiguous_answer_scores_zero_even_if_gold_is_a_candidate(self):
        vocab = helpers.tiny_vocab()
        result = self.score(
            [("s1", "T-1", "x"), ("s2", "T-1", "x")],
            [("s1", ["Pox"]), ("s2", ["Pox", "Alpha"])],
            primary={"metric": "acc@1"},
            secondary=[{"metric": "acc@2"}],
            vocab=vocab,
        )
        self.assertEqual(result["primary"]["value"], 0.0)
        self.assertEqual(result["secondary"][0]["value"], 0.5)
        self.assertIsNone(hit_rank(["Pox"], "T-1", vocab))
        self.assertEqual(hit_rank(["Pox", "alpha"], "T-1", vocab), 2)

    def test_rounding_to_six_decimals(self):
        self.assertEqual(round_value(Fraction(2, 3)), 0.666667)
        self.assertEqual(round_value(Fraction(1, 7)), 0.142857)
        self.assertEqual(round_value(Fraction(1, 1)), 1.0)
        self.assertIsInstance(round_value(Fraction(0)), float)
        self.assertIsNone(round_value(None))


class InputConsistencyTest(unittest.TestCase):
    def config_errors(self, **overrides):
        with self.assertRaises(InputError) as ctx:
            parse_config(helpers.config_dict(**overrides), helpers.real_vocab())
        self.assertEqual(ctx.exception.source, "config")
        return [e["code"] for e in ctx.exception.errors]

    def test_vocabulary_version_mismatch(self):
        codes = self.config_errors(vocabulary={"id": "superderm-diagnoses", "version": "2.0.0"})
        self.assertEqual(codes, ["vocabulary_mismatch"])

    def test_vocabulary_id_mismatch(self):
        codes = self.config_errors(vocabulary={"id": "another-vocab", "version": "1.0.0"})
        self.assertEqual(codes, ["vocabulary_mismatch"])

    def test_config_against_another_vocabulary_file(self):
        with self.assertRaises(InputError) as ctx:
            parse_config(helpers.config_dict(), helpers.tiny_vocab())
        self.assertIn("vocabulary_mismatch", [e["code"] for e in ctx.exception.errors])

    def test_scorer_version_mismatch(self):
        self.assertEqual(self.config_errors(scorer_version="0.9.0"), ["scorer_version_mismatch"])

    def test_reference_for_another_challenge_version(self):
        ref = helpers.reference([("s1", "DX-070", "x")], version="dev-2")
        data = helpers.jsonl({"case_id": "s1", "predictions": []})
        with self.assertRaises(InputError) as ctx:
            score_predictions(data, ref, helpers.real_vocab(), helpers.config())
        self.assertEqual((ctx.exception.source, ctx.exception.errors[0]["code"]), ("reference", "challenge_version_mismatch"))

    def test_metric_definition_errors(self):
        cases = {
            "invalid_metric": [{"metric": "top5"}, {"metric": "acc@4"}, {"metric": "acc@0"}, {"metric": "acc@01"}],
            "missing_diagnoses": [{"metric": "macro_recall@3"}],
            "unknown_term_id": [{"metric": "acc@1", "diagnoses": ["DX-999"]}],
            "invalid_diagnoses": [{"metric": "acc@1", "diagnoses": "rare"}, {"metric": "acc@1", "diagnoses": []}],
            "duplicate_term_id": [{"metric": "macro_recall@1", "diagnoses": ["DX-070", "DX-070"]}],
            "unknown_key": [{"metric": "acc@1", "cohorts": "x"}],
            "invalid_cohort": [{"metric": "acc@1", "cohort": ""}],
        }
        for code, specs in cases.items():
            for spec in specs:
                with self.subTest(spec=spec):
                    self.assertIn(code, self.config_errors(primary=spec))

    def test_config_shape_errors(self):
        self.assertEqual(self.config_errors(extra=1), ["unknown_key"])
        self.assertEqual(self.config_errors(secondary={"metric": "acc@1"}), ["invalid_field"])
        self.assertEqual(self.config_errors(secondary=[{"metric": "acc@1"}, {"metric": "acc@1"}]), ["duplicate_metric"])
        data = helpers.config_dict()
        del data["challenge_id"]
        with self.assertRaises(InputError) as ctx:
            parse_config(data, helpers.real_vocab())
        self.assertEqual([e["code"] for e in ctx.exception.errors], ["missing_key"])

    def test_invalid_predictions_are_never_scored(self):
        ref = helpers.reference([("s1", "DX-074", "x")])
        data = helpers.jsonl({"case_id": "s1", "predictions": ["Smallpox", "Variola"]})
        with self.assertRaises(SubmissionError) as ctx:
            score_predictions(data, ref, helpers.real_vocab(), helpers.config())
        self.assertEqual(ctx.exception.errors[0]["positions"], [1, 2])


class DeterminismTest(unittest.TestCase):
    def test_same_bytes_give_identical_json(self):
        first = dumps(dev_result(per_case=True))
        second = dumps(dev_result(per_case=True))
        self.assertEqual(first, second)
        self.assertEqual(json.loads(first), dev_result(per_case=True))

    def test_row_order_does_not_change_scores(self):
        rows = (F / "predictions.jsonl").read_bytes().splitlines(keepends=True)
        ref = helpers.reference(
            [("s_demo_001", "DX-070", "a"), ("s_demo_002", "DX-074", "a"), ("s_demo_003", "DX-111", "b"),
             ("s_demo_004", "DX-076", "b"), ("s_demo_005", "DX-067", "b")]
        )
        forward = score_predictions(b"".join(rows), ref, helpers.real_vocab(), helpers.config())
        backward = score_predictions(b"".join(reversed(rows)), ref, helpers.real_vocab(), helpers.config())
        self.assertNotEqual(forward["predictions_sha256"], backward["predictions_sha256"])
        forward.pop("predictions_sha256"), backward.pop("predictions_sha256")
        self.assertEqual(dumps(forward), dumps(backward))

    def test_canonical_json(self):
        text = dumps({"b": 1, "a": ["é"], "c": 0.4})
        self.assertEqual(text, '{\n  "a": [\n    "\\u00e9"\n  ],\n  "b": 1,\n  "c": 0.4\n}\n')


if __name__ == "__main__":
    unittest.main()
