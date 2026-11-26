import unittest

import helpers
from superderm_eval.validate import (
    PREDICTION_ERROR_CODES,
    load_case_list,
    load_reference,
    parse_case_list,
    parse_reference,
    validate_predictions,
    validate_predictions_file,
)
from superderm_eval.vocab import InputError

CASES = ["s_demo_001", "s_demo_002", "s_demo_003"]


def row(case_id, predictions):
    return {"case_id": case_id, "predictions": predictions}


class PredictionValidationTest(unittest.TestCase):
    def validate(self, data, expected=CASES):
        return validate_predictions(data, expected, helpers.real_vocab())

    def errors(self, data, expected=CASES):
        report = self.validate(data, expected)
        self.assertFalse(report.ok)
        self.assertLessEqual({e["code"] for e in report.errors}, PREDICTION_ERROR_CODES)
        return report.errors

    def only_error(self, data, expected=CASES):
        errors = self.errors(data, expected)
        self.assertEqual(len(errors), 1, errors)
        return errors[0]

    def test_dev_fixture_is_valid_against_case_list_and_reference(self):
        vocab = helpers.real_vocab()
        from_list = load_case_list(helpers.FIXTURES / "cases.txt")
        from_reference = load_reference(helpers.FIXTURES / "reference.csv", vocab).case_ids
        self.assertEqual(from_list, from_reference)
        report = validate_predictions_file(helpers.FIXTURES / "predictions.jsonl", from_list, vocab)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.rows["s_demo_005"], ())
        self.assertEqual(report.to_report()["n_cases"], 5)

    def test_valid_rows_including_abstention_and_unknown_names(self):
        data = helpers.jsonl(
            row("s_demo_001", ["Mpox", "Varicella", "Herpes Zoster"]),
            row("s_demo_002", ["Not A Real Diagnosis", "Smallpox Variola"]),  # unmapped: not an error
            row("s_demo_003", []),
        )
        report = self.validate(data)
        self.assertTrue(report.ok, report.errors)

    def test_missing_row_fails_and_lists_the_case(self):
        data = helpers.jsonl(row("s_demo_001", ["Mpox"]), row("s_demo_002", []))
        error = self.only_error(data)
        self.assertEqual(error["code"], "missing_case_ids")
        self.assertIsNone(error["line"])
        self.assertEqual(error["case_ids"], ["s_demo_003"])
        self.assertIn("s_demo_003", error["message"])

    def test_empty_file_reports_every_case_missing(self):
        self.assertEqual(self.only_error(b"")["case_ids"], CASES)

    def test_two_names_of_one_term_are_a_duplicate_naming_both_positions(self):
        data = helpers.jsonl(row("s_demo_001", ["Smallpox", "Variola"]), row("s_demo_002", []), row("s_demo_003", []))
        error = self.only_error(data)
        self.assertEqual(error["code"], "duplicate_prediction")
        self.assertEqual(error["kind"], "same_term")
        self.assertEqual(error["positions"], [1, 2])
        self.assertEqual(error["term_id"], "DX-074")
        self.assertEqual(error["line"], 1)
        self.assertEqual(error["case_id"], "s_demo_001")
        self.assertIn("predictions 1", error["message"])
        self.assertIn("2 ('Variola')", error["message"])

    def test_same_normalized_text_twice_is_a_duplicate(self):
        data = helpers.jsonl(row("s_demo_001", ["Mpox", "Varicella", "  MPOX "]), row("s_demo_002", []), row("s_demo_003", []))
        error = self.only_error(data)
        self.assertEqual((error["code"], error["kind"], error["positions"]), ("duplicate_prediction", "same_text", [1, 3]))

    def test_unknown_name_repeated_is_a_duplicate(self):
        data = helpers.jsonl(row("s_demo_001", ["Foo", "foo"]), row("s_demo_002", []), row("s_demo_003", []))
        error = self.only_error(data)
        self.assertEqual((error["kind"], error["positions"], error["term_id"]), ("same_text", [1, 2], None))

    def test_four_predictions_fail(self):
        data = helpers.jsonl(
            row("s_demo_001", ["Mpox", "Varicella", "Herpes Zoster", "Smallpox"]), row("s_demo_002", []), row("s_demo_003", [])
        )
        error = self.only_error(data)
        self.assertEqual((error["code"], error["count"], error["line"]), ("too_many_predictions", 4, 1))

    def test_non_string_and_blank_entries_fail_with_positions(self):
        data = helpers.jsonl(row("s_demo_001", ["Mpox", 7]), row("s_demo_002", ["", "Mpox"]), row("s_demo_003", [None, " \t　"]))
        errors = self.errors(data)
        self.assertEqual(
            [(e["code"], e["line"], e["position"]) for e in errors],
            [
                ("prediction_not_string", 1, 2),
                ("blank_prediction", 2, 1),
                ("prediction_not_string", 3, 1),
                ("blank_prediction", 3, 2),
            ],
        )

    def test_repeated_case_id_fails(self):
        data = helpers.jsonl(row("s_demo_001", []), row("s_demo_001", ["Mpox"]), row("s_demo_002", []), row("s_demo_003", []))
        error = self.only_error(data)
        self.assertEqual((error["code"], error["line"], error["first_line"]), ("duplicate_case_id", 2, 1))

    def test_unknown_case_id_fails(self):
        data = helpers.jsonl(row("s_demo_001", []), row("s_demo_002", []), row("s_demo_003", []), row("s_other", []))
        error = self.only_error(data)
        self.assertEqual((error["code"], error["line"], error["case_id"]), ("unknown_case_id", 4, "s_other"))

    def test_case_ids_are_exact(self):
        data = helpers.jsonl(row("S_DEMO_001", []), row("s_demo_002", []), row("s_demo_003", []))
        codes = [e["code"] for e in self.errors(data)]
        self.assertEqual(codes, ["unknown_case_id", "missing_case_ids"])

    def test_extra_key_fails(self):
        data = helpers.jsonl(
            {"case_id": "s_demo_001", "predictions": [], "model": "x"}, row("s_demo_002", []), row("s_demo_003", [])
        )
        error = self.only_error(data)
        self.assertEqual(error["code"], "invalid_keys")
        self.assertEqual((error["unexpected_keys"], error["missing_keys"]), (["model"], []))

    def test_missing_key_fails(self):
        data = helpers.jsonl({"case_id": "s_demo_001"}, row("s_demo_002", []), row("s_demo_003", []))
        error = self.only_error(data)
        self.assertEqual((error["code"], error["missing_keys"]), ("invalid_keys", ["predictions"]))

    def test_wrong_types(self):
        data = helpers.jsonl(row(1, []), row("s_demo_002", "Mpox"), row("s_demo_003", {"a": 1}))
        codes = [(e["code"], e["line"]) for e in self.errors(data, ["s_demo_002", "s_demo_003"])]
        self.assertEqual(
            codes,
            [("case_id_not_string", 1), ("predictions_not_array", 2), ("predictions_not_array", 3)],
        )

    def test_invalid_json_row_not_object_and_duplicate_key(self):
        data = (
            b'{"case_id": "s_demo_001", "predictions": [}\n'
            b'["s_demo_002", []]\n'
            b'{"case_id": "s_demo_003", "case_id": "s_demo_003", "predictions": []}\n'
            b'{"case_id": "s_demo_003", "predictions": [NaN]}\n'
        )
        errors = self.errors(data)
        self.assertEqual(
            [(e["code"], e["line"]) for e in errors],
            [("invalid_json", 1), ("row_not_object", 2), ("duplicate_key", 3), ("invalid_json", 4), ("missing_case_ids", None)],
        )
        self.assertEqual(errors[-1]["case_ids"], CASES)

    def test_invalid_utf8_is_reported_per_line(self):
        data = helpers.jsonl(row("s_demo_001", [])) + b'{"case_id": "s_demo_002", "predictions": ["\xff"]}\n'
        data += helpers.jsonl(row("s_demo_003", []))
        codes = [(e["code"], e["line"]) for e in self.errors(data)]
        self.assertEqual(codes, [("invalid_utf8", 2), ("missing_case_ids", None)])

    def test_bom_blank_lines_and_crlf(self):
        data = (
            b"\xef\xbb\xbf"
            + b'{"case_id": "s_demo_001", "predictions": ["Mpox"]}\r\n'
            + b"\r\n   \n"
            + b'{"case_id": "s_demo_002", "predictions": []}\r\n'
            + b'{"case_id": "s_demo_003", "predictions": ["a", "b", "c", "d"]}\r\n'
            + b"\n"
        )
        error = self.only_error(data)
        self.assertEqual((error["code"], error["line"]), ("too_many_predictions", 5))  # physical line number

    def test_bom_in_the_middle_is_invalid_json(self):
        data = helpers.jsonl(row("s_demo_001", [])) + "﻿".encode() + helpers.jsonl(row("s_demo_002", []), row("s_demo_003", []))
        codes = [(e["code"], e["line"]) for e in self.errors(data)]
        self.assertEqual(codes, [("invalid_json", 2), ("missing_case_ids", None)])

    def test_every_problem_in_a_file_is_reported(self):
        expected = load_case_list(helpers.FIXTURES / "cases.txt")
        report = validate_predictions_file(helpers.FIXTURES / "predictions_invalid.jsonl", expected, helpers.real_vocab())
        self.assertEqual(
            [(e["code"], e["line"]) for e in report.errors],
            [
                ("duplicate_prediction", 1),
                ("too_many_predictions", 2),
                ("prediction_not_string", 3),
                ("blank_prediction", 3),
                ("duplicate_case_id", 4),
                ("invalid_keys", 5),
                ("unknown_case_id", 5),
                ("invalid_json", 6),
                ("missing_case_ids", None),
            ],
        )
        self.assertEqual(report.errors[-1]["case_ids"], ["s_demo_004", "s_demo_005"])
        self.assertEqual(report.to_report()["error_count"], 9)


class CaseListTest(unittest.TestCase):
    def test_parsing(self):
        data = b"\xef\xbb\xbf s_a \r\n\n s_b\n\t\n"
        self.assertEqual(parse_case_list(data), ["s_a", "s_b"])

    def test_errors(self):
        with self.assertRaises(InputError) as ctx:
            parse_case_list(b"s_a\ns_b\ns_a\n")
        self.assertEqual(ctx.exception.source, "cases")
        self.assertEqual([(e["code"], e["line"]) for e in ctx.exception.errors], [("duplicate_case_id", 3)])
        with self.assertRaises(InputError) as ctx:
            parse_case_list(b"\n \n")
        self.assertEqual(ctx.exception.errors[0]["code"], "empty_case_list")


class ReferenceTest(unittest.TestCase):
    def errors(self, data):
        with self.assertRaises(InputError) as ctx:
            parse_reference(data, helpers.real_vocab())
        self.assertEqual(ctx.exception.source, "reference")
        return [(e["code"], e["line"]) for e in ctx.exception.errors]

    def test_fixture(self):
        ref = load_reference(helpers.FIXTURES / "reference.csv", helpers.real_vocab())
        self.assertEqual(ref.challenge_version, "dev-1")
        self.assertEqual([(c.case_id, c.gold_term_id, c.cohort) for c in ref.cases][1], ("s_demo_002", "DX-074", "rare_high_consequence"))

    def test_bom_and_crlf_accepted(self):
        data = b"\xef\xbb\xbfcase_id,gold_term_id,cohort,challenge_version\r\ns_a,DX-070,,dev-1\r\n"
        ref = parse_reference(data, helpers.real_vocab())
        self.assertEqual((ref.case_ids, ref.cases[0].cohort), (["s_a"], ""))

    def test_header_must_match(self):
        self.assertEqual(self.errors(b"case_id,gold,cohort,challenge_version\ns_a,DX-070,x,dev-1\n"), [("invalid_header", 1)])

    def test_row_errors(self):
        data = (
            b"case_id,gold_term_id,cohort,challenge_version\n"
            b"s_a,DX-070,x,dev-1\n"
            b"s_a,DX-074,x,dev-1\n"
            b"s_b,DX-999,x,dev-1\n"
            b"s_c,DX-070,x,dev-2\n"
            b"s_d,DX-070,x\n"
            b"s_e , DX-070,x,dev-1\n"
            b",DX-070,x,dev-1\n"
        )
        self.assertEqual(
            self.errors(data),
            [
                ("duplicate_case_id", 3),
                ("unknown_gold_term_id", 4),
                ("mixed_challenge_version", 5),
                ("wrong_field_count", 6),
                ("field_whitespace", 7),
                ("field_whitespace", 7),
                ("empty_field", 8),
            ],
        )

    def test_empty(self):
        self.assertEqual(self.errors(b""), [("empty_reference", None)])
        self.assertEqual(self.errors(b"case_id,gold_term_id,cohort,challenge_version\n"), [("empty_reference", None)])


if __name__ == "__main__":
    unittest.main()
