import json
import tempfile
import unittest
from pathlib import Path

import helpers
from superderm_eval.vocab import (
    DEFAULT_VOCAB_PATH,
    InputError,
    load_vocabulary,
    normalize,
    parse_vocabulary,
)


class RealVocabularyTest(unittest.TestCase):
    def test_default_path_is_the_repository_vocabulary(self):
        self.assertEqual(DEFAULT_VOCAB_PATH, helpers.REPO_VOCAB_PATH.resolve())
        self.assertTrue(DEFAULT_VOCAB_PATH.is_file())

    def test_loads_with_expected_shape(self):
        vocab = helpers.real_vocab()
        self.assertEqual((vocab.id, vocab.version), ("superderm-diagnoses", "1.0.0"))
        self.assertEqual(vocab.synonyms_status, "draft")
        self.assertEqual(len(vocab.terms), 214)
        self.assertEqual(len(vocab.rare_and_high_consequence()), 70)
        self.assertIn("DX-070", vocab.rare_and_high_consequence())
        self.assertIn("DX-074", vocab.rare_and_high_consequence())
        self.assertNotIn("DX-076", vocab.rare_and_high_consequence())  # varicella: not rare

    def test_current_vocabulary_has_no_ambiguous_names(self):
        self.assertEqual(helpers.real_vocab().ambiguous, {})

    def test_override_path_loads_another_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            path.write_text(json.dumps(helpers.tiny_vocab_dict()), encoding="utf-8")
            vocab = load_vocabulary(path)
        self.assertEqual((vocab.id, len(vocab.terms)), ("tiny-test-vocab", 3))

    def test_content_hash_ignores_formatting(self):
        data = helpers.tiny_vocab_dict()
        with tempfile.TemporaryDirectory() as tmp:
            compact, pretty = Path(tmp) / "a.json", Path(tmp) / "b.json"
            compact.write_bytes(json.dumps(data, separators=(",", ":")).encode())
            pretty.write_bytes(json.dumps(data, indent=4).replace("\n", "\r\n").encode())
            self.assertEqual(load_vocabulary(compact).content_sha256, load_vocabulary(pretty).content_sha256)


class NormalizationTest(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(normalize("  Herpes \t\n Zoster  "), "herpes zoster")
        self.assertEqual(normalize("ＭＰＯＸ"), "mpox")  # full-width letters (NFKC)
        self.assertEqual(normalize("Herpes Zoster"), "herpes zoster")  # no-break space
        self.assertEqual(normalize("ﬁfth disease"), "fifth disease")  # "fi" ligature
        self.assertEqual(normalize("STRASSE"), normalize("straße"))  # casefold, not lower
        self.assertEqual(normalize(" 　\t"), "")
        self.assertEqual(normalize("Smallpox (Variola)"), "smallpox (variola)")  # punctuation kept

    def test_idempotent_for_every_code_point(self):
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            once = normalize(chr(cp))
            self.assertEqual(normalize(once), once, f"U+{cp:04X}")

    def test_case_whitespace_and_nfkc_variants_match(self):
        vocab = helpers.real_vocab()
        variants = {
            "DX-070": ["mpox", "MPOX", "  Mpox ", "ＭＰＯＸ", "monkeypox", "MonkeyPox"],
            "DX-067": ["herpes   zoster", "Herpes\tZoster", "Herpes Zoster", "HERPES ZOSTER", "shingles"],
            "DX-074": ["smallpox", "Smallpox (Variola)", "  SMALLPOX  (variola) ", "variola"],
            "DX-111": ["insect bite", "INSECT　BITE", "Arthropod Bite Reaction (Insect Bite)"],
        }
        for term_id, texts in variants.items():
            for text in texts:
                with self.subTest(text=text):
                    self.assertEqual(vocab.term_id_for(text), term_id)

    def test_near_misses_do_not_match(self):
        vocab = helpers.real_vocab()
        for text in ["Smallpox Variola", "Mpox.", "M-pox", "Not A Real Diagnosis", "insectbite"]:
            with self.subTest(text=text):
                resolution = vocab.resolve(text)
                self.assertEqual(resolution.status, "unknown")
                self.assertIsNone(resolution.term_id)
        self.assertEqual(vocab.resolve("   ").status, "blank")


class AmbiguityTest(unittest.TestCase):
    def test_ambiguous_names_are_reported_and_map_to_nothing(self):
        vocab = helpers.tiny_vocab()
        self.assertEqual(vocab.ambiguous, {"pox": ("T-1", "T-2")})
        resolution = vocab.resolve(" POX")
        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.term_id)
        self.assertEqual(resolution.candidates, ("T-1", "T-2"))
        self.assertEqual(vocab.term_id_for("alpha"), "T-1")

        report = vocab.check_report()
        self.assertEqual(report["n_ambiguous"], 1)
        self.assertEqual(report["ambiguous"][0]["normalized"], "pox")
        self.assertEqual(report["ambiguous"][0]["term_ids"], ["T-1", "T-2"])
        self.assertEqual(
            report["ambiguous"][0]["sources"],
            [{"term_id": "T-1", "text": "Pox"}, {"term_id": "T-2", "text": "pox "}],
        )
        self.assertTrue(any("score zero" in w for w in report["warnings"]))

    def test_draft_synonyms_are_flagged(self):
        report = helpers.real_vocab().check_report()
        self.assertTrue(any("draft" in w for w in report["warnings"]))
        approved = helpers.tiny_vocab().check_report()["warnings"]  # approved, but has an ambiguous name
        self.assertFalse(any("synonymsStatus" in w for w in approved))


class StructureErrorsTest(unittest.TestCase):
    def codes(self, data):
        with self.assertRaises(InputError) as ctx:
            parse_vocabulary(data)
        self.assertEqual(ctx.exception.source, "vocabulary")
        return [e["code"] for e in ctx.exception.errors]

    def test_duplicate_term_id(self):
        data = helpers.tiny_vocab_dict()
        data["terms"][1]["id"] = "T-1"
        self.assertEqual(self.codes(data), ["duplicate_term_id"])

    def test_bad_fields(self):
        data = helpers.tiny_vocab_dict()
        del data["terms"][0]["rare"]
        data["terms"][1]["synonyms"] = "Pox"
        data["terms"][2]["name"] = "  "
        self.assertEqual(self.codes(data), ["invalid_field"] * 3)

    def test_not_an_object_and_bad_json(self):
        self.assertEqual(self.codes([]), ["not_an_object"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.json"
            path.write_text('{"id": "a", "id": "b"}', encoding="utf-8")
            with self.assertRaises(InputError) as ctx:
                load_vocabulary(path)
            self.assertEqual(ctx.exception.errors[0]["code"], "invalid_json")
            with self.assertRaises(InputError) as ctx:
                load_vocabulary(Path(tmp) / "missing.json")
            self.assertEqual(ctx.exception.errors[0]["code"], "unreadable_file")


if __name__ == "__main__":
    unittest.main()
