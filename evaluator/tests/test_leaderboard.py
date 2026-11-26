import unittest

import helpers
from superderm_eval.leaderboard import metric_entry, rank_runs
from superderm_eval.score import score_predictions


def result(value, secondary=None, label="macro_recall@3[diagnoses=DX-070,DX-074]", version="dev-1"):
    return {
        "challenge_id": "rare-and-mimics",
        "challenge_version": version,
        "primary": {"label": label, "value": value},
        "secondary": [{"label": "acc@1", "value": v} for v in ([] if secondary is None else [secondary])],
    }


def ranks(ranking):
    return [(r["run_id"], r["rank"]) for r in ranking]


class RankRunsTest(unittest.TestCase):
    def test_standard_competition_ranking(self):
        ranking = rank_runs({"run-a": result(0.6), "run-b": result(0.4), "run-c": result(0.6)})
        self.assertEqual(ranks(ranking), [("run-a", 1), ("run-c", 1), ("run-b", 3)])
        self.assertEqual([r["value"] for r in ranking], [0.6, 0.6, 0.4])

    def test_several_tie_groups(self):
        runs = {"e": result(0.25), "d": result(0.5), "c": result(0.75), "b": result(0.5), "a": result(0.25), "f": result(0.1)}
        self.assertEqual(
            ranks(rank_runs(runs)),
            [("c", 1), ("b", 2), ("d", 2), ("a", 4), ("e", 4), ("f", 6)],
        )

    def test_all_tied(self):
        self.assertEqual(ranks(rank_runs([("x", result(0.5)), ("y", result(0.5))])), [("x", 1), ("y", 1)])

    def test_null_values_are_unranked_and_last(self):
        ranking = rank_runs({"a": result(None), "b": result(0.0), "c": result(0.3)})
        self.assertEqual(ranks(ranking), [("c", 1), ("b", 2), ("a", None)])

    def test_rank_by_secondary_label(self):
        runs = {"a": result(0.9, secondary=0.1), "b": result(0.1, secondary=0.9)}
        self.assertEqual(ranks(rank_runs(runs, "acc@1")), [("b", 1), ("a", 2)])
        self.assertEqual(metric_entry(runs["a"], "acc@1")["value"], 0.1)
        with self.assertRaises(KeyError):
            rank_runs(runs, "acc@3")

    def test_incomparable_runs_are_rejected(self):
        with self.assertRaises(ValueError):
            rank_runs({"a": result(0.5), "b": result(0.5, version="dev-2")})
        with self.assertRaises(ValueError):
            rank_runs({"a": result(0.5), "b": result(0.5, label="acc@1")})
        with self.assertRaises(ValueError):
            rank_runs([("a", result(0.5)), ("a", result(0.4))])
        with self.assertRaises(ValueError):
            rank_runs({"a": result(float("nan"))})

    def test_ranks_real_score_results(self):
        vocab = helpers.real_vocab()
        ref = helpers.reference([("s1", "DX-070", "x"), ("s2", "DX-074", "x")])
        cfg = helpers.config()

        def run(first, second):
            data = helpers.jsonl({"case_id": "s1", "predictions": first}, {"case_id": "s2", "predictions": second})
            return score_predictions(data, ref, vocab, cfg)

        results = {
            "perfect": run(["Mpox"], ["Smallpox"]),
            "late": run(["Varicella", "Mpox"], ["Varicella", "Herpes Zoster", "Variola"]),  # same macro_recall@3
            "half": run(["Mpox"], []),
            "none": run([], []),
        }
        self.assertEqual(ranks(rank_runs(results)), [("late", 1), ("perfect", 1), ("half", 3), ("none", 4)])
        self.assertEqual(ranks(rank_runs(results, "acc@1")), [("perfect", 1), ("half", 2), ("late", 3), ("none", 3)])


if __name__ == "__main__":
    unittest.main()
