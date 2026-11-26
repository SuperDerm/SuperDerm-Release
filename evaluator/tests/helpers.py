"""Shared test helpers (not a test module)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

EVALUATOR_DIR = Path(__file__).resolve().parent.parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from superderm_eval.score import parse_config  # noqa: E402
from superderm_eval.validate import parse_reference  # noqa: E402
from superderm_eval.vocab import load_vocabulary, parse_vocabulary  # noqa: E402

FIXTURES = EVALUATOR_DIR / "fixtures" / "dev"
# The platform repository keeps the taxonomy in content/vocab/; the public SuperDerm-Release repository in taxonomy/.
REPO_VOCAB_PATH = next(
    (p for p in (EVALUATOR_DIR.parent / "content" / "vocab" / "diagnoses.v1.json", EVALUATOR_DIR.parent / "taxonomy" / "diagnoses.v1.json") if p.exists()),
    EVALUATOR_DIR.parent / "content" / "vocab" / "diagnoses.v1.json",
)

_real_vocab = None


def real_vocab():
    """The repository vocabulary, loaded through the package-relative default path."""
    global _real_vocab
    if _real_vocab is None:
        _real_vocab = load_vocabulary()
    return _real_vocab


def jsonl(*rows: Any) -> bytes:
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def reference_bytes(rows: list[tuple[str, str, str]], version: str = "dev-1") -> bytes:
    lines = ["case_id,gold_term_id,cohort,challenge_version"]
    lines += [f"{case_id},{gold},{cohort},{version}" for case_id, gold, cohort in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def reference(rows: list[tuple[str, str, str]], vocab=None, version: str = "dev-1"):
    return parse_reference(reference_bytes(rows, version), vocab or real_vocab())


def config_dict(primary: dict | None = None, secondary: list | None = None, **overrides: Any) -> dict:
    data = {
        "challenge_id": "rare-and-mimics",
        "challenge_version": "dev-1",
        "scorer_version": "1.0.0",
        "vocabulary": {"id": "superderm-diagnoses", "version": "1.0.0"},
        "primary": primary or {"metric": "macro_recall@3", "diagnoses": ["DX-070", "DX-074"]},
        "secondary": secondary if secondary is not None else [{"metric": "acc@1"}, {"metric": "acc@3"}],
    }
    data.update(overrides)
    return data


def config(primary: dict | None = None, secondary: list | None = None, vocab=None, **overrides: Any):
    return parse_config(config_dict(primary, secondary, **overrides), vocab or real_vocab())


def tiny_vocab_dict(**overrides: Any) -> dict:
    """A synthetic vocabulary in which the synonym "Pox" belongs to two terms."""
    data = {
        "id": "tiny-test-vocab",
        "version": "0.0.1",
        "synonymsStatus": "approved",
        "terms": [
            {"id": "T-1", "name": "Alpha Pox", "rare": True, "highConsequence": True, "synonyms": ["Pox", "Alpha"]},
            {"id": "T-2", "name": "Beta Pox", "rare": False, "highConsequence": True, "synonyms": ["pox "]},
            {"id": "T-3", "name": "Gamma Rash", "rare": True, "highConsequence": False, "synonyms": []},
        ],
    }
    data.update(overrides)
    return data


def tiny_vocab():
    return parse_vocabulary(tiny_vocab_dict())
