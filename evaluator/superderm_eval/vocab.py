"""Diagnosis vocabulary: loading, name normalization and lookup.

Normalization is applied identically to vocabulary names, vocabulary synonyms
and submitted answers:

1. Unicode NFKC normalization (full-width letters, ligatures, non-breaking
   spaces and other compatibility forms become their plain equivalents);
2. case folding with ``str.casefold`` (stronger than lower-casing);
3. leading and trailing whitespace removed;
4. every internal run of whitespace replaced by one ASCII space.

"Whitespace" means any character for which ``str.isspace()`` is true. Nothing
else is changed: punctuation, hyphens, parentheses and accents (after NFKC)
are kept, so "Smallpox (Variola)" and "Smallpox Variola" are different strings.

A normalized string that comes from the names or synonyms of more than one
term is *ambiguous*: it maps to no term and therefore scores zero.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# evaluator/superderm_eval/vocab.py -> <repo>/content/vocab/diagnoses.v1.json in the platform repository,
# or <repo>/taxonomy/diagnoses.v1.json in the public SuperDerm-Release repository.
_REPO = Path(__file__).resolve().parent.parent.parent
_VOCAB_CANDIDATES = (_REPO / "content" / "vocab" / "diagnoses.v1.json", _REPO / "taxonomy" / "diagnoses.v1.json")
DEFAULT_VOCAB_PATH = next((p for p in _VOCAB_CANDIDATES if p.exists()), _VOCAB_CANDIDATES[0])

RARE_AND_HIGH_CONSEQUENCE = "rare_and_high_consequence"


class InputError(Exception):
    """An input (vocabulary, config, reference, case list, predictions) cannot be used.

    ``source`` names the input. ``errors`` is a list of issue dicts, each with at
    least ``code`` (stable identifier), ``line`` (1-based, or None when the
    problem has no line) and ``message``.
    """

    def __init__(self, source: str, errors: list[dict[str, Any]]):
        self.source = source
        self.errors = errors
        summary = "; ".join(str(e.get("message")) for e in errors[:3])
        more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
        super().__init__(f"invalid {source}: {summary}{more}")

    def to_report(self) -> dict[str, Any]:
        return {
            "ok": False,
            "input": self.source,
            "error_count": len(self.errors),
            "errors": self.errors,
        }


class SubmissionError(InputError):
    """The predictions file is invalid. Raised only for problems in its content."""

    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__("predictions", errors)


def issue(code: str, message: str, line: int | None = None, **extra: Any) -> dict[str, Any]:
    """Build one error entry for a JSON error report."""
    entry: dict[str, Any] = {"code": code, "line": line, "message": message}
    entry.update(extra)
    return entry


def read_input(path: str | Path, source: str) -> bytes:
    """Read a whole input file as bytes; unreadable files raise InputError."""
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        reason = exc.strerror or exc.__class__.__name__
        raise InputError(
            source, [issue("unreadable_file", f"cannot read {source} file {str(path)!r}: {reason}")]
        ) from None


def json_type(value: Any) -> str:
    """JSON type name of a parsed value, for error messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


class DuplicateKeyError(ValueError):
    def __init__(self, key: str):
        self.key = key
        super().__init__(f"duplicate key {key!r}")


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise DuplicateKeyError(key)
        obj[key] = value
    return obj


def strict_json_loads(text: str) -> Any:
    """``json.loads`` that rejects NaN/Infinity and duplicate object keys.

    Raises ``DuplicateKeyError`` (a ValueError) for repeated keys, ValueError for
    any other invalid JSON and RecursionError for absurdly deep nesting.
    """
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def normalize(text: str) -> str:
    """NFKC, casefold, strip, collapse internal whitespace runs to one space."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


@dataclass(frozen=True)
class Term:
    id: str
    name: str
    synonyms: tuple[str, ...]
    rare: bool
    high_consequence: bool


@dataclass(frozen=True)
class Resolution:
    """How one answer string resolves against the vocabulary.

    status: "mapped" (exactly one term), "ambiguous" (several terms),
    "unknown" (no term) or "blank" (empty after normalization).
    ``term_id`` is set only for "mapped"; ``candidates`` only for "ambiguous".
    """

    status: str
    normalized: str
    term_id: str | None = None
    candidates: tuple[str, ...] = ()


class Vocabulary:
    def __init__(
        self,
        vocab_id: str,
        version: str,
        terms: list[Term],
        synonyms_status: str | None = None,
        content_sha256: str | None = None,
    ):
        self.id = vocab_id
        self.version = version
        self.synonyms_status = synonyms_status
        self.content_sha256 = content_sha256
        self.terms: dict[str, Term] = {term.id: term for term in terms}  # file order

        targets: dict[str, set[str]] = {}
        sources: dict[str, list[tuple[str, str]]] = {}
        for term in terms:
            for text in (term.name, *term.synonyms):
                key = normalize(text)
                targets.setdefault(key, set()).add(term.id)
                sources.setdefault(key, []).append((term.id, text))
        self._lookup = {key: next(iter(ids)) for key, ids in targets.items() if len(ids) == 1}
        self.ambiguous: dict[str, tuple[str, ...]] = {
            key: tuple(sorted(ids)) for key, ids in sorted(targets.items()) if len(ids) > 1
        }
        self._sources = sources
        self.n_names = len(targets)

    def resolve(self, text: str) -> Resolution:
        key = normalize(text)
        if not key:
            return Resolution("blank", key)
        term_id = self._lookup.get(key)
        if term_id is not None:
            return Resolution("mapped", key, term_id=term_id)
        if key in self.ambiguous:
            return Resolution("ambiguous", key, candidates=self.ambiguous[key])
        return Resolution("unknown", key)

    def term_id_for(self, text: str) -> str | None:
        """The term an answer maps to unambiguously, else None."""
        return self.resolve(text).term_id

    def rare_and_high_consequence(self) -> tuple[str, ...]:
        """IDs of terms flagged both rare and high-consequence, in vocabulary order."""
        return tuple(t.id for t in self.terms.values() if t.rare and t.high_consequence)

    def summary(self) -> dict[str, Any]:
        return {
            "content_sha256": self.content_sha256,
            "id": self.id,
            "synonyms_status": self.synonyms_status,
            "version": self.version,
        }

    def check_report(self) -> dict[str, Any]:
        """Report printed by ``vocab-check``."""
        ambiguous = [
            {
                "normalized": key,
                "sources": [
                    {"term_id": term_id, "text": text}
                    for term_id, text in sorted(self._sources[key])
                ],
                "term_ids": list(term_ids),
            }
            for key, term_ids in self.ambiguous.items()
        ]
        warnings = []
        if self.synonyms_status != "approved":
            warnings.append(
                f"synonymsStatus is {self.synonyms_status!r}, not 'approved': synonym matches "
                "are provisional until the clinical lead approves this vocabulary version"
            )
        if ambiguous:
            warnings.append(
                f"{len(ambiguous)} normalized name(s) belong to more than one term; "
                "answers using them map to no term and score zero"
            )
        return {
            "ambiguous": ambiguous,
            "n_ambiguous": len(ambiguous),
            "n_names": self.n_names,
            "n_rare_and_high_consequence": len(self.rare_and_high_consequence()),
            "n_terms": len(self.terms),
            "ok": True,
            "vocabulary": self.summary(),
            "warnings": warnings,
        }


def canonical_sha256(data: Any) -> str:
    """SHA-256 of a JSON value's canonical form (sorted keys, no spaces, ASCII escapes).

    Independent of indentation and line endings, so a checkout with CRLF line
    endings hashes the same as the original file.
    """
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _clean_string(value: Any) -> bool:
    return isinstance(value, str) and value != "" and value == value.strip()


def parse_vocabulary(data: Any, content_sha256: str | None = None) -> Vocabulary:
    """Build a Vocabulary from the parsed JSON; structural problems raise InputError."""
    errors: list[dict[str, Any]] = []

    def bad(field: str, message: str, code: str = "invalid_field") -> None:
        errors.append(issue(code, f"{field}: {message}", None, field=field))

    if not isinstance(data, dict):
        raise InputError("vocabulary", [issue("not_an_object", "vocabulary must be a JSON object")])
    for field in ("id", "version"):
        if not _clean_string(data.get(field)):
            bad(field, "must be a non-empty string without surrounding whitespace")
    synonyms_status = data.get("synonymsStatus")
    if synonyms_status is not None and not isinstance(synonyms_status, str):
        bad("synonymsStatus", "must be a string when present")
    raw_terms = data.get("terms")
    if not isinstance(raw_terms, list) or not raw_terms:
        bad("terms", "must be a non-empty array")
        raw_terms = []

    terms: list[Term] = []
    seen_ids: dict[str, int] = {}
    for index, raw in enumerate(raw_terms):
        where = f"terms[{index}]"
        if not isinstance(raw, dict):
            bad(where, "must be an object")
            continue
        ok = True
        term_id = raw.get("id")
        if not _clean_string(term_id):
            bad(f"{where}.id", "must be a non-empty string without surrounding whitespace")
            ok = False
        elif term_id in seen_ids:
            bad(f"{where}.id", f"{term_id!r} repeats terms[{seen_ids[term_id]}].id", "duplicate_term_id")
            ok = False
        else:
            seen_ids[term_id] = index
        name = raw.get("name")
        if not isinstance(name, str) or not normalize(name):
            bad(f"{where}.name", "must be a non-blank string")
            ok = False
        synonyms = raw.get("synonyms")
        if not isinstance(synonyms, list):
            bad(f"{where}.synonyms", "must be an array of strings")
            ok = False
            synonyms = []
        for s_index, synonym in enumerate(synonyms):
            if not isinstance(synonym, str) or not normalize(synonym):
                bad(f"{where}.synonyms[{s_index}]", "must be a non-blank string")
                ok = False
        for flag in ("rare", "highConsequence"):
            if not isinstance(raw.get(flag), bool):
                bad(f"{where}.{flag}", "must be true or false")
                ok = False
        if ok:
            terms.append(
                Term(
                    id=term_id,
                    name=name,
                    synonyms=tuple(synonyms),
                    rare=raw["rare"],
                    high_consequence=raw["highConsequence"],
                )
            )
    if errors:
        raise InputError("vocabulary", errors)
    return Vocabulary(data["id"], data["version"], terms, synonyms_status, content_sha256)


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    """Load the vocabulary JSON (default: the repository's diagnoses.v1.json)."""
    path = DEFAULT_VOCAB_PATH if path is None else Path(path)
    raw = read_input(path, "vocabulary")
    try:
        data = strict_json_loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise InputError("vocabulary", [issue("invalid_utf8", f"vocabulary is not UTF-8: {exc.reason}")]) from None
    except (ValueError, RecursionError) as exc:
        raise InputError("vocabulary", [issue("invalid_json", f"vocabulary is not valid JSON: {exc}")]) from None
    return parse_vocabulary(data, content_sha256=canonical_sha256(data))
