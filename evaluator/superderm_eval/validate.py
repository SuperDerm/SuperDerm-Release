"""Validation of predictions files, reference files and case lists.

A predictions file is either entirely valid or rejected; it is never partly
scored. Every problem found is reported with its 1-based physical line number
(``line``; None for whole-file problems such as missing cases) and a stable
``code``:

    invalid_utf8           line is not valid UTF-8
    invalid_json           line is not valid JSON (NaN/Infinity are rejected)
    duplicate_key          a key appears twice in one JSON object
    row_not_object         row is valid JSON but not an object
    invalid_keys           keys are not exactly "case_id" and "predictions"
    case_id_not_string     case_id is not a string
    unknown_case_id        case_id is not in the expected case list
    duplicate_case_id      case_id already appeared on an earlier line
    missing_case_ids       expected cases with no row (listed in "case_ids")
    predictions_not_array  predictions is not an array
    too_many_predictions   more than three entries
    prediction_not_string  an entry is not a string ("position" = 1-based rank)
    blank_prediction       an entry is empty after normalization ("position")
    duplicate_prediction   two entries have the same normalized text, or map to
                           the same term ("positions" = both 1-based ranks)

Unknown, unmapped and ambiguous answers are not errors; they score zero.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .vocab import (
    DuplicateKeyError,
    InputError,
    SubmissionError,
    Vocabulary,
    issue,
    json_type,
    read_input,
    strict_json_loads,
)

PREDICTION_ERROR_CODES = frozenset(
    {
        "invalid_utf8",
        "invalid_json",
        "duplicate_key",
        "row_not_object",
        "invalid_keys",
        "case_id_not_string",
        "unknown_case_id",
        "duplicate_case_id",
        "missing_case_ids",
        "predictions_not_array",
        "too_many_predictions",
        "prediction_not_string",
        "blank_prediction",
        "duplicate_prediction",
    }
)
PREDICTION_KEYS = ("case_id", "predictions")
MAX_PREDICTIONS = 3
REFERENCE_HEADER = ("case_id", "gold_term_id", "cohort", "challenge_version")
UTF8_BOM = b"\xef\xbb\xbf"


def _physical_lines(data: bytes) -> list[tuple[int, bytes]]:
    """(1-based line number, bytes) for each "\\n"-separated line, leading BOM removed."""
    if data.startswith(UTF8_BOM):
        data = data[len(UTF8_BOM):]
    return list(enumerate(data.split(b"\n"), start=1))


# --------------------------------------------------------------------------- predictions


@dataclass
class ValidationReport:
    errors: list[dict[str, Any]]
    rows: dict[str, tuple[str, ...]]  # case_id -> predictions; complete only when ok
    sha256: str  # of the file bytes exactly as received

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_report(self) -> dict[str, Any]:
        if not self.ok:
            return SubmissionError(self.errors).to_report()
        return {"n_cases": len(self.rows), "ok": True, "predictions_sha256": self.sha256}

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise SubmissionError(self.errors)


def _check_predictions(
    predictions: Any, vocab: Vocabulary, line: int, context: dict[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        return [
            issue(
                "predictions_not_array",
                f"'predictions' must be an array of at most {MAX_PREDICTIONS} strings, "
                f"got {json_type(predictions)}",
                line,
                **context,
            )
        ]
    problems = []
    if len(predictions) > MAX_PREDICTIONS:
        problems.append(
            issue(
                "too_many_predictions",
                f"'predictions' has {len(predictions)} entries; at most {MAX_PREDICTIONS} are allowed",
                line,
                count=len(predictions),
                **context,
            )
        )
    first_text: dict[str, int] = {}  # normalized text -> first rank
    first_term: dict[str, int] = {}  # term id -> first rank
    for rank, entry in enumerate(predictions, start=1):
        if not isinstance(entry, str):
            problems.append(
                issue(
                    "prediction_not_string",
                    f"prediction {rank} must be a string, got {json_type(entry)}",
                    line,
                    position=rank,
                    **context,
                )
            )
            continue
        resolution = vocab.resolve(entry)
        key = resolution.normalized
        if not key:
            problems.append(
                issue("blank_prediction", f"prediction {rank} is blank", line, position=rank, **context)
            )
            continue
        term_id = resolution.term_id
        if key in first_text:
            earlier = first_text[key]
            problems.append(
                issue(
                    "duplicate_prediction",
                    f"predictions {earlier} and {rank} are the same after normalization ({key!r})",
                    line,
                    kind="same_text",
                    positions=[earlier, rank],
                    term_id=term_id,
                    **context,
                )
            )
            continue
        first_text[key] = rank
        if term_id is None:
            continue
        if term_id in first_term:
            earlier = first_term[term_id]
            problems.append(
                issue(
                    "duplicate_prediction",
                    f"predictions {earlier} ({predictions[earlier - 1]!r}) and {rank} ({entry!r}) "
                    f"both map to {term_id} ({vocab.terms[term_id].name})",
                    line,
                    kind="same_term",
                    positions=[earlier, rank],
                    term_id=term_id,
                    **context,
                )
            )
        else:
            first_term[term_id] = rank
    return problems


def validate_predictions(
    data: bytes, expected_case_ids: Iterable[str], vocab: Vocabulary
) -> ValidationReport:
    """Validate predictions JSONL bytes against the expected case IDs."""
    expected = set(expected_case_ids)
    errors: list[dict[str, Any]] = []
    rows: dict[str, tuple[str, ...]] = {}
    first_line: dict[str, int] = {}

    for line, raw in _physical_lines(data):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(
                issue("invalid_utf8", f"line is not valid UTF-8 ({exc.reason} at byte {exc.start + 1})", line)
            )
            continue
        if not text.strip():
            continue
        try:
            row = strict_json_loads(text)
        except DuplicateKeyError as exc:
            errors.append(
                issue("duplicate_key", f"key {exc.key!r} appears more than once in one object", line, key=exc.key)
            )
            continue
        except RecursionError:
            errors.append(issue("invalid_json", "line is not valid JSON: nesting too deep", line))
            continue
        except ValueError as exc:
            detail = f"{exc.msg} at column {exc.colno}" if hasattr(exc, "colno") else str(exc)
            errors.append(issue("invalid_json", f"line is not valid JSON: {detail}", line))
            continue
        if not isinstance(row, dict):
            errors.append(issue("row_not_object", f"row must be a JSON object, got {json_type(row)}", line))
            continue

        context: dict[str, Any] = {}
        if isinstance(row.get("case_id"), str):
            context["case_id"] = row["case_id"]
        keys_ok = set(row) == set(PREDICTION_KEYS)
        if not keys_ok:
            missing = [key for key in PREDICTION_KEYS if key not in row]
            unexpected = sorted(key for key in row if key not in PREDICTION_KEYS)
            parts = []
            if missing:
                parts.append("missing " + ", ".join(repr(k) for k in missing))
            if unexpected:
                parts.append("unexpected " + ", ".join(repr(k) for k in unexpected))
            errors.append(
                issue(
                    "invalid_keys",
                    "row must have exactly the keys 'case_id' and 'predictions' (" + "; ".join(parts) + ")",
                    line,
                    missing_keys=missing,
                    unexpected_keys=unexpected,
                    **context,
                )
            )

        accepted: str | None = None
        if "case_id" in row:
            case_id = row["case_id"]
            if not isinstance(case_id, str):
                errors.append(
                    issue("case_id_not_string", f"'case_id' must be a string, got {json_type(case_id)}", line)
                )
            elif case_id not in expected:
                errors.append(
                    issue("unknown_case_id", f"case_id {case_id!r} is not in the expected case list", line, case_id=case_id)
                )
            elif case_id in first_line:
                errors.append(
                    issue(
                        "duplicate_case_id",
                        f"case_id {case_id!r} already appeared on line {first_line[case_id]}",
                        line,
                        case_id=case_id,
                        first_line=first_line[case_id],
                    )
                )
            else:
                first_line[case_id] = line
                accepted = case_id

        if "predictions" in row:
            problems = _check_predictions(row["predictions"], vocab, line, context)
            errors.extend(problems)
            if accepted is not None and keys_ok and not problems:
                rows[accepted] = tuple(row["predictions"])

    missing_ids = sorted(expected.difference(first_line))
    if missing_ids:
        shown = ", ".join(missing_ids[:10])
        if len(missing_ids) > 10:
            shown += f", ... ({len(missing_ids) - 10} more)"
        errors.append(
            issue(
                "missing_case_ids",
                f"{len(missing_ids)} expected case ID(s) have no row: {shown}",
                None,
                case_ids=missing_ids,
            )
        )

    return ValidationReport(errors=errors, rows=rows, sha256=hashlib.sha256(data).hexdigest())


def validate_predictions_file(
    path: str | Path, expected_case_ids: Iterable[str], vocab: Vocabulary
) -> ValidationReport:
    return validate_predictions(read_input(path, "predictions"), expected_case_ids, vocab)


# --------------------------------------------------------------------------- case list


def parse_case_list(data: bytes) -> list[str]:
    """One case_id per line; surrounding whitespace and empty lines are ignored."""
    errors: list[dict[str, Any]] = []
    case_ids: list[str] = []
    first_line: dict[str, int] = {}
    for line, raw in _physical_lines(data):
        try:
            case_id = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            errors.append(issue("invalid_utf8", f"line is not valid UTF-8 ({exc.reason})", line))
            continue
        if not case_id:
            continue
        if case_id in first_line:
            errors.append(
                issue(
                    "duplicate_case_id",
                    f"case_id {case_id!r} already appeared on line {first_line[case_id]}",
                    line,
                    case_id=case_id,
                    first_line=first_line[case_id],
                )
            )
            continue
        first_line[case_id] = line
        case_ids.append(case_id)
    if not case_ids and not errors:
        errors.append(issue("empty_case_list", "case list contains no case IDs"))
    if errors:
        raise InputError("cases", errors)
    return case_ids


def load_case_list(path: str | Path) -> list[str]:
    return parse_case_list(read_input(path, "cases"))


# --------------------------------------------------------------------------- reference


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    gold_term_id: str
    cohort: str
    line: int


@dataclass(frozen=True)
class Reference:
    cases: tuple[ReferenceCase, ...]  # file order
    challenge_version: str

    @property
    def case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases]


def parse_reference(data: bytes, vocab: Vocabulary) -> Reference:
    """Parse and check reference CSV bytes (header case_id,gold_term_id,cohort,challenge_version)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = data.count(b"\n", 0, exc.start) + 1
        raise InputError("reference", [issue("invalid_utf8", f"reference is not valid UTF-8 ({exc.reason})", line)]) from None
    if text.startswith("﻿"):
        text = text[1:]

    errors: list[dict[str, Any]] = []
    cases: list[ReferenceCase] = []
    first_line: dict[str, int] = {}
    version: str | None = None
    header: list[str] | None = None
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        for row in reader:
            line = reader.line_num
            if not row or (len(row) == 1 and not row[0].strip()):
                continue  # blank line
            if header is None:
                header = row
                if tuple(row) != REFERENCE_HEADER:
                    raise InputError(
                        "reference",
                        [
                            issue(
                                "invalid_header",
                                f"header must be exactly {','.join(REFERENCE_HEADER)!r}, got {','.join(row)!r}",
                                line,
                            )
                        ],
                    )
                continue
            if len(row) != len(REFERENCE_HEADER):
                errors.append(
                    issue(
                        "wrong_field_count",
                        f"row has {len(row)} fields; expected {len(REFERENCE_HEADER)}",
                        line,
                    )
                )
                continue
            case_id, gold, cohort, row_version = row
            # Content checks below only look at fields that are clean, to avoid cascades.
            padded = [name for name, value in zip(REFERENCE_HEADER, row) if value != value.strip()]
            for name in padded:
                value = row[REFERENCE_HEADER.index(name)]
                errors.append(
                    issue("field_whitespace", f"{name} {value!r} has surrounding whitespace", line, field=name)
                )
            row_ok = not padded
            for name, value in (("case_id", case_id), ("gold_term_id", gold), ("challenge_version", row_version)):
                if not value:
                    errors.append(issue("empty_field", f"{name} is empty", line, field=name))
                    row_ok = False
            if case_id in first_line:
                errors.append(
                    issue(
                        "duplicate_case_id",
                        f"case_id {case_id!r} already appeared on line {first_line[case_id]}",
                        line,
                        case_id=case_id,
                        first_line=first_line[case_id],
                    )
                )
                row_ok = False
            elif case_id:
                first_line[case_id] = line
            if gold and "gold_term_id" not in padded and gold not in vocab.terms:
                errors.append(
                    issue(
                        "unknown_gold_term_id",
                        f"gold_term_id {gold!r} is not a term of vocabulary {vocab.id} {vocab.version}",
                        line,
                        case_id=case_id,
                    )
                )
                row_ok = False
            if row_version and "challenge_version" not in padded:
                if version is None:
                    version = row_version
                elif row_version != version:
                    errors.append(
                        issue(
                            "mixed_challenge_version",
                            f"challenge_version {row_version!r} differs from {version!r} on earlier rows",
                            line,
                        )
                    )
                    row_ok = False
            if row_ok:
                cases.append(ReferenceCase(case_id, gold, cohort, line))
    except csv.Error as exc:
        errors.append(issue("invalid_csv", f"reference is not valid CSV: {exc}", reader.line_num))
    if header is None and not errors:
        errors.append(issue("empty_reference", "reference file is empty"))
    elif not first_line and not errors:
        errors.append(issue("empty_reference", "reference has a header but no case rows"))
    if errors:
        raise InputError("reference", errors)
    assert version is not None
    return Reference(tuple(cases), version)


def load_reference(path: str | Path, vocab: Vocabulary) -> Reference:
    return parse_reference(read_input(path, "reference"), vocab)
