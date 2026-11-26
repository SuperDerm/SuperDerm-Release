"""Challenge configuration, metrics and the result JSON.

Definitions (U is a set of reference cases; k is 1, 2 or 3):

    m(answer, gold)     1 if the answer maps unambiguously to gold, else 0
    hit@k(case)         max of m over the case's first k answers (at most one hit per case)
    acc@k(U)            (1/|U|) * sum of hit@k over U; |U| counts every reference case
                        in U, so abstentions stay in the denominator
    macro_recall@k(S)   mean of acc@k(cases whose gold is c) over the diagnoses c in S
                        that have at least one reference case

A metric may be restricted to one ``cohort`` (reference rows with that label)
and/or to a ``diagnoses`` set (reference rows whose gold is in the set). When
no reference case is left, the value is null and ``reason`` says why.

Arithmetic is exact (fractions); each value is rounded once, to 6 decimals
(round-half-even on the exact value), when the result is built.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .validate import Reference, load_reference, validate_predictions
from .vocab import (
    RARE_AND_HIGH_CONSEQUENCE,
    InputError,
    Vocabulary,
    issue,
    json_type,
    load_vocabulary,
    read_input,
    strict_json_loads,
)

MAX_K = 3
DECIMALS = 6
METRIC_PATTERN = re.compile(r"(acc|macro_recall)@([1-9][0-9]*)")
CONFIG_KEYS = ("challenge_id", "challenge_version", "scorer_version", "vocabulary", "primary", "secondary")
REQUIRED_CONFIG_KEYS = ("challenge_id", "challenge_version", "scorer_version", "vocabulary", "primary")
METRIC_KEYS = ("metric", "cohort", "diagnoses")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class MetricSpec:
    metric: str  # e.g. "acc@3"
    kind: str  # "acc" or "macro_recall"
    k: int
    cohort: str | None
    selector: str | tuple[str, ...] | None  # "diagnoses" exactly as written in the config
    diagnoses: frozenset[str] | None  # resolved term IDs
    label: str  # unique within a config, e.g. "acc@1[cohort=common_mimic]"

    def describe(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "diagnoses": list(self.selector) if isinstance(self.selector, tuple) else self.selector,
            "label": self.label,
            "metric": self.metric,
            "n_diagnoses_in_set": None if self.diagnoses is None else len(self.diagnoses),
        }


@dataclass(frozen=True)
class ChallengeConfig:
    challenge_id: str
    challenge_version: str
    scorer_version: str
    vocabulary_id: str
    vocabulary_version: str
    primary: MetricSpec
    secondary: tuple[MetricSpec, ...]


def _clean_string(value: Any) -> bool:
    return isinstance(value, str) and value != "" and value == value.strip()


def metric_label(metric: str, cohort: str | None, selector: str | Sequence[str] | None) -> str:
    qualifiers = []
    if cohort is not None:
        qualifiers.append(f"cohort={cohort}")
    if selector is not None:
        qualifiers.append("diagnoses=" + (selector if isinstance(selector, str) else ",".join(selector)))
    return metric + (f"[{';'.join(qualifiers)}]" if qualifiers else "")


def _parse_metric(raw: Any, where: str, vocab: Vocabulary, errors: list[dict[str, Any]]) -> MetricSpec | None:
    def bad(code: str, message: str, at: str = where) -> None:
        errors.append(issue(code, f"{at}: {message}", None, field=at))

    if not isinstance(raw, dict):
        bad("invalid_metric", f"must be an object, got {json_type(raw)}")
        return None
    ok = True
    for key in sorted(set(raw) - set(METRIC_KEYS)):
        bad("unknown_key", "unknown key (allowed: metric, cohort, diagnoses)", f"{where}.{key}")
        ok = False

    metric = raw.get("metric")
    match = METRIC_PATTERN.fullmatch(metric) if isinstance(metric, str) else None
    if match is None or int(match.group(2)) > MAX_K:
        bad(
            "invalid_metric",
            f"must be 'acc@k' or 'macro_recall@k' with k from 1 to {MAX_K}, got {metric!r}",
            f"{where}.metric",
        )
        return None
    kind, k = match.group(1), int(match.group(2))

    cohort = raw.get("cohort")
    if cohort is not None and not _clean_string(cohort):
        bad("invalid_cohort", "must be a non-empty string without surrounding whitespace", f"{where}.cohort")
        ok = False

    selector = raw.get("diagnoses")
    diagnoses: frozenset[str] | None = None
    if selector is None:
        if kind == "macro_recall":
            bad(
                "missing_diagnoses",
                f"{metric} needs 'diagnoses': a list of term IDs or {RARE_AND_HIGH_CONSEQUENCE!r}",
            )
            ok = False
    elif isinstance(selector, str):
        if selector != RARE_AND_HIGH_CONSEQUENCE:
            bad(
                "invalid_diagnoses",
                f"unknown diagnosis set {selector!r}; use a list of term IDs or {RARE_AND_HIGH_CONSEQUENCE!r}",
                f"{where}.diagnoses",
            )
            ok = False
        else:
            diagnoses = frozenset(vocab.rare_and_high_consequence())
            if not diagnoses:
                bad("invalid_diagnoses", "the vocabulary has no term that is both rare and high-consequence", f"{where}.diagnoses")
                ok = False
    elif isinstance(selector, list):
        if not selector:
            bad("invalid_diagnoses", "must not be empty", f"{where}.diagnoses")
            ok = False
        seen: set[str] = set()
        for index, term_id in enumerate(selector):
            at = f"{where}.diagnoses[{index}]"
            if not isinstance(term_id, str):
                bad("invalid_diagnoses", f"must be a term ID string, got {json_type(term_id)}", at)
                ok = False
            elif term_id not in vocab.terms:
                bad("unknown_term_id", f"{term_id!r} is not a term of vocabulary {vocab.id} {vocab.version}", at)
                ok = False
            elif term_id in seen:
                bad("duplicate_term_id", f"{term_id!r} is listed more than once", at)
                ok = False
            else:
                seen.add(term_id)
        selector = tuple(selector)
        diagnoses = frozenset(seen)
    else:
        bad(
            "invalid_diagnoses",
            f"must be a list of term IDs or {RARE_AND_HIGH_CONSEQUENCE!r}, got {json_type(selector)}",
            f"{where}.diagnoses",
        )
        ok = False

    if not ok:
        return None
    return MetricSpec(
        metric=metric,
        kind=kind,
        k=k,
        cohort=cohort,
        selector=selector,
        diagnoses=diagnoses,
        label=metric_label(metric, cohort, selector),
    )


def parse_config(data: Any, vocab: Vocabulary) -> ChallengeConfig:
    """Check a parsed challenge config against the loaded vocabulary and this scorer."""
    errors: list[dict[str, Any]] = []

    def bad(code: str, message: str, at: str) -> None:
        errors.append(issue(code, f"{at}: {message}", None, field=at))

    if not isinstance(data, dict):
        raise InputError("config", [issue("not_an_object", f"config must be a JSON object, got {json_type(data)}")])
    for key in sorted(set(data) - set(CONFIG_KEYS)):
        bad("unknown_key", "unknown key (allowed: " + ", ".join(CONFIG_KEYS) + ")", key)
    for key in REQUIRED_CONFIG_KEYS:
        if key not in data:
            bad("missing_key", "is required", key)
    for key in ("challenge_id", "challenge_version", "scorer_version"):
        if key in data and not _clean_string(data[key]):
            bad("invalid_field", "must be a non-empty string without surrounding whitespace", key)
    if _clean_string(data.get("scorer_version")) and data["scorer_version"] != __version__:
        bad(
            "scorer_version_mismatch",
            f"config requires scorer version {data['scorer_version']!r} but this is superderm-eval {__version__}",
            "scorer_version",
        )

    if "vocabulary" in data:
        declared = data["vocabulary"]
        if (
            not isinstance(declared, dict)
            or set(declared) != {"id", "version"}
            or not all(isinstance(declared[k], str) for k in ("id", "version"))
        ):
            bad("invalid_field", "must be an object with exactly the string keys 'id' and 'version'", "vocabulary")
        elif (declared["id"], declared["version"]) != (vocab.id, vocab.version):
            bad(
                "vocabulary_mismatch",
                f"config expects vocabulary {declared['id']!r} version {declared['version']!r} but the "
                f"loaded vocabulary is {vocab.id!r} version {vocab.version!r}",
                "vocabulary",
            )

    named: list[tuple[str, MetricSpec]] = []
    primary = None
    if "primary" in data:
        primary = _parse_metric(data["primary"], "primary", vocab, errors)
        if primary is not None:
            named.append(("primary", primary))
    secondary: list[MetricSpec] = []
    raw_secondary = data.get("secondary", [])
    if not isinstance(raw_secondary, list):
        bad("invalid_field", f"must be an array of metric objects, got {json_type(raw_secondary)}", "secondary")
    else:
        for index, raw in enumerate(raw_secondary):
            spec = _parse_metric(raw, f"secondary[{index}]", vocab, errors)
            if spec is not None:
                secondary.append(spec)
                named.append((f"secondary[{index}]", spec))
    defined_at: dict[str, str] = {}
    for at, spec in named:
        if spec.label in defined_at:
            bad("duplicate_metric", f"{spec.label!r} is already defined at {defined_at[spec.label]}", at)
        else:
            defined_at[spec.label] = at

    if errors:
        raise InputError("config", errors)
    assert primary is not None
    return ChallengeConfig(
        challenge_id=data["challenge_id"],
        challenge_version=data["challenge_version"],
        scorer_version=data["scorer_version"],
        vocabulary_id=data["vocabulary"]["id"],
        vocabulary_version=data["vocabulary"]["version"],
        primary=primary,
        secondary=tuple(secondary),
    )


def load_config(path: str | Path, vocab: Vocabulary) -> ChallengeConfig:
    raw = read_input(path, "config")
    try:
        data = strict_json_loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise InputError("config", [issue("invalid_utf8", f"config is not UTF-8: {exc.reason}")]) from None
    except (ValueError, RecursionError) as exc:
        raise InputError("config", [issue("invalid_json", f"config is not valid JSON: {exc}")]) from None
    return parse_config(data, vocab)


# --------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    gold_term_id: str
    cohort: str
    hit_rank: int | None  # 1-based rank of the answer that maps to gold; None if none does

    def hit(self, k: int) -> int:
        return 1 if self.hit_rank is not None and self.hit_rank <= k else 0


def hit_rank(predictions: Sequence[str], gold_term_id: str, vocab: Vocabulary) -> int | None:
    """Rank of the first answer that maps unambiguously to the gold term, else None."""
    for rank, answer in enumerate(predictions, start=1):
        if vocab.term_id_for(answer) == gold_term_id:
            return rank
    return None


def evaluate_cases(
    rows: Mapping[str, Sequence[str]], reference: Reference, vocab: Vocabulary
) -> list[CaseOutcome]:
    """One outcome per reference case, in reference order."""
    return [
        CaseOutcome(case.case_id, case.gold_term_id, case.cohort, hit_rank(rows[case.case_id], case.gold_term_id, vocab))
        for case in reference.cases
    ]


def round_value(value: Fraction | None) -> float | None:
    return None if value is None else float(round(value, DECIMALS))


def _empty_reason(spec: MetricSpec) -> str:
    cohort = f" with cohort {spec.cohort!r}" if spec.cohort is not None else ""
    if spec.diagnoses is not None:
        n = len(spec.diagnoses)
        return f"none of the {n} diagnos{'is' if n == 1 else 'es'} in the set has a reference case{cohort}"
    return f"no reference case{cohort}"


def compute_metric(spec: MetricSpec, outcomes: Sequence[CaseOutcome]) -> dict[str, Any]:
    """Exact value of one metric over the reference cases; see the module docstring."""
    cases = [
        o
        for o in outcomes
        if (spec.cohort is None or o.cohort == spec.cohort)
        and (spec.diagnoses is None or o.gold_term_id in spec.diagnoses)
    ]
    by_diagnosis: dict[str, list[CaseOutcome]] = {}
    for o in cases:
        by_diagnosis.setdefault(o.gold_term_id, []).append(o)

    value: Fraction | None = None
    reason: str | None = None
    if not cases:
        reason = _empty_reason(spec)
    elif spec.kind == "acc":
        value = Fraction(sum(o.hit(spec.k) for o in cases), len(cases))
    else:
        recalls = [
            Fraction(sum(o.hit(spec.k) for o in group), len(group))
            for _, group in sorted(by_diagnosis.items())
        ]
        value = sum(recalls, Fraction(0)) / len(recalls)

    result = spec.describe()
    result.update(
        {
            "n_cases": len(cases),
            "n_diagnoses": len(by_diagnosis),
            "reason": reason,
            "value": round_value(value),
        }
    )
    return result


# --------------------------------------------------------------------------- result


def _per_case(outcomes: Sequence[CaseOutcome], rows: Mapping[str, Sequence[str]], vocab: Vocabulary) -> list[dict[str, Any]]:
    details = []
    for o in sorted(outcomes, key=lambda o: o.case_id):
        answers = []
        for rank, text in enumerate(rows[o.case_id], start=1):
            resolution = vocab.resolve(text)
            answers.append(
                {
                    "normalized": resolution.normalized,
                    "rank": rank,
                    "status": resolution.status,
                    "term_id": resolution.term_id,
                    "text": text,
                }
            )
        entry: dict[str, Any] = {
            "case_id": o.case_id,
            "cohort": o.cohort,
            "gold_term_id": o.gold_term_id,
            "hit_rank": o.hit_rank,
            "predictions": answers,
        }
        for k in range(1, MAX_K + 1):
            entry[f"hit@{k}"] = o.hit(k)
        details.append(entry)
    return details


def score_predictions(
    data: bytes,
    reference: Reference,
    vocab: Vocabulary,
    config: ChallengeConfig,
    *,
    per_case: bool = False,
) -> dict[str, Any]:
    """Validate predictions bytes against the reference, then score them.

    Raises SubmissionError when the predictions are invalid (nothing is scored)
    and InputError when the reference does not belong to the config.
    """
    if (config.vocabulary_id, config.vocabulary_version) != (vocab.id, vocab.version):
        raise InputError(
            "config",
            [
                issue(
                    "vocabulary_mismatch",
                    f"config expects vocabulary {config.vocabulary_id!r} version {config.vocabulary_version!r} "
                    f"but the loaded vocabulary is {vocab.id!r} version {vocab.version!r}",
                )
            ],
        )
    if reference.challenge_version != config.challenge_version:
        raise InputError(
            "reference",
            [
                issue(
                    "challenge_version_mismatch",
                    f"reference rows are for challenge_version {reference.challenge_version!r} but the "
                    f"config is for {config.challenge_version!r}",
                )
            ],
        )
    report = validate_predictions(data, reference.case_ids, vocab)
    report.raise_if_invalid()
    outcomes = evaluate_cases(report.rows, reference, vocab)
    result: dict[str, Any] = {
        "challenge_id": config.challenge_id,
        "challenge_version": config.challenge_version,
        "n_cases": len(outcomes),
        "predictions_sha256": report.sha256,
        "primary": compute_metric(config.primary, outcomes),
        "scorer_version": __version__,
        "secondary": [compute_metric(spec, outcomes) for spec in config.secondary],
        "vocabulary": vocab.summary(),
    }
    if per_case:
        result["per_case"] = _per_case(outcomes, report.rows, vocab)
    return result


def score_files(
    predictions: str | Path,
    reference: str | Path,
    config: str | Path,
    vocab: str | Path | None = None,
    *,
    per_case: bool = False,
) -> dict[str, Any]:
    """Load every input from disk and score; ``vocab=None`` uses the repository vocabulary."""
    vocabulary = load_vocabulary(vocab)
    challenge = load_config(config, vocabulary)
    ref = load_reference(reference, vocabulary)
    return score_predictions(read_input(predictions, "predictions"), ref, vocabulary, challenge, per_case=per_case)


def dumps(result: Any) -> str:
    """Canonical JSON text: sorted keys, 2-space indent, ASCII only, trailing newline."""
    return json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
