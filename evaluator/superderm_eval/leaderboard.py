"""Ranking of scored runs.

Runs are ranked on the published (6-decimal) value of one metric, highest
first, with standard competition ranking: runs with equal values share a rank
and the next rank skips accordingly (1, 1, 3). Ties are listed by run ID.
Runs whose metric is null cannot be compared; they get rank None and come last.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


def metric_entry(result: Mapping[str, Any], metric: str = "primary") -> Mapping[str, Any]:
    """The metric object named ``metric`` in a score result.

    ``metric`` is "primary" or the ``label`` of a metric in the result, e.g.
    "acc@3" or "acc@1[cohort=common_mimic]".
    """
    if metric == "primary":
        return result["primary"]
    entries = [result["primary"], *result.get("secondary", [])]
    for entry in entries:
        if entry.get("label") == metric:
            return entry
    available = ", ".join(["primary", *(str(e.get("label")) for e in entries)])
    raise KeyError(f"metric {metric!r} is not in the result (available: {available})")


def rank_runs(
    results: Mapping[Any, Mapping[str, Any]] | Iterable[tuple[Any, Mapping[str, Any]]],
    metric: str = "primary",
) -> list[dict[str, Any]]:
    """Rank runs on one metric.

    ``results`` maps run ID -> score result (the JSON printed by ``score``), or
    is an iterable of (run ID, result) pairs. Run IDs must be unique and
    mutually orderable (e.g. all strings). Returns, best first, one
    ``{"rank", "run_id", "value"}`` dict per run.

    Raises ValueError when runs come from different challenges or challenge
    versions, were scored with different definitions of the metric, or carry a
    value that is not a finite number or null.
    """
    pairs = list(results.items()) if isinstance(results, Mapping) else [tuple(p) for p in results]
    values: list[tuple[Any, float | None]] = []
    seen: set[Any] = set()
    challenges: set[tuple[Any, Any]] = set()
    labels: set[Any] = set()
    for run_id, result in pairs:
        if run_id in seen:
            raise ValueError(f"run {run_id!r} is listed more than once")
        seen.add(run_id)
        entry = metric_entry(result, metric)
        challenges.add((result.get("challenge_id"), result.get("challenge_version")))
        labels.add(entry.get("label"))
        value = entry.get("value")
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ValueError(f"run {run_id!r}: metric value must be a finite number or null, got {value!r}")
        values.append((run_id, value))
    if len(challenges) > 1:
        raise ValueError(f"runs come from different challenges/versions: {sorted(map(repr, challenges))}")
    if len(labels) > 1:
        raise ValueError(f"runs define the metric differently: {sorted(map(repr, labels))}")

    ranking: list[dict[str, Any]] = []
    scored = sorted((p for p in values if p[1] is not None), key=lambda p: (-p[1], p[0]))
    rank, previous = 0, None
    for position, (run_id, value) in enumerate(scored, start=1):
        if value != previous:
            rank, previous = position, value
        ranking.append({"rank": rank, "run_id": run_id, "value": value})
    for run_id, _ in sorted((p for p in values if p[1] is None), key=lambda p: p[0]):
        ranking.append({"rank": None, "run_id": run_id, "value": None})
    return ranking
