"""Reproducible scorer for the SuperDerm community challenges.

Modules:
    vocab        diagnosis vocabulary, name normalization and lookup
    validate     predictions / reference / case-list validation
    score        challenge config, metrics and the result JSON
    leaderboard  ranking of scored runs
    cli          ``python3 -m superderm_eval {validate,score,vocab-check}``
"""

__version__ = "1.0.0"  # scorer version; recorded in every result

__all__ = ["__version__"]
