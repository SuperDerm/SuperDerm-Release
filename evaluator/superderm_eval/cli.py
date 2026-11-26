"""Command line: ``python3 -m superderm_eval {validate,score,vocab-check}``.

Every command prints exactly one JSON document on stdout (sorted keys, ASCII,
"\\n" line endings on every platform). Exit codes:

    0  success: file valid / scored / vocabulary loaded
    1  the predictions file is invalid; stdout holds the row-level error report
    2  anything that is not the participant's fault: bad arguments, unreadable
       file, invalid vocabulary, config, reference or case list, a version
       mismatch, or an internal error (traceback on stderr)
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Sequence

from . import __version__
from .score import dumps, load_config, score_predictions
from .validate import load_case_list, load_reference, validate_predictions
from .vocab import DEFAULT_VOCAB_PATH, InputError, SubmissionError, Vocabulary, issue, load_vocabulary, read_input

EXIT_OK = 0
EXIT_INVALID_SUBMISSION = 1
EXIT_OTHER_ERROR = 2


def emit(obj: Any) -> None:
    text = dumps(obj)
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:  # e.g. stdout redirected to a StringIO in tests
        sys.stdout.write(text)
        return
    sys.stdout.flush()
    buffer.write(text.encode("ascii"))  # bytes: no newline translation on Windows
    buffer.flush()


def _error_report(source: str, code: str, message: str) -> dict[str, Any]:
    return InputError(source, [issue(code, message)]).to_report()


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # JSON on stdout instead of text on stderr
        emit(_error_report("arguments", "usage", f"{self.prog}: {message}"))
        self.exit(EXIT_OTHER_ERROR)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="superderm_eval",
        description="Validate and score SuperDerm community-challenge submissions.",
    )
    parser.add_argument("--version", action="version", version=f"superderm-eval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{validate,score,vocab-check}")
    vocab_help = f"vocabulary JSON (default: {DEFAULT_VOCAB_PATH})"

    validate = commands.add_parser("validate", help="check a predictions file without scoring it")
    validate.add_argument("--predictions", required=True, help="predictions JSONL")
    expected = validate.add_mutually_exclusive_group(required=True)
    expected.add_argument("--cases", help="case list, one case_id per line (participants)")
    expected.add_argument("--reference", help="reference CSV (maintainers)")
    validate.add_argument("--vocab", help=vocab_help)

    score = commands.add_parser("score", help="validate, then score a predictions file")
    score.add_argument("--predictions", required=True, help="predictions JSONL")
    score.add_argument("--reference", required=True, help="reference CSV")
    score.add_argument("--config", required=True, help="challenge config JSON")
    score.add_argument("--vocab", help=vocab_help)
    score.add_argument(
        "--per-case",
        action="store_true",
        help="include per-case details (reveals gold labels: maintainers only)",
    )

    check = commands.add_parser("vocab-check", help="load the vocabulary and list ambiguous names")
    check.add_argument("--vocab", help=vocab_help)
    return parser


def _warn_ambiguous(vocab: Vocabulary) -> None:
    if vocab.ambiguous:
        print(
            f"superderm_eval: warning: {len(vocab.ambiguous)} vocabulary name(s) map to more than one "
            "term and score zero; run vocab-check for the list",
            file=sys.stderr,
        )


def _run(args: argparse.Namespace) -> int:
    vocab = load_vocabulary(args.vocab)
    if args.command == "vocab-check":
        emit(vocab.check_report())
        return EXIT_OK
    _warn_ambiguous(vocab)
    if args.command == "validate":
        if args.cases is not None:
            expected = load_case_list(args.cases)
        else:
            expected = load_reference(args.reference, vocab).case_ids
        report = validate_predictions(read_input(args.predictions, "predictions"), expected, vocab)
        emit(report.to_report())
        return EXIT_OK if report.ok else EXIT_INVALID_SUBMISSION
    config = load_config(args.config, vocab)
    reference = load_reference(args.reference, vocab)
    data = read_input(args.predictions, "predictions")
    emit(score_predictions(data, reference, vocab, config, per_case=args.per_case))
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except SubmissionError as exc:
        emit(exc.to_report())
        return EXIT_INVALID_SUBMISSION
    except InputError as exc:
        emit(exc.to_report())
        return EXIT_OTHER_ERROR
    except Exception as exc:  # never let a crash look like exit 1 ("invalid submission")
        traceback.print_exc()
        emit(_error_report("internal", "internal_error", f"{type(exc).__name__}: {exc}"))
        return EXIT_OTHER_ERROR
