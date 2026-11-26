# superderm-eval

The reproducible scorer for the SuperDerm community challenges (open-ended
dermatology diagnosis). Participants submit up to three ranked, free-text
diagnoses per test case; this package validates the submission file and
scores it against a hidden reference. The same code is published so that
anyone holding the same inputs can reproduce a published score byte for byte.

> **For research discussion only.** Scores are research measurements, not evidence of clinical safety or performance. Model
> outputs scored here are not medical advice and do not constitute a medical diagnosis.

Python 3.10 or newer, standard library only, no dependencies.

```sh
cd evaluator
python3 -m superderm_eval validate --predictions fixtures/dev/predictions.jsonl --cases fixtures/dev/cases.txt
python3 -m superderm_eval score --predictions fixtures/dev/predictions.jsonl \
    --reference fixtures/dev/reference.csv --config fixtures/dev/config.json
python3 -m superderm_eval vocab-check
python3 -m unittest discover -s tests -v
```

`--vocab` defaults to the repository's taxonomy: `../taxonomy/diagnoses.v1.json` in SuperDerm-Release, or
`../content/vocab/diagnoses.v1.json` in the platform repository. Pass `--vocab PATH` when the package runs from
anywhere else; `pyproject.toml` also declares a `superderm-eval` console script
for installed use.

## File formats

### Predictions (participants)

UTF-8 JSON Lines, one object per test case:

```json
{"case_id":"s_demo_001","predictions":["Mpox","Varicella","Herpes Zoster"]}
{"case_id":"s_demo_005","predictions":[]}
```

- `predictions` holds 0 to 3 strings, most likely first. An empty array is an
  explicit abstention: it scores zero and still counts in every denominator.
- Each row has exactly the keys `case_id` and `predictions`, and every expected
  case has exactly one row. Row order does not matter.
- A leading UTF-8 byte-order mark is removed. Lines that are empty after
  removing whitespace are ignored. Lines end with `\n` (`\r\n` also works).
  Line numbers in error reports are physical line numbers, counting ignored
  lines.
- `case_id` is compared exactly (no case folding or trimming).

### Case list (participants)

Plain UTF-8 text, one `case_id` per line. It lets participants validate a file
without the answers. Surrounding whitespace and empty lines are ignored; a
repeated ID is an error. There is no header line.

### Reference (maintainers; hidden in production)

UTF-8 CSV with exactly this header:

```
case_id,gold_term_id,cohort,challenge_version
s_demo_001,DX-070,rare_high_consequence,dev-1
```

One row per case. `gold_term_id` must be a term ID of the vocabulary.
`cohort` is a free label (it may be empty). Every row carries the same
`challenge_version`, which must equal the config's. Fields may not have
surrounding whitespace; a leading byte-order mark and blank lines are allowed.

### Challenge config (maintainers)

```json
{"challenge_id": "rare-and-mimics", "challenge_version": "dev-1", "scorer_version": "1.0.0",
 "vocabulary": {"id": "superderm-diagnoses", "version": "1.0.0"},
 "primary": {"metric": "macro_recall@3", "diagnoses": ["DX-070", "DX-074"]},
 "secondary": [{"metric": "acc@1"}, {"metric": "acc@3"},
               {"metric": "acc@1", "cohort": "common_mimic"}, {"metric": "acc@3", "cohort": "common_mimic"}]}
```

- `scorer_version` must equal this package's version, and `vocabulary.id` /
  `vocabulary.version` must equal the loaded vocabulary file's. A mismatch is
  an error, not a warning.
- A metric object has `metric` (`acc@k` or `macro_recall@k`, k = 1, 2 or 3)
  and optionally `cohort` (keep only reference rows with that label) and
  `diagnoses` (keep only reference rows whose gold is in the set).
  `macro_recall@k` requires `diagnoses`.
- `diagnoses` is a list of term IDs or the string `"rare_and_high_consequence"`:
  every term whose `rare` and `highConsequence` flags are both true (70 terms
  in vocabulary 1.0.0).
- `secondary` is optional. Unknown keys are errors, so a typo cannot silently
  change a metric. Two metrics with the same definition are an error.

## Matching answers to diagnoses

Each answer, and every vocabulary name and synonym, is normalized the same way:

1. Unicode NFKC (full-width letters, ligatures, no-break spaces and other
   compatibility characters become their plain forms);
2. case folding (`str.casefold`, stronger than lower-casing);
3. leading and trailing whitespace removed;
4. each internal run of whitespace (any character for which `str.isspace()`
   is true) replaced by one space.

Nothing else is changed: punctuation, hyphens and parentheses are kept, and
there is no spelling correction or fuzzy matching. `"  MPOX "`, `"ＭＰＯＸ"`
and `"monkeypox"` all match DX-070; `"M-pox"` does not.

The lookup maps each normalized canonical name and synonym to its term ID. A
normalized string that belongs to more than one term is **ambiguous** and maps
to nothing, even if one of its terms is the gold: it scores zero. Canonical
names get no priority over synonyms. `vocab-check` lists every ambiguous
string (vocabulary 1.0.0 has none). Unknown, unmapped and ambiguous answers
are not validation errors; they score zero.

Normalization uses the Unicode database of the running Python (13.0 in
Python 3.10, 15.1 in 3.13). This only matters for characters assigned after
Unicode 13.0, which no vocabulary name uses.

## Validation

A predictions file is either valid or rejected; an invalid file is never
scored. Every problem is reported, each with a stable `code`, the 1-based
`line` (null for whole-file problems), a `message` and, where relevant,
`case_id`, `position` or `positions` (1-based ranks).

| code | meaning |
|---|---|
| `invalid_utf8` | the line is not valid UTF-8 |
| `invalid_json` | the line is not valid JSON (`NaN` and `Infinity` are rejected) |
| `duplicate_key` | a key appears twice in one object |
| `row_not_object` | the line is valid JSON but not an object |
| `invalid_keys` | keys are not exactly `case_id` and `predictions` (`missing_keys`, `unexpected_keys`) |
| `case_id_not_string` | `case_id` is not a string |
| `unknown_case_id` | `case_id` is not in the expected case list |
| `duplicate_case_id` | `case_id` already appeared (`first_line`) |
| `missing_case_ids` | expected cases without a row (`case_ids`, sorted) |
| `predictions_not_array` | `predictions` is not an array |
| `too_many_predictions` | more than three entries (`count`) |
| `prediction_not_string` | an entry is not a string (`position`) |
| `blank_prediction` | an entry is empty after normalization (`position`) |
| `duplicate_prediction` | two entries are the same after normalization (`kind: "same_text"`) or map to the same term (`kind: "same_term"`, e.g. "Smallpox" and "Variola", both DX-074); `positions` names both ranks |

Error report (stdout, exit code 1):

```json
{"ok": false, "input": "predictions", "error_count": 1,
 "errors": [{"case_id": "s_demo_001", "code": "duplicate_prediction", "kind": "same_term", "line": 1,
             "message": "predictions 1 ('Smallpox') and 2 ('Variola') both map to DX-074 (Smallpox (Variola))",
             "positions": [1, 2], "term_id": "DX-074"}]}
```

The other inputs use the same report shape with `input` set to `vocabulary`,
`config`, `reference`, `cases`, `arguments` or `internal`. Their codes include
`unreadable_file`, `invalid_header`, `wrong_field_count`, `field_whitespace`,
`empty_field`, `unknown_gold_term_id`, `mixed_challenge_version`,
`challenge_version_mismatch`, `vocabulary_mismatch`, `scorer_version_mismatch`,
`unknown_key`, `missing_key`, `invalid_metric`, `missing_diagnoses`,
`unknown_term_id` and `duplicate_metric`.

## Metrics

For a set U of reference cases and k = 1, 2 or 3:

- `m(answer, gold) = 1` if the answer maps unambiguously to `gold_term_id`, else 0.
- `hit@k(case)` = the maximum of `m` over the case's first k answers. A case
  scores at most one hit at each k (duplicates are rejected anyway).
- `acc@k(U) = (1/|U|) * sum of hit@k over U`. |U| is the fixed number of
  reference cases in U; abstentions stay in it.
- `macro_recall@k(S)` = the mean, over diagnoses c in S that have at least one
  reference case, of acc@k over the cases whose gold is c. Diagnoses of S
  without a case are left out of the average. `n_cases` and `n_diagnoses`
  report what was used; `n_diagnoses_in_set` reports |S|.
- `acc@k` with `diagnoses` is the micro average over the same cases. For three
  mpox cases all found and one smallpox case missed, macro recall is
  (1 + 0) / 2 = 0.5 and micro accuracy is 3/4 = 0.75.
- If no reference case is left after the `cohort` / `diagnoses` filters, the
  value is `null` and `reason` says why; nothing is divided by zero.

Arithmetic is exact (Python fractions). Each value is rounded once, to 6
decimals (half-to-even on the exact value), and emitted as a JSON number.

## Output

`score` prints one JSON object with sorted keys, 2-space indentation, ASCII
only (other characters are `\u` escaped) and `\n` line endings on every
platform:

```json
{
  "challenge_id": "rare-and-mimics",
  "challenge_version": "dev-1",
  "n_cases": 5,
  "predictions_sha256": "<sha256 of the predictions file bytes, as received>",
  "primary": {"cohort": null, "diagnoses": ["DX-070", "DX-074"],
              "label": "macro_recall@3[diagnoses=DX-070,DX-074]", "metric": "macro_recall@3",
              "n_cases": 2, "n_diagnoses": 2, "n_diagnoses_in_set": 2, "reason": null, "value": 1.0},
  "scorer_version": "1.0.0",
  "secondary": [{"cohort": null, "diagnoses": null, "label": "acc@1", "metric": "acc@1",
                 "n_cases": 5, "n_diagnoses": 5, "n_diagnoses_in_set": null, "reason": null, "value": 0.4},
                "..."],
  "vocabulary": {"content_sha256": "<sha256>", "id": "superderm-diagnoses",
                 "synonyms_status": "draft", "version": "1.0.0"}
}
```

- `label` names a metric uniquely within a config: the metric, then any
  filters, e.g. `acc@1[cohort=common_mimic]` or
  `macro_recall@3[diagnoses=rare_and_high_consequence]`.
- `vocabulary.content_sha256` is the SHA-256 of the vocabulary's canonical JSON
  (sorted keys, no whitespace, ASCII escapes). It pins the exact names and
  synonyms used, and does not change with indentation or line endings, so it
  differs from `shasum` of the file.
- `--per-case` adds `per_case`: for each case (sorted by `case_id`) its
  `cohort`, `gold_term_id`, `hit_rank` (or null), `hit@1`..`hit@3`, and each
  answer's `text`, `normalized` form, `status` (`mapped`, `ambiguous`,
  `unknown`) and `term_id`. It reveals the gold labels, so it is for
  maintainers only and is never on by default.

## Command line

```
python3 -m superderm_eval validate --predictions P (--cases LIST | --reference REF) [--vocab V]
python3 -m superderm_eval score    --predictions P --reference REF --config C [--vocab V] [--per-case]
python3 -m superderm_eval vocab-check [--vocab V]
```

`score` validates first. Every command prints exactly one JSON document on
stdout, including on failure (argument errors too), so a worker can always
parse it. Warnings, if any, go to stderr.

| exit | meaning |
|---|---|
| 0 | valid / scored / vocabulary loaded |
| 1 | the predictions file is invalid; stdout holds the error report for the participant |
| 2 | anything that is not the participant's fault: bad arguments, unreadable file, invalid vocabulary, config, reference or case list, a version mismatch, or an internal error (traceback on stderr) |

## Leaderboard ranking

`superderm_eval.leaderboard.rank_runs(results, metric="primary")` takes a
mapping of run ID to score result (or a list of `(run_id, result)` pairs) and
returns `[{"rank", "run_id", "value"}, ...]`, best first. `metric` is
`"primary"` or a metric `label`. Ranking uses the published 6-decimal values
and standard competition ranking: equal values share a rank and the next rank
skips (1, 1, 3). Tied runs are listed by run ID. Runs with a null value get
rank `null` and come last. Runs from different challenges or versions, or with
different metric definitions, raise `ValueError`.

## Reproducing a published score

A published result names everything that determines it: `scorer_version`,
`challenge_id` and `challenge_version`, the vocabulary `id`, `version` and
`content_sha256`, and `predictions_sha256`. With the same scorer version,
vocabulary, config, reference and predictions bytes, `score` prints
byte-identical output: the output is written as bytes (no newline
translation), arithmetic is exact, and nothing depends on row order,
dictionary order or hash seeds (the tests compare runs under different hash
seeds). The only Python-version dependence is the Unicode note under
"Matching answers to diagnoses".

## Development fixture and tests

`fixtures/dev/` is synthetic: case IDs are fictitious (`s_demo_001` ...);
term IDs are real vocabulary IDs.

| case | gold | answers | hit@1 | hit@3 |
|---|---|---|---|---|
| s_demo_001 | DX-070 Mpox | Mpox, Varicella, Herpes Zoster | 1 | 1 |
| s_demo_002 | DX-074 Smallpox | Varicella, Mpox, Smallpox | 0 | 1 |
| s_demo_003 | DX-111 Arthropod bite | insect bite (synonym) | 1 | 1 |
| s_demo_004 | DX-076 Varicella | Not A Real Diagnosis | 0 | 0 |
| s_demo_005 | DX-067 Herpes zoster | (abstains) | 0 | 0 |

acc@1 = 2/5 = 0.4, acc@3 = 3/5 = 0.6, and the primary macro_recall@3 over
{DX-070, DX-074} is 1.0. `predictions_invalid.jsonl` shows one example of most
error codes. Run the tests with `python3 -m unittest discover -s tests -v`.

## How this differs from the paper's protocol

Both protocols score **free-text** differentials with the same metric definitions: top-k accuracy over all cases,
and macro top-k recall averaged over diagnoses, with a missing answer scoring zero. They differ in how an answer is
matched to the documented diagnosis.

**Preprint protocol** (SuperDerm preprint, Sec. 3.2 "Scoring Free-Text Answers"):

- an exact name or a synonym from the preprint dictionary, which maps 21,026 source names to the 214 diagnoses, scores 1;
- when the case is labelled NOS, a more specific diagnosis in the same subgroup is also accepted;
- an answer that maps to another diagnosis scores 0;
- an unresolved answer is classified by an LLM judge (Qwen3.8-27B, given the subgroup and category) as equivalent,
  subtype, parent, sibling or different; equivalents and subtypes are accepted;
- each case is also asked as a paired four-option question (MCQ) with the same inputs.

**Community protocol** (this package):

1. **No LLM judge.** Names are matched deterministically after normalization; unmapped or ambiguous names score 0.
   Anyone can reproduce a score offline with the published files.
2. **Smaller synonym list.** Matching uses the canonical names plus the platform's synonyms in
   `content/vocab/diagnoses.v1.json` (63, marked draft until clinically approved), not the 21,026-name preprint
   dictionary. Scores therefore depend on the vocabulary version, which every result records with a content hash.
3. **No NOS-subtype credit.** Naming a more specific diagnosis than an NOS reference scores 0.
4. **Strict files.** Two names of the same term in one list, more than three answers, or unknown, repeated or missing
   case IDs reject the whole file instead of being scored.
5. **Explicit abstention.** An empty list is allowed and counts as a miss; it stays in the denominator.
6. **Open-ended only.** Challenges do not include the paired four-option question.
7. **Cohorts are challenge labels** assigned per case in the reference file (e.g. `common_mimic`), and macro recall
   covers only diagnoses present in the challenge's reference set (`n_diagnoses` versus `n_diagnoses_in_set`).
8. **Point estimates only.** No confidence intervals; leaderboard ties are exact ties of the published values.

A challenge score is therefore not directly comparable with a number in the preprint.
