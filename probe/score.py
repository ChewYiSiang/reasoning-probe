"""Scoring predicted profiles against the traced ground truth.

Reads the profiles logged by the model wrapper and the ground truth written by
`build_truth_mucoco`, matches them by program and input, and reports how often the model
got the control flow right.

Counts are reported separately for constructs the input exercised and constructs it did
not. Most CRUXEval branches never run one of their sides, so a model that answers 0
everywhere would otherwise look accurate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_truth(path: str) -> dict[tuple[str, str], dict]:
    """{(task id, version): {construct id: counts}} plus the code, for matching.

    A branch is recorded as (taken, not taken), where "not taken" means the condition was
    false. The trace stores how often the else branch ran, which is the same number when
    there is an else and always zero when there is not; counting the false outcomes
    instead is well defined either way, and is what the question asks for.
    """
    truth: dict[tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version_name, version in row["versions"].items():
                if "profile" not in version:
                    continue
                counts = {key: (value,) for key, value in version["profile"]["loops"].items()}
                counts.update(branch_counts(version))
                truth[(row["task_id"], version_name)] = {
                    "counts": counts,
                    "code": version.get("code"),
                }
    return truth


def branch_counts(version: dict) -> dict[str, tuple[int, int]]:
    """Per branch: (times the condition was true, times it was false)."""
    profile = version["profile"]
    statements = {int(line): count
                  for line, count in profile.get("statement_counts", {}).items()}
    headers = {c["id"]: c["header_line"] for c in version.get("constructs", [])}

    counts: dict[str, tuple[int, int]] = {}
    for name, (taken, else_taken) in profile["branches"].items():
        header = headers.get(name)
        if header is None or not statements:
            counts[name] = (taken, else_taken)      # older truth file: fall back
            continue
        evaluated = statements.get(header, taken + else_taken)
        counts[name] = (taken, max(0, evaluated - taken))
    return counts


def load_predictions(path: str) -> list[dict]:
    """Read a profile log, re-parsing any reply whose profile came out empty.

    Replies are stored in full, so a parser improvement can be applied to logs that were
    written before it without re-running anything.
    """
    from probe.parse import parse_profile, split_sections

    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("profile") and row.get("raw_reply"):
                _, profile_text = split_sections(row["raw_reply"])
                row["profile"] = {key: list(value)
                                  for key, value in parse_profile(profile_text).items()}
            rows.append(row)
    return rows


def score(truth: dict, predictions: list[dict], code_index: dict[str, tuple[str, str]]) -> Counter:
    """Compare every logged prediction with the traced counts for the same version."""
    stats = Counter()
    for prediction in predictions:
        key = code_index.get(prediction["full_sol"])
        if key is None:
            stats["prediction with no matching ground truth"] += 1
            continue

        real = truth[key]["counts"]
        claimed = {name: tuple(value) for name, value in prediction["profile"].items()}
        stats["replies scored"] += 1
        if not real:
            # Some programs are straight-line code: nothing to count, nothing to score.
            stats["programs with no loop or branch"] += 1
            continue
        if not claimed:
            stats["replies with no readable profile"] += 1
            continue

        correct_here = True
        for name, real_counts in real.items():
            claim = claimed.get(name)
            exercised = any(count > 0 for count in real_counts)
            bucket = "exercised" if exercised else "never ran"
            if claim is None:
                stats[f"{bucket}: not answered"] += 1
                correct_here = False
            elif claim == real_counts:
                stats[f"{bucket}: correct"] += 1
            else:
                stats[f"{bucket}: wrong"] += 1
                correct_here = False

        stats["profiles entirely correct" if correct_here else "profiles with an error"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default="results/truth/cruxeval.jsonl")
    parser.add_argument("--predictions", default="results/probe/profiles.jsonl")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    # The model log holds the program text, not the task id, so match on the text.
    code_index = {value["code"]: key for key, value in truth.items() if value["code"]}
    if not code_index:
        raise SystemExit(
            f"{args.truth} has no program text in it, so predictions cannot be matched.\n"
            "It was built by an older version of build_truth_mucoco; rebuild it (a few seconds)."
        )
    predictions = load_predictions(args.predictions)

    stats = score(truth, predictions, code_index)
    width = max(len(key) for key in stats) if stats else 10
    for key, value in sorted(stats.items()):
        print(f"{key.ljust(width)}  {value}")


if __name__ == "__main__":
    main()
