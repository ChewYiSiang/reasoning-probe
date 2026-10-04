"""Are the branch "errors" real, or are we asking an ambiguous question?

For an `if` with no `else`, the ground truth records else = 0, because there is no else
branch to execute. A model may reasonably read "else = ?" as "how many times was the
condition false", which is a different number.

This checks that directly: for every branch the model answered, it compares the claimed
else count with both readings.

    python -m probe.branch_audit --truth results/truth/cruxeval.jsonl \
        --probe-log results/probe/profiles_no_mutation.jsonl --version no_mutation
"""

from __future__ import annotations

import argparse
from collections import Counter

from probe.score import load_predictions, load_truth
from probe.stats import rate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--probe-log", required=True)
    parser.add_argument("--version", default="no_mutation")
    args = parser.parse_args()

    # The full rows are needed here, not just the counts: statement and arc counts say
    # how often each condition was evaluated.
    import json
    rows = {}
    with open(args.truth, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            version = row["versions"].get(args.version)
            if version and "profile" in version:
                rows[version["code"]] = version

    predictions = load_predictions(args.probe_log)
    stats = Counter()

    for prediction in predictions:
        version = rows.get(prediction["full_sol"])
        if version is None:
            continue
        constructs = {c["id"]: c for c in version["constructs"]}
        profile = version["profile"]
        statements = {int(line): count for line, count in profile["statement_counts"].items()}
        claimed = {name: tuple(value) for name, value in prediction["profile"].items()}

        for name, counts in profile["branches"].items():
            construct = constructs.get(name)
            claim = claimed.get(name)
            if construct is None or claim is None or len(claim) != 2:
                continue

            taken, else_taken = counts
            evaluated = statements.get(construct["header_line"], 0)
            condition_false = max(0, evaluated - taken)
            has_else = construct["else_line"] is not None
            group = "with else" if has_else else "without else"

            stats[f"{group}: branches answered"] += 1
            if claim == (taken, else_taken):
                stats[f"{group}: matches the trace"] += 1
            elif claim == (taken, condition_false):
                # Right about the branch, but counting "condition was false" rather than
                # "the else branch ran".
                stats[f"{group}: taken right, else read as condition-false"] += 1
            elif claim[0] == taken:
                stats[f"{group}: taken right, else neither reading"] += 1
            else:
                stats[f"{group}: taken wrong"] += 1

    width = max(len(key) for key in stats) if stats else 10
    for key in sorted(stats):
        print(f"{key.ljust(width)}  {stats[key]}")

    for group in ("with else", "without else"):
        answered = stats[f"{group}: branches answered"]
        if not answered:
            continue
        strict = stats[f"{group}: matches the trace"]
        lenient = strict + stats[f"{group}: taken right, else read as condition-false"]
        print(f"\n{group}: strict {rate(strict, answered)}")
        print(f"{group}: allowing the condition-false reading {rate(lenient, answered)}")


if __name__ == "__main__":
    main()
