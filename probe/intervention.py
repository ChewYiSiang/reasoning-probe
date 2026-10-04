"""Preparing and scoring the execution-facts intervention.

Two jobs:

  `prepare`   turns the traced ground truth into the facts block for every task, in the
              three flavours, and marks which tasks would have the answer handed to them.
  `compare`   scores one intervention run against the baseline run, task by task.

The primary comparison is deliberately narrow: tasks that have control flow, whose counts
do not give the answer away, judged by MuCoCo's own correctness oracle, paired with the
same task in the baseline run and tested with exact McNemar. Everything else is reported
alongside as secondary.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from probe.analyse import read_results_csv, verdict
from probe.facts import corrupt, leaks_answer, placebo, render
from probe.score import load_truth
from probe.stats import mcnemar, rate

FLAVOURS = ("true", "placebo", "corrupted")


def prepare(truth_path: str, version: str = "no_mutation") -> dict:
    """Build the facts blocks and the leak flags for every task of one version."""
    truth = load_truth(truth_path)
    rows: dict[str, dict] = {}

    with open(truth_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            entry = row["versions"].get(version)
            if not entry or "profile" not in entry:
                continue
            key = (row["task_id"], version)
            counts = truth[key]["counts"]
            if not counts:
                continue                      # straight-line program: nothing to supply

            expected = entry["profile"]["output"]
            try:
                expected_value = ast.literal_eval(expected)
            except Exception:
                expected_value = expected

            rows[row["task_id"]] = {
                "code": entry["code"],
                "leaks": leaks_answer(expected_value, counts),
                "true": render(counts, entry["constructs"]),
                "placebo": placebo(entry["code"]),
                "corrupted": render(corrupt(counts), entry["constructs"]),
            }
    return rows


def facts_by_code(prepared: dict, flavour: str) -> dict[str, str]:
    """The lookup the prompt wrapper needs: program text -> facts block."""
    return {row["code"]: row[flavour] for row in prepared.values()}


def compare(baseline_csv: str, intervention_csv: str, prepared: dict) -> None:
    """Paired comparison, reported for the eligible tasks and then for the rest."""
    baseline = read_results_csv(baseline_csv)
    intervention = read_results_csv(intervention_csv)

    groups = {"eligible (control flow, no leak)": Counter(),
              "leak-prone (reported separately)": Counter()}

    for task_id, row in prepared.items():
        first, second = baseline.get(task_id), intervention.get(task_id)
        if first is None or second is None:
            continue
        before, after = verdict(first), verdict(second)
        if before not in ("correct", "incorrect") or after not in ("correct", "incorrect"):
            continue

        group = groups["leak-prone (reported separately)" if row["leaks"]
                       else "eligible (control flow, no leak)"]
        group["tasks"] += 1
        group["right before"] += before == "correct"
        group["right after"] += after == "correct"
        if before == "incorrect" and after == "correct":
            group["fixed"] += 1
        elif before == "correct" and after == "incorrect":
            group["broken"] += 1
        if before == "incorrect":
            group["wrong before"] += 1

    for name, stats in groups.items():
        if not stats["tasks"]:
            continue
        print(f"\n=== {name} ===")
        print(f"  tasks compared        {stats['tasks']}")
        print(f"  correct before        {rate(stats['right before'], stats['tasks'])}")
        print(f"  correct after         {rate(stats['right after'], stats['tasks'])}")
        print(f"  fixed by the facts    {stats['fixed']}")
        print(f"  broken by the facts   {stats['broken']}")
        if stats["wrong before"]:
            # The number a tool user cares about: of what it got wrong, how much is
            # recovered by being told how the program runs.
            print(f"  fix rate              {rate(stats['fixed'], stats['wrong before'])}")
        print(f"  exact McNemar p       {mcnemar(stats['broken'], stats['fixed']):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--version", default="no_mutation")
    parser.add_argument("--baseline-csv", help="the run without facts")
    parser.add_argument("--intervention-csv", help="the run with facts")
    parser.add_argument("--summary", action="store_true",
                        help="just describe the prepared tasks and exit")
    args = parser.parse_args()

    prepared = prepare(args.truth, args.version)
    leaks = sum(1 for row in prepared.values() if row["leaks"])
    print(f"tasks with control flow: {len(prepared)}")
    print(f"  leak-prone (excluded from the main analysis): {leaks}")
    print(f"  eligible: {len(prepared) - leaks}")

    if args.summary:
        return
    if not (args.baseline_csv and args.intervention_csv):
        parser.error("--baseline-csv and --intervention-csv are needed unless --summary")
    compare(args.baseline_csv, args.intervention_csv, prepared)


if __name__ == "__main__":
    main()
