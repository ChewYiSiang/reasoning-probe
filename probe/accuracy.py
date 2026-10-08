"""Pooled reasoning accuracy: the loop, branch and variable questions scored together.

The two fields are reported separately everywhere else, which is right for diagnosis but
awkward as a headline, where a reader wants one number for how much of the run the model
reported correctly. Pooling needs care: the fields have different denominators and are
asked at different rates, so this counts items rather than averaging two percentages.

Predictions are matched to the ground truth by program text, the same route the other
scorers use, so a mutant is scored against its own traced counts rather than the original's.

    python -m probe.accuracy --truth cruxeval_all.jsonl --writes writes_all.jsonl \
        --probe-dir results/qwen14b_both --versions no_mutation sequential
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe.dataflow import parse_writes
from probe.score import load_predictions, load_truth
from probe.stats import rate


def pooled(truth_path: str, writes_path: str, probe_dir: str, versions: list[str]) -> None:
    # the same loader the other scorers use, so the counts and the matching agree with them
    truth = load_truth(truth_path)
    by_code = {entry["code"]: key for key, entry in truth.items() if entry.get("code")}

    wanted: dict[tuple[str, str], dict] = {}
    with open(writes_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                if entry["asked"]:
                    wanted[(row["task_id"], version)] = entry

    totals = dict.fromkeys(
        ("construct asked", "construct right", "variable asked", "variable right"), 0)

    for version in versions:
        log = Path(probe_dir) / f"profiles_{version}.jsonl"
        if not log.exists():
            print(f"  no log for {version}, skipped")
            continue
        for prediction in load_predictions(str(log)):
            key = by_code.get(prediction.get("full_sol", ""))
            if key is None:
                continue

            claimed = {name: tuple(value)
                       for name, value in (prediction.get("profile") or {}).items()}
            for name, counts in (truth[key].get("counts") or {}).items():
                totals["construct asked"] += 1
                totals["construct right"] += claimed.get(name) == tuple(counts)

            entry = wanted.get(key)
            if entry:
                said = parse_writes(prediction.get("raw_reply") or "")
                for variable in entry["asked"]:
                    totals["variable asked"] += 1
                    totals["variable right"] += (
                        said.get(variable) == entry["informative"].get(variable))

    asked = totals["construct asked"] + totals["variable asked"]
    right = totals["construct right"] + totals["variable right"]
    print(f"\n=== pooled over {', '.join(versions)} ===")
    print(f"  loops and branches  {rate(totals['construct right'], totals['construct asked'])}")
    print(f"  variables           {rate(totals['variable right'], totals['variable asked'])}")
    print(f"  reasoning accuracy  {rate(right, asked)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--writes", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--versions", nargs="+", default=["no_mutation"])
    args = parser.parse_args()
    pooled(args.truth, args.writes, args.probe_dir, args.versions)


if __name__ == "__main__":
    main()
