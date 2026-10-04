"""Did asking for a profile change the answers?

The probe adds a section to MuCoCo's prompt, so their task is not quite the task they
measured. This compares the two runs task by task: the baseline model against the same
model with the profile question attached.

    python -m probe.contamination \
        --baseline results/output_prediction/deepseek-v4-flash \
        --probe    results/output_prediction/deepseek-v4-flash-probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

from probe.analyse import read_results_csv, verdict
from probe.stats import mcnemar, rate


def compare(baseline_csv: str, probe_csv: str) -> None:
    base = read_results_csv(baseline_csv)
    probe = read_results_csv(probe_csv)
    shared = sorted(set(base) & set(probe))

    counts = {"both right": 0, "both wrong": 0, "baseline only": 0, "probe only": 0,
              "not comparable": 0}
    changed_answer = 0

    for task_id in shared:
        first, second = verdict(base[task_id]), verdict(probe[task_id])
        if "skipped" in (first, second) or "invalid" in (first, second):
            counts["not comparable"] += 1
            continue
        if str(base[task_id].get("model_output", "")).strip() != \
           str(probe[task_id].get("model_output", "")).strip():
            changed_answer += 1
        if first == second == "correct":
            counts["both right"] += 1
        elif first == second == "incorrect":
            counts["both wrong"] += 1
        elif first == "correct":
            counts["baseline only"] += 1
        else:
            counts["probe only"] += 1

    comparable = sum(v for k, v in counts.items() if k != "not comparable")
    base_right = counts["both right"] + counts["baseline only"]
    probe_right = counts["both right"] + counts["probe only"]

    print(f"tasks in both runs: {len(shared)} | comparable: {comparable}")
    print(f"  baseline correct  {rate(base_right, comparable)}")
    print(f"  with profile      {rate(probe_right, comparable)}")
    print(f"  right only without the profile question: {counts['baseline only']}")
    print(f"  right only with it:                      {counts['probe only']}")
    print(f"  coin-flip test on that difference: p = "
          f"{mcnemar(counts['baseline only'], counts['probe only']):.4f}")
    print(f"  answers that changed at all: {rate(changed_answer, comparable)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="folder with the baseline CSVs")
    parser.add_argument("--probe", required=True, help="folder with the probe CSVs")
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    parser.add_argument("--mutation", default="no_mutation")
    args = parser.parse_args()

    name = f"{args.benchmark}_{args.prompt_type}_{args.mutation}.csv"
    compare(str(Path(args.baseline) / name), str(Path(args.probe) / name))


if __name__ == "__main__":
    main()
