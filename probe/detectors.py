"""How good is each detector, measured against the tracer?

Two of MuCoCo's checks need no oracle: comparing two answers, and comparing two profiles.
Both can be run where the code cannot be executed, which is the case that matters in
practice. The tracer can tell us how well they do.

Ground truth here is "did the model misread the execution in at least one version of this
pair", decided by comparing each profile against the traced counts. Against that label:

  * output inconsistency   - MuCoCo's existing check, applied unchanged
  * profile inconsistency  - the same rule applied to the profile field
  * the two combined       - flag if either fires

Precision says whether a flag is trustworthy; recall says how much is found. A profile
disagreement should have precision 1.0, because a semantics-preserving mutation leaves the
execution identical: two different accounts of it cannot both be right.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from probe.analyse import (inconsistent, profile_text, profile_verdict, read_results_csv,
                           verdict)
from probe.score import load_predictions, load_truth
from probe.stats import rate


def evaluate(truth: dict, code_index: dict, results_dir: str, probe_dir: str,
             mutation: str, benchmark: str, prompt_type: str) -> tuple[Counter, int]:
    """Cross-tabulate each detector against the traced label, pair by pair."""

    def load(name: str):
        csv_path = Path(results_dir) / f"{benchmark}_{prompt_type}_{name}.csv"
        log_path = Path(probe_dir) / f"profiles_{name}.jsonl"
        rows = read_results_csv(csv_path)
        by_task: dict[str, dict] = {}
        if log_path.exists():
            for prediction in load_predictions(log_path):
                key = code_index.get(prediction["full_sol"])
                if key:
                    by_task[key[0]] = prediction
        return rows, by_task

    base_rows, base_profiles = load("no_mutation")
    mutant_rows, mutant_profiles = load(mutation)

    stats = Counter()
    pairs = 0

    for task_id, base_row in base_rows.items():
        mutant_row = mutant_rows.get(task_id)
        if mutant_row is None:
            continue

        real_base = truth.get((task_id, "no_mutation"), {}).get("counts")
        real_mutant = truth.get((task_id, mutation), {}).get("counts")
        if not real_base or not real_mutant:
            continue                       # straight-line program: nothing to misread

        base_state = profile_verdict(base_profiles.get(task_id), real_base)
        mutant_state = profile_verdict(mutant_profiles.get(task_id), real_mutant)
        if "skipped" in (base_state, mutant_state):
            continue

        # The label: the tracer says at least one version was misread.
        misread = "incorrect" in (base_state, mutant_state)

        output_flag = inconsistent(verdict(base_row), verdict(mutant_row),
                                   str(base_row.get("model_output", "")),
                                   str(mutant_row.get("model_output", "")))
        profile_flag = inconsistent(base_state, mutant_state,
                                    profile_text(base_profiles.get(task_id)),
                                    profile_text(mutant_profiles.get(task_id)))
        if output_flag is None or profile_flag is None:
            continue

        pairs += 1
        stats["misread pairs"] += misread
        for name, flag in (("output", output_flag),
                           ("profile", profile_flag),
                           ("either", output_flag or profile_flag)):
            if flag:
                stats[f"{name}: flagged"] += 1
                stats[f"{name}: true positive" if misread else f"{name}: false positive"] += 1
            elif misread:
                stats[f"{name}: missed"] += 1

    return stats, pairs


def report(stats: Counter, pairs: int, mutation: str) -> None:
    misread = stats["misread pairs"]
    print(f"\n=== {mutation}: {pairs} pairs, {rate(misread, pairs)} misread by the tracer ===")
    print(f"{'detector':24s}{'flagged':>10}{'precision':>22}{'recall':>22}")
    for name, label in (("output", "output inconsistency"),
                        ("profile", "profile inconsistency"),
                        ("either", "either check")):
        flagged = stats[f"{name}: flagged"]
        true_positive = stats[f"{name}: true positive"]
        print(f"  {label:22s}{flagged:>10}"
              f"{rate(true_positive, flagged) if flagged else 'n/a':>22}"
              f"{rate(true_positive, misread) if misread else 'n/a':>22}")
    print(f"  {'profile correctness':22s}{misread:>10}{'by definition 100%':>22}{'by definition 100%':>22}")
    print("\n  (profile correctness is the label itself: it needs the code to be run, "
          "the other three do not)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--mutations", nargs="*", default=["sequential", "constant_unfold_add"])
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    code_index = {value["code"]: key for key, value in truth.items() if value["code"]}
    if not code_index:
        raise SystemExit(f"{args.truth} has no program text; rebuild it with the current builder")

    for mutation in args.mutations:
        stats, pairs = evaluate(truth, code_index, args.results_dir, args.probe_dir,
                                mutation, args.benchmark, args.prompt_type)
        if pairs:
            report(stats, pairs, mutation)
        else:
            print(f"\n{mutation}: no comparable pairs found, check the file names")


if __name__ == "__main__":
    main()
