"""Put the two signals side by side.

Reads, for each mutation configuration, MuCoCo's own result CSV (their answer, their
correctness verdict) and the probe's profile log (what the model believed about the
execution), and compares both against the traced ground truth.

Produces three things:
  * profile accuracy, split into constructs the input exercised and constructs it did not
  * output inconsistency and profile inconsistency per mutation, using MuCoCo's own
    three-way rule in both cases
  * the 2x2 between them, plus the silent-failure count: pairs where the answers agree
    but the execution was misunderstood
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from probe.score import load_predictions, load_truth
from probe.stats import mcnemar, rate, two_by_two

INVALID_PREFIXES = ("LLMExecutionRuntimeError", "PicklingError", "TimeoutError")


def read_results_csv(path: str) -> dict[str, dict]:
    """{task_id: row} from a MuCoCo result log."""
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["task_id"]] = row
    return rows


def verdict(row: dict) -> str:
    """MuCoCo's view of one answer: correct, incorrect, invalid, or skipped."""
    failure = (row.get("failure_type") or "").strip()
    if not failure:
        return "correct"
    if failure.startswith(INVALID_PREFIXES):
        return "invalid"
    if failure.startswith("AssertionError"):
        return "incorrect"
    return "skipped"          # mutation did not apply, or the task was rejected


def inconsistent(first: str, second: str, first_answer: str, second_answer: str) -> bool | None:
    """MuCoCo's three-way rule. None when the pair cannot be compared."""
    if "skipped" in (first, second):
        return None
    if {first, second} == {"correct", "incorrect"}:
        return True                                     # correctness-based
    if first == second == "incorrect":
        return first_answer.strip() != second_answer.strip()   # incorrectness-based
    if (first == "invalid") != (second == "invalid"):
        return True                                     # invalidity-based
    return False


def profile_verdict(prediction: dict | None, real: dict | None) -> str:
    """The same three categories, applied to the execution profile."""
    if real is None:
        return "skipped"
    if not real:
        return "skipped"                                # nothing to count in this program
    if prediction is None or not prediction.get("profile"):
        return "invalid"                                # no readable profile in the reply
    claimed = {name: tuple(value) for name, value in prediction["profile"].items()}
    return "correct" if all(claimed.get(name) == counts for name, counts in real.items()) \
        else "incorrect"


def profile_text(prediction: dict | None) -> str:
    if prediction is None:
        return ""
    return json.dumps(prediction.get("profile", {}), sort_keys=True)


def construct_accuracy(predictions: list[dict], truth: dict, code_index: dict) -> Counter:
    """Per-construct correctness, split by whether the input exercised the construct."""
    stats = Counter()
    for prediction in predictions:
        key = code_index.get(prediction["full_sol"])
        if key is None:
            continue
        real = truth[key]["counts"]
        claimed = {name: tuple(value) for name, value in prediction["profile"].items()}
        for name, counts in real.items():
            bucket = "exercised" if any(c > 0 for c in counts) else "never ran"
            if name not in claimed:
                stats[f"{bucket}: not answered"] += 1
            elif claimed[name] == counts:
                stats[f"{bucket}: correct"] += 1
            else:
                stats[f"{bucket}: wrong"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--results-dir", required=True,
                        help="folder holding MuCoCo's result CSVs for this model")
    parser.add_argument("--probe-dir", required=True,
                        help="folder holding profiles_<mutation>.jsonl")
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    parser.add_argument("--mutations", nargs="*", default=["sequential", "constant_unfold_add"])
    args = parser.parse_args()

    truth = load_truth(args.truth)
    code_index = {value["code"]: key for key, value in truth.items() if value["code"]}
    if not code_index:
        raise SystemExit(f"{args.truth} has no program text; rebuild it with the current builder")

    def load_config(name: str) -> tuple[dict, dict]:
        csv_path = Path(args.results_dir) / f"{args.benchmark}_{args.prompt_type}_{name}.csv"
        log_path = Path(args.probe_dir) / f"profiles_{name}.jsonl"
        rows = read_results_csv(csv_path)
        predictions = load_predictions(log_path) if log_path.exists() else []
        # Profiles are logged per program text; turn them into task_id -> prediction.
        by_task: dict[str, dict] = {}
        for prediction in predictions:
            key = code_index.get(prediction["full_sol"])
            if key:
                by_task[key[0]] = prediction
        return rows, by_task

    base_rows, base_profiles = load_config("no_mutation")
    print(f"baseline: {len(base_rows)} logged rows, {len(base_profiles)} profiles matched\n")

    print("=== profile accuracy, baseline ===")
    base_predictions = [p for p in base_profiles.values()]
    for key, value in sorted(construct_accuracy(base_predictions, truth, code_index).items()):
        print(f"  {key:28s} {value}")

    for mutation in args.mutations:
        rows, profiles = load_config(mutation)
        if not rows:
            print(f"\n{mutation}: no result CSV found, skipped")
            continue

        table = Counter()
        silent = 0
        output_only = Counter()       # output signal over every comparable pair
        for task_id, base_row in base_rows.items():
            mutant_row = rows.get(task_id)
            if mutant_row is None:
                continue

            output_bad = inconsistent(verdict(base_row), verdict(mutant_row),
                                      str(base_row.get("model_output", "")),
                                      str(mutant_row.get("model_output", "")))
            if output_bad is None:
                continue
            # Counted before the profile filter below, so this rate is comparable with
            # MuCoCo's own: it covers every pair, including straight-line programs where
            # there is no control flow to profile.
            output_only[output_bad] += 1

            real_base = truth.get((task_id, "no_mutation"), {}).get("counts")
            real_mut = truth.get((task_id, mutation), {}).get("counts")
            base_profile = profile_verdict(base_profiles.get(task_id), real_base)
            mut_profile = profile_verdict(profiles.get(task_id), real_mut)
            profile_bad = inconsistent(base_profile, mut_profile,
                                       profile_text(base_profiles.get(task_id)),
                                       profile_text(profiles.get(task_id)))
            if profile_bad is None:
                continue

            table[(output_bad, profile_bad)] += 1
            # A silent failure: MuCoCo sees nothing wrong, but the execution was
            # misunderstood in at least one of the two versions.
            if not output_bad and "incorrect" in (base_profile, mut_profile):
                silent += 1

        pairs = sum(table.values())
        all_pairs = sum(output_only.values())
        print(f"\n=== {mutation} ===")
        print(f"  all comparable pairs: {all_pairs}   "
              f"(of which scorable for profiles: {pairs})")
        print(f"  output inconsistency, all pairs   {rate(output_only[True], all_pairs)}")
        print(f"\n  among the {pairs} pairs where both signals exist:")
        print(f"{'':22s}{'profiles agree':>16}{'profiles differ':>17}")
        for output_bad, label in ((False, "answers agree"), (True, "answers differ")):
            agree = table[(output_bad, False)]
            differ = table[(output_bad, True)]
            print(f"  {label:20s}{agree:>16}{differ:>17}")

        out_bad = table[(True, False)] + table[(True, True)]
        prof_bad = table[(False, True)] + table[(True, True)]
        print(f"  output inconsistency   {rate(out_bad, pairs)}")
        print(f"  profile inconsistency  {rate(prof_bad, pairs)}")
        print(f"  silent failures        {rate(silent, pairs)}")
        # Is one signal flagging more than the other? Only the pairs where they disagree
        # carry information, so this is the coin-flip test.
        print(f"  profile vs output flags: p = "
              f"{mcnemar(table[(True, False)], table[(False, True)]):.4f}")
        print("  " + two_by_two(table[(False, False)], table[(True, False)],
                                table[(False, True)], table[(True, True)]).replace("\n", "\n  "))

        print(f"\n  profile accuracy, {mutation}:")
        for key, value in sorted(construct_accuracy(list(profiles.values()), truth, code_index).items()):
            print(f"    {key:28s} {value}")


if __name__ == "__main__":
    main()
