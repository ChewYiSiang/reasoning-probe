"""Which loops and branches does the model misread, and how often while still answering
the question correctly?

Everything here comes from files already on disk: the traced ground truth, the profile
logs, and MuCoCo's own result CSVs. No model calls.

Two tables:
  * per-construct accuracy broken down by the shape of the construct
  * profile correctness conditioned on whether the answer was right, which is the
    "stable and wrong" number a user cares about
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path

from probe.analyse import read_results_csv, verdict
from probe.constructs import find_constructs
from probe.score import load_predictions, load_truth
from probe.stats import rate


def construct_features(code: str) -> dict[str, dict]:
    """Describe every construct in a program: kind, nesting, and how it can exit.

    Keys match the construct ids used everywhere else (loop1, if2, ...).
    """
    tree = ast.parse(code)
    features: dict[str, dict] = {}
    counters = {"loop": 0, "if": 0}

    def exits(node: ast.AST) -> bool:
        """Does this body contain a break or a return, so it can stop early?"""
        return any(isinstance(inner, (ast.Break, ast.Return))
                   for inner in ast.walk(node))

    def walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                counters["loop"] += 1
                features[f"loop{counters['loop']}"] = {
                    "kind": "loop",
                    "loop_type": "while" if isinstance(child, ast.While) else "for",
                    "depth": depth,
                    "early_exit": exits(child),
                }
                walk(child, depth + 1)
            elif isinstance(child, ast.If):
                counters["if"] += 1
                features[f"if{counters['if']}"] = {
                    "kind": "branch",
                    "has_else": bool(child.orelse),
                    "depth": depth,
                    "early_exit": exits(child),
                }
                walk(child, depth + 1)
            else:
                walk(child, depth)

    walk(tree, 0)
    return features


def bucket(construct_id: str, counts: tuple[int, ...], feature: dict) -> list[str]:
    """The categories one construct belongs to, for the breakdown table."""
    labels = []
    if feature["kind"] == "loop":
        iterations = counts[0]
        size = ("0 iterations" if iterations == 0 else
                "1 iteration" if iterations == 1 else
                "2 to 5 iterations" if iterations <= 5 else
                "6 to 20 iterations" if iterations <= 20 else
                "over 20 iterations")
        labels += [f"loop, {feature['loop_type']}", f"loop, {size}"]
        if feature["early_exit"]:
            labels.append("loop with break or return")
    else:
        taken, not_taken = counts
        outcome = ("never evaluated" if not taken and not not_taken else
                   "both outcomes seen" if taken and not_taken else
                   "condition always true" if taken else "condition always false")
        labels += ["branch", f"branch, {outcome}"]
        labels.append("branch with else" if feature["has_else"] else "branch without else")
    labels.append("nested" if feature["depth"] > 0 else "top level")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--probe-log", required=True,
                        help="profiles_<mutation>.jsonl for the version to diagnose")
    parser.add_argument("--results-csv", default=None,
                        help="the matching MuCoCo CSV, to split by answer correctness")
    parser.add_argument("--version", default="no_mutation",
                        help="which version in the truth file these profiles belong to")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    code_index = {value["code"]: key for key, value in truth.items() if value["code"]}
    predictions = load_predictions(args.probe_log)
    answers = read_results_csv(args.results_csv) if args.results_csv else {}

    correct = Counter()
    total = Counter()
    by_answer = defaultdict(lambda: [0, 0])      # answer verdict -> [wrong profiles, profiles]
    per_program = Counter()

    for prediction in predictions:
        key = code_index.get(prediction["full_sol"])
        if key is None or key[1] != args.version:
            continue
        task_id = key[0]
        real = truth[key]["counts"]
        if not real:
            continue                              # straight-line program, nothing to read

        features = construct_features(truth[key]["code"])
        claimed = {name: tuple(value) for name, value in prediction["profile"].items()}

        program_wrong = False
        for name, counts in real.items():
            feature = features.get(name)
            if feature is None:
                continue
            right = claimed.get(name) == counts
            program_wrong |= not right
            for label in bucket(name, counts, feature):
                total[label] += 1
                correct[label] += right

        per_program["profiles with an error" if program_wrong else "profiles fully correct"] += 1

        if answers:
            row = answers.get(task_id)
            if row is not None:
                state = verdict(row)
                if state in ("correct", "incorrect"):
                    by_answer[state][0] += program_wrong
                    by_answer[state][1] += 1

    print(f"=== {args.version}: accuracy by construct shape ===")
    width = max(len(label) for label in total) if total else 10
    for label in sorted(total, key=lambda l: (-total[l], l)):
        print(f"  {label.ljust(width)}  {rate(correct[label], total[label])}")

    print(f"\n=== per program ===")
    for label, count in sorted(per_program.items()):
        print(f"  {label:26s} {count}")

    if by_answer:
        print(f"\n=== profile errors, split by whether the answer was right ===")
        for state in ("correct", "incorrect"):
            wrong, n = by_answer[state]
            if n:
                print(f"  answer {state:9s}: profile wrong in {rate(wrong, n)}")


if __name__ == "__main__":
    main()
