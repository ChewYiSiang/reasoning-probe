"""Do different models get the same constructs wrong?

Three separate error rates say how often each model is wrong. They do not say whether the
models are wrong about the same things. If a construct defeats every model, that is a
property of the construct rather than of any one model, and it is a far more useful thing
to tell a user than three percentages.

For every construct the ground truth knows about, this records which models got it right
and which got it wrong, then reports:

  * how many constructs every model got right, and how many every model got wrong
  * the agreement between each pair of models
  * what the constructs nobody gets right have in common, by shape
  * the worst of them, with the numbers each model gave

A construct with no answer from a model is left out of that model's column rather than
counted as wrong, so a missing reply never looks like a mistake.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from probe.diagnose import bucket, construct_features
from probe.score import load_predictions, load_truth
from probe.stats import rate


def answers(truth: dict, log_path: Path, version: str) -> dict[tuple[str, str], tuple[int, ...]]:
    """What one model said for each construct. Constructs it did not answer are left out."""
    by_code = {entry["code"]: task for (task, ver), entry in truth.items()
               if ver == version and entry["code"]}
    said: dict[tuple[str, str], tuple[int, ...]] = {}
    if not log_path.exists():
        return said
    for prediction in load_predictions(log_path):
        task = by_code.get(prediction["full_sol"])
        if task is None:
            continue
        claimed = prediction.get("profile") or {}
        for name in truth[(task, version)]["counts"]:
            if name in claimed:
                said[(task, name)] = tuple(claimed[name])
    return said


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("NAME", "LOG"),
                        required=True, help="a label and its profile log; repeat per model")
    parser.add_argument("--version", default="no_mutation")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    truth = load_truth(args.truth)
    said = {name: answers(truth, Path(log), args.version) for name, log in args.model}
    names = list(said)

    def right(model: str, key: tuple[str, str]) -> bool:
        return said[model][key] == tuple(truth[(key[0], args.version)]["counts"][key[1]])

    models = {name: {key: right(name, key) for key in said[name]} for name in names}
    shared = set.intersection(*(set(v) for v in models.values())) if models else set()
    print(f"models: {', '.join(names)}")
    print(f"constructs every model answered: {len(shared)}")

    all_right = {key for key in shared if all(models[name][key] for name in names)}
    all_wrong = {key for key in shared if not any(models[name][key] for name in names)}
    print(f"  every model right: {rate(len(all_right), len(shared))}")
    print(f"  every model wrong: {rate(len(all_wrong), len(shared))}")
    print(f"  models disagree:   {rate(len(shared) - len(all_right) - len(all_wrong), len(shared))}")

    print("\nagreement between pairs")
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            a, b = names[first], names[second]
            agree = sum(1 for key in shared if models[a][key] == models[b][key])
            both_wrong = sum(1 for key in shared
                             if not models[a][key] and not models[b][key])
            either_wrong = sum(1 for key in shared
                               if not models[a][key] or not models[b][key])
            print(f"  {a} and {b}: agree on {rate(agree, len(shared))}")
            if either_wrong:
                # of everything at least one of them got wrong, how much they share
                print(f"{'':4s}of the {either_wrong} either got wrong, both got "
                      f"{rate(both_wrong, either_wrong)}")

    if all_wrong:
        print("\nwhat the constructs nobody gets right have in common")
        shapes: Counter = Counter()
        shapes_all: Counter = Counter()
        for task, name in shared:
            entry = truth[(task, args.version)]
            feature = construct_features(entry["code"]).get(name)
            if feature is None:
                continue
            labels = bucket(name, entry["counts"][name], feature)
            for label in labels:
                shapes_all[label] += 1
                if (task, name) in all_wrong:
                    shapes[label] += 1
        print(f"  {'shape':40s}{'nobody right':>22}{'share of that shape':>24}")
        for label, count in shapes.most_common(12):
            print(f"  {label:40s}{count:>8} of {shapes_all[label]:<6}"
                  f"{count / shapes_all[label]:>22.0%}")

        print("\n  the worst of them, by how far the models were out")

        def distance(key: tuple[str, str]) -> int:
            real = tuple(truth[(key[0], args.version)]["counts"][key[1]])
            return max(abs(sum(said[name][key]) - sum(real)) for name in names)

        for key in sorted(all_wrong, key=distance, reverse=True)[:args.examples]:
            real = tuple(truth[(key[0], args.version)]["counts"][key[1]])
            claims = ", ".join(f"{name} said {said[name][key]}" for name in names)
            print(f"    {key[0]} {key[1]}: really {real} | {claims}")


if __name__ == "__main__":
    main()
