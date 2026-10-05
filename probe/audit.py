"""Re-score a run over a fixed denominator, and reconcile every reply.

The accuracy tables in `analyse.py` and `diagnose.py` iterate over the replies that could be
matched to a task, so a run that lost some replies quietly shrinks its own denominator: one
model was scored over 866 constructs and another over 877 of the same 877. That hides the
loss rather than reporting it.

This module iterates over the constructs in the ground truth instead, so every run is scored
over the same denominator, and each construct lands in one of four outcomes:

    correct        the reported counts match the trace
    wrong          they do not
    not answered   the reply came back but said nothing about this construct
    reply missing  no reply could be matched to this task at all

It also reconciles the reply log against the task list, which is where the missing
constructs come from in the first place.

`--compare <mutation>` answers a different question: does rewriting the program change how
well the model reads it? That needs a matched comparison, because the two versions do not
hold the same constructs. Mutation passes every program through `ast.unparse`, which puts a
one-liner's body on its own line and so makes a construct measurable in the mutant that was
not measurable in the original. The comparison therefore runs over the tasks where the
mutation applies and the constructs measurable in both versions, so the only thing differing
between the two columns is the rewriting itself.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from probe.analyse import read_results_csv, verdict
from probe.score import load_predictions, load_truth
from probe.stats import rate

OUTCOMES = ("correct", "wrong", "not answered", "reply missing")


def audit_version(truth: dict, predictions_path: Path, version: str) -> tuple[Counter, Counter]:
    """Score one version over every construct the ground truth knows about."""
    wanted = {task: entry for (task, ver), entry in truth.items()
              if ver == version and entry["counts"]}
    by_code = {entry["code"]: task for task, entry in wanted.items() if entry["code"]}
    # Tasks of this version with no loop or branch at all. Their replies are not losses:
    # there is simply nothing to score, so they are counted separately from genuine misses.
    straight_line = {entry["code"] for (task, ver), entry in truth.items()
                     if ver == version and not entry["counts"] and entry["code"]}

    replies: dict[str, dict] = {}
    logged = unmatched = no_constructs = 0
    if predictions_path.exists():
        for prediction in load_predictions(predictions_path):
            logged += 1
            task = by_code.get(prediction["full_sol"])
            if task is not None:
                replies[task] = prediction
            elif prediction["full_sol"] in straight_line:
                no_constructs += 1
            else:
                unmatched += 1

    outcomes: Counter = Counter()
    for task, entry in wanted.items():
        prediction = replies.get(task)
        for construct, real in entry["counts"].items():
            if prediction is None:
                outcomes["reply missing"] += 1
                continue
            claimed = (prediction.get("profile") or {}).get(construct)
            if claimed is None:
                outcomes["not answered"] += 1
            elif tuple(claimed) == tuple(real):
                outcomes["correct"] += 1
            else:
                outcomes["wrong"] += 1

    reconciliation = Counter({
        "tasks with constructs": len(wanted),
        "replies logged (all tasks)": logged,
        "replies matched": len(replies),
        "replies for straight-line tasks": no_constructs,
        "replies matching no task": unmatched,
        "tasks with constructs but no reply": len(wanted) - len(replies),
    })
    return outcomes, reconciliation


def compare_versions(truth: dict, probe_dir: Path, mutation: str,
                     baseline: str = "no_mutation") -> dict:
    """Score the original and the mutant over exactly the same constructs.

    Only tasks the mutation applies to, and within those only constructs present and
    measurable in both versions. Anything else would mix the effect of the rewriting with a
    change in what is being counted.
    """
    def replies_for(version: str) -> dict[str, dict]:
        by_code = {entry["code"]: task for (task, ver), entry in truth.items()
                   if ver == version and entry["code"]}
        found: dict[str, dict] = {}
        path = probe_dir / f"profiles_{version}.jsonl"
        if path.exists():
            for prediction in load_predictions(path):
                task = by_code.get(prediction["full_sol"])
                if task:
                    found[task] = prediction
        return found

    base_replies, mutant_replies = replies_for(baseline), replies_for(mutation)

    tasks = sorted({task for (task, ver) in truth if ver == mutation}
                   & {task for (task, ver) in truth if ver == baseline})

    stats = {"tasks": 0, "constructs": 0,
             "base correct": 0, "mutant correct": 0,
             "both correct": 0, "neither correct": 0,
             "only base correct": 0, "only mutant correct": 0,
             "dropped, construct not in both": 0,
             "skipped, a reply missing": 0}

    for task in tasks:
        base_counts = truth[(task, baseline)]["counts"]
        mutant_counts = truth[(task, mutation)]["counts"]
        if not base_counts or not mutant_counts:
            continue
        base, mutant = base_replies.get(task), mutant_replies.get(task)
        if base is None or mutant is None:
            stats["skipped, a reply missing"] += 1
            continue

        shared = set(base_counts) & set(mutant_counts)
        stats["dropped, construct not in both"] += len(set(base_counts) ^ set(mutant_counts))
        if not shared:
            continue
        stats["tasks"] += 1

        for name in sorted(shared):
            stats["constructs"] += 1
            base_right = tuple((base.get("profile") or {}).get(name, ())) == tuple(base_counts[name])
            mutant_right = tuple((mutant.get("profile") or {}).get(name, ())) == tuple(mutant_counts[name])
            stats["base correct"] += base_right
            stats["mutant correct"] += mutant_right
            if base_right and mutant_right:
                stats["both correct"] += 1
            elif base_right:
                stats["only base correct"] += 1
            elif mutant_right:
                stats["only mutant correct"] += 1
            else:
                stats["neither correct"] += 1
    return stats


def answer_accuracy(csv_path: Path, tasks: set[str]) -> tuple[int, int]:
    """Their correctness oracle, restricted to the same tasks, for the audit table."""
    if not csv_path.exists():
        return 0, 0
    rows = read_results_csv(csv_path)
    judged = [verdict(row) for task, row in rows.items() if task in tasks]
    scorable = [state for state in judged if state in ("correct", "incorrect")]
    return sum(1 for state in scorable if state == "correct"), len(scorable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--probe-dir", required=True,
                        help="folder holding profiles_<version>.jsonl")
    parser.add_argument("--results-dir", help="folder holding the result CSVs, for answer accuracy")
    parser.add_argument("--versions", nargs="*",
                        default=["no_mutation", "sequential", "constant_unfold_add"])
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    parser.add_argument("--compare", nargs="*", default=None,
                        help="mutations to compare against the originals on matched constructs")
    args = parser.parse_args()

    truth = load_truth(args.truth)

    if args.compare is not None:
        for mutation in (args.compare or ["sequential", "constant_unfold_add"]):
            stats = compare_versions(truth, Path(args.probe_dir), mutation)
            if not stats["constructs"]:
                print(f"\n=== {mutation} against the originals: nothing comparable ===")
                continue
            total = stats["constructs"]
            print(f"\n=== {mutation} against the originals, matched ===")
            print(f"  {stats['tasks']} tasks where the mutation applies, "
                  f"{total} constructs measurable in both versions")
            print(f"  original  {rate(stats['base correct'], total)}")
            print(f"  mutant    {rate(stats['mutant correct'], total)}")
            difference = (stats['mutant correct'] - stats['base correct']) / total * 100
            print(f"  change    {difference:+.1f} points")
            print(f"  right in both        {rate(stats['both correct'], total)}")
            print(f"  right only before    {stats['only base correct']}")
            print(f"  right only after     {stats['only mutant correct']}")
            print(f"  wrong in both        {rate(stats['neither correct'], total)}")
            print(f"  constructs dropped because they are not in both versions: "
                  f"{stats['dropped, construct not in both']}")
            if stats["skipped, a reply missing"]:
                print(f"  tasks skipped because one reply never arrived: "
                      f"{stats['skipped, a reply missing']}")
        return

    for version in args.versions:
        log = Path(args.probe_dir) / f"profiles_{version}.jsonl"
        outcomes, reconciliation = audit_version(truth, log, version)
        total = sum(outcomes[name] for name in OUTCOMES)
        if not total:
            print(f"\n=== {version}: nothing to score (is {log} there?) ===")
            continue

        print(f"\n=== {version}: {total} constructs, fixed denominator ===")
        for name in OUTCOMES:
            print(f"  {name:16s} {rate(outcomes[name], total)}")
        print(f"  {'answered':16s} {rate(outcomes['correct'] + outcomes['wrong'], total)}"
              "   <- the denominator the old tables used")
        answered = outcomes["correct"] + outcomes["wrong"]
        if answered:
            print(f"  accuracy over answered constructs only: {rate(outcomes['correct'], answered)}")

        print("  reconciliation")
        for name, count in reconciliation.items():
            print(f"    {name:26s} {count}")

        if args.results_dir:
            tasks = {task for (task, ver) in truth if ver == version}
            csv_path = Path(args.results_dir) / f"{args.benchmark}_{args.prompt_type}_{version}.csv"
            correct, scorable = answer_accuracy(csv_path, tasks)
            if scorable:
                print(f"  answer accuracy (same tasks):   {rate(correct, scorable)}")


if __name__ == "__main__":
    main()
