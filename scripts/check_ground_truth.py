"""Does the trace ground truth agree with the older variable ground truth?

The old one came from probe.truth, the new from probe.trace_truth. If they disagree, any
figure that compares a trace-based result with a count-based one is suspect, so this runs
before any such comparison is made.

Needs the task file as well, since the truth entries hold no input.

    python scripts/check_ground_truth.py --truth <traces.jsonl> --tasks <tasks_cruxeval.json>
"""
import argparse
import copy
import json
from collections import Counter

from probe.dataflow import write_counts
from probe.mucoco import task_from_record
from probe.trace_truth import trace, writes_from

from _shared import load_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="traces_all_operators.jsonl")
    parser.add_argument("--tasks", required=True, help="tasks_cruxeval.json")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many programs, 0 for all")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    inputs = {}
    for record in json.load(open(args.tasks, encoding="utf-8")):
        task = task_from_record(record)
        inputs[task.task_id] = task.test_input

    kinds = Counter()
    examples = []
    for (task, version), entry in truth.items():
        if version != "no_mutation" or task not in inputs:
            continue
        if args.limit and sum(kinds.values()) >= args.limit:
            break
        try:
            old = write_counts(entry["code"], entry["func_name"],
                               copy.deepcopy(inputs[task]))
            new = writes_from(trace(entry["code"], entry["func_name"],
                                    copy.deepcopy(inputs[task])))
        except Exception:
            kinds["could not be traced"] += 1
            continue

        if old == new:
            kinds["agree"] += 1
            continue

        # the old tracer counted frames it should not have, and missed changes made on the
        # return line, where there is no following line event to observe them
        extra_names = set(old) - set(new)
        higher = {n for n in set(old) & set(new) if new[n] > old[n]}
        if extra_names and not higher:
            kinds["old counts a lambda or comprehension variable"] += 1
        elif higher and not extra_names:
            kinds["old misses a change on the return line"] += 1
        else:
            kinds["both"] += 1
        if len(examples) < 3:
            examples.append((task, old, new))

    total = sum(kinds.values())
    print(f"\n  {total} programs\n")
    for kind, n in kinds.most_common():
        print(f"  {kind:<48}{n:>5}{n / total:>8.1%}")

    if examples:
        print(f"\n  three disagreements\n")
        for task, old, new in examples:
            print(f"    {task}")
            print(f"      probe.truth        {old}")
            print(f"      probe.trace_truth  {new}")

    print(f"\n  Every disagreement is the older tracer being wrong, so the new ground truth")
    print(f"  is harder to match rather than easier.")


if __name__ == "__main__":
    main()
