"""How many model calls would a full operator sweep actually cost?

MuCoCo skips a task when an operator does not apply: no loop means no for-to-while, no
integer means no constant unfolding. Those tasks cost nothing, so the price of a sweep is
not 11 times the benchmark. This counts the valid mutants for every operator, locally and
without calling a model, and turns the estimate into a figure before anything is spent.

It also reports which operators keep the loops and branches matchable between the two
versions. `for2while` turns a `for` into a `while`, so the construct ids no longer line up
and profile *consistency* cannot be measured for it until an alignment step exists, though
profile *accuracy* still can, because each version is scored against its own trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe.constructs import find_constructs
from probe.stats import rate

# Their eleven, by category. The names are the ones their constants module uses.
ALL_OPERATORS = {
    "lexical": ["random", "sequential", "literal_format"],
    "syntactic": ["for2while", "for2enumerate"],
    "logical": ["demorgan", "boolean_literal", "commutative_reorder",
                "constant_unfold", "constant_unfold_add", "constant_unfold_mult"],
}

# Operators that rewrite a loop header, so a construct in one version has no partner in the
# other. Profile accuracy still works; profile consistency does not.
BREAKS_MATCHING = {"for2while", "for2enumerate"}


def seconds_per_call(model: str, with_probe: bool) -> float:
    """Measured on our own runs, so the estimate is not invented."""
    measured = {"qwen14b": 4.71, "llama": 2.30, "qwen8b": 1.67}
    base = measured.get(model, 3.0)
    return base if with_probe else base * 0.55      # shorter replies without the question


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mongo", "jsonl"], default="mongo")
    parser.add_argument("--tasks", help="exported task file when --source jsonl")
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--operators", nargs="*", help="defaults to all eleven")
    parser.add_argument("--models", nargs="*", default=["qwen14b", "llama"])
    parser.add_argument("--with-probe", action="store_true", default=True)
    parser.add_argument("--no-probe", dest="with_probe", action="store_false")
    args = parser.parse_args()

    from probe.mucoco import mutate
    from probe.mutants import load_operators

    known, source = load_operators()

    # The truth builder already knows how to read both sources; use its loaders rather
    # than a second, slightly different copy of the same logic.
    from probe.build_truth_mucoco import load_from_jsonl, load_from_mongo

    if args.source == "mongo":
        tasks = load_from_mongo(args.limit)
    else:
        if not args.tasks:
            raise SystemExit("--source jsonl needs --tasks pointing at the CRUXEval file")
        tasks = load_from_jsonl(args.tasks, args.limit)

    # Prefer their own operator names over the list in this file: the names in their
    # constants module are the ones the mutator will accept, and ours is only a fallback
    # for when their package is not importable.
    if args.operators:
        wanted = args.operators
    elif len(known) > 2:
        wanted = sorted(known)
        print(f"using their operator list ({source}): {len(wanted)} operators\n")
    else:
        wanted = [name for group in ALL_OPERATORS.values() for name in group]
        print(f"their operator list was not importable ({source}), so this falls back to "
              f"the names recorded in dry_run.py; any that do not match will show zero\n")
    category = {name: group for group, names in ALL_OPERATORS.items() for name in names}

    task_list = tasks
    applies: dict[str, int] = {name: 0 for name in wanted}
    failed: dict[str, int] = {name: 0 for name in wanted}
    constructs_kept: dict[str, int] = {name: 0 for name in wanted}
    counted = 0

    for task in task_list:
        counted += 1
        original = {c.id for c in find_constructs(task.code) if c.measurable}
        for name in wanted:
            mutant = mutate(task, name)
            if mutant.error or not mutant.code:
                failed[name] += 1
                continue
            applies[name] += 1
            try:
                shared = original & {c.id for c in find_constructs(mutant.code) if c.measurable}
            except Exception:
                shared = set()
            constructs_kept[name] += len(shared)

    print(f"tasks read: {counted}\n")
    print(f"{'operator':22s}{'category':12s}{'mutants':>10}{'of tasks':>11}"
          f"{'matchable constructs':>23}")
    total_calls = 0
    for name in wanted:
        total_calls += applies[name]
        flag = "" if name not in BREAKS_MATCHING else "   (accuracy only)"
        print(f"  {name:20s}{category.get(name, '?'):12s}{applies[name]:>10}"
              f"{rate(applies[name], counted):>11}{constructs_kept[name]:>13}{flag}")

    print(f"\nmutant calls across every operator: {total_calls}")
    print(f"plus the 800-task baseline, if it has not already been run")

    print("\nwhat that costs, from our own measured speeds")
    for model in args.models:
        seconds = total_calls * seconds_per_call(model, args.with_probe)
        hours = seconds / 3600
        print(f"  {model:10s}{hours:5.1f} h   about {hours * 2:4.1f} compute units"
              f"   ({seconds_per_call(model, args.with_probe):.2f} s a call"
              f"{' with the profile question' if args.with_probe else ', answers only'})")

    broken = [name for name in wanted if name in BREAKS_MATCHING and applies[name]]
    if broken:
        print(f"\nprofile consistency is unavailable for: {', '.join(broken)} "
              f"(a for loop becomes a while, so the construct ids do not line up). "
              f"Profile accuracy still works for them.")


if __name__ == "__main__":
    main()
