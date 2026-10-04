"""Build the ground-truth execution profiles for a benchmark slice.

For every item: take the original program, apply each mutation operator, run all
versions under the tracer, and record what really happened. Two checks run on the way:
the mutant must return the same output as the original, and it must produce the same
profile. A mutant that fails either is not semantics-preserving for that input and is
dropped, with the reason kept in the output file.

    python -m probe.build_truth --dataset data/cruxeval.jsonl --limit 200 \
        --operators sequential_rename constant_unfold_add --out results/truth.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from probe.constructs import find_constructs
from probe.dataset import call_source, load_cruxeval
from probe.coverage_check import available as coverage_available
from probe.coverage_check import disagreements
from probe.mutants import load_operators
from probe.truth import Profile, profile


def build_versions(code: str, operators: dict) -> list[tuple[str, str, dict]]:
    """Return [(version name, code, name mapping)], starting with the original."""
    versions = [("original", code, {})]
    for name, operator in operators.items():
        try:
            mutated, mapping = operator(code)
        except Exception:
            continue          # operator does not apply to this program, which is normal
        versions.append((name, mutated, mapping))
    return versions


def profile_version(code: str, call_source: str, **kwargs) -> tuple[Profile, list]:
    constructs = find_constructs(code)
    return profile(code, call_source, constructs, **kwargs), constructs


def _is_recursive(code: str) -> bool:
    """Flag self-calls: arc counting is less reliable across nested frames."""
    tree = ast.parse(code)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in names
               for n in ast.walk(tree))


def _same_profile(base: Profile, mutant: Profile) -> bool:
    """Compare the constructs both versions can measure.

    One-liners such as `if x: return 1` are excluded from scoring, and ast.unparse
    rewrites them over several lines, so a mutant can measure a construct the original
    cannot. Those are ignored rather than counted as a disagreement.
    """
    base_counts, mutant_counts = base.counts_only(), mutant.counts_only()
    shared = set(base_counts) & set(mutant_counts)
    return all(base_counts[key] == mutant_counts[key] for key in shared)


def process_item(item, operators: dict, cross_check: bool = False, **kwargs) -> dict:
    """Profile one benchmark item and all of its mutants."""
    row: dict = {"id": item.id, "input": item.input_source,
                 "expected_output": item.expected_output, "versions": {}}

    try:
        base_profile, base_constructs = profile_version(item.code, item.call_source, **kwargs)
    except Exception as error:
        row["status"] = "original failed"
        row["error"] = f"{type(error).__name__}: {error}"
        return row

    row["status"] = "ok"
    row["recursive"] = _is_recursive(item.code)
    row["versions"]["original"] = {
        "code": item.code,
        "call": item.call_source,
        "constructs": [c.__dict__ for c in base_constructs],
        "profile": base_profile.as_dict(),
    }

    if cross_check:
        row["coverage_disagreements"] = disagreements(
            base_profile, base_constructs, item.code, item.call_source)

    for name, code, mapping in build_versions(item.code, operators)[1:]:
        entry: dict = {"code": code, "name_mapping": mapping}
        try:
            # The call is rebuilt here: lexical mutations rename the function itself.
            call = call_source(code, item.input_source)
            entry["call"] = call
            mutant_profile, mutant_constructs = profile_version(code, call, **kwargs)
        except Exception as error:
            entry["dropped"] = f"{type(error).__name__}: {error}"
            row["versions"][name] = entry
            continue

        # A semantics-preserving mutation must not change the answer or the control flow.
        if repr(mutant_profile.output) != repr(base_profile.output):
            entry["dropped"] = "output differs from the original"
        elif not _same_profile(base_profile, mutant_profile):
            entry["dropped"] = "execution profile differs from the original"
        else:
            entry["constructs"] = [c.__dict__ for c in mutant_constructs]
            entry["profile"] = mutant_profile.as_dict()
        row["versions"][name] = entry

    return row


def summarise(rows: list[dict]) -> str:
    """The numbers worth reading after a run, including how much of each program ran."""
    stats = Counter()
    for row in rows:
        stats["items"] += 1
        if row.get("recursive"):
            stats["recursive programs (counts less reliable)"] += 1
        if row["status"] != "ok":
            stats["items skipped"] += 1
            continue

        original = row["versions"]["original"]
        constructs = original["constructs"]
        prof = original["profile"]

        stats["loops"] += len(prof["loops"])
        stats["branches"] += len(prof["branches"])
        stats["one-liners excluded"] += sum(1 for c in constructs if not c["measurable"])
        stats["loops running more than once"] += sum(1 for n in prof["loops"].values() if n > 1)
        stats["loops never entered"] += sum(1 for n in prof["loops"].values() if n == 0)

        if prof["branches"]:
            stats["items with a branch"] += 1
            if any(taken == 0 or else_taken == 0 for taken, else_taken in prof["branches"].values()):
                stats["items where one branch side never ran"] += 1

        if "coverage_disagreements" in row:
            stats["cross-checked against coverage.py"] += 1
            stats["cross-check disagreements"] += bool(row["coverage_disagreements"])

        for name, version in row["versions"].items():
            if name == "original":
                continue
            stats[f"{name}: applied"] += 1
            if "dropped" in version:
                stats[f"{name}: dropped ({version['dropped']})"] += 1

    width = max(len(k) for k in stats)
    lines = [f"{key.ljust(width)}  {value}" for key, value in stats.items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="CRUXEval jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="only the first N items")
    parser.add_argument("--operators", nargs="*", default=None,
                        help="operator names to apply (default: all available)")
    parser.add_argument("--out", default="results/truth.jsonl")
    parser.add_argument("--step-budget", type=int, default=200_000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--cross-check", action="store_true",
                        help="verify the branch counts against coverage.py (slower)")
    args = parser.parse_args()

    operators, source = load_operators()
    if args.operators:
        missing = set(args.operators) - set(operators)
        if missing:
            parser.error(f"unknown operators: {', '.join(sorted(missing))}")
        operators = {name: operators[name] for name in args.operators}
    print(f"operators from {source}: {', '.join(operators)}")

    items = load_cruxeval(args.dataset, args.limit)
    if args.cross_check and not coverage_available():
        parser.error("--cross-check needs coverage.py (pip install coverage)")

    rows = [process_item(item, operators, cross_check=args.cross_check,
                         step_budget=args.step_budget, timeout_s=args.timeout)
            for item in items]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(rows)} items to {out_path}\n")
    print(summarise(rows))


if __name__ == "__main__":
    main()
