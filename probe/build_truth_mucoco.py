"""Build execution-profile ground truth for MuCoCo's CRUXEval tasks.

For every task: take the original program and each mutant produced by MuCoCo's own
operators, run them under the tracer, and record how many times each loop and branch
ran. No model is involved and nothing is sent anywhere.

Reads tasks either from the MongoDB collection the MuCoCo notebooks build, or from a
CRUXEval jsonl file when the database is not to hand.

    python -m probe.build_truth_mucoco --source mongo --limit 800 \
        --mutations sequential constant_unfold_add --out results/truth/cruxeval.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from probe.constructs import find_constructs
from probe.mucoco import Task, task_from_record, versions_for
from probe.truth import Profile, profile_call


def load_from_mongo(limit: int | None) -> list[Task]:
    """Read the collection the MuCoCo database builder writes."""
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()
    client = MongoClient(os.environ["MONGODB_URI"])
    collection = client[os.environ.get("MONGODB_BENCHMARK_DATABASE", "Base_Questions_DB")][
        os.environ.get("MONGODB_CRUXEVAL_COLLECTION", "CruxEval_Input_Output")
    ]
    cursor = collection.find({}).sort("_id", 1)
    if limit:
        cursor = cursor.limit(limit)
    return [task_from_record(record) for record in cursor]


def load_from_jsonl(path: str, limit: int | None) -> list[Task]:
    """Fallback source: the raw CRUXEval file, shaped like a database record."""
    tasks: list[Task] = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if not row["input"].strip():          # a handful of tasks take no argument
                test_input = None
            else:
                # Some inputs name a fixture the program defines at module level, so the
                # argument text is evaluated with the program already in scope.
                namespace: dict = {}
                exec(row["code"], namespace)
                # A few inputs already end with a comma, which would make an empty slot
                # once this wraps them in a tuple.
                raw_input = row["input"].strip().rstrip(",")
                arguments = eval("(" + raw_input + ",)", namespace)
                test_input = arguments[0] if len(arguments) == 1 else list(arguments)
            tasks.append(Task(task_id=row["id"], code=row["code"], test_input=test_input,
                              input_metadata=type(test_input).__name__,
                              expected_output=eval(row["output"])))
            if limit and len(tasks) == limit:
                break
    return tasks


def shared_counts_match(first: Profile, second: Profile) -> bool:
    """Compare the constructs both versions can measure.

    A mutant goes through `ast.unparse`, which spreads one-liners such as
    `if x: return 1` over two lines. That makes a construct measurable in the mutant that
    is not measurable in the original, so only shared constructs are compared.
    """
    a, b = first.counts_only(), second.counts_only()
    shared = set(a) & set(b)
    return all(a[key] == b[key] for key in shared)


def process(task: Task, mutations: list[str], verify_with_mucoco: bool = False,
            **kwargs) -> tuple[dict, dict[str, Profile]]:
    """Trace every version of one task. Returns the record to save and the profiles."""
    row: dict = {"task_id": task.task_id, "versions": {}}
    profiles: dict[str, Profile] = {}

    for version in versions_for(task, mutations, verify_with_mucoco):
        entry: dict = {"func_name": version.func_name}
        if version.error:
            entry["skipped"] = version.error
            row["versions"][version.name] = entry
            continue

        try:
            constructs = find_constructs(version.code)
            prof = profile_call(version.code, version.func_name, task.test_input,
                                constructs, **kwargs)
        except Exception as error:
            entry["skipped"] = f"trace failed: {type(error).__name__}: {error}"
            row["versions"][version.name] = entry
            continue

        entry["code"] = version.code          # scoring matches replies back by program text
        entry["constructs"] = [c.__dict__ for c in constructs]
        entry["profile"] = prof.as_dict()
        row["versions"][version.name] = entry
        profiles[version.name] = prof

    row["status"] = "ok" if "no_mutation" in profiles else "original failed"
    row["mutation_warnings"] = check_mutants(profiles)
    return row, profiles


def check_mutants(profiles: dict[str, Profile]) -> list[str]:
    """A semantics-preserving mutation must not change the answer or the control flow."""
    problems = []
    base = profiles.get("no_mutation")
    if base is None:
        return problems
    for name, prof in profiles.items():
        if name == "no_mutation":
            continue
        if repr(prof.output) != repr(base.output):
            problems.append(f"{name}: output differs")
        elif not shared_counts_match(base, prof):
            problems.append(f"{name}: profile differs")
    return problems


def summarise(rows: list[dict]) -> str:
    stats = Counter()
    for row in rows:
        stats["tasks"] += 1
        if row["status"] != "ok":
            stats["original failed"] += 1
            continue

        base = row["versions"]["no_mutation"]
        prof, constructs = base["profile"], base["constructs"]
        stats["loops"] += len(prof["loops"])
        stats["branches"] += len(prof["branches"])
        stats["one-liners excluded"] += sum(1 for c in constructs if not c["measurable"])
        stats["loops running more than once"] += sum(1 for n in prof["loops"].values() if n > 1)
        if prof["branches"]:
            stats["tasks with a branch"] += 1
            if any(t == 0 or e == 0 for t, e in prof["branches"].values()):
                stats["tasks where one branch side never ran"] += 1

        for name, version in row["versions"].items():
            if name == "no_mutation":
                continue
            if "skipped" in version:
                stats[f"{name}: not applicable"] += 1
            else:
                stats[f"{name}: traced"] += 1
        for problem in row.get("mutation_warnings", []):
            stats[f"WARNING {problem}"] += 1

    width = max(len(key) for key in stats)
    return "\n".join(f"{key.ljust(width)}  {value}" for key, value in stats.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("mongo", "jsonl"), default="mongo")
    parser.add_argument("--jsonl", help="path to cruxeval.jsonl when --source jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mutations", nargs="*", default=["sequential", "constant_unfold_add"],
                        help="MuCoCo operator names, as spelled in utility/constants.py")
    parser.add_argument("--out", default="results/truth/cruxeval.jsonl")
    parser.add_argument("--step-budget", type=int, default=200_000)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--verify-with-mucoco", action="store_true",
                        help="also run MuCoCo's own validity check on each mutant. Much "
                             "slower: it spawns a subprocess per mutant")
    args = parser.parse_args()

    if args.source == "mongo":
        tasks = load_from_mongo(args.limit)
    else:
        if not args.jsonl:
            parser.error("--source jsonl needs --jsonl")
        tasks = load_from_jsonl(args.jsonl, args.limit)
    print(f"{len(tasks)} tasks | mutations: {', '.join(args.mutations)}")

    rows = []
    for task in tasks:
        row, _ = process(task, args.mutations, verify_with_mucoco=args.verify_with_mucoco,
                         step_budget=args.step_budget, timeout_s=args.timeout)
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(rows)} tasks to {out_path}\n")
    print(summarise(rows))


if __name__ == "__main__":
    main()
