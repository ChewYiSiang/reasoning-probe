"""Build and score the write-count ground truth.

`build` traces every task and every mutant a second time, recording how many times each
variable changed, then keeps only the variables worth asking about. `score` compares a run's
answers against it. Both are reached through `python -m probe.dataflow`.
"""

from __future__ import annotations

import json
from pathlib import Path

from probe.dataflow import checklist, informative, parse_writes, score, write_counts
from probe.score import load_predictions, load_truth
from probe.stats import rate


def build_writes(args) -> None:
    from probe.constructs import find_constructs
    from probe.mucoco import mutate
    from probe.truth import profile_call

    truth = load_truth(args.truth)

    # The truth builder already knows how to read both sources; use its loaders rather
    # than a second, slightly different copy of the same logic.
    from probe.build_truth_mucoco import load_from_jsonl, load_from_mongo

    if args.source == "mongo":
        tasks = load_from_mongo(args.limit)
    else:
        if not args.tasks:
            raise SystemExit("--source jsonl needs --tasks pointing at the CRUXEval file")
        tasks = load_from_jsonl(args.tasks, args.limit)

    rows, traced, skipped = [], 0, 0
    for task in tasks:
        versions = {"no_mutation": (task.code, task.func_name)}
        for mutation in args.mutations:
            try:
                mutated = mutate(task, mutation)
            except Exception:
                continue
            if mutated is not None:
                versions[mutation] = (mutated.code, mutated.func_name)

        entry = {"task_id": task.task_id, "versions": {}}
        for version, (code, func_name) in versions.items():
            key = (task.task_id, version)
            if key not in truth:
                continue
            try:
                counts = write_counts(code, func_name, task.test_input)
            except Exception:
                skipped += 1
                continue
            useful = informative(counts, truth[key]["counts"], code)
            entry["versions"][version] = {
                "code": code,
                "all_writes": counts,
                "informative": useful,
                "asked": checklist(useful),
            }
            traced += 1
        if entry["versions"]:
            rows.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    asked = sum(len(v["asked"]) for row in rows for v in row["versions"].values())
    every = sum(len(v["all_writes"]) for row in rows for v in row["versions"].values())
    useful = sum(len(v["informative"]) for row in rows for v in row["versions"].values())
    print(f"traced {traced} versions across {len(rows)} tasks ({skipped} could not be traced)")
    print(f"  variables seen:                  {every}")
    print(f"  carrying new information:        {rate(useful, every)}")
    print(f"  asked about (at most 3 a task):  {asked}")
    print(f"written to {out}")


def score_writes(args) -> None:
    rows = {}
    with open(args.writes, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                rows[(row["task_id"], version)] = entry

    outcomes = {"correct": 0, "wrong": 0, "not answered": 0, "no reply": 0}
    errors = []

    for log_path, version in zip(args.log, args.version):
        by_code = {entry["code"]: task for (task, ver), entry in rows.items()
                   if ver == version}
        answered = {}
        for prediction in load_predictions(Path(log_path)):
            task = by_code.get(prediction["full_sol"])
            if task is not None:
                answered[task] = parse_writes(prediction.get("raw_reply") or "")

        for (task, ver), entry in rows.items():
            if ver != version or not entry["asked"]:
                continue
            wanted = {name: entry["informative"][name] for name in entry["asked"]}
            claimed = answered.get(task)
            if claimed is None:
                outcomes["no reply"] += len(wanted)
                continue
            for name, verdict in score(claimed, wanted).items():
                outcomes[verdict] += 1
                if verdict == "wrong":
                    errors.append((task, name, claimed.get(name), wanted[name]))

    total = sum(outcomes.values())
    print(f"=== write counts: {total} variables asked about ===")
    for name, count in outcomes.items():
        print(f"  {name:14s} {rate(count, total)}")
    if errors:
        print("\n  the worst of them")
        for task, name, said, really in sorted(errors, key=lambda e: -abs((e[2] or 0) - e[3]))[:8]:
            print(f"    {task} {name}: said {said}, really {really}")
