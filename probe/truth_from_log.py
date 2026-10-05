"""Build the ground truth from the programs the model was actually shown.

Until now a mutant was scored by regenerating it locally and matching it to the logged
reply by program text. That silently requires the two copies to be character-identical,
and it fails outright for `random`, whose renaming picks fresh names on every run: the
model saw one program and the truth builder traced another, so nothing matched and 800
calls scored as "no reply".

The log already holds the exact program in the prompt and the exact input, so the tracer
can run those instead. Nothing has to match, which removes the assumption entirely rather
than working around it, and makes scoring possible for mutations that change behaviour,
where an original's truth does not apply at all.

    python -m probe.truth_from_log --log results/llama_sweep/profiles_random.jsonl \\
        --version random --out results/truth/random_from_log.jsonl

The output is in the same shape as `build_truth_mucoco` writes, so every scoring tool
reads it unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from probe.constructs import find_constructs
from probe.dataset import entry_function
from probe.stats import rate
from probe.truth import profile_call


def recover_input(text: str):
    """The log stores the input as text; turn it back into the value it was."""
    if text is None:
        return None, "no input recorded"
    try:
        return ast.literal_eval(text), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def trace_one(code: str, test_input, step_budget: int = 200_000) -> tuple[dict | None, str | None]:
    """Trace one logged program, returning the same shape build_truth_mucoco writes."""
    try:
        func_name = entry_function(code)
    except Exception as error:
        return None, f"no entry function: {error}"

    try:
        constructs = find_constructs(code)
    except SyntaxError as error:
        return None, f"could not parse: {error}"

    try:
        profile = profile_call(code, func_name, test_input, constructs,
                               step_budget=step_budget)
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"

    return {
        "func_name": func_name,
        "code": code,
        "constructs": [
            {"id": c.id, "kind": c.kind, "header_line": c.header_line,
             "body_line": c.body_line, "else_line": c.else_line,
             "header_text": c.header_text, "measurable": c.measurable}
            for c in constructs
        ],
        # Profile.as_dict is what build_truth_mucoco writes, so the two files are the
        # same shape and every scoring tool reads either without knowing the difference.
        "profile": profile.as_dict(),
    }, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, nargs="+",
                        help="one or more profile logs")
    parser.add_argument("--version", required=True, nargs="+",
                        help="the version name each log belongs to, in the same order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step-budget", type=int, default=200_000)
    args = parser.parse_args()

    if len(args.log) != len(args.version):
        parser.error("give one --version for each --log, in the same order")

    rows: dict[str, dict] = {}
    outcomes: Counter = Counter()
    failures: Counter = Counter()

    for log_path, version in zip(args.log, args.version):
        path = Path(log_path)
        if not path.exists():
            raise SystemExit(f"no such log: {path}")

        for index, line in enumerate(open(path, encoding="utf-8")):
            entry = json.loads(line)
            code = entry.get("full_sol")
            if not code:
                outcomes["no program in the log"] += 1
                continue

            test_input, problem = recover_input(entry.get("test_input"))
            if problem:
                outcomes["input could not be read back"] += 1
                failures[problem[:60]] += 1
                continue

            traced, problem = trace_one(code, test_input, args.step_budget)
            if traced is None:
                outcomes["could not be traced"] += 1
                failures[problem[:60]] += 1
                continue

            # The log carries no task id, so one is made from the reply's position. It is
            # only used to pair a construct with its counts, never to join to another file.
            task_id = f"{version}#{index}"
            rows.setdefault(task_id, {"task_id": task_id, "versions": {}})
            rows[task_id]["versions"][version] = traced
            outcomes["traced"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for row in rows.values():
            handle.write(json.dumps(row) + "\n")

    total = sum(outcomes.values())
    print(f"=== {total} logged replies ===")
    for name, count in outcomes.most_common():
        print(f"  {name:28s} {rate(count, total)}")
    if failures:
        print("\n  why tracing failed")
        for reason, count in failures.most_common(6):
            print(f"    {count:5d}  {reason}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
