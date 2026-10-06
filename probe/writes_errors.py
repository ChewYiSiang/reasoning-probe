"""Why a variable update was wrong, and which kinds of variable models lose track of.

The control-flow side reports the kind of mistake, how far out it was and sometimes its
cause, which is what makes it useful to a reader rather than just a rate. This does the same
for the data-flow side.

Two things are reported.

**What kind of mistake it was.** The errors are not mostly off-by-one slips, as they are for
loop counts. A model says `b = 372359` when the variable changed twice, and 372359 is what
`b` ended up holding. So the first question is whether the model answered a different
question: the variable's value, or its length, rather than how many times it changed.

**Which kinds of variable go wrong.** A counter built with `+=`, a list built by `append`, a
string built by concatenation and a dictionary filled by subscript are four different jobs,
and a model may track some and not others. The role is read from the syntax tree.

    python -m probe.writes_errors --writes results/truth/writes.jsonl \\
        --log results/qwen14b_writes/profiles_no_mutation.jsonl --version no_mutation
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from probe.dataflow import parse_writes, trace_writes
from probe.score import load_predictions
from probe.stats import rate

UNKNOWN = "miscounted, no pattern"


# --------------------------------------------------------------------------- the mistake

def sized(value) -> int | None:
    """len(value) where that means something, ignoring numbers and booleans."""
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        try:
            return len(value)
        except TypeError:
            return None
    return None


def kind_of_error(said: int, really: int, final_value) -> str:
    """One label per wrong count. Checked in this order; the first that fits wins.

    Out by one comes first on purpose. A counter written as `n = 0` then `n += 1` per pass
    has a variable update of one more than its final value, every time, so "said the value" and
    "out by one" would otherwise coincide for every counter in the benchmark and the value
    category would swallow them. The simpler explanation wins when both fit.
    """
    if said == really:
        return "not a mistake"      # the caller skips these; guarded so reuse cannot mislead

    if said == really - 1 or said == really + 1:
        return "out by one"

    if isinstance(final_value, bool):
        final_value = None                      # True == 1 would match far too easily

    # Only a match above 1 is informative. Saying 0 or 1 is a common failure on its own, and
    # a one-element list or a one-character string has length 1, so without this guard every
    # "said 1" against a short container was credited to the length category.
    if isinstance(final_value, int) and said == final_value and said > 1:
        return "gave the variable's value, not how often it changed"

    # the same mistake when the variable holds the digits as text: b ended as '372359'
    if isinstance(final_value, str) and said > 1 and final_value.strip() == str(said):
        return "gave the variable's value, not how often it changed"

    length = sized(final_value)
    if length is not None and said == length and said > 1:
        return "gave the variable's length, not how often it changed"

    if said == 1 and really > 1:
        return "said it changed once when it changed many times"
    if said == 0 and really > 0:
        return "said it never changed when it did"
    if said > really * 2:
        return "far too high"
    if said * 2 < really:
        return "far too low"
    return UNKNOWN


# --------------------------------------------------------------------------- the variable

# Weakest last. A variable is nearly always assigned once at the top before being used, so
# first-match ordering labelled everything "reassigned outright"; the strongest signal wins.
ROLE_PRIORITY = [
    "the loop's own variable",
    "collection built by a method call",
    "counter stepped by a constant",
    "accumulated with +=",
    "accumulated with another operator",
    "filled by index or key",
    "assigned as part of a tuple",
    "reassigned outright",
]


def roles(code: str) -> dict[str, str]:
    """What each variable is used for, read from the syntax tree.

    Every signal is collected and the strongest kept, so `total = 0` followed by `total += 1`
    inside a loop is a counter rather than a plain assignment.
    """
    signals: dict[str, set[str]] = {}

    def note(name: str, role: str) -> None:
        signals.setdefault(name, set()).add(role)

    for node in ast.walk(ast.parse(code)):
        # out.append(x), seen.add(x): a collection being built up
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name)
                    and node.func.attr in {"append", "add", "extend", "insert", "update", "pop"}):
                note(node.func.value.id, "collection built by a method call")
        # total += 1, out += ch
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.op, ast.Add) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, (int, float)):
                note(node.target.id, "counter stepped by a constant")
            elif isinstance(node.op, ast.Add):
                note(node.target.id, "accumulated with +=")
            else:
                note(node.target.id, "accumulated with another operator")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    base = target.value
                    while isinstance(base, ast.Subscript):
                        base = base.value
                    if isinstance(base, ast.Name):
                        note(base.id, "filled by index or key")
                elif isinstance(target, ast.Name):
                    note(target.id, "reassigned outright")
                elif isinstance(target, ast.Tuple):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            note(element.id, "assigned as part of a tuple")
        elif isinstance(node, ast.For):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    note(name.id, "the loop's own variable")

    return {name: next(role for role in ROLE_PRIORITY if role in found)
            for name, found in signals.items()}


# --------------------------------------------------------------------------- the report

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--writes", required=True, help="the file written by dataflow build")
    parser.add_argument("--log", required=True, nargs="+")
    parser.add_argument("--version", required=True, nargs="+")
    parser.add_argument("--examples", type=int, default=4)
    args = parser.parse_args()

    if len(args.log) != len(args.version):
        parser.error("give one --version for each --log, in the same order")

    wanted: dict[tuple[str, str], dict] = {}
    with open(args.writes, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                wanted[(row["task_id"], version)] = entry

    kinds: Counter = Counter()
    by_role: dict[str, list[int]] = {}
    examples: dict[str, list[str]] = {}
    could_not_trace = 0

    for log_path, version in zip(args.log, args.version):
        by_code = {entry["code"]: task for (task, ver), entry in wanted.items() if ver == version}
        for prediction in load_predictions(Path(log_path)):
            task = by_code.get(prediction["full_sol"])
            if task is None:
                continue
            entry = wanted[(task, version)]
            if not entry["asked"]:
                continue
            claimed = parse_writes(prediction.get("raw_reply") or "")

            # the final values, needed to tell a miscount from an answer to a different question
            try:
                test_input = ast.literal_eval(prediction.get("test_input") or "")
                from probe.dataset import entry_function
                _, final = trace_writes(entry["code"], entry_function(entry["code"]), test_input)
            except Exception:
                could_not_trace += 1
                final = {}

            where = roles(entry["code"])
            for name in entry["asked"]:
                really = entry["informative"][name]
                role = where.get(name, "something else")
                right, seen = by_role.get(role, (0, 0))
                if name not in claimed:
                    by_role[role] = (right, seen)       # unanswered counts in neither column
                    continue
                said = claimed[name]
                by_role[role] = (right + (said == really), seen + 1)
                if said == really:
                    continue
                label = kind_of_error(said, really, final.get(name))
                kinds[label] += 1
                if len(examples.setdefault(label, [])) < args.examples:
                    value = final.get(name)
                    shown = repr(value)[:40] if value is not None else "not traced"
                    examples[label].append(
                        f"{task} {name}: said {said}, really {really}, ended as {shown}")

    total = sum(kinds.values())
    if not total:
        print("nothing to classify: check the file names and versions")
        return

    print(f"=== {total} wrong variable updates, by kind of mistake ===")
    for label, count in kinds.most_common():
        print(f"  {label:52s} {rate(count, total)}")
    if could_not_trace:
        print(f"\n  ({could_not_trace} programs could not be re-traced, so the value checks "
              f"were skipped for them)")

    print("\n=== accuracy by what the variable is used for ===")
    for role, (right, seen) in sorted(by_role.items(), key=lambda item: -item[1][1]):
        if seen:
            print(f"  {role:40s} {rate(right, seen)}")

    for label, lines in examples.items():
        if label == UNKNOWN:
            continue
        print(f"\n  {label}")
        for line in lines:
            print(f"    {line}")


if __name__ == "__main__":
    main()
