"""Why did this count come out wrong?

The taxonomy in `errors.py` says what shape the mistake had. This says, where the evidence
is unambiguous, what caused it. Four causes, all decided from the code, the input and the
traced counts, with no reading of the model's prose.

  the number written in the code   the model gave the figure a reader can see in the source,
                                   such as the 200 in range(200), rather than the number of
                                   times the loop really ran
  the inner loop's count           the model gave the inner loop's figure for the outer one
  one too many or one too few      the model gave the written figure plus or minus one
  follows from an earlier mistake  the construct sits inside a loop or branch the model had
                                   already got wrong, so the mistake may not be its own

Anything else is reported as "wrong, cause unknown". A wrong cause is worse than none, so a
cause is only given when nothing else fits, and the share we could explain is always printed
beside the share we could not.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

from functools import lru_cache

from probe.constructs import find_constructs
from probe.errors import classify
from probe.score import load_predictions, load_truth
from probe.stats import rate

UNKNOWN = "wrong, cause unknown"


def written_number(code: str, header_line: int, test_input, parameters: list[str]) -> int | None:
    """The figure a reader would take from the source without running it.

    Deliberately narrow. The length of the input only counts when the function takes a
    single argument and the loop walks that argument, because a multi-argument input is a
    tuple whose length means nothing here. Guessing produces false explanations, which are
    worse than none.
    """
    tree = ast.parse(code)
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.For) and n.lineno == header_line), None)
    if node is None:
        return None

    single_argument = len(parameters) == 1

    def length_of_input() -> int | None:
        if not single_argument:
            return None
        try:
            return len(test_input)
        except TypeError:
            return None

    iterated = node.iter
    if isinstance(iterated, ast.Name):
        return length_of_input() if iterated.id in parameters else None

    if (isinstance(iterated, ast.Call) and isinstance(iterated.func, ast.Name)
            and iterated.func.id == "range" and len(iterated.args) == 1):
        argument = iterated.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, int):
            return argument.value
        if (isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name)
                and argument.func.id == "len" and len(argument.args) == 1
                and isinstance(argument.args[0], ast.Name)
                and argument.args[0].id in parameters):
            return length_of_input()
    return None


@lru_cache(maxsize=4096)
def constructs_of(code: str) -> tuple[dict, ...]:
    """The loops and branches of a program, in the shape this module wants them."""
    return tuple({"id": c.id, "kind": c.kind, "header_line": c.header_line}
                 for c in find_constructs(code))


def parameters_of(code: str) -> list[str]:
    """The function's argument names, needed before the length of an input means anything."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return [argument.arg for argument in node.args.args]
    return []


def enclosing_constructs(code: str, constructs: list[dict], name: str) -> list[str]:
    """Which constructs this one sits inside, worked out from the source line ranges."""
    target = next((c for c in constructs if c["id"] == name), None)
    if target is None:
        return []
    spans = {}
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While, ast.If)):
            spans[node.lineno] = (node.lineno, getattr(node, "end_lineno", node.lineno))

    target_span = spans.get(target["header_line"])
    if target_span is None:
        return []
    inside = []
    for other in constructs:
        if other["id"] == name:
            continue
        span = spans.get(other["header_line"])
        if span and span[0] < target_span[0] and span[1] >= target_span[1]:
            inside.append(other["id"])
    return inside


def as_value(test_input):
    """The log keeps the input as text; the length of a string is not the length of a list."""
    if isinstance(test_input, str):
        try:
            return ast.literal_eval(test_input)
        except Exception:
            return None
    return test_input


def explain(name: str, claimed: tuple[int, ...], real: tuple[int, ...], entry: dict,
            test_input, wrong_elsewhere: set[str],
            claimed_elsewhere: dict[str, tuple[int, ...]] | None = None) -> str:
    """One cause, or UNKNOWN. Only one cause is ever given."""
    claimed_elsewhere = claimed_elsewhere or {}
    constructs = list(constructs_of(entry["code"]))
    construct = next((c for c in constructs if c["id"] == name), None)
    if construct is None:
        return UNKNOWN

    # inside something the model already got wrong: the mistake may not be its own
    outer_wrong = [outer for outer in enclosing_constructs(entry["code"], constructs, name)
                   if outer in wrong_elsewhere]
    if outer_wrong:
        # the strong case: the numbers given here match the wrong figure given for the loop
        # around them, so the earlier mistake was carried forward rather than made again
        if len(claimed) == 2:
            said_total = claimed[0] + claimed[1]
            for outer in outer_wrong:
                said_outer = claimed_elsewhere.get(outer)
                if said_outer and len(said_outer) == 1 and said_total == said_outer[0]:
                    return "carried forward from the loop count it already got wrong"
        return "sits inside a loop or branch that was also wrong"

    if construct["kind"] != "loop":
        return UNKNOWN

    said, really = claimed[0], real[0]
    parameters = parameters_of(entry["code"])
    written = written_number(entry["code"], construct["header_line"], test_input, parameters)
    if written is not None and written != really:
        if said == written:
            return "gave the number written in the code"
        if abs(said - written) == 1:
            return "one off the number written in the code"

    # the outer loop given the inner loop's figure, or the other way round
    for other in constructs:
        if other["id"] == name or other["kind"] != "loop":
            continue
        other_real = real_counts_of(entry, other["id"])
        if other_real is not None and said == other_real and other_real != really:
            inner = other["header_line"] > construct["header_line"]
            return ("gave the inner loop's count for the outer loop" if inner
                    else "gave the outer loop's count for the inner loop")
    return UNKNOWN


def real_counts_of(entry: dict, name: str) -> int | None:
    counts = entry["counts"].get(name)
    return counts[0] if counts and len(counts) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--probe-log", required=True, nargs="+")
    parser.add_argument("--version", required=True, nargs="+")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    if len(args.probe_log) != len(args.version):
        parser.error("give one --version for each --probe-log, in the same order")

    truth = load_truth(args.truth)
    causes: Counter = Counter()
    examples: dict[str, list[str]] = {}
    total = unreadable_inputs = 0

    for log_path, version in zip(args.probe_log, args.version):
        by_code = {value["code"]: key for key, value in truth.items()
                   if value["code"] and key[1] == version}
        for prediction in load_predictions(Path(log_path)):
            key = by_code.get(prediction["full_sol"])
            if key is None:
                continue
            entry = truth[key]
            claimed_all = prediction.get("profile") or {}
            test_input = as_value(prediction.get("test_input"))
            if test_input is None and prediction.get("test_input") not in (None, "None"):
                unreadable_inputs += 1

            wrong = {name for name, real in entry["counts"].items()
                     if name in claimed_all
                     and classify(name, tuple(claimed_all[name]), tuple(real)) is not None}

            for name, real in entry["counts"].items():
                claimed = claimed_all.get(name)
                if claimed is None or classify(name, tuple(claimed), tuple(real)) is None:
                    continue
                total += 1
                others = wrong - {name}
                cause = explain(name, tuple(claimed), tuple(real), entry, test_input,
                                others, {other: tuple(value) for other, value
                                         in claimed_all.items()})
                causes[cause] += 1
                if len(examples.setdefault(cause, [])) < args.examples:
                    examples[cause].append(
                        f"{key[0]} {name}: said {tuple(claimed)}, really {tuple(real)}")

    if unreadable_inputs:
        print(f"note: the input could not be read back for {unreadable_inputs} replies, so the "
              f"checks that need it were skipped for those")

    print(f"=== {total} wrong counts, by cause ===")
    explained = total - causes[UNKNOWN]
    for cause, count in causes.most_common():
        print(f"  {cause:52s} {rate(count, total)}")
    print(f"\n  explained: {rate(explained, total)}")

    for cause, lines in examples.items():
        if cause == UNKNOWN:
            continue
        print(f"\n  {cause}")
        for line in lines:
            print(f"    {line}")


if __name__ == "__main__":
    main()
