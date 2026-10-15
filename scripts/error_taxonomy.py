"""The error taxonomy: where a failed trace first parts from the run, and what kind of
failure it was.

Following Abdollahi et al. 2025, who built the same taxonomy by hand. Ours is rules, so it
runs on every failure rather than a sample. The rules are first-match-wins and the order is
a judgement: a mechanism before its consequence. An omitted step makes a branch count wrong,
so testing the count first would report control flow where the diagnosis is a skipped
statement. Changing the order changes the shares.

    python scripts/error_taxonomy.py --results <folder> --truth <traces.jsonl>
"""
import argparse
import ast
from collections import Counter

from _shared import MARK, add_arguments, align, counts_from, load_all, parse_trace


def statement_kinds(code):
    """{line: the kind of statement on it}, from the syntax tree.

    Headers claim their line outright; assignments only claim a line nobody else has, since
    a `for` line also contains Name nodes.
    """
    kinds = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return kinds
    for node in ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            kinds.setdefault(node.lineno, "assignment")
        elif isinstance(node, ast.Expr):
            kinds.setdefault(node.lineno, "call with no assignment")
        elif isinstance(node, ast.For):
            kinds[node.lineno] = "for header"
        elif isinstance(node, ast.While):
            kinds[node.lineno] = "while header"
        elif isinstance(node, ast.If):
            kinds[node.lineno] = "if header"
        elif isinstance(node, ast.Return):
            kinds.setdefault(node.lineno, "return")
        elif isinstance(node, ast.FunctionDef):
            kinds.setdefault(node.lineno, "parameter binding")
    return kinds


def where_it_diverged(claimed, real, code):
    """(statement kind, whether the path or a value went wrong), or None.

    A divergence at the last step or the one before is reported as stopping early rather
    than as a return error: 87 of 101 return divergences were there, and running out before
    the end is a different diagnosis from misreading the return itself.
    """
    result = align(claimed, real)
    position, nature = result["first_path_error"], "took the wrong path"
    if position is None:
        position, nature = result["first_value_error"], "right path, wrong value"
    if position is None:
        return None
    lineno = real[position][0] if position < len(real) else real[-1][0]
    kind = statement_kinds(code).get(lineno, "other")
    if kind == "return" and position >= len(real) - 2:
        kind = "stopped before the end"
    return kind, nature


def why_it_failed(claimed, real, code, answer_correct):
    """The category, or None when nothing went wrong."""
    result = align(claimed, real)
    path_exact = result["path_exact"]
    values_wrong = result["values_wrong"] > 0

    if path_exact and not values_wrong and answer_correct:
        return None                       # not a failure, so not in the taxonomy

    claimed_lines = {ln for ln, _ in claimed}
    real_lines = {ln for ln, _ in real}
    source_lines = set(statement_kinds(code))

    if claimed_lines - source_lines - real_lines:
        return "hallucination", "partial"

    # a parameter reported with the wrong value, only where the tracer never reassigns that
    # name: comparing a later value would catch a reassignment rather than a misread input
    if real:
        for name, real_value in real[0][1].items():
            if sum(1 for _, changed in real if name in changed) > 1:
                continue
            claimed_value = next((changed[name] for _, changed in claimed
                                  if name in changed), None)
            if claimed_value is not None and claimed_value != real_value:
                return "input misread", "exact"

    if path_exact and not values_wrong and not answer_correct:
        return "misreporting the final output", "exact"

    # shorter than the real trace and using only real lines, so steps were left out rather
    # than the model going somewhere else. Counting absent LINES does not work: a loop body
    # line is almost never absent entirely, only one occurrence is dropped
    if not path_exact and len(claimed) < len(real) and claimed_lines <= real_lines:
        return "skipping statements", "exact"

    claimed_counts = counts_from(claimed, code)
    real_counts = counts_from(real, code)
    if any(claimed_counts.get(n) != c for n, c in real_counts.items() if n.startswith("if")):
        return "control flow", "exact"
    if any(claimed_counts.get(n) != c for n, c in real_counts.items() if n.startswith("loop")):
        return "indexing", "partial"

    if path_exact and values_wrong:
        position = result["first_value_error"]
        if position is not None and position < len(real):
            lines = code.splitlines()
            lineno = real[position][0]
            text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if any(f"{name}(" in text for name in
                   ("len", "sorted", "str", "int", "list", "dict", "set", "range",
                    "enumerate", "zip", "map", "filter", "sum", "max", "min", "abs")):
                return "native API misevaluation", "partial"

    if values_wrong:
        return "computation", "exact"
    return "unclassified", "exact"


def main():
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    where, why = Counter(), Counter()
    scored = no_failure = 0
    for code, reply in originals_replies.items():
        found = by_code.get(("no_mutation", code))
        if not found:
            continue
        task, entry = found
        text = reply.get("raw_reply") or ""
        if MARK not in text:
            continue
        claimed = parse_trace(text)
        real = parse_trace(entry["text"])
        if not claimed:
            continue

        scored += 1
        answer_ok = originals_csv.get(task, ("", False))[1]
        outcome = why_it_failed(claimed, real, entry["code"], answer_ok)
        if outcome is None:
            no_failure += 1
            continue
        why[outcome] += 1
        diverged = where_it_diverged(claimed, real, entry["code"])
        if diverged:
            where[diverged] += 1

    failures = scored - no_failure
    print(f"\n  {scored} traces, {no_failure} with nothing wrong, {failures} failures\n")

    print(f"  Where the reasoning first parts from the real run\n")
    print(f"    {'statement the model was at':<30}{'wrong path':>13}"
          f"{'wrong value':>14}{'failures':>11}")
    kinds = sorted({k for k, _ in where},
                   key=lambda k: -sum(n for (kind, _), n in where.items() if kind == k))
    for kind in kinds:
        a = where[(kind, "took the wrong path")]
        b = where[(kind, "right path, wrong value")]
        print(f"    {kind:<30}{a:>13}{b:>14}{a + b:>11}")
    path_total = sum(n for (_, nature), n in where.items() if nature == "took the wrong path")
    total = sum(where.values())
    print(f"    {'all':<30}{path_total:>13}{total - path_total:>14}{total:>11}")
    print(f"\n    {failures - total} failures have no divergence in the trace itself:")
    print(f"    the answer was wrong and the trace was not")

    print(f"\n\n  What kind of failure\n")
    print(f"    {'category':<34}{'detection':<12}{'of ' + str(failures) + ' failures':>18}")
    for (category, detection), n in why.most_common():
        print(f"    {category:<34}{detection:<12}{n:>8}  {n / failures:>7.1%}")


if __name__ == "__main__":
    main()
