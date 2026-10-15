"""What a developer can check without running the program.

Six checks. The source supplies the statements, the control-flow graph and the branch
conditions; the model's own trace supplies the values. Nothing executes the program: the
statement check runs one statement against the state the model itself claimed.

Each check is reported separately, because they are not mutually exclusive and a profile of
what fired says more than one label.

    python scripts/tracer_free_checks.py --results <folder> --truth <traces.jsonl>
"""
import argparse
import ast
import copy
from collections import Counter

from probe.constructs import find_constructs

from _shared import MARK, add_arguments, counts_from, load_all, parse_trace
from error_taxonomy import why_it_failed

# Only the builtins a traced program plausibly uses, so a statement cannot reach anything else
SAFE = {n: (getattr(__builtins__, n, None) if not isinstance(__builtins__, dict)
            else __builtins__.get(n))
        for n in ("len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
                  "sorted", "reversed", "enumerate", "zip", "range", "sum", "max", "min",
                  "abs", "round", "any", "all", "map", "filter", "ord", "chr", "repr",
                  "isinstance", "type")}


def successors(code):
    """{line: the lines that may run next}, from the source alone."""
    edges = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return edges

    def walk(body, after):
        for index, node in enumerate(body):
            line = node.lineno
            following = ({body[index + 1].lineno} if index + 1 < len(body) else set(after))
            out = edges.setdefault(line, set())
            if isinstance(node, (ast.For, ast.While)):
                out |= {node.body[0].lineno} | following
                if node.orelse:
                    out.add(node.orelse[0].lineno)
                walk(node.body, {line})            # the body loops back to the header
                if node.orelse:
                    walk(node.orelse, following)
            elif isinstance(node, ast.If):
                out.add(node.body[0].lineno)
                if node.orelse:
                    out.add(node.orelse[0].lineno)
                else:
                    out |= following
                walk(node.body, following)
                if node.orelse:
                    walk(node.orelse, following)
            elif isinstance(node, (ast.Return, ast.Raise)):
                pass                               # nothing follows
            elif isinstance(node, ast.FunctionDef):
                out.add(node.body[0].lineno)
                walk(node.body, set())
            else:
                out |= following
    walk(tree.body, set())
    return edges


def uses_jump(code):
    """break and continue are approximated above, so a hit on such a program is weaker."""
    try:
        return any(isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(ast.parse(code)))
    except Exception:
        return True


def illegal_transition(code, claimed):
    edges = successors(code)
    for position in range(1, len(claimed)):
        allowed = edges.get(claimed[position - 1][0])
        if allowed and claimed[position][0] not in allowed:
            return position
    return None


def branch_arms(code):
    """{line inside a branch: (its test, which arm, the header line)}.

    Every line of each arm, not only the one after the header: the compact format makes the
    header step optional, so requiring it caught almost nothing.
    """
    out = {}
    try:
        tree = ast.parse(code)
    except Exception:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for arm, body in (("true", node.body), ("false", node.orelse)):
            for statement in body:
                for inner in ast.walk(statement):
                    line = getattr(inner, "lineno", None)
                    if line is not None:
                        out.setdefault(line, (node.test, arm, node.lineno))
    return out


def wrong_arm(code, claimed):
    """A step inside a branch whose condition, on the model's own values, required the other."""
    arms = branch_arms(code)
    state = {}
    for position, (lineno, changed) in enumerate(claimed):
        found = arms.get(lineno)
        if found is not None:
            test, arm, _ = found
            namespace = {"__builtins__": SAFE}
            try:
                namespace.update(copy.deepcopy(state))
                verdict = bool(eval(compile(ast.Expression(test), "<cond>", "eval"),
                                    namespace))
            except Exception:
                verdict = None
            if verdict is not None and verdict != (arm == "true"):
                return position
        state.update(changed)
    return None


def statement_at(code, lineno):
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) == lineno and isinstance(
                node, (ast.Assign, ast.AugAssign, ast.Expr, ast.Return)):
            return node
    return None


def bad_step(code, claimed):
    """(the first failing step, how many fail, how many were checkable).

    State is carried forward, since a compact scratchpad omits whatever did not change.
    """
    state, first, failing, checked = {}, None, 0, 0
    for position, (lineno, changed) in enumerate(claimed):
        node = statement_at(code, lineno)
        if node is not None and not isinstance(node, ast.Return) and changed:
            namespace = {"__builtins__": SAFE}
            try:
                namespace.update(copy.deepcopy(state))
                exec(compile(ast.unparse(node), "<step>", "exec"), namespace)
            except Exception:
                namespace = None
            if namespace is not None and all(n in namespace for n in changed):
                checked += 1
                if any(namespace[n] != v for n, v in changed.items()):
                    failing += 1
                    first = first if first is not None else position
        state.update(changed)
    return first, failing, checked


def return_disagrees(code, claimed, answer_text):
    """Does the return expression on the trace's final state match the answer given?"""
    try:
        tree = ast.parse(code)
    except Exception:
        return None
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None]
    if len(returns) != 1:
        return None
    state = {}
    for _, changed in claimed:
        state.update(changed)
    if not state:
        return None
    namespace = {"__builtins__": SAFE}
    try:
        namespace.update(copy.deepcopy(state))
        implied = eval(compile(ast.Expression(returns[0].value), "<ret>", "eval"), namespace)
    except Exception:
        return None
    answer = (answer_text or "").strip()
    if answer.startswith("(") and "," in answer:
        answer = answer[1:answer.rindex(",")].strip()      # their (value, type) wrapper
    try:
        given = ast.literal_eval(answer)
    except Exception:
        return None
    return implied != given


def invented_line(code, claimed):
    try:
        source_lines = {n.lineno for n in ast.walk(ast.parse(code)) if hasattr(n, "lineno")}
    except Exception:
        return False
    return bool({ln for ln, _ in claimed} - source_lines)


def numbers_contradict(code, claimed):
    """A branch inside a loop is tested once per pass, so its counts must sum to the loop's."""
    try:
        tree, constructs = ast.parse(code), find_constructs(code)
    except Exception:
        return False
    by_line = {c.header_line: c for c in constructs}
    pairs = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        loop = by_line.get(node.lineno)
        if loop is None:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.If) and inner.lineno != node.lineno:
                branch = by_line.get(inner.lineno)
                if branch is not None:
                    pairs.setdefault(branch.id, loop.id)
    counts = counts_from(claimed, code)
    for branch_id, loop_id in pairs.items():
        branch, loop = counts.get(branch_id), counts.get(loop_id)
        if branch and loop and len(branch) == 2 and sum(branch) != loop[0]:
            return True
    return False


def main():
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    fired, profile, cascade = Counter(), Counter(), Counter()
    failures = 0
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
        answer_text, answer_ok = originals_csv.get(task, ("", False))
        if why_it_failed(claimed, real, entry["code"], answer_ok) is None:
            continue
        failures += 1
        program = entry["code"]

        checks = {
            "the answer does not follow from the trace":
                return_disagrees(program, claimed, answer_text) is True,
            "a transition the program does not allow":
                illegal_transition(program, claimed) is not None and not uses_jump(program),
            "its own branch numbers do not add up": numbers_contradict(program, claimed),
            "a statement does not give what it claims": bad_step(program, claimed)[0] is not None,
            "a line the program does not have": invented_line(program, claimed),
            "the wrong arm, on the model's own values": wrong_arm(program, claimed) is not None,
        }
        for name, hit in checks.items():
            fired[name] += hit
        profile[sum(checks.values())] += 1

        _, failing, checked = bad_step(program, claimed)
        if checked:
            cascade["every step holds" if failing == 0 else
                    "one step fails" if failing == 1 else "several steps fail"] += 1

    print(f"\n  {failures} failures\n")
    print(f"  {'check':<46}{'fires on':>10}{'share':>8}")
    for name, n in fired.most_common():
        print(f"  {name:<46}{n:>10}{n / failures:>7.0%}")

    print(f"\n  How many checks fire at once\n")
    for count in sorted(profile):
        print(f"    {count} check{'s' if count != 1 else ' '}  {profile[count]:>5}"
              f"  {profile[count] / failures:>5.0%}")
    print(f"\n    at least one  {failures - profile[0]:>5}"
          f"  {(failures - profile[0]) / failures:>5.0%}")

    print(f"\n  One mistake or many, among traces whose steps could be checked\n")
    for label, n in cascade.most_common():
        print(f"    {label:<24}{n:>5}")
    print(f"\n    'every step holds' means the arithmetic was right and the trace was")
    print(f"    still wrong: the model went to the wrong places, not miscalculated")


if __name__ == "__main__":
    main()
