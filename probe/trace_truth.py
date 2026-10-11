"""Per-step execution traces: the path and the values, in one record.

The existing profile question asks how many times each construct ran, and the writes question
how many times each variable changed. Both are totals, so neither can say where in the run a
model went wrong. A trace carries the same information in order: one entry per executed line,
listing only the variables whose value changed at that step.

Only the changed variables are listed. The full locals at every step would repeat everything
that did not move, which on these programs is most of it, and the cost is what decides whether
the question can be asked at all.
"""
from __future__ import annotations

import ast
import copy
import inspect
import signal
import sys

PROGRAM_FILENAME = "<program>"
DEFAULT_CAP = 50


class TraceTooLong(Exception):
    """The run has more steps than the cap allows, so the question is not asked for it."""


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


# SIGALRM is Unix-only. The runs happen on Linux, but the tests are also run on Windows, and
# the step cap already bounds execution, so the alarm is a second guard rather than the only
# one: it is used where the platform has it and skipped where it does not.
_CAN_ALARM = hasattr(signal, "SIGALRM")


def _short(value, limit: int = 40) -> str:
    """A value as the model would be asked to write it, truncated where it is unreasonable."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _call_like_mucoco(func, test_input):
    """Call the function the way MuCoCo's own validity check does.

    Their helper passes no argument for a stored input of None, unpacks when the function
    takes several parameters, and passes the value whole otherwise. A call made any other way
    would trace a different run from the one the model is asked about, so this mirrors
    probe.truth.call_like_mucoco rather than inventing a second convention.
    """
    if test_input is None:
        return func()
    signature = inspect.signature(func)
    if not isinstance(test_input, int) and len(signature.parameters) > 1:
        return func(*test_input)
    return func(test_input)


def trace(code: str, func_name: str, test_input, *, cap: int = DEFAULT_CAP,
          timeout_s: float = 5.0) -> list[tuple[int, dict]]:
    """Every executed line, with the variables that changed at it.

    Raises TraceTooLong past the cap rather than returning a truncated trace, because a
    truncated one would be scored as if the model had stopped early.
    """
    namespace: dict = {}
    exec(compile(code, PROGRAM_FILENAME, "exec"), namespace)
    if func_name not in namespace:
        raise NameError(f"{func_name} is not defined by this program")

    steps: list[list] = []
    previous: dict = {}

    def snapshot(frame) -> dict:
        now = {}
        for name, value in frame.f_locals.items():
            if name.startswith("."):          # CPython's internal comprehension variable
                continue
            try:
                now[name] = copy.deepcopy(value)
            except Exception:
                now[name] = repr(value)
        return now

    def tracer(frame, event, arg):
        """Record the path as it happens, and attach each change to the line that caused it.

        f_locals is read at the start of a line, so what it shows was done by the line
        before. Attaching a change to the line currently about to run would put every value
        one step late, and a model writing a correct trace would be marked wrong everywhere.
        """
        nonlocal previous
        if frame.f_code.co_filename != PROGRAM_FILENAME:
            return None
        if event == "call":
            # the parameters are bound before any statement runs, so they belong to the def
            previous = snapshot(frame)
            steps.append([frame.f_code.co_firstlineno, dict(previous)])
            return tracer
        if event == "return":
            now = snapshot(frame)
            if steps:
                steps[-1][1].update({n: v for n, v in now.items()
                                     if n not in previous or previous[n] != v})
            return tracer
        if event != "line":
            return tracer
        if len(steps) > cap:
            raise TraceTooLong(f"more than {cap} steps")
        now = snapshot(frame)
        if steps:
            steps[-1][1].update({n: v for n, v in now.items()
                                 if n not in previous or previous[n] != v})
        previous = now
        steps.append([frame.f_lineno, {}])
        return tracer

    if _CAN_ALARM:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(timeout_s) or 1)
    try:
        sys.settrace(tracer)
        _call_like_mucoco(namespace[func_name], test_input)
    finally:
        sys.settrace(None)
        if _CAN_ALARM:
            signal.alarm(0)
    return [(lineno, changed) for lineno, changed in steps]


def as_text(steps: list[tuple[int, dict]]) -> str:
    """The trace in the form the model is asked to produce."""
    lines = []
    for lineno, changed in steps:
        shown = ", ".join(f"{name}={_short(value)}" for name, value in changed.items())
        lines.append(f"line {lineno} | {shown}")
    return "\n".join(lines)


def path_of(steps) -> list[int]:
    """Just the line sequence, which is what the execution path question asks for."""
    return [lineno for lineno, _ in steps]


def counts_from(steps, code: str) -> dict[str, tuple[int, ...]]:
    """Recover the construct counts from a trace, so existing figures stay computable.

    A loop iterates once per execution of its first body statement; a branch's true count is
    executions of its first body statement and its false count of its first orelse statement.
    Deriving them here rather than asking separately keeps one question where there were two.
    """
    tree = ast.parse(code)
    loops = sorted((n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))),
                   key=lambda n: n.lineno)
    branches = sorted((n for n in ast.walk(tree) if isinstance(n, ast.If)),
                      key=lambda n: n.lineno)
    watch: dict[int, tuple[str, str | None]] = {}
    for i, node in enumerate(loops, start=1):
        if node.body:
            watch.setdefault(node.body[0].lineno, (f"loop{i}", None))
    for i, node in enumerate(branches, start=1):
        if node.body:
            watch.setdefault(node.body[0].lineno, (f"if{i}", "true"))
        if node.orelse:
            watch.setdefault(node.orelse[0].lineno, (f"if{i}", "false"))

    hits: dict[str, int] = {}
    for lineno in path_of(steps):
        if lineno in watch:
            name, label = watch[lineno]
            key = name if label is None else f"{name}_{label}"
            hits[key] = hits.get(key, 0) + 1

    out: dict[str, tuple[int, ...]] = {}
    for i, _ in enumerate(loops, start=1):
        out[f"loop{i}"] = (hits.get(f"loop{i}", 0),)
    for i, _ in enumerate(branches, start=1):
        out[f"if{i}"] = (hits.get(f"if{i}_true", 0), hits.get(f"if{i}_false", 0))
    return out


def writes_from(steps) -> dict[str, int]:
    """Recover the variable change counts from a trace, for the same reason."""
    counts: dict[str, int] = {}
    for _, changed in steps:
        for name in changed:
            counts[name] = counts.get(name, 0) + 1
    return counts
