"""Record what a program really does when it runs, without touching the program.

Two mechanisms, same result:
  * sys.monitoring - Python 3.12 and later, the cheap one, used when available
  * sys.settrace   - the older hook, used as a fallback

Both hook every line event. Raw line events are not quite what we want: a statement
spread over several lines fires several events, so the counts are taken per *statement*
instead. Each line is mapped to the statement that owns it, and a statement only scores
when execution enters it, not while it moves within it.

What gets recorded is a *counted arc*: how many times control moved from one statement
to another. Coverage tools record the same arcs as a set ("this branch was taken at some
point"); counting them is what turns coverage into profiling. A loop's iteration count is
the number of times control moved from the loop header into its body, and a branch's
count is the number of times it moved from the condition into that branch.

Counting arcs rather than statement hits matters for nested loops: the inner loop header
runs once per inner iteration, so counting hits of the outer loop's first statement would
report the inner iterations as well.
"""

from __future__ import annotations

import ast
import copy
import inspect
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

PROGRAM_FILENAME = "<program>"   # tag used to tell the program's lines from our own

USE_MONITORING = hasattr(sys, "monitoring")


class ExecutionBudgetExceeded(RuntimeError):
    """Raised when a program runs too long to be worth tracing."""


@dataclass
class Profile:
    """What happened when the program ran, at the profiling level."""
    output: Any
    loops: dict[str, int] = field(default_factory=dict)          # construct id -> iterations
    branches: dict[str, tuple[int, int]] = field(default_factory=dict)  # id -> (taken, else)
    statement_counts: dict[int, int] = field(default_factory=dict)      # statement line -> times run
    arc_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    steps: int = 0

    def as_dict(self) -> dict:
        return {
            "output": repr(self.output),
            "loops": self.loops,
            "branches": {k: list(v) for k, v in self.branches.items()},
            "statement_counts": self.statement_counts,
            "arc_counts": [[src, dst, n] for (src, dst), n in self.arc_counts.items()],
            "steps": self.steps,
        }

    def counts_only(self) -> dict[str, tuple[int, ...]]:
        """The part that is compared between a program and its mutants."""
        out: dict[str, tuple[int, ...]] = {k: (v,) for k, v in self.loops.items()}
        out.update({k: v for k, v in self.branches.items()})
        return out


def statement_lines(code: str) -> dict[int, int]:
    """Map every line of the program to the first line of the statement that owns it.

    Inner statements override the outer ones they sit inside, so the deepest statement
    covering a line wins.
    """
    owner: dict[int, int] = {}

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.stmt) and node.end_lineno is not None:
            for line in range(node.lineno, node.end_lineno + 1):
                owner[line] = node.lineno
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(code))
    return owner


class _Counter:
    """Counts statement entries and stops runaway programs."""

    def __init__(self, step_budget: int, timeout_s: float, owner: dict[int, int]):
        self.counts: dict[int, int] = {}
        self.arcs: dict[tuple[int, int], int] = {}
        self.owner = owner
        self.last_owner: int | None = None
        self.steps = 0
        self.step_budget = step_budget
        self.deadline = time.monotonic() + timeout_s

    def hit(self, lineno: int) -> None:
        owner = self.owner.get(lineno, lineno)
        # Only count when execution moves into a different statement; a multi-line
        # statement fires one event per line and must still count as one step.
        if owner != self.last_owner:
            self.counts[owner] = self.counts.get(owner, 0) + 1
            if self.last_owner is not None:
                arc = (self.last_owner, owner)
                self.arcs[arc] = self.arcs.get(arc, 0) + 1
            self.last_owner = owner
        self.steps += 1
        if self.steps > self.step_budget:
            raise ExecutionBudgetExceeded(f"more than {self.step_budget} lines executed")
        # Checking the clock on every line would dominate the cost, so do it rarely.
        if self.steps % 5000 == 0 and time.monotonic() > self.deadline:
            raise ExecutionBudgetExceeded("timed out")


def _run_with_monitoring(thunk: Callable[[], Any], counter: _Counter) -> Any:
    mon = sys.monitoring
    tool_id = mon.PROFILER_ID
    mon.use_tool_id(tool_id, "reasoning-probe")
    try:
        def on_line(code, lineno):
            if code.co_filename == PROGRAM_FILENAME:
                counter.hit(lineno)
            return None          # returning DISABLE would switch off further events

        mon.register_callback(tool_id, mon.events.LINE, on_line)
        mon.set_events(tool_id, mon.events.LINE)
        try:
            return thunk()
        finally:
            mon.set_events(tool_id, 0)
            mon.register_callback(tool_id, mon.events.LINE, None)
    finally:
        mon.free_tool_id(tool_id)


def _run_with_settrace(thunk: Callable[[], Any], counter: _Counter) -> Any:
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != PROGRAM_FILENAME:
            return None
        if event == "line":
            counter.hit(frame.f_lineno)
        return tracer

    sys.settrace(tracer)
    try:
        return thunk()
    finally:
        sys.settrace(None)


def call_like_mucoco(func: Callable, test_input: Any) -> Any:
    """Call a benchmark function the way MuCoCo's own validity check does.

    Mirrors `PredictionInconsistencyHelper.check_input_output`: no argument when the
    stored input is None, unpacked when the function takes several parameters, and
    passed whole otherwise. Tracing a call made any other way would not match the run
    the model is being asked about.
    """
    if test_input is None:
        return func()
    signature = inspect.signature(func)
    if not isinstance(test_input, int) and len(signature.parameters) > 1:
        return func(*test_input)
    return func(test_input)


def run_traced(code: str, call_source: str, *, step_budget: int = 200_000,
               timeout_s: float = 5.0) -> tuple[Any, _Counter]:
    """Run `call_source` against `code` under the tracer and return (result, counter)."""
    namespace: dict = {}
    exec(compile(code, PROGRAM_FILENAME, "exec"), namespace)

    counter = _Counter(step_budget, timeout_s, statement_lines(code))
    runner = _run_with_monitoring if USE_MONITORING else _run_with_settrace
    result = runner(lambda: eval(compile(call_source, "<call>", "eval"), namespace), counter)
    return result, counter


def run_traced_call(code: str, func_name: str, test_input: Any, *,
                    step_budget: int = 200_000, timeout_s: float = 5.0) -> tuple[Any, _Counter]:
    """Same as `run_traced`, but calls a function with real argument values.

    Used for MuCoCo tasks, where the input comes out of the database as Python objects
    rather than as source text.
    """
    namespace: dict = {}
    exec(compile(code, PROGRAM_FILENAME, "exec"), namespace)
    if func_name not in namespace:
        raise NameError(f"{func_name} is not defined by this program")

    counter = _Counter(step_budget, timeout_s, statement_lines(code))
    runner = _run_with_monitoring if USE_MONITORING else _run_with_settrace
    arguments = copy.deepcopy(test_input)          # the program may mutate its input
    result = runner(lambda: call_like_mucoco(namespace[func_name], arguments), counter)
    return result, counter


def _to_profile(result: Any, counter: "_Counter", constructs) -> Profile:
    prof = Profile(output=result, statement_counts=counter.counts,
                   arc_counts=counter.arcs, steps=counter.steps)

    for c in constructs:
        if not c.measurable:
            continue
        entries = counter.arcs.get((c.header_line, c.body_line), 0)
        if c.kind == "loop":
            prof.loops[c.id] = entries
        else:
            else_taken = counter.arcs.get((c.header_line, c.else_line), 0) if c.else_line else 0
            prof.branches[c.id] = (entries, else_taken)
    return prof


def profile(code: str, call_source: str, constructs, **kwargs) -> Profile:
    """Ground truth for one program, called through a source expression."""
    result, counter = run_traced(code, call_source, **kwargs)
    return _to_profile(result, counter, constructs)


def profile_call(code: str, func_name: str, test_input: Any, constructs, **kwargs) -> Profile:
    """Ground truth for one program, called with real argument values (MuCoCo tasks)."""
    result, counter = run_traced_call(code, func_name, test_input, **kwargs)
    return _to_profile(result, counter, constructs)
