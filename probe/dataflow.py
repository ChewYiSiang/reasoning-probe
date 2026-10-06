"""Data flow: how many times each variable changed while the program ran.

The control-flow side asks which way execution went and how often. This asks what happened
to the values as it went, at the same level of detail: a count per variable, so the same
parser, the same scoring and the same error taxonomy all apply unchanged.

Why counts rather than values. RQ5 showed that handing a model the loop and branch figures
does not repair its answers, so the next thing to test is the layer underneath. Values are
the obvious candidate but they need formatting rules, they often reveal the answer outright,
and a wrong type is a different failure from a wrong value. A variable update is an integer, it
is almost never the answer, and it pairs directly with the loop counts: a variable assigned
once per pass is a check on whether the model tracked the loop at all.

Measured on 167 CRUXEval programs, about 30% of variables carry information that cannot be
read off the control-flow counts; the rest are loop variables, or written once, or equal to
a number already reported. Only the informative ones are asked about, and the others are
reported separately so the accuracy figure is not padded with freebies.

A write is counted when a name appears for the first time or its value changes, which
catches rebinding (`x = x + 1`) and mutation in place (`out.append(c)`) alike.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path

PROGRAM_FILENAME = "<program>"
MARK = "### WRITES"
ANSWER_MARK = "### ANSWER"


# --------------------------------------------------------------------------- tracing

def trace_writes(code: str, func_name: str, test_input, *, step_budget: int = 200_000
                 ) -> tuple[dict[str, int], dict[str, object]]:
    """How many times each local variable changed, and what it held at the end.

    The final values are needed to tell apart two very different mistakes: a model that
    miscounted, and a model that reported the variable's value or its length instead of how
    many times it changed.

    A second tracing pass rather than an extension of the control-flow tracer: that one uses
    `sys.monitoring` where available, whose line callback does not hand back a frame, and the
    two passes together cost about a millisecond on these programs.
    """
    namespace: dict = {}
    exec(compile(code, PROGRAM_FILENAME, "exec"), namespace)
    if func_name not in namespace:
        raise NameError(f"{func_name} is not defined by this program")

    counts: dict[str, int] = {}
    previous: dict[str, object] = {}
    final: dict[str, object] = {}
    steps = 0

    def tracer(frame, event, arg):
        nonlocal previous, steps
        if frame.f_code.co_filename != PROGRAM_FILENAME:
            return None
        if event == "call":
            previous = {}
            return tracer
        if event != "line":
            return tracer

        steps += 1
        if steps > step_budget:
            raise RuntimeError("variable-update budget exceeded")

        now: dict[str, object] = {}
        for name, value in frame.f_locals.items():
            try:
                now[name] = copy.deepcopy(value)
            except Exception:
                now[name] = repr(value)        # uncopyable: compare its text instead
        for name, value in now.items():
            if name not in previous or previous[name] != value:
                counts[name] = counts.get(name, 0) + 1
            final[name] = value
        previous = now
        return tracer

    from probe.truth import call_like_mucoco

    arguments = copy.deepcopy(test_input)
    sys.settrace(tracer)
    try:
        call_like_mucoco(namespace[func_name], arguments)
    finally:
        sys.settrace(None)
    return counts, final


def write_counts(code: str, func_name: str, test_input, *, step_budget: int = 200_000
                 ) -> dict[str, int]:
    """Just the counts, for callers that do not need the values."""
    return trace_writes(code, func_name, test_input, step_budget=step_budget)[0]


# --------------------------------------------------------------------------- selection

def loop_targets(code: str) -> dict[str, int]:
    """Loop variables and the header line of the loop that assigns them.

    `for x in items` rebinds x once per pass, so its variable update is that loop's iteration
    count by construction. Asking about it would be asking the same question twice.
    """
    targets: dict[str, int] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.For):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    targets[name.id] = node.lineno
    return targets


def informative(writes: dict[str, int], control_flow_counts: dict[str, tuple[int, ...]],
                code: str) -> dict[str, int]:
    """Drop the variables whose count is already known from the control flow.

    Two exclusions, both structural rather than numerical:

      * written once, so the count carries nothing and the answer is trivially 1
      * a loop variable, whose count is that loop's iteration count by construction

    Everything else is kept, including a list built by `append` inside a loop, which is
    changed by the run without ever being assigned inside it.

    A variable is **not** dropped merely because its count happens to equal some unrelated
    figure elsewhere in the program. On programs this short small integers collide often,
    and dropping on coincidence threw away most of the data. How often such a collision
    occurs is reported separately by `coincidental`, so the caveat is visible without the
    measurement being thrown away.
    """
    targets = set(loop_targets(code))
    return {name: count for name, count in writes.items()
            if count > 1 and name not in targets}


def coincidental(writes: dict[str, int],
                 control_flow_counts: dict[str, tuple[int, ...]]) -> set[str]:
    """Variables whose count happens to equal a number the control-flow side reports.

    Not excluded, because the coincidence is usually just small integers colliding, but
    worth counting: a model that guessed from the loop figures would get these right.
    """
    known = {value for counts in control_flow_counts.values() for value in counts}
    known |= {sum(counts) for counts in control_flow_counts.values() if len(counts) == 2}
    return {name for name, count in writes.items() if count in known}


def checklist(writes: dict[str, int], limit: int = 3) -> list[str]:
    """The variables to ask about, in a stable order, at most `limit` of them."""
    return sorted(writes, key=lambda name: (-writes[name], name))[:limit]


# --------------------------------------------------------------------------- the prompt

# Mirrors the profile question's wording: the model is told to write two sections with the
# answer first, so MuCoCo's task is unchanged and their parser still finds the answer.
# Without the ANSWER section the model answers only the question we appended, which costs
# the whole run: LLaMA scored 6.4% instead of its usual 27.1%.
INSTRUCTIONS = """
After the answer, report how the variables changed while the program ran.

Write exactly two sections and nothing else:

{answer_mark}
<your answer, in the format described above, on one line>

{writes_mark}
{checklist}

For each variable, give how many times its value changed during the run, counting the
first time it was set. Write a single number for each.
"""


def section(names: list[str]) -> str:
    """The question itself, or nothing when this program has no variable worth asking about."""
    if not names:
        return ""
    lines = "\n".join(f"{name} = ?" for name in names)
    return INSTRUCTIONS.format(answer_mark=ANSWER_MARK, writes_mark=MARK, checklist=lines)


class WritesPrompt:
    """MuCoCo's prompt with the variable-update question appended.

    Used in place of their prompt helper, exactly like `ProfilePrompt`. The variables to ask
    about are looked up by program text, because the tester does not pass the task id down.
    """

    def __init__(self, base_helper, names_by_code: dict[str, list[str]]):
        self.base_helper = base_helper
        self.names_by_code = names_by_code

    def __call__(self) -> "WritesPrompt":
        return self

    def format(self, **input_variables) -> str:
        template = self.base_helper() if callable(self.base_helper) else self.base_helper
        base = template.format(**input_variables)
        names = self.names_by_code.get(input_variables.get("full_sol") or "", [])
        return base.rstrip() + "\n" + section(names) if names else base


_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(\d+)", re.M)


def parse_writes(reply: str) -> dict[str, int]:
    """Read the variable updates out of a reply, ignoring anything before the marker."""
    if MARK not in reply:
        return {}
    tail = reply.split(MARK, 1)[1]
    return {name: int(value) for name, value in _LINE.findall(tail)}


# --------------------------------------------------------------------------- scoring

def score(claimed: dict[str, int], real: dict[str, int]) -> dict[str, str]:
    """correct, wrong or not answered, per variable asked about."""
    result = {}
    for name, count in real.items():
        if name not in claimed:
            result[name] = "not answered"
        elif claimed[name] == count:
            result[name] = "correct"
        else:
            result[name] = "wrong"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="trace variable updates for every task and version")
    build.add_argument("--source", choices=["mongo", "jsonl"], default="mongo")
    build.add_argument("--tasks", help="exported task file when --source jsonl")
    build.add_argument("--truth", required=True, help="the control-flow truth file")
    build.add_argument("--limit", type=int, default=800)
    build.add_argument("--mutations", nargs="*", default=["sequential", "constant_unfold_add"])
    build.add_argument("--out", required=True)

    report = sub.add_parser("score", help="compare a run's answers with the traced counts")
    report.add_argument("--writes", required=True, help="the file written by `build`")
    report.add_argument("--log", required=True, nargs="+")
    report.add_argument("--version", required=True, nargs="+")

    args = parser.parse_args()
    if args.command == "build":
        from probe.build_writes import build_writes
        build_writes(args)
    else:
        from probe.build_writes import score_writes
        score_writes(args)


if __name__ == "__main__":
    main()


class BothPrompt:
    """Their prompt, with the loop and branch question and the variable-update question.

    Two separate runs cannot answer whether the two signals overlap: they are two
    generations, so a difference between them might only be the model reading the program
    differently on the second occasion. Both sections have to come from one reply about one
    program, which is what this does.

    Section order is answer, then loops and branches, then variable updates. The answer
    stays first so MuCoCo's task is unchanged.
    """

    def __init__(self, base_helper, names_by_code: dict[str, list[str]]):
        self.base_helper = base_helper
        self.names_by_code = names_by_code

    def __call__(self) -> "BothPrompt":
        return self

    def format(self, **input_variables) -> str:
        from probe.prompt import ANSWER_MARK as PROFILE_ANSWER_MARK
        from probe.prompt import PROFILE_MARK, checklist

        template = self.base_helper() if callable(self.base_helper) else self.base_helper
        base = template.format(**input_variables)
        code = input_variables.get("full_sol") or ""
        names = self.names_by_code.get(code, [])

        parts = [f"{PROFILE_ANSWER_MARK}", "<your answer, in the format described above, on one line>",
                 "", PROFILE_MARK, checklist(code)]
        if names:
            parts += ["", MARK] + [f"{name} = ?" for name in names]

        question = (
            "\nAfter the answer, report how the program ran for this input.\n\n"
            "Write exactly " + ("three" if names else "two") + " sections and nothing else:\n\n"
            + "\n".join(parts) + "\n\n"
            "For a loop, give the number of iterations. For a branch, give how many times the\n"
            "condition was true and how many times it was false. Write 0 where something never\n"
            "happens."
        )
        if names:
            question += ("\n\nFor each variable, give how many times its value changed during the\n"
                         "run, counting the first time it was set. Write a single number for each.")
        return base.rstrip() + "\n" + question + "\n"
