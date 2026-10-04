"""Supplying the model with the true execution facts, instead of asking for them.

This is the intervention: MuCoCo's task, unchanged, with a block stating how the program
really runs. If a model answers better when told, then execution understanding was part of
what it was missing; if not, its failures lie elsewhere.

Three flavours, because an improvement on its own proves little:
  * true       - the traced counts
  * placebo    - true but useless facts, matched in shape, to control for prompt bulk
  * corrupted  - wrong counts, to see whether supplied facts are used at all

`leaks_answer` marks tasks where the counts give the answer away. Roughly 43% of
control-flow tasks in CRUXEval are like that, so they are excluded from the main analysis
rather than quietly inflating it.
"""

from __future__ import annotations

import ast
from typing import Any

FACTS_MARK = "### EXECUTION FACTS"

HEADER = f"""
{FACTS_MARK}
These facts describe what happens when this program runs on the input above.

"""


def render(counts: dict[str, tuple[int, ...]], constructs: list[dict]) -> str:
    """The facts block, one line per construct, in source order."""
    by_id = {construct["id"]: construct for construct in constructs}
    lines = []
    for name, values in sorted(counts.items(), key=lambda item: by_id.get(item[0], {}).get("header_line", 0)):
        construct = by_id.get(name, {})
        where = f"line {construct.get('header_line', '?')}"
        if len(values) == 1:
            lines.append(f"- the loop at {where} runs {values[0]} times")
        else:
            taken, not_taken = values
            lines.append(f"- the condition at {where} is true {taken} times "
                         f"and false {not_taken} times")
    return HEADER + "\n".join(lines) + "\n"


def placebo(code: str) -> str:
    """True statements about the program that say nothing about how it runs.

    Matched roughly in length and shape to the real facts, so that any benefit of the real
    facts cannot be explained by the prompt simply being longer or more structured.
    """
    tree = ast.parse(code)
    statements = sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    calls = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))
    lines = [
        f"- the program contains {statements} statements",
        f"- the program refers to {len(names)} distinct names",
        f"- the program makes {calls} function calls",
    ]
    return HEADER + "\n".join(lines) + "\n"


def corrupt(counts: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Plausible but wrong counts: each number shifted, never left unchanged."""
    wrong: dict[str, tuple[int, ...]] = {}
    for name, values in counts.items():
        wrong[name] = tuple(value + 2 if value == 0 else value + 1 for value in values)
    return wrong


def leaks_answer(expected: Any, counts: dict[str, tuple[int, ...]]) -> bool:
    """Would the counts hand the model the answer?

    Two ways it happens in practice: the answer is one of the counts (a function that
    counts matches), or the answer's length is one of the counts (a list or string built
    one item per iteration). Deliberately generous, since a false positive only costs
    sample size while a false negative corrupts the result.
    """
    numbers = [value for values in counts.values() for value in values]
    if not numbers:
        return False
    if isinstance(expected, bool):
        return False                      # True/False cannot be read off a count
    if isinstance(expected, int) and expected in numbers:
        return True
    if hasattr(expected, "__len__") and len(expected) in numbers:
        return True
    return False


class FactsPrompt:
    """MuCoCo's prompt with an execution-facts block appended.

    Used in place of their prompt helper, exactly like `ProfilePrompt`. The facts for the
    current task are looked up by program text, because the tester does not pass the task
    id down to the model.
    """

    def __init__(self, base_helper, facts_by_code: dict[str, str], where: str = "after"):
        self.base_helper = base_helper
        self.facts_by_code = facts_by_code
        # Their template ends with "# Your answer", so facts appended after it sit where
        # the model's own output would begin. "before" puts them at the top instead,
        # which is the control for that.
        self.where = where

    def __call__(self) -> "FactsPrompt":
        return self

    def format(self, **input_variables) -> str:
        template = self.base_helper() if callable(self.base_helper) else self.base_helper
        base = template.format(**input_variables)
        facts = self.facts_by_code.get(input_variables.get("full_sol") or "", "")
        if not facts:
            return base
        if self.where == "before":
            return facts.strip() + "\n\n" + base
        return base.rstrip() + "\n" + facts
