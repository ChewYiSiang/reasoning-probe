"""Find the loops and branches in a program and give each one a stable name.

The names have to survive mutation: `loop1` in the original must be `loop1` in the
renamed or constant-unfolded version, otherwise the two profiles cannot be compared.
Line numbers are no good for that, so constructs are numbered in source order and the
line number is kept only for display.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class Construct:
    id: str                 # "loop1", "if2" - stable across versions of the same program
    kind: str               # "loop" or "if"
    header_line: int        # the `for`/`while`/`if` line itself
    body_line: int          # first statement inside the body
    else_line: int | None   # first statement in the else branch, if there is one
    header_text: str        # e.g. "for x in text:" - shown to the model
    measurable: bool        # False for one-liners, see below


def _walk_in_source_order(tree: ast.AST):
    """Depth-first walk, children visited in the order they appear in the source."""
    yield tree
    for child in ast.iter_child_nodes(tree):
        yield from _walk_in_source_order(child)


def _header_text(node: ast.AST, source_lines: list[str]) -> str:
    line = source_lines[node.lineno - 1].strip()
    return line


def find_constructs(code: str) -> list[Construct]:
    """Return every loop and if statement, numbered per kind in source order."""
    tree = ast.parse(code)
    lines = code.splitlines()
    found: list[Construct] = []
    counters = {"loop": 0, "if": 0}

    for node in _walk_in_source_order(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            kind = "loop"
        elif isinstance(node, ast.If):
            kind = "if"
        else:
            continue

        counters[kind] += 1
        body_line = node.body[0].lineno
        else_line = node.orelse[0].lineno if node.orelse else None

        # A one-liner such as `if x: return 1` puts the body on the header line, so the
        # line counter cannot tell "condition evaluated" from "branch taken". Those are
        # recorded but excluded from scoring.
        measurable = body_line != node.lineno

        found.append(
            Construct(
                id=f"{kind}{counters[kind]}",
                kind=kind,
                header_line=node.lineno,
                body_line=body_line,
                else_line=else_line,
                header_text=_header_text(node, lines),
                measurable=measurable,
            )
        )

    found.sort(key=lambda c: (c.header_line, c.id))
    return found


def checklist(constructs: list[Construct]) -> str:
    """The list of constructs shown in the prompt, so the model knows what to answer."""
    rows = []
    for c in constructs:
        if not c.measurable:
            continue
        if c.kind == "loop":
            rows.append(f"{c.id}  (line {c.header_line}: {c.header_text})  iterations = ?")
        else:
            rows.append(f"{c.id}  (line {c.header_line}: {c.header_text})  taken = ?, else = ?")
    return "\n".join(rows)
