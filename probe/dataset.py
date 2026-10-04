"""Reading benchmark items.

CRUXEval ships one JSON object per line with a function, one input and the output that
input produces. MuCoCo's own loaders replace this module once the repo is wired in.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


def entry_function(code: str) -> str:
    """Name of the function a benchmark item is called through.

    MuCoCo's lexical mutations rename function definitions (`f` becomes
    `generic_function1`), so the call has to be rebuilt for every version rather than
    hard-coded to `f`. The last top-level definition is the entry point; helper
    functions defined above it keep their place.
    """
    defs = [node.name for node in ast.parse(code).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not defs:
        raise ValueError("no function definition found")
    return defs[-1]


def call_source(code: str, input_source: str) -> str:
    """The call expression for this version of the program."""
    return f"{entry_function(code)}({input_source})"


@dataclass(frozen=True)
class Item:
    id: str
    code: str
    input_source: str        # the argument list as written in the benchmark, e.g. "'abc', 2"
    expected_output: str     # repr of the expected result

    @property
    def call_source(self) -> str:
        return call_source(self.code, self.input_source)


def load_cruxeval(path: str | Path, limit: int | None = None) -> list[Item]:
    items: list[Item] = []
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            items.append(
                Item(
                    id=record["id"],
                    code=record["code"],
                    input_source=record["input"],
                    expected_output=record["output"],
                )
            )
            if limit and len(items) == limit:
                break
    return items
