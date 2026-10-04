"""Semantics-preserving mutations.

The real operators live in the group's MuCoCo repository. Until that is wired in,
this module provides two of them so the rest of the pipeline can be built and tested:
sequential renaming (lexical) and constant unfold add (logical). Both keep every
statement in place, which is what lets a construct keep its name across versions.

`load_operators()` prefers MuCoCo's implementations and falls back to these.
"""

from __future__ import annotations

import ast
import builtins

BUILTIN_NAMES = set(dir(builtins)) | {"ANCHOR", "locals"}


def _module_fixtures(tree: ast.Module) -> set[str]:
    """Module-level names that benchmark inputs refer to, so they must not be renamed."""
    names: set[str] = set()
    for node in tree.body:
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


class _Renamer(ast.NodeTransformer):
    """Rename user-defined names the way MuCoCo's sequential mutation does.

    Functions become generic_function1, generic_function2, ...; everything else becomes
    var1, var2, ... in order of first appearance.
    """

    def __init__(self, keep: set[str] | None = None,
                 functions: set[str] | None = None) -> None:
        self.mapping: dict[str, str] = {}
        self.keep = keep or set()
        self.functions = functions or set()
        self._counts = {"function": 0, "var": 0}

    def _new_name(self, old: str) -> str:
        if old not in self.mapping:
            if old in self.functions:
                self._counts["function"] += 1
                self.mapping[old] = f"generic_function{self._counts['function']}"
            else:
                self._counts["var"] += 1
                self.mapping[old] = f"var{self._counts['var']}"
        return self.mapping[old]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._new_name(node.name)
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        # Covers function and lambda parameters alike. Arguments are visited before the
        # body, so parameter order decides v1, v2, ...
        node.arg = self._new_name(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Built-ins keep their names unless the program shadowed one with a parameter.
        if node.id in self.keep:
            return node
        if node.id in BUILTIN_NAMES and node.id not in self.mapping:
            return node
        node.id = self._new_name(node.id)
        return node


def sequential_rename(code: str) -> tuple[str, dict[str, str]]:
    """Lexical mutation, MuCoCo's sequential flavour.

    Returns the new code and {new name: original name}. Function names change too, so
    callers must rebuild the call expression from the mutated code.
    """
    tree = ast.parse(code)
    renamer = _Renamer(keep=_module_fixtures(tree), functions=_function_names(tree))
    tree = renamer.visit(tree)
    ast.fix_missing_locations(tree)
    reverse = {new: old for old, new in renamer.mapping.items()}
    return ast.unparse(tree), reverse


class _ConstantUnfold(ast.NodeTransformer):
    """Rewrite every whole number n as (n - 1 + 1)."""

    def __init__(self) -> None:
        self.changes = 0

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        # `type(...) is int` because booleans are ints in Python and must be left alone.
        if type(node.value) is int:
            self.changes += 1
            return ast.BinOp(left=ast.Constant(node.value - 1), op=ast.Add(),
                             right=ast.Constant(1))
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        return node          # f-strings break if their parts are rewritten


def constant_unfold_add(code: str) -> tuple[str, dict[str, str]]:
    """Logical mutation. Returns the new code and an empty name mapping."""
    unfolder = _ConstantUnfold()
    tree = unfolder.visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    if unfolder.changes == 0:
        raise ValueError("no integer constant to unfold")
    return ast.unparse(tree), {}


FALLBACK_OPERATORS = {
    "sequential_rename": sequential_rename,
    "constant_unfold_add": constant_unfold_add,
}


def load_operators() -> tuple[dict, str]:
    """Return ({name: operator}, source) with MuCoCo's operators when importable."""
    try:
        from mucoco import mutations            # type: ignore
    except ImportError:
        return FALLBACK_OPERATORS, "fallback"
    # The import path and names are confirmed once the group's repo is available;
    # until then the fallback is used and the run log says so.
    operators = getattr(mutations, "OPERATORS", None)
    if not operators:
        return FALLBACK_OPERATORS, "fallback"
    return operators, "mucoco"
