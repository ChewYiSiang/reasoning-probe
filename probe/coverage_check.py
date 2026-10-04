"""Cross-check the tracer against coverage.py.

coverage.py records the same branch arcs we do, but as a set: it knows an arc was taken,
not how often. That makes it a good independent check of *which* branches ran, and a
ready-made baseline for the coverage level of the study.

coverage.py is an optional dependency; when it is missing, `available()` returns False.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

try:
    import coverage
except ImportError:            # optional
    coverage = None


def available() -> bool:
    return coverage is not None


def executed_arcs(code: str, call_source: str) -> set[tuple[int, int]]:
    """Arcs coverage.py saw, as (from line, to line). Needs the program on disk."""
    if coverage is None:
        raise RuntimeError("coverage.py is not installed")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "program.py"
        path.write_text(code)
        namespace: dict = {}
        source = compile(code, str(path), "exec")

        cov = coverage.Coverage(branch=True, data_file=None)
        exec(source, namespace)
        cov.start()
        try:
            eval(compile(call_source, "<call>", "eval"), namespace)
        finally:
            cov.stop()
        arcs = cov.get_data().arcs(os.path.abspath(path)) or []
    return {(src, dst) for src, dst in arcs}


def disagreements(profile, constructs, code: str, call_source: str) -> list[str]:
    """Construct ids where our counts and coverage.py's arcs tell different stories."""
    arcs = executed_arcs(code, call_source)
    problems = []
    for c in constructs:
        if not c.measurable:
            continue
        ours = profile.loops.get(c.id) if c.kind == "loop" else profile.branches.get(c.id, (0, 0))[0]
        theirs = (c.header_line, c.body_line) in arcs
        if bool(ours) != theirs:
            problems.append(f"{c.id}: counted {ours}, coverage.py says taken={theirs}")
    return problems
