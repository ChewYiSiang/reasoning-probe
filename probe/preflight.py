"""Check an operator will actually be asked the variable question, before any model call.

A run of ten operators cost three hours and produced the variable question for only two of
them. The lookup that decides which variables to ask about is keyed by the mutant's exact
program text, and the file holding it had been built for two operators only. Every other
operator silently got a prompt with the loops-and-branches question and nothing else. The
run printed a healthy-looking count the whole time, because that count was over the versions
in the file rather than over the operator being run.

Two checks, both free:

1. Is the operator even in the file? A version that was never traced can never be matched.
2. For an operator that is, do freshly generated mutants actually find their entry? This
   catches a file built from a different task set or a different library version.

    from probe.preflight import check
    check("/content/writes.jsonl", ["random", "demorgan"], tasks)   # raises if it would be silent
"""

from __future__ import annotations

import json
from pathlib import Path


class WouldBeSilent(Exception):
    """Raised when a run would ask no variable question for an operator."""


def versions_in(writes_path: str | Path) -> set[str]:
    """Which versions the lookup file actually covers."""
    found: set[str] = set()
    with open(writes_path, encoding="utf-8") as handle:
        for line in handle:
            found.update(json.loads(line)["versions"])
    return found


def lookup_from(writes_path: str | Path) -> dict[str, list[str]]:
    """The code to variable-names table the prompt uses."""
    table: dict[str, list[str]] = {}
    with open(writes_path, encoding="utf-8") as handle:
        for line in handle:
            for version in json.loads(line)["versions"].values():
                if version["asked"]:
                    table[version["code"]] = version["asked"]
    return table


def sample_hit_rate(table: dict[str, list[str]], tasks, mutations, sample: int = 25) -> float:
    """Generate a few real mutants and see how many find their entry.

    Needs MuCoCo, so it is skipped where that is not importable; the version check above
    still runs and is the one that catches a missing operator.
    """
    from probe.mucoco import mutate

    hits = tried = 0
    for task in tasks[:sample * 4]:
        if tried >= sample:
            break
        try:
            mutant = mutate(task, mutations)
        except Exception:
            continue
        tried += 1
        hits += mutant.code in table
    return hits / tried if tried else 0.0


def check(writes_path: str | Path, operators: list[str], tasks=None,
          mutation_objects: dict | None = None, minimum: float = 0.05) -> None:
    """Raise unless every operator would really be asked the variable question.

    `minimum` is deliberately low: some operators apply to a fifth of the benchmark, and a
    few per cent of hits is enough to show the lookup works. Zero is the failure worth
    catching.
    """
    covered = versions_in(writes_path)
    missing = [name for name in operators if name not in covered]
    if missing:
        raise WouldBeSilent(
            f"{Path(writes_path).name} has no entry for: {', '.join(missing)}.\n"
            f"    it covers: {', '.join(sorted(covered))}\n"
            f"    the run would ask the loops-and-branches question only, and the variable\n"
            f"    question would be silently dropped for those operators.\n"
            f"    rebuild it with:  python -m probe.dataflow build --mutations "
            f"{' '.join(sorted(set(operators) - {'no_mutation'}))} ...")

    if tasks is None or mutation_objects is None:
        print(f"preflight: {writes_path} covers all {len(operators)} operators "
              f"(mutants not sampled)")
        return

    table = lookup_from(writes_path)
    for name in operators:
        if name == "no_mutation":
            continue
        rate = sample_hit_rate(table, tasks, mutation_objects[name])
        if rate < minimum:
            raise WouldBeSilent(
                f"{name}: freshly generated mutants do not match the lookup "
                f"({rate:.0%} of a sample). The file may have been built from a different "
                f"task set or a different version of their mutator.")
        print(f"preflight: {name:22s} {rate:.0%} of sampled mutants find their entry")
