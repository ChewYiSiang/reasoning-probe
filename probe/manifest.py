"""A record of the conditions a run was produced under.

Results folders accumulate: several models, different quantisation, the probe on or off,
and a ground truth that has already changed twice. Two months later the numbers are still
there and the conditions are not, so a re-score can quietly use a different truth file and
every figure moves with nothing to say why.

This writes one `manifest.json` beside a run's results, recording what was asked for, what
produced it, and what came out. The field that earns its keep is `truth_sha256`: it makes a
changed ground truth impossible to miss.

    from probe.manifest import write
    write(RESULTS, model=MODEL, mutation="sequential", probe=True,
          truth="/content/truth_cruxeval.json")

Most of it fills itself in. For runs already finished, `describe` reconstructs what it can
from the files and leaves the rest blank rather than guessing.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def checksum(path: str | Path) -> str | None:
    """A file's SHA-256, or None if it is not there."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def outcomes(csv_path: Path) -> dict[str, int]:
    """How each task ended, read from the result CSV rather than remembered."""
    if not csv_path.exists():
        return {}
    counts: Counter = Counter()
    with open(csv_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            failure = (row.get("failure_type") or "").strip()
            counts["correct" if not failure else failure.split(">")[0].strip()] += 1
    return dict(counts.most_common())


def package_size() -> int | None:
    """The probe's own size, as a rough version marker when there is no commit to hand."""
    folder = Path(__file__).resolve().parent
    try:
        return sum(f.stat().st_size for f in folder.glob("*.py"))
    except OSError:
        return None


def describe(results_dir: str | Path, mutation: str = "no_mutation", *,
             benchmark: str = "CruxEval", prompt_type: str = "zero_shot",
             model: str | None = None, probe: bool | None = None,
             quantisation: str | None = None, max_new_tokens: int | None = None,
             tasks_requested: int | None = None, temperature: float = 0.0,
             truth: str | None = None, notes: str | None = None) -> dict:
    """Build the record. Anything not supplied and not derivable is left as None."""
    results_dir = Path(results_dir)
    csv_path = results_dir / f"{benchmark}_{prompt_type}_{mutation}.csv"
    log_path = results_dir / f"profiles_{mutation}.jsonl"

    counts = outcomes(csv_path)
    replies = sum(1 for _ in open(log_path, encoding="utf-8")) if log_path.exists() else None

    return {
        "run": f"{results_dir.name}_{mutation}",
        "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "quantisation": quantisation,
        "probe": probe,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "benchmark": benchmark,
        "task": "output_prediction",
        "prompt_type": prompt_type,
        "mutation": mutation,
        "tasks_requested": tasks_requested,
        "truth_file": str(truth) if truth else None,
        "truth_sha256": checksum(truth) if truth else None,
        "probe_package_bytes": package_size(),
        "results_csv": csv_path.name if csv_path.exists() else None,
        "profile_log": log_path.name if log_path.exists() else None,
        "profiles_logged": replies,
        "outcomes": counts,
        "notes": notes,
    }


def write(results_dir: str | Path, mutation: str = "no_mutation", **details) -> Path:
    """Write or update `manifest.json` in the results folder, one entry per run."""
    results_dir = Path(results_dir)
    path = results_dir / "manifest.json"

    existing: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            existing = loaded if isinstance(loaded, list) else [loaded]
        except json.JSONDecodeError:
            existing = []

    record = describe(results_dir, mutation, **details)
    # one entry per run: replace rather than append when the same run is written twice
    existing = [entry for entry in existing if entry.get("run") != record["run"]]
    existing.append(record)
    existing.sort(key=lambda entry: entry.get("run") or "")

    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {len(existing)} runs recorded in {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    parser.add_argument("--mutation", nargs="+", default=["no_mutation"])
    parser.add_argument("--model")
    parser.add_argument("--truth")
    parser.add_argument("--quantisation")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--no-probe", dest="probe", action="store_false")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--tasks", type=int)
    parser.add_argument("--notes")
    parser.set_defaults(probe=None)
    args = parser.parse_args()

    for mutation in args.mutation:
        write(args.results_dir, mutation, model=args.model, truth=args.truth,
              quantisation=args.quantisation, probe=args.probe,
              max_new_tokens=args.max_new_tokens, tasks_requested=args.tasks,
              notes=args.notes)


if __name__ == "__main__":
    main()
