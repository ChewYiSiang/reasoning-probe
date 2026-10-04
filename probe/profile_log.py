"""Where the profiles go.

Both model wrappers write here: one line per call, holding the program, the input, the
parsed profile and the full reply. The reply is kept so a later parser fix can be applied
without re-running anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def log_path() -> str:
    # Read at call time, not import time, so a notebook can change it between runs.
    return os.environ.get("PROBE_PROFILE_LOG", "results/probe/profiles.jsonl")


def record(input_variables: dict, profile: dict, reply: str) -> None:
    path = Path(log_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "full_sol": input_variables.get("full_sol"),
        "test_input": str(input_variables.get("test_input")),
        "profile": {key: list(value) for key, value in profile.items()},
        "raw_reply": reply,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
