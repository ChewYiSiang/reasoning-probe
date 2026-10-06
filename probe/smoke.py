"""Run the whole path once, locally, before spending a GPU session on it.

Three bugs in a row reached Colab because each piece was tested alone: the model was
reloaded on every run until the card filled, the reply splitter left the variable-update section
attached to the answer, and the variable-update prompt never asked for an answer at all. Each
would have shown up the moment the full path was run once with a stand-in model.

This renders the real prompt for real programs, fakes a well-formed reply from what the
prompt asks for, and pushes it through the same splitter, logger and scorer a run uses. If
any step loses the answer or the extra section, it fails loudly here instead of after an
hour on a GPU.

    python -m probe.smoke

Exits non-zero on the first failure, so it can gate a notebook cell.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

# a stand-in for their template: what matters is that it ends with the answer cue, as theirs does
BASE_TEMPLATE = ("Given the following Python code and input, predict the output.\n\n"
                 "{full_sol}\n\nInput: {test_input}\n\n# Your answer")

PROGRAMS = [
    ("def f(rows):\n    found = 0\n    out = []\n    for row in rows:\n        for cell in row:\n"
     "            if cell > 0:\n                found += 1\n                out.append(cell)\n"
     "    return found", "[[0, 5], [-1, -2]]", "1"),
    ("def f(s):\n    total = 0\n    for ch in s:\n        if ch.isdigit():\n            total += int(ch)\n"
     "    return total", "'a1b2'", "3"),
    ("def f(n):\n    return n * 2", "4", "8"),          # straight-line: no extra question
]


class Failure(Exception):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def fake_reply(prompt: str, answer: str, extra_mark: str | None) -> str:
    """What a model that follows the prompt would write."""
    if extra_mark is None or extra_mark not in prompt:
        return f"### ANSWER\n{answer}\n"
    # only the question block after the marker, so the program text is never mistaken for it
    asked_block = prompt.split(extra_mark, 1)[1]
    lines = []
    for line in asked_block.splitlines():
        construct = re.match(r"^\s*(loop\d+|if\d+)\b", line)
        variable = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*\?\s*$", line)
        if construct and construct.group(1).startswith("loop"):
            lines.append(f"{construct.group(1)} iterations = 2")
        elif construct:
            lines.append(f"{construct.group(1)} taken = 1, not taken = 0")
        elif variable:
            lines.append(f"{variable.group(1)} = 3")
    return f"### ANSWER\n{answer}\n\n{extra_mark}\n" + "\n".join(lines) + "\n"


def run_one(label: str, wrapper, extra_mark: str, parse_extra) -> None:
    from probe.parse import parse_reply
    from probe.profile_log import record

    print(f"\n--- {label} ---")
    log = Path(tempfile.mkdtemp()) / "profiles_smoke.jsonl"
    import os
    os.environ["PROBE_PROFILE_LOG"] = str(log)

    prompt_object = wrapper()            # their tester calls the helper to get the template
    for code, test_input, answer in PROGRAMS:
        prompt = prompt_object.format(full_sol=code, test_input=test_input)

        # 1. the prompt still contains the program and their answer cue
        check(code in prompt, f"{label}: program text missing from the prompt")
        check("# Your answer" in prompt, f"{label}: their answer cue was lost")

        # 2. if we added a question, the model is told to write the answer first
        if extra_mark in prompt:
            check("### ANSWER" in prompt,
                  f"{label}: the prompt adds {extra_mark} but never asks for ### ANSWER, "
                  f"so the model will answer only the added question")
            check(prompt.index("### ANSWER") < prompt.index(extra_mark),
                  f"{label}: ### ANSWER must come before {extra_mark}")

        # 3. a well-behaved reply splits back into exactly the answer
        reply = fake_reply(prompt, answer, extra_mark)
        got_answer, profile = parse_reply(reply)
        check(got_answer == answer,
              f"{label}: the answer handed to their parser was {got_answer!r}, expected {answer!r}")
        check(extra_mark not in got_answer, f"{label}: {extra_mark} leaked into the answer")

        # 4. the extra section survives logging and parses back
        record({"full_sol": code, "test_input": test_input}, profile, reply)
        extracted = parse_extra(reply)
        # A straight-line program still gets the profile section, with nothing to fill in, so
        # an empty result is only a failure when the prompt actually asked for something.
        asked_something = extra_mark in prompt and "= ?" in prompt.split(extra_mark, 1)[1]
        if asked_something:
            check(bool(extracted), f"{label}: nothing parsed back from the {extra_mark} section")

    logged = [json.loads(line) for line in open(log, encoding="utf-8")]
    check(len(logged) == len(PROGRAMS), f"{label}: logged {len(logged)} replies, expected {len(PROGRAMS)}")
    print(f"    prompt, split, log and parse all correct on {len(PROGRAMS)} programs")
    print(f"    sample prompt tail:\n" + "\n".join("      " + l for l in prompt.splitlines()[-8:]))


def main() -> int:
    from probe.dataflow import MARK as WRITES_MARK
    from probe.dataflow import WritesPrompt, informative, parse_writes, write_counts
    from probe.parse import parse_profile, split_sections
    from probe.prompt import PROFILE_MARK, ProfilePrompt

    names_by_code = {}
    for code, test_input, _ in PROGRAMS:
        import ast
        counts = write_counts(code, "f", ast.literal_eval(test_input))
        chosen = sorted(informative(counts, {}, code))[:3]
        if chosen:
            names_by_code[code] = chosen

    try:
        run_one("profile question", lambda: ProfilePrompt(lambda: BASE_TEMPLATE), PROFILE_MARK,
                lambda reply: parse_profile(split_sections(reply)[1]))
        run_one("variable-update question", lambda: WritesPrompt(lambda: BASE_TEMPLATE, names_by_code),
                WRITES_MARK, parse_writes)
        # the combined prompt carries both sections, so check each survives the split
        from probe.dataflow import BothPrompt
        run_one("both questions, loops and branches", lambda: BothPrompt(lambda: BASE_TEMPLATE, names_by_code),
                PROFILE_MARK, lambda reply: parse_profile(split_sections(reply)[1]))
        run_one("both questions, variable updates", lambda: BothPrompt(lambda: BASE_TEMPLATE, names_by_code),
                WRITES_MARK, parse_writes)
    except Failure as problem:
        print(f"\nFAILED: {problem}")
        return 1
    print("\nall checks passed: safe to run on a GPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
