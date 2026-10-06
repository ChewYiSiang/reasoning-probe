"""Reading the two sections back out of a model reply.

The answer is handed to MuCoCo's own parser untouched, so their correctness and
consistency numbers are unaffected by the profile question. The profile is parsed here.
"""

from __future__ import annotations

import re

from probe.dataflow import MARK as WRITES_MARK
from probe.prompt import ANSWER_MARK, PROFILE_MARK

# loop3 iterations = 4      |      if2 taken = 1, else = 0
#
# The model usually repeats the construct's source line, which can itself contain "=",
# as in `key=lambda x: ...` or `while a == b:`. The patterns therefore anchor on the
# words "iterations", "taken" and "else" rather than on the first equals sign.
_LOOP = re.compile(r"^\s*(loop\d+)\b.*?iterations\s*=\s*(-?\d+)", re.IGNORECASE)
# "not taken" is the wording now used; "else" is accepted so that logs written by the
# earlier prompt can still be read.
_BRANCH = re.compile(
    r"^\s*(if\d+)\b.*?\btaken\s*=\s*(-?\d+).*?\b(?:not[ _]?taken|else)\s*=\s*(-?\d+)",
    re.IGNORECASE)
# Fallback for a line that names a construct and ends in a number, e.g. "loop1: 4".
_TRAILING = re.compile(r"^\s*(loop\d+|if\d+)\b.*?(-?\d+)\s*$", re.IGNORECASE)


def split_sections(reply: str) -> tuple[str, str]:
    """Return (answer text, profile text). Either can be empty if the model ignored the format.

    Any section we appended has to be cut off before the answer goes to their parser, not
    only the profile: a data-flow run appends `### WRITES` instead, and leaving it attached
    would hand their correctness oracle the whole block and score every answer wrong.
    """
    profile = ""
    if PROFILE_MARK in reply:
        reply, profile = reply.split(PROFILE_MARK, 1)
    # the variable-update section, when a data-flow run put one there
    reply = reply.split(WRITES_MARK, 1)[0]
    profile = profile.split(WRITES_MARK, 1)[0]
    answer = reply.split(ANSWER_MARK, 1)[-1] if ANSWER_MARK in reply else reply
    return answer.strip(), profile.strip()


def parse_profile(profile_text: str) -> dict[str, tuple[int, ...]]:
    """{construct id: counts}. Loops give one number, branches give (taken, else)."""
    counts: dict[str, tuple[int, ...]] = {}
    for line in profile_text.splitlines():
        branch = _BRANCH.match(line)
        if branch:
            counts[branch.group(1).lower()] = (int(branch.group(2)), int(branch.group(3)))
            continue
        loop = _LOOP.match(line)
        if loop:
            counts[loop.group(1).lower()] = (int(loop.group(2)),)
            continue
        trailing = _TRAILING.match(line)
        if trailing and trailing.group(1).lower().startswith("loop"):
            counts[trailing.group(1).lower()] = (int(trailing.group(2)),)
    return counts


def parse_reply(reply: str) -> tuple[str, dict[str, tuple[int, ...]]]:
    answer, profile_text = split_sections(reply)
    return answer, parse_profile(profile_text)
