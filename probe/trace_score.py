"""Reading a model's trace back, and saying where it diverged from the real one.

The point of asking for a trace rather than totals is that two sequences can be aligned, so a
wrong answer has a position. Everything here is about making that position meaningful: the
parser has to be forgiving enough that a formatting slip is not read as a reasoning failure,
and the alignment has to distinguish a model that skipped a step from one that got a step
wrong.
"""
from __future__ import annotations

import ast
import re

# "line 6 | even='M', i=2"  with the separator, the spacing and the keyword all optional,
# because a model that writes "6: even = 'M'" has not made a reasoning mistake
_STEP = re.compile(
    r"""^\s*
        (?:line\s*)?            # the word is optional
        (\d+)                   # the line number
        \s*[|:\-]?\s*           # whatever separator it chose
        (.*)$""",
    re.X)


def _drop_degenerate_tail(steps, run: int = 4):
    """Cut a trace where it stops saying anything new.

    A correct trace repeats line numbers, since a loop body runs again and again, which is
    the whole point. What is not informative is the same line repeated with no value
    changing, which is what greedy decoding produces when it runs past the end of a program
    and keeps emitting until the token budget stops it. The degenerate tail carries no
    values; a real loop does.
    """
    if len(steps) <= run:
        return steps
    for start in range(len(steps) - run + 1):
        window = steps[start:start + run]
        if all(not changed for _, changed in window) and \
                len({lineno for lineno, _ in window}) == 1:
            return steps[:start]
    return steps


def parse_trace(reply: str, marker: str = "### TRACE",
                drop_degenerate: bool = True) -> list[tuple[int, dict]]:
    """Read a trace out of a reply, keeping every line that can be read at all.

    Parsed per line rather than per trace: one unreadable step costs that step, not the whole
    sequence. A step whose assignments cannot be read is kept with an empty dict, because the
    line number is still evidence about the path even when the values are lost.
    """
    if marker in reply:
        reply = reply.split(marker, 1)[1]
    steps: list[tuple[int, dict]] = []
    for raw in reply.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _STEP.match(raw)
        if not match:
            continue
        lineno, rest = int(match.group(1)), match.group(2)
        steps.append((lineno, _assignments(rest)))
    return _drop_degenerate_tail(steps) if drop_degenerate else steps


def _assignments(text: str) -> dict:
    """The name=value pairs on one step, as far as they can be read.

    Splitting on commas would break a list or a dict, so the split is on commas that sit
    outside any bracket or quote.
    """
    out: dict = {}
    for piece in _split_top_level(text):
        if "=" not in piece:
            continue
        name, _, value = piece.partition("=")
        name = name.strip()
        if not name.isidentifier():
            continue
        value = value.strip()
        try:
            out[name] = ast.literal_eval(value)
        except Exception:
            out[name] = value          # kept as text, so a near miss is still comparable
    return out


def _split_top_level(text: str) -> list[str]:
    pieces, depth, quote, current = [], 0, None, []
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
            current.append(char)
        elif char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            pieces.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        pieces.append("".join(current))
    return pieces


# ------------------------------------------------------------------------------- alignment

def _state_at(steps: list[tuple[int, dict]]) -> list[dict]:
    """What every variable holds after each step, built up from the changes.

    The traces record changes, but a model will often restate a value that has not moved:
    a loop variable rebound to the same value is reported every pass, while the trace has
    nothing there because nothing changed. Comparing state rather than changes scores that
    as right, which it is, and still catches a value that is genuinely wrong.
    """
    state, out = {}, []
    for _, changed in steps:
        state.update(changed)
        out.append(dict(state))
    return out


def align(claimed: list[tuple[int, dict]], real: list[tuple[int, dict]]) -> dict:
    """Where the model's trace first parts from the real one, and how.

    Edit distance over the line sequence, because a model that skips a step should be scored
    as having skipped one rather than as having got every later step wrong. Values are
    compared only at steps the alignment matched, so a path error and a value error are
    separable: a model can follow the right statements and still carry the wrong numbers.
    """
    claimed_path = [lineno for lineno, _ in claimed]
    real_path = [lineno for lineno, _ in real]

    # a truth step that records no change is one the model may reasonably leave out: the
    # loop header re-evaluated, a condition re-tested, the return. Skipping one costs
    # nothing, so a trace is not punished for being terser than the tracer.
    free_to_skip = [not changed for _, changed in real]
    if free_to_skip:
        # the first step binds the parameters before any statement runs. A model would not
        # call that an executed statement, but the values have to stay in the truth's state
        # or every mention of a parameter later reads as wrong.
        free_to_skip[0] = True

    # standard edit distance, kept explicit so the backtrace can report the operations
    rows, cols = len(claimed_path) + 1, len(real_path) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for j in range(1, cols):
        cost[0][j] = cost[0][j - 1] + (0 if free_to_skip[j - 1] else 1)
    for i in range(1, rows):
        for j in range(1, cols):
            same = claimed_path[i - 1] == real_path[j - 1]
            skip = 0 if free_to_skip[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j] + 1,
                             cost[i][j - 1] + skip,
                             cost[i - 1][j - 1] + (0 if same else 1))

    pairs, i, j = [], len(claimed_path), len(real_path)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + \
                (0 if claimed_path[i - 1] == real_path[j - 1] else 1):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))          # a step the model invented
            i -= 1
        else:
            skipped_freely = free_to_skip[j - 1]
            pairs.append((None, j - 1) if not skipped_freely else (None, None))
            j -= 1
    pairs.reverse()

    claimed_state, real_state = _state_at(claimed), _state_at(real)

    first_path_error = None
    matched = value_right = value_wrong = 0
    first_value_error = None
    for position, (ci, ri) in enumerate(pairs):
        if ci is None and ri is None:
            continue                             # a no-change step the model left out
        if ci is None or ri is None or claimed_path[ci] != real_path[ri]:
            if first_path_error is None:
                first_path_error = position
            continue
        matched += 1
        # only the variables the model mentioned by this point are judged: it is not wrong
        # for staying silent about one, only for getting one it named wrong
        mine, truth = claimed_state[ci], real_state[ri]
        if all(name in truth and truth[name] == value for name, value in mine.items()):
            value_right += 1
        else:
            value_wrong += 1
            if first_value_error is None:
                first_value_error = position

    return {
        "steps_real": len(real_path),
        "steps_claimed": len(claimed_path),
        "path_distance": cost[len(claimed_path)][len(real_path)],
        "path_exact": cost[len(claimed_path)][len(real_path)] == 0,
        "first_path_error": first_path_error,
        "matched_steps": matched,
        "values_right": value_right,
        "values_wrong": value_wrong,
        "first_value_error": first_value_error,
        "values_exact": value_wrong == 0 and matched == len(real_path),
    }
