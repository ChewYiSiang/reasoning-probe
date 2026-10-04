import pytest

from probe.constructs import find_constructs
from probe.truth import ExecutionBudgetExceeded, profile, statement_lines

COUNTING = """
def f(items):
    total = 0
    for x in items:
        if x % 2 == 0:
            total += x
        else:
            total -= x
    return total
""".strip()

MULTILINE = """
def f(text):
    if text:
        return (text
                + text)
    return text
""".strip()

NESTED = """
def f(rows):
    seen = 0
    for row in rows:
        for cell in row:
            seen += 1
    return seen
""".strip()


def run(code, call):
    return profile(code, call, find_constructs(code))


def test_loop_and_branch_counts():
    result = run(COUNTING, "f([1, 2, 3, 4])")
    assert result.output == 2
    assert result.loops == {"loop1": 4}
    assert result.branches == {"if1": (2, 2)}      # two even, two odd


def test_branch_never_taken_is_zero_not_missing():
    result = run(COUNTING, "f([2, 4])")
    assert result.branches == {"if1": (2, 0)}


def test_multiline_statement_counts_once():
    result = run(MULTILINE, "f('ab')")
    assert result.branches == {"if1": (1, 0)}


def test_nested_loops():
    result = run(NESTED, "f([[1, 2], [3]])")
    assert result.loops == {"loop1": 2, "loop2": 3}


def test_statement_lines_maps_continuation_lines_to_their_statement():
    owner = statement_lines(MULTILINE)
    assert owner[3] == owner[4] == 3               # the return spans lines 3 and 4


def test_runaway_program_is_stopped():
    code = "def f():\n    n = 0\n    while True:\n        n += 1\n    return n"
    with pytest.raises(ExecutionBudgetExceeded):
        profile(code, "f()", find_constructs(code), step_budget=5_000)
