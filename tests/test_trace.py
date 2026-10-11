"""Tests for the trace design, written against cases whose answer is known by hand."""
import pytest

from probe.trace_score import align, parse_trace
from probe.trace_truth import TraceTooLong, counts_from, trace, writes_from

MAMMOTH = """def f(text):
    odd = ''
    even = ''
    for i, c in enumerate(text):
        if i % 2 == 0:
            even += c
        else:
            odd += c
    return even + odd.lower()
"""


def real_trace():
    return trace(MAMMOTH, "f", "Mammoth")


def test_a_change_is_attached_to_the_line_that_caused_it():
    """f_locals is read at the start of a line, so a naive tracer puts every value one step
    late and a model writing a correct trace would be marked wrong everywhere."""
    steps = dict((lineno, changed) for lineno, changed in real_trace() if changed)
    assert steps[2] == {"odd": ""}
    assert steps[3] == {"even": ""}


def test_a_change_on_the_return_line_is_not_lost():
    """There is no line event after the last statement, so the return event has to be used."""
    steps = trace("def f(xs):\n    return xs.pop()\n", "f", [1, 2, 3])
    assert any("xs" in changed for _, changed in steps), "the pop was never recorded"


def test_the_path_alternates_with_the_branch():
    path = [lineno for lineno, _ in real_trace()]
    assert path[3:9] == [4, 5, 6, 4, 5, 8], path[3:9]


def test_construct_counts_are_recoverable():
    """If they were not, replacing the two current questions would orphan every figure."""
    assert counts_from(real_trace(), MAMMOTH) == {"loop1": (7,), "if1": (4, 3)}


def test_variable_counts_are_recoverable():
    counts = writes_from(real_trace())
    assert counts["even"] == 5 and counts["odd"] == 4


def test_the_cap_refuses_rather_than_truncates():
    """A truncated trace would be scored as a model that stopped early."""
    with pytest.raises(TraceTooLong):
        trace("def f(n):\n    for i in range(n):\n        pass\n", "f", 500, cap=20)


def test_loose_formatting_is_not_a_reasoning_failure():
    strict = parse_trace("### TRACE\nline 4 | i=0, c='M'")
    loose = parse_trace("### TRACE\n4: i = 0, c = 'M'")
    assert strict == loose == [(4, {"i": 0, "c": "M"})]


def test_a_list_value_is_not_split_on_its_commas():
    assert parse_trace("### TRACE\nline 2 | xs=[1, 2, 3]") == [(2, {"xs": [1, 2, 3]})]


def test_a_skipped_step_costs_one_step_not_all_of_them():
    real = [(2, {}), (3, {}), (4, {}), (5, {})]
    claimed = [(2, {}), (4, {}), (5, {})]
    result = align(claimed, real)
    assert result["path_distance"] == 1
    assert result["first_path_error"] == 1


def test_the_path_can_be_right_while_the_values_are_wrong():
    real = [(2, {"x": 1}), (3, {"x": 2})]
    claimed = [(2, {"x": 1}), (3, {"x": 9})]
    result = align(claimed, real)
    assert result["path_exact"] and result["values_wrong"] == 1
    assert result["first_value_error"] == 1


def test_a_perfect_trace_reports_no_error():
    steps = real_trace()
    result = align(steps, steps)
    assert result["path_exact"] and result["values_exact"]
    assert result["first_path_error"] is None and result["first_value_error"] is None
