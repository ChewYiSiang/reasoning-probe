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
    """Steps carrying a change, so this measures the skip rather than the free-omission
    rule below."""
    real = [(2, {"a": 1}), (3, {"b": 2}), (4, {"c": 3}), (5, {"d": 4})]
    claimed = [(2, {"a": 1}), (4, {"c": 3}), (5, {"d": 4})]
    result = align(claimed, real)
    assert result["path_distance"] == 1
    assert result["first_path_error"] == 1


def test_omitting_a_step_that_changed_nothing_is_free():
    """The tracer records the loop header re-evaluated and the return, where nothing moves.
    A model that leaves those out is being terser, not wrong, and the pilot showed every
    model doing it."""
    real = [(2, {"a": 1}), (3, {}), (4, {"b": 2}), (3, {}), (5, {})]
    claimed = [(2, {"a": 1}), (4, {"b": 2})]
    result = align(claimed, real)
    assert result["path_exact"], result
    assert result["values_wrong"] == 0, result


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


# ----------------------------------------------------------------- from a real pilot reply

APPEND = """def f(nums):
    output = []
    for n in nums:
        output.append((nums.count(n), n))
    output.sort(reverse=True)
    return output
"""


def test_restating_an_unchanged_value_is_not_an_error():
    """A model reports the loop variable on every pass; the trace records it only when it
    changes. Comparing changes marks the model wrong for being more informative, so the
    comparison is on state. Taken from a real reply in the first pilot."""
    real = trace(APPEND, "f", [1, 1, 3])
    claimed = parse_trace("""### TRACE
line 1 | nums=[1, 1, 3]
line 2 | output=[]
line 3 | n=1
line 4 | output=[(2, 1)]
line 3 | n=1
line 4 | output=[(2, 1), (2, 1)]
line 3 | n=3
line 4 | output=[(2, 1), (2, 1), (1, 3)]
line 3 |
line 5 | output=[(2, 1), (2, 1), (1, 3)]
line 6 |""")
    result = align(claimed, real)
    assert result["path_exact"], result
    assert result["values_wrong"] == 0, result


def test_a_trace_numbered_by_step_is_caught():
    """The first pilot reply numbered its steps 1..17 instead of using the program's line
    numbers, which loses the path while leaving the values looking plausible."""
    real = trace(APPEND, "f", [1, 1, 3])
    claimed = parse_trace("### TRACE\n" + "\n".join(
        f"line {i} | " for i in range(1, len(real) + 1)))
    result = align(claimed, real)
    assert not result["path_exact"]
    assert result["path_distance"] > 0


def test_the_stored_truth_round_trips_through_the_parser():
    """The scorer parses the truth back and compares it against the model's parsed value, so
    a truncated literal would come back as a bare string and poison every later comparison
    built on it. Found in the first pilot, where long lists were cut at 40 characters."""
    from probe.trace_truth import as_text
    code = ("def f(nums):\n    out = []\n    for n in nums:\n"
            "        out.append((nums.count(n), n))\n    return out\n")
    real = trace(code, "f", [1, 1, 3, 1, 3, 1])
    assert parse_trace(as_text(real)) == real


def test_a_lambda_on_the_same_line_is_not_a_step():
    """sorted(key=lambda ...) runs the lambda in its own frame on the same source line. The
    tracer used to record each call as a step on that line, which no model would report."""
    code = ("def f(dic):\n"
            "    for k, v in sorted(dic.items(), key=lambda x: len(str(x)))[:-1]:\n"
            "        dic.pop(k)\n"
            "    return list(dic.items())\n")
    steps = trace(code, "f", {"a": 1, "bb": 2, "ccc": 3})
    on_line_two = [changed for lineno, changed in steps if lineno == 2]
    assert all("x" not in changed for changed in on_line_two), steps


def test_a_parameter_stays_in_the_truth_state_even_though_its_step_is_optional():
    """The first step binds the parameters. A model does not call that an executed statement,
    but if the binding is dropped entirely then any later mention of a parameter reads as
    wrong, which is what the first pilot showed."""
    code = "def f(xs):\n    n = len(xs)\n    xs = []\n    return n\n"
    real = trace(code, "f", [1, 2, 3])
    claimed = parse_trace("### TRACE\nline 2 | n=3\nline 3 | xs=[]\nline 4 |")
    result = align(claimed, real)
    assert result["path_exact"], result
    assert result["values_wrong"] == 0, result
