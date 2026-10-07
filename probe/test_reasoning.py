"""The reasoning check: one check, two fields, and the arithmetic that relates them."""

from __future__ import annotations

from collections import Counter


def test_the_two_fields_are_not_additive():
    """RQ2's table showed control flow and both fields together, so a reader had to work out
    the data-flow field by subtracting. That is wrong: the fields overlap, so subtracting
    undercounts it. This pins the identity the scorer now prints instead."""
    # (control flow disagrees, variable updates disagree) for eight pairs
    pairs = [(True, False), (True, False), (False, True), (False, True),
             (False, True), (True, True), (True, True), (False, False)]

    stats = Counter()
    for profiles, counts in pairs:
        reasoning = profiles or counts          # the check fires on either field
        stats["control"] += profiles
        stats["updates"] += counts
        stats["reasoning"] += reasoning

    assert stats["control"] == 4
    assert stats["updates"] == 5
    assert stats["reasoning"] == 7              # not 9: two pairs are counted twice

    both = stats["control"] + stats["updates"] - stats["reasoning"]
    assert both == 2                            # inclusion and exclusion recovers the overlap
    assert both / stats["reasoning"] == 2 / 7   # the share the scorer reports

    # the trap this guards against: inferring the data-flow field by subtraction
    inferred = stats["reasoning"] - stats["control"]
    assert inferred == 3
    assert inferred != stats["updates"]         # it really does undercount


def test_the_check_fires_on_either_field_alone():
    """A disagreement in either field is a reasoning disagreement. Four pairs, four cases."""
    cases = {(False, False): False, (True, False): True,
             (False, True): True, (True, True): True}
    for (profiles, counts), expected in cases.items():
        assert (bool(profiles) or bool(counts)) is expected
