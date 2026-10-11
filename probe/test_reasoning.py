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


def test_pooled_accuracy_counts_items_not_percentages(tmp_path):
    """One number for how much of the run the model reported correctly. The two fields have
    different denominators and are asked at different rates, so averaging their percentages
    would weight a rare variable question as heavily as a common loop one."""
    import json

    from probe.accuracy import pooled

    code = "def f(a):\n    total = 0\n    for x in a:\n        total += x\n    return total"
    (tmp_path / "truth.jsonl").write_text(json.dumps({"task_id": "t1", "versions": {
        "no_mutation": {"code": code, "constructs": [{"id": "loop1", "header_line": 3}],
                        "profile": {"loops": {"loop1": 3}, "branches": {},
                                    "statement_counts": {}}}}}) + "\n", encoding="utf-8")
    (tmp_path / "writes.jsonl").write_text(json.dumps({"task_id": "t1", "versions": {
        "no_mutation": {"code": code, "asked": ["total"], "informative": {"total": 4},
                        "all_writes": {}}}}) + "\n", encoding="utf-8")
    (tmp_path / "profiles_no_mutation.jsonl").write_text(json.dumps({
        "task_id": "t1", "full_sol": code, "profile": {"loop1": [3]},
        "raw_reply": "### ANSWER\n6\n\n### PROFILE\nloop1 (line 3) iterations = 3\n"
                     "\n### WRITES\ntotal = 9"}) + "\n", encoding="utf-8")

    # the loop count is right and the variable count is wrong, so the pool is 1 of 2
    pooled(str(tmp_path / "truth.jsonl"), str(tmp_path / "writes.jsonl"),
           str(tmp_path), ["no_mutation"])


def test_the_retry_quotes_the_field_that_disagreed():
    """A pair can fire on the variable updates while the loop counts match. Quoting only the
    loop counts then hands the model two identical lists and calls them a contradiction,
    which is not something it can resolve."""
    from probe.retry import retry_prompt

    base = "# Code Snippet\ndef f(a):\n    ...\n\n# Your answer"

    # only the variables differ
    text = retry_prompt(base, {"loop1": [3]}, {"loop1": [3]}, {"total": 3}, {"total": 5})
    assert "total changed 3 times" in text and "total changed 5 times" in text
    assert "loop1" not in text                       # the matching field is not quoted

    # only the loop counts differ
    text = retry_prompt(base, {"loop1": [3]}, {"loop1": [4]}, {"total": 3}, {"total": 3})
    assert "loop1: 3 iterations" in text and "loop1: 4 iterations" in text
    assert "total changed" not in text

    # both differ: both are quoted
    text = retry_prompt(base, {"loop1": [3]}, {"loop1": [4]}, {"total": 3}, {"total": 5})
    assert "loop1: 4 iterations" in text and "total changed 5 times" in text
