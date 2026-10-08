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


def test_straight_line_programs_still_reach_the_data_check():
    """251 of the 800 CRUXEval programs have no loop or branch, but most still have a
    variable worth asking about. An early guard was dropping those pairs before the
    data-flow half of the check ever ran, which cost about a third of the triggers."""
    from probe.analyse import inconsistent, profile_verdict

    # a program with nothing to count: the control-flow half returns skipped, not a firing
    assert profile_verdict({"profile": {}}, {}) == "skipped"
    assert inconsistent("skipped", "skipped", "", "") is None

    # so it cannot fire on control flow, and the data half decides the pair on its own
    assert bool(inconsistent("skipped", "skipped", "", "")) is False
    assert inconsistent("correct", "incorrect", "3", "5") is True


def test_renamed_versions_quote_counts_not_names():
    """Under sequential renaming the second version uses different variable names, and the
    model is shown the first version's code. Quoting the second version's names would name
    variables that do not appear in the program it is looking at."""
    from probe.retry import retry_prompt

    base = "# Code Snippet\ndef f(a):\n    ...\n\n# Your answer"

    text = retry_prompt(base, {"loop1": [4]}, {"loop1": [5]},
                        {"dic": 5, "x": 0}, {"var1": 5, "var2": 5})
    assert "the variables changed 5, 0 times" in text
    assert "the variables changed 5, 5 times" in text
    assert "var1" not in text and "dic" not in text      # no name the reader cannot place

    # same names, as under constant unfolding: the names are useful and are kept
    text = retry_prompt(base, {"loop1": [3]}, {"loop1": [3]}, {"total": 3}, {"total": 5})
    assert "total changed 3 times" in text and "total changed 5 times" in text

    # renamed but the counts agree: that field has no conflict to show
    text = retry_prompt(base, {"loop1": [4]}, {"loop1": [5]}, {"dic": 5}, {"var1": 5})
    assert "the variables changed" not in text


def test_the_retry_run_writes_a_scorable_csv(tmp_path):
    """The retry prompts are already complete, so the run does not go through their tester.
    It still has to write their CSV shape, or the retry cannot be scored against the run
    that produced it."""
    import csv
    import json

    from probe.analyse import read_results_csv, verdict
    from probe.retry_run import run, same_answer

    assert same_answer("[1,2]", "[1, 2]")          # parsed as values, not as text
    assert not same_answer("4", "5")
    assert same_answer("abc", "abc")               # falls back to text when it will not parse

    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"task_id": "t1", "prompt": "p", "answer_before": "incorrect"}) + "\n",
        encoding="utf-8")
    with open(tmp_path / "before.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "prompt", "model_output", "expected_output",
                         "failure_type"])
        writer.writerow(["t1", "", "3", "4", "AssertionError: "])

    class Stub:
        def invoke(self, variables, template):
            assert template.format(**variables) == "p"
            return "4"                              # the re-ask settles on the right answer

    counts = run(str(tmp_path / "prompts.jsonl"), str(tmp_path / "before.csv"),
                 str(tmp_path / "after.csv"), Stub())
    assert counts == {"written": 1, "correct": 1, "incorrect": 0}

    # and the scorer reads it back as their own result log
    rows = read_results_csv(str(tmp_path / "after.csv"))
    assert verdict(rows["t1"]) == "correct"


def test_the_comparison_matches_what_their_tester_writes(tmp_path):
    """Their tester writes the whole output record into expected_output, while the value it
    compares against is the record's 'args' field; and the wrapper logs the answer as a
    (value, type) pair whose value is itself a repr. Comparing the two as written never
    matches, which silently scores every answer wrong."""
    import csv

    from probe.analyse import read_results_csv, verdict
    from probe.retry_run import rescore, same_answer

    # three shapes appear in the result files and all three must unwrap
    assert same_answer("{'ans': '4', 'geom_mean_prob': 0.1}", "{'args': '4', 'metadata': 'int'}")
    assert not same_answer("{'ans': '5', 'geom_mean_prob': 0.1}", "{'args': '4', 'metadata': 'int'}")
    assert same_answer("{'ans': '{1: None}', 'geom_mean_prob': 0.0}",
                       "{'args': '{1: None}', 'metadata': 'dict'}")
    assert same_answer("(4, <class 'int'>)", "{'args': '4', 'metadata': 'int'}")
    assert not same_answer("(5, <class 'int'>)", "{'args': '4', 'metadata': 'int'}")
    assert same_answer("('[1, 2]', <class 'list'>)", "{'args': '[1, 2]', 'metadata': 'list'}")
    assert same_answer("(True, <class 'bool'>)", "True")        # input prediction
    assert same_answer("4", "{'args': '4', 'metadata': 'int'}")  # a bare answer still works

    header = ["task_id", "prompt", "model_output", "expected_output", "failure_type"]
    want = "{'args': '4', 'metadata': 'int'}"
    with open(tmp_path / "before.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["t1", "", "(3, <class 'int'>)", want, "AssertionError: "])
    with open(tmp_path / "after.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(["t1", "", "(4, <class 'int'>)", want, "AssertionError: "])

    counts = rescore(str(tmp_path / "after.csv"), str(tmp_path / "before.csv"))
    assert counts["correct"] == 1
    assert verdict(read_results_csv(str(tmp_path / "after.csv"))["t1"]) == "correct"


def test_the_filter_drops_what_each_check_flags(tmp_path):
    """MuCoCo's Appendix I keeps only answers the model was confident about and reports what
    that does to inconsistency and accuracy. This is the same table with the reasoning check
    as the signal, so each filter has to drop exactly the pairs its own check fires on."""
    import csv
    import json

    from probe.filter import kept
    from probe.reasoning import load_side
    from probe.score import load_truth

    programs = {f"t{i}": f"def f():\n    return {i}" for i in range(1, 5)}
    for name, build in (
            ("truth.jsonl", lambda code, v: {
                "code": f"{code}  # {v}", "constructs": [{"id": "loop1", "header_line": 2}],
                "profile": {"loops": {"loop1": 2}, "branches": {}, "statement_counts": {}}}),
            ("writes.jsonl", lambda code, v: {
                "code": f"{code}  # {v}", "asked": ["x"], "informative": {"x": 2},
                "all_writes": {}})):
        with open(tmp_path / name, "w", encoding="utf-8") as handle:
            for task, code in programs.items():
                handle.write(json.dumps({"task_id": task, "versions": {
                    v: build(code, v) for v in ("no_mutation", "sequential")}}) + "\n")

    # t1 agrees throughout; t2's answers differ; t3's loop counts differ; t4 both
    replies = {("no_mutation", "t1"): ("1", 2), ("sequential", "t1"): ("1", 2),
               ("no_mutation", "t2"): ("2", 2), ("sequential", "t2"): ("9", 2),
               ("no_mutation", "t3"): ("3", 2), ("sequential", "t3"): ("3", 5),
               ("no_mutation", "t4"): ("4", 2), ("sequential", "t4"): ("8", 7)}
    for version in ("no_mutation", "sequential"):
        with open(tmp_path / f"CruxEval_zero_shot_{version}.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task_id", "prompt", "model_output", "expected_output",
                             "failure_type"])
            for task in programs:
                answer = replies[(version, task)][0]
                writer.writerow([task, "", answer, task[1],
                                 "" if answer == task[1] else "AssertionError: "])
        with open(tmp_path / f"profiles_{version}.jsonl", "w", encoding="utf-8") as handle:
            for task, code in programs.items():
                answer, loops = replies[(version, task)]
                handle.write(json.dumps({
                    "task_id": task, "full_sol": f"{code}  # {version}",
                    "profile": {"loop1": [loops]},
                    "raw_reply": f"### ANSWER\n{answer}\n\n### PROFILE\n"
                                 f"loop1 iterations = {loops}\n\n### WRITES\nx = 2"}) + "\n")

    truth = load_truth(str(tmp_path / "truth.jsonl"))
    writes = {}
    with open(tmp_path / "writes.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                writes[(row["task_id"], version)] = entry

    sides = tuple(load_side(truth, writes, tmp_path, tmp_path, version, "CruxEval",
                            "zero_shot") for version in ("no_mutation", "sequential"))

    survives = lambda drop_a, drop_r: {t for t in programs if kept(sides, t, drop_a, drop_r)}
    assert survives(False, False) == {"t1", "t2", "t3", "t4"}    # no filter
    assert survives(True, False) == {"t1", "t3"}                 # the answers drop t2 and t4
    assert survives(False, True) == {"t1", "t2"}                 # the reasoning drops t3 and t4
    assert survives(True, True) == {"t1"}                        # either drops three
