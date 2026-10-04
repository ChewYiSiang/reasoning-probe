import json

from probe.parse import parse_profile, parse_reply, split_sections
from probe.prompt import ProfilePrompt, checklist

CODE = """
def f(text):
    b = True
    for x in text:
        if x.isdigit():
            b = True
        else:
            b = False
            break
    return b
""".strip()


def fake_helper():
    """Stands in for their prompt helper, which returns the template text when called."""
    return "Program:\n{full_sol}\nInput: {test_input}\nAnswer:"


def test_checklist_lists_every_measurable_construct():
    rows = checklist(CODE).splitlines()
    assert any(row.startswith("loop1") for row in rows)
    assert any(row.startswith("if1") and "taken" in row for row in rows)


def test_prompt_keeps_the_original_and_appends_the_profile_question():
    # Their tester calls the helper, then formats what it returns.
    template = ProfilePrompt(fake_helper)()
    prompt = template.format(full_sol=CODE, test_input="'-1-3'")
    assert prompt.startswith("Program:")          # their part comes first, unchanged
    assert "### ANSWER" in prompt and "### PROFILE" in prompt
    assert "loop1" in prompt


def test_wrapper_can_also_take_a_plain_template_string():
    template = ProfilePrompt("Program:\n{full_sol}\nAnswer:")()
    assert "### PROFILE" in template.format(full_sol=CODE)


def test_reply_splits_into_answer_and_profile():
    reply = "### ANSWER\nFalse\n\n### PROFILE\nloop1 iterations = 1\nif1 taken = 0, not taken = 1"
    answer, profile = parse_reply(reply)
    assert answer == "False"
    assert profile == {"loop1": (1,), "if1": (0, 1)}


def test_profile_survives_loose_formatting():
    # "else" is the earlier wording and is still read, so old logs stay usable.
    text = """
    loop1  (line 3: for x in text:)  iterations = 12
    if1 taken = 3, else = 0
    """
    assert parse_profile(text) == {"loop1": (12,), "if1": (3, 0)}


def test_reply_without_a_profile_still_yields_the_answer():
    answer, profile = parse_reply("False")
    assert answer == "False" and profile == {}


def test_counts_are_read_even_when_the_source_line_contains_equals_signs():
    # The model echoes the construct's own line, which can contain "=" inside a lambda
    # or a comparison. Real replies from the first probe run:
    text = (
        "loop1 (line 2: for k,v in sorted(dic.items(), key=lambda x: len(str(x)))[:-1]:) iterations = 4\n"
        "loop2 (line 3: while s[:len(x)] == x and count < len(s)-len(x):) iterations = 0\n"
        "if1 (line 4: if a == b and c != d:) taken = 2, not taken = 1"
    )
    assert parse_profile(text) == {"loop1": (4,), "loop2": (0,), "if1": (2, 1)}


def test_a_bare_count_is_still_read():
    assert parse_profile("loop2: 7") == {"loop2": (7,)}


def test_no_constructs_line_yields_nothing():
    assert parse_profile("(this program has no loops or branches)") == {}


def test_mucoco_consistency_rule_matches_their_three_categories():
    from probe.analyse import inconsistent, verdict

    assert verdict({"failure_type": ""}) == "correct"
    assert verdict({"failure_type": "AssertionError > "}) == "incorrect"
    assert verdict({"failure_type": "LLMExecutionRuntimeError > x"}) == "invalid"
    assert verdict({"failure_type": "NoConstantUnfoldError > x"}) == "skipped"

    assert inconsistent("correct", "incorrect", "a", "b") is True     # correctness-based
    assert inconsistent("incorrect", "incorrect", "a", "b") is True   # incorrectness-based
    assert inconsistent("incorrect", "incorrect", "a", "a") is False  # wrong the same way
    assert inconsistent("invalid", "correct", "", "a") is True        # invalidity-based
    assert inconsistent("correct", "correct", "a", "a") is False
    assert inconsistent("skipped", "correct", "", "") is None         # not comparable


def test_profile_verdict_uses_the_same_categories():
    from probe.analyse import profile_verdict

    assert profile_verdict({"profile": {"loop1": [3]}}, {"loop1": (3,)}) == "correct"
    assert profile_verdict({"profile": {"loop1": [2]}}, {"loop1": (3,)}) == "incorrect"
    assert profile_verdict({"profile": {}}, {"loop1": (3,)}) == "invalid"
    assert profile_verdict({"profile": {}}, {}) == "skipped"


def test_statistics_match_values_that_can_be_checked_by_hand():
    from probe.stats import fisher_exact, mcnemar, wilson

    rate_, low, high = wilson(13, 50)
    assert round(rate_, 2) == 0.26 and round(low, 3) == 0.159 and round(high, 3) == 0.396

    assert round(mcnemar(3, 7), 4) == 0.3438        # same as a two-sided binomial test
    assert mcnemar(0, 0) == 1.0
    assert round(fisher_exact(9, 1, 2, 8), 5) == 0.00548
    assert fisher_exact(5, 5, 5, 5) == 1.0


def test_gpu_wrapper_returns_only_the_answer_and_logs_the_profile(tmp_path, monkeypatch):
    """The Colab path builds its own model object, so the wrapper is checked in isolation."""
    import sys, types

    gpu = types.ModuleType("llm_models.gpu_code_llms")

    class FakeTransformers:
        def obtain_max_new_tokens(self, answers):
            self.max_new_token = 32

    gpu.TransformersCodeLLM = FakeTransformers

    # The wrapper loads the model itself, so torch and transformers are stubbed too.
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.float16 = "fp16"
    transformers = types.ModuleType("transformers")
    class FakeTokenizer:
        chat_template = "yes"
        eos_token_id = 0
        seen = ""

        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                enable_thinking=None):
            return "<chat>" + messages[0]["content"]

        def __call__(self, text, return_tensors=None):
            FakeTokenizer.seen = text
            return types.SimpleNamespace(
                to=lambda device: {"input_ids": types.SimpleNamespace(shape=[1, 3])})

        def decode(self, ids, skip_special_tokens=True):
            return "### ANSWER\n42\n\n### PROFILE\nloop1 iterations = 3"

        def encode(self, text):
            return [0] * len(text)

    tokenizer = FakeTokenizer()
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: tokenizer)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda name, **kwargs: types.SimpleNamespace(
            device="cpu", generate=lambda **kwargs_: [[0, 0, 0, 1, 2, 3]]))

    class NoGrad:
        def __enter__(self): return None
        def __exit__(self, *args): return False

    torch.no_grad = lambda: NoGrad()
    transformers.BitsAndBytesConfig = lambda **kwargs: kwargs

    monkeypatch.setitem(sys.modules, "llm_models", types.ModuleType("llm_models"))
    monkeypatch.setitem(sys.modules, "llm_models.gpu_code_llms", gpu)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setenv("PROBE_PROFILE_LOG", str(tmp_path / "profiles.jsonl"))

    from probe.probe_llm_gpu import profile_transformers_class

    llm = profile_transformers_class()(model_name="fake/model", answers=["123"])
    assert llm.max_new_token > 32                 # room for the profile as well

    prompt_template = types.SimpleNamespace(format=lambda **kwargs: "PROMPT")
    result = llm.invoke({"full_sol": "def f(): pass", "test_input": "1"}, prompt_template)
    assert result["ans"] == "42"                  # their parser sees only the answer
    assert FakeTokenizer.seen.startswith("<chat>")   # the chat template was applied

    import json
    logged = json.loads((tmp_path / "profiles.jsonl").read_text().strip())
    assert logged["profile"] == {"loop1": [3]}


def test_construct_features_describe_shape_not_just_kind():
    from probe.diagnose import bucket, construct_features

    code = (
        "def f(rows):\n"
        "    total = 0\n"
        "    for row in rows:\n"
        "        while total < 10:\n"
        "            total += 1\n"
        "            break\n"
        "        if total > 5:\n"
        "            total = 0\n"
        "        else:\n"
        "            total += 2\n"
        "    return total\n"
    )
    features = construct_features(code)
    assert features["loop1"] == {"kind": "loop", "loop_type": "for", "depth": 0, "early_exit": True}
    assert features["loop2"]["loop_type"] == "while" and features["loop2"]["depth"] == 1
    assert features["if1"]["has_else"] is True and features["if1"]["depth"] == 1

    labels = bucket("if1", (0, 0), features["if1"])
    assert "branch, never evaluated" in labels and "nested" in labels
    assert "branch, condition always false" in bucket("if1", (0, 3), features["if1"])


def test_offline_collection_serves_what_the_tester_asks_for():
    from probe.offline_db import OfflineCollection

    documents = [{"_id": "CruxEvalTF0", "full_sol": "def f(): pass",
                  "input": {"args": "1", "metadata": "int"},
                  "output": {"args": "1", "metadata": "int"}},
                 {"_id": "CruxEvalTF1", "full_sol": "def f(x): return x",
                  "input": {"args": "2", "metadata": "int"},
                  "output": {"args": "2", "metadata": "int"}}]
    collection = OfflineCollection(documents)

    assert collection.count_documents({}) == 2
    assert collection.find_one({"_id": "CruxEvalTF1"})["full_sol"] == "def f(x): return x"
    assert collection.find_one({"_id": "missing"}) is None

    projected = list(collection.find({}, {"_id": 0, "output": 1}))
    assert projected == [{"output": d["output"]} for d in documents]


def test_make_tester_builds_one_without_touching_mongo(tmp_path):
    import json

    from probe.offline_db import make_tester

    class FakeTester:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("would connect to MongoDB")

        def run(self):
            return self.question_database.count_documents({})

    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{"_id": "CruxEvalTF0", "full_sol": "def f(): pass"}]))

    tester = make_tester(FakeTester, str(path))
    assert tester.run() == 1


def test_facts_block_reads_like_a_statement_of_what_happened():
    from probe.facts import corrupt, render

    constructs = [{"id": "loop1", "header_line": 3}, {"id": "if1", "header_line": 4}]
    text = render({"loop1": (4,), "if1": (2, 2)}, constructs)
    assert "the loop at line 3 runs 4 times" in text
    assert "the condition at line 4 is true 2 times and false 2 times" in text

    # Corrupted facts must differ from the real ones everywhere, including zeros.
    wrong = corrupt({"loop1": (0,), "if1": (2, 0)})
    assert wrong == {"loop1": (2,), "if1": (3, 2)}


def test_leak_detection_catches_the_counting_functions():
    from probe.facts import leaks_answer

    counts = {"loop1": (4,), "if1": (2, 2)}
    assert leaks_answer(2, counts) is True            # answer is one of the counts
    assert leaks_answer([1, 2, 3, 4], counts) is True # answer's length is a count
    assert leaks_answer(99, counts) is False
    assert leaks_answer(True, counts) is False        # booleans are not read off counts


def test_facts_prompt_appends_only_for_the_task_it_knows():
    from probe.facts import FactsPrompt

    code = "def f(n):\n    return n"
    prompt = FactsPrompt(lambda: "CODE:\n{full_sol}\nANSWER:", {code: "### EXECUTION FACTS\nx"})()
    assert "EXECUTION FACTS" in prompt.format(full_sol=code)
    assert "EXECUTION FACTS" not in prompt.format(full_sol="def g(): pass")


def test_facts_can_be_placed_before_the_task():
    from probe.facts import FactsPrompt

    code = "def f(n):\n    return n"
    facts = {code: "### EXECUTION FACTS\nx"}
    after = FactsPrompt(lambda: "CODE:\n{full_sol}\n# Your answer", facts)()
    before = FactsPrompt(lambda: "CODE:\n{full_sol}\n# Your answer", facts, where="before")()

    assert after.format(full_sol=code).rstrip().endswith("x")
    assert before.format(full_sol=code).startswith("### EXECUTION FACTS")


def test_detector_evaluation_counts_precision_and_recall_correctly(tmp_path):
    """A small synthetic set where the right answer is known by hand."""
    import csv as csv_module
    import json

    from probe.detectors import evaluate

    # Three pairs, all with one loop that really runs 3 times:
    #   t1  both profiles right, answers agree          -> not misread, no flag
    #   t2  profiles disagree (one wrong), answers agree -> misread, profile flag only
    #   t3  both profiles wrong the same way, answers differ -> misread, output flag only
    truth, code_index, codes = {}, {}, {}
    for task, code in (("t1", "def f(a):\n    for x in a:\n        pass"),
                       ("t2", "def f(b):\n    for y in b:\n        pass"),
                       ("t3", "def f(c):\n    for z in c:\n        pass")):
        for version in ("no_mutation", "sequential"):
            marked = code if version == "no_mutation" else code + "  # mutant"
            truth[(task, version)] = {"counts": {"loop1": (3,)}, "code": marked}
            code_index[marked] = (task, version)
            codes[(task, version)] = marked

    claims = {("t1", "no_mutation"): [3], ("t1", "sequential"): [3],
              ("t2", "no_mutation"): [3], ("t2", "sequential"): [4],
              ("t3", "no_mutation"): [9], ("t3", "sequential"): [9]}
    answers = {("t1", "no_mutation"): "", ("t1", "sequential"): "",
               ("t2", "no_mutation"): "", ("t2", "sequential"): "",
               ("t3", "no_mutation"): "", ("t3", "sequential"): "AssertionError > "}

    results_dir, probe_dir = tmp_path / "results", tmp_path / "probe"
    results_dir.mkdir(), probe_dir.mkdir()
    for version in ("no_mutation", "sequential"):
        with open(results_dir / f"CruxEval_zero_shot_{version}.csv", "w",
                  newline="", encoding="utf-8") as handle:
            writer = csv_module.DictWriter(handle, ["task_id", "model_output", "failure_type"])
            writer.writeheader()
            for task in ("t1", "t2", "t3"):
                writer.writerow({"task_id": task, "model_output": "x",
                                 "failure_type": answers[(task, version)]})
        with open(probe_dir / f"profiles_{version}.jsonl", "w", encoding="utf-8") as handle:
            for task in ("t1", "t2", "t3"):
                handle.write(json.dumps({"full_sol": codes[(task, version)],
                                         "test_input": "x",
                                         "profile": {"loop1": claims[(task, version)]},
                                         "raw_reply": ""}) + "\n")

    stats, pairs = evaluate(truth, code_index, str(results_dir), str(probe_dir),
                            "sequential", "CruxEval", "zero_shot")

    assert pairs == 3
    assert stats["misread pairs"] == 2                  # t2 and t3
    assert stats["profile: flagged"] == 1               # t2 only
    assert stats["profile: false positive"] == 0        # a disagreement is always a real error
    assert stats["output: flagged"] == 1                # t3 only
    assert stats["either: true positive"] == 2          # together they find both
