from probe.dataset import call_source
from probe.mutants import constant_unfold_add, sequential_rename
from probe.truth import run_traced

RECURSIVE = """
def f(text, suffix):
    if suffix and suffix[-1] in text:
        return f(text.rstrip(suffix[-1]), suffix[:-1])
    return text
""".strip()

WITH_FIXTURE = """
fixture = [1, 2, 3]
def f(items):
    return sum(items)
""".strip()

WITH_LAMBDA = """
def f(rows):
    return sorted(rows, key=lambda row: len(row))
""".strip()


def behaves_the_same(original, mutated, arguments):
    """Run both versions on the same arguments.

    The call is built per version because sequential renaming changes the function name.
    """
    first, _ = run_traced(original, call_source(original, arguments))
    second, _ = run_traced(mutated, call_source(mutated, arguments))
    return repr(first) == repr(second)


def test_rename_keeps_the_function_name_so_recursion_still_works():
    mutated, _ = sequential_rename(RECURSIVE)
    assert behaves_the_same(RECURSIVE, mutated, "'hello!!', '!'")


def test_rename_keeps_module_level_fixtures():
    mutated, _ = sequential_rename(WITH_FIXTURE)
    assert "fixture" in mutated                      # benchmark inputs refer to it by name
    assert behaves_the_same(WITH_FIXTURE, mutated, "fixture")


def test_rename_handles_lambda_parameters():
    mutated, _ = sequential_rename(WITH_LAMBDA)
    assert behaves_the_same(WITH_LAMBDA, mutated, "['abc', 'a']")


def test_rename_reports_the_mapping_back_to_original_names():
    mutated, mapping = sequential_rename("def f(text):\n    total = len(text)\n    return total")
    assert mapping["generic_function1"] == "f"
    assert mapping["var1"] == "text" and mapping["var2"] == "total"


def test_rename_changes_the_function_name_like_mucoco_does():
    mutated, _ = sequential_rename("def f(n):\n    return n")
    assert mutated.startswith("def generic_function1(")


def test_unfold_preserves_behaviour():
    code = "def f(n):\n    return n * 3"
    mutated, _ = constant_unfold_add(code)
    assert behaves_the_same(code, mutated, "4")
