from probe.constructs import checklist, find_constructs
from probe.mutants import constant_unfold_add, sequential_rename

LOOP_WITH_BRANCH = """
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

ONE_LINER = """
def f(n):
    if n > 0: return 1
    return 0
""".strip()


def test_constructs_are_numbered_in_source_order():
    ids = [c.id for c in find_constructs(LOOP_WITH_BRANCH)]
    assert ids == ["loop1", "if1"]


def test_one_liner_is_flagged_unmeasurable():
    ifs = [c for c in find_constructs(ONE_LINER) if c.kind == "if"]
    assert ifs[0].measurable is False
    assert "if1" not in checklist(find_constructs(ONE_LINER))


def test_ids_survive_mutation():
    renamed, _ = sequential_rename(LOOP_WITH_BRANCH)
    original_ids = [c.id for c in find_constructs(LOOP_WITH_BRANCH)]
    assert [c.id for c in find_constructs(renamed)] == original_ids


def test_elif_becomes_its_own_construct():
    code = "def f(n):\n    if n > 0:\n        return 1\n    elif n < 0:\n        return -1\n    return 0"
    assert [c.id for c in find_constructs(code)] == ["if1", "if2"]


def test_constant_unfold_needs_an_integer():
    code = "def f(s):\n    return s.upper()"
    try:
        constant_unfold_add(code)
    except ValueError:
        return
    raise AssertionError("expected ValueError when there is no integer to unfold")


def test_unfold_rewrites_integers_but_not_booleans():
    mutated, _ = constant_unfold_add("def f(n):\n    return n + 2")
    assert mutated == "def f(n):\n    return n + (1 + 1)"

    # Booleans are integers in Python; rewriting them would change the source in a way
    # a reader would notice, so the operator leaves them alone.
    try:
        constant_unfold_add("def f():\n    return True")
    except ValueError:
        return
    raise AssertionError("True should not count as an integer constant")
