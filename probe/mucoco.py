"""Bridge to the MuCoCo repository.

The tracer and construct extractor know nothing about MuCoCo. This module is the only
place that does: it reads their task records, applies their mutation operators, and hands
back something the tracer can run.

Nothing here modifies their code. `CodeMutator` is imported and called exactly as
`prediction_inconsistency_tester.py` calls it, so the mutants we trace are the same
mutants the model is shown.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from code_mutation.mutation_functions import (
    CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT, COMMUTATIVE_REORDER,
    BOOLEAN_LITERAL, DEMORGAN, FOR2ENUMERATE, FOR2WHILE, LITERAL_FORMAT,
    CodeMutator, IdenticalMutationError,
)

# Operators that must change the source to count as applied; MuCoCo checks this too.
MUST_DIFFER = (FOR2ENUMERATE, FOR2WHILE, DEMORGAN, LITERAL_FORMAT, BOOLEAN_LITERAL,
               COMMUTATIVE_REORDER, CONSTANT_UNFOLD, CONSTANT_UNFOLD_ADD, CONSTANT_UNFOLD_MULT)

CRUXEVAL_ENTRY_FUNCTION = "f"        # CRUXEval tasks always define f; the tester hardcodes this too


@dataclass
class Task:
    """One benchmark record, normalised out of MongoDB."""
    task_id: str
    code: str
    test_input: Any
    input_metadata: str
    expected_output: Any
    func_name: str = CRUXEVAL_ENTRY_FUNCTION


@dataclass
class Version:
    """A program version to trace: the original, or one mutant of it."""
    name: str                # "no_mutation", or the mutation's own name
    code: str
    func_name: str           # lexical mutations rename the function, so this can change
    error: str | None = None # set when the operator does not apply to this program


def task_from_record(record: dict) -> Task:
    """Normalise a CRUXEval document the way the tester does before use.

    MongoDB cannot store Python objects, so inputs and outputs come back as text plus a
    metadata field naming the original type. Strings are left alone; everything else is
    evaluated back into an object.
    """
    test_input = record["input"]["args"]
    input_metadata = record["input"]["metadata"]
    output_args = record["output"]["args"]
    output_metadata = record["output"]["metadata"]

    if isinstance(test_input, str) and input_metadata != str.__name__:
        test_input = eval(test_input)
    expected_output = (output_args if output_metadata == str.__name__
                       else ast.literal_eval(output_args))

    return Task(
        task_id=record["_id"],
        code=record["full_sol"],
        test_input=test_input,
        input_metadata=input_metadata,
        expected_output=expected_output,
    )


def mutate(task: Task, mutation: str, verify_with_mucoco: bool = False) -> Version:
    """Apply one MuCoCo operator, or report why it does not apply.

    The mutated dictionary has the same shape the tester builds. After a lexical
    mutation the entry function has been renamed, and the mutator records the new name
    on itself, which is what we read back.

    `verify_with_mucoco` runs their validity check as well. It spawns a subprocess per
    mutant, which on Windows re-imports their whole dependency chain each time and makes
    a full pass take hours. It is off by default because the builder verifies every
    mutant anyway, and more strictly: same answer *and* same control flow, measured by
    the tracer.
    """
    mutator = CodeMutator(
        func_name=task.func_name,
        mutated_dict={
            "question": task.code,
            "full_sol": task.code,
            "qn_desc": None,
            "examples": None,
        },
        benchmark_set="CruxEval",
    )
    try:
        if verify_with_mucoco:
            mutator.mutate_for_prediction_inconsistency_test(
                mutation_type=mutation,
                input_args=task.test_input,
                output_args=task.expected_output,
                input_metadata=task.input_metadata,
                task_set="CruxEval",
            )
        else:
            _mutate_without_their_validity_check(mutator, task, mutation)
    except Exception as error:
        # Operators legitimately fail on some programs: no loop for for2while, no integer
        # for constant unfolding. The tester skips those tasks, and so do we.
        return Version(name=mutation, code="", func_name=task.func_name,
                       error=f"{type(error).__name__}: {error}")

    return Version(name=mutation, code=mutator.mutated_dict["full_sol"],
                   func_name=mutator.func_name)


def _mutate_without_their_validity_check(mutator: CodeMutator, task: Task, mutation: str) -> None:
    """The same steps as `mutate_for_prediction_inconsistency_test`, minus the subprocess.

    Kept deliberately close to their method so the mutants are identical: parse, sanitise
    through the AST, check the operator applies, apply it, then check the source actually
    changed. Only their `check_solution_validity` call is left out.
    """
    tree = ast.parse(mutator.mutated_dict["full_sol"])
    sanitised = CodeMutator.parse_through_ast(tree)
    tree = ast.parse(sanitised)

    mutator.check_mutation_validity(tree=tree, mutation_type=mutation,
                                    task_set="CruxEval", input_args=task.test_input)
    mutator.handle_mutation(mutation_type=mutation, task_set="CruxEval",
                            tree=tree, input_args=task.test_input)

    mutated = mutator.mutated_dict["full_sol"]
    mutator.mutated_dict["question"] = mutated
    if mutation in MUST_DIFFER and (CodeMutator.standardize_program(mutated)
                                    == CodeMutator.standardize_program(sanitised)):
        raise IdenticalMutationError(mutation_type=mutation)


def versions_for(task: Task, mutations: list[str], verify_with_mucoco: bool = False) -> list[Version]:
    """The original plus one entry per requested mutation, in a fixed order."""
    out = [Version(name="no_mutation", code=task.code, func_name=task.func_name)]
    out.extend(mutate(task, mutation, verify_with_mucoco) for mutation in mutations)
    return out
