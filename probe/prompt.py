"""The profile question, added to MuCoCo's own prompt.

The model is asked for MuCoCo's answer first and the execution profile second, in one
reply. Answer first so their task is unchanged: asking for the profile first would turn
it into reasoning-before-answering and would no longer be the same task.

Nothing in their repository is edited. `ProfilePrompt` wraps their template and adds the
profile section at format time, which is possible because the model adapters call
`prompt_template.format(**input_variables)`.
"""

from __future__ import annotations

from probe.constructs import find_constructs

ANSWER_MARK = "### ANSWER"
PROFILE_MARK = "### PROFILE"

INSTRUCTIONS = """
After the answer, report how the program runs for this input.

Write exactly two sections and nothing else:

{answer_mark}
<your answer, in the format described above, on one line>

{profile_mark}
{checklist}

For a loop, give the number of iterations. For a branch, give how many times the
condition was true and how many times it was false. Write 0 where something never
happens.
"""


def checklist(code: str) -> str:
    """One line per loop and branch, generated from the version of the code being shown."""
    rows = []
    for construct in find_constructs(code):
        if not construct.measurable:
            continue          # one-liners cannot be counted separately from their header
        if construct.kind == "loop":
            rows.append(f"{construct.id} (line {construct.header_line}: "
                        f"{construct.header_text}) iterations = ?")
        else:
            rows.append(f"{construct.id} (line {construct.header_line}: "
                        f"{construct.header_text}) taken = ?, not taken = ?")
    return "\n".join(rows) if rows else "(this program has no loops or branches)"


def profile_section(code: str) -> str:
    return INSTRUCTIONS.format(answer_mark=ANSWER_MARK, profile_mark=PROFILE_MARK,
                               checklist=checklist(code))


class ProfilePrompt:
    """Their prompt, with the profile question appended.

    Passed to `run_code_consistency_test` in place of the usual prompt helper. Their
    tester does two things with it: calls it to get the template, then formats that
    template with the task's variables. This class covers both, so it can stand in
    wherever their helper is used.
    """

    def __init__(self, base_helper):
        # Their helper is a callable returning the template text, but a plain string or
        # an object with .format is accepted too, so this works from a notebook either way.
        self.base_helper = base_helper

    def __call__(self) -> "ProfilePrompt":
        return self

    def format(self, **input_variables) -> str:
        template = self.base_helper() if callable(self.base_helper) else self.base_helper
        base = template.format(**input_variables)
        # The profile question is appended after formatting, so braces inside the
        # program text are never treated as format fields.
        code = input_variables.get("full_sol") or ""
        return base.rstrip() + "\n" + profile_section(code)
