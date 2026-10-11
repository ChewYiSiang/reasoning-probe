"""The question, appended to MuCoCo's prompt.

Their task is untouched and their answer still comes first, so their parser sees what it
always saw and their accuracy figures stay comparable. The trace follows it.

The instruction names the line numbers explicitly. A model left to choose will sometimes
number the function from 1 and sometimes from the def, and a trace that is right but offset
by one would be scored as a path error on every step.
"""

ANSWER_MARK = "### ANSWER"
TRACE_MARK = "### TRACE"

INSTRUCTIONS = """
After the answer, write the execution trace for this input.

Write exactly two sections and nothing else:

{answer_mark}
<your answer, in the format described above, on one line>

{trace_mark}
<one line per statement executed, in order>

Number the lines as they are numbered in the program above, starting at 1 for the `def`
line. Write one entry per statement executed, in the order it ran, like this:

line 4 | i=0, c='M'
line 6 | even='M'

After the vertical bar, list only the variables whose value changed at that statement, as
name=value. Write nothing after the bar when no value changed. A statement that runs more
than once appears once per run.
"""


def numbered(code: str) -> str:
    """The program with line numbers, so the trace can refer to them unambiguously."""
    return "\n".join(f"{i:>3}  {line}" for i, line in enumerate(code.splitlines(), start=1))


def section() -> str:
    return INSTRUCTIONS.format(answer_mark=ANSWER_MARK, trace_mark=TRACE_MARK)


class TracePrompt:
    """MuCoCo's prompt with the trace question appended.

    Mirrors the shape of the existing prompt helpers: callable, and formatting through the
    base helper so their template work is not duplicated here.
    """

    def __init__(self, base_helper):
        self.base_helper = base_helper

    def __call__(self) -> "TracePrompt":
        return self

    def format(self, **input_variables) -> str:
        template = self.base_helper() if callable(self.base_helper) else self.base_helper
        base = template.format(**input_variables)
        return _append_to_user_turn(base, section())


# chat templates put the question in a user turn and then open an assistant turn for the
# reply. Appending to the end would land the instructions inside the assistant turn, where
# the model reads them as the opening of its own answer and continues from them instead of
# following them.
_TURN_ENDS = ("<|im_end|>",                      # ChatML, used by Qwen
              "<|eot_id|>",                      # Llama 3
              "<end_of_turn>",                   # Gemma
              "[/INST]")                         # Mistral


def _append_to_user_turn(base: str, addition: str) -> str:
    """Put the addition at the end of the user's turn, not after it.

    Falls back to appending when the template carries no chat markers, which is the case
    for a plain completion prompt.
    """
    for marker in _TURN_ENDS:
        position = base.rfind(marker)
        if position != -1:
            return base[:position].rstrip() + "\n" + addition + "\n" + base[position:]
    return base.rstrip() + "\n" + addition + "\n"
