"""Model wrapper that keeps the profile out of MuCoCo's way.

MuCoCo parses a reply with `ast.literal_eval`, so a reply carrying two sections would
fail to parse and every task would count as wrong. This wrapper takes the reply apart
first: the answer goes back to their pipeline exactly as before, and the profile is
written to a side file keyed by task.

Their code is untouched. The only change needed on their side is one entry in
`utility/constants.py` pointing at the class built here.
"""

from __future__ import annotations

from llm_models.code_llms import DeepSeekLLM

from probe.parse import parse_reply
from probe.profile_log import record


PROBE_SUFFIX = "-probe"


class ProfileDeepSeekLLM(DeepSeekLLM):
    """DeepSeek, asked for a profile as well, returning only the answer to the caller.

    Registered under a name ending in `-probe` so its results land in their own folder,
    separate from the baseline run. The suffix is stripped before the API sees it.
    """

    def __init__(self, model_name: str = "deepseek-v4-flash" + PROBE_SUFFIX):
        super().__init__(model_name.removesuffix(PROBE_SUFFIX))

    def invoke(self, input_variables, prompt_template):
        prompt = prompt_template.format(**input_variables)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.return_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0,
        )
        reply = response.choices[0].message.content
        answer, profile = parse_reply(reply)

        # The profile is stored against the program and input, which together identify
        # the version of the task. The tester does not pass the task id down to the model.
        record(input_variables, profile, reply)

        # Only the answer goes back, so their parser and their metrics see what they
        # always saw.
        return answer
