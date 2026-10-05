"""The same profile probe for models run locally through Hugging Face.

The GPU path in MuCoCo does not look a model up in their registry: the tester builds a
`TransformersCodeLLM` itself. So instead of registering a class, this module swaps the
name in the tester's module for a subclass, which is a one-line change made from a
notebook and leaves their files untouched.

Kept separate from `probe_llm` because importing it pulls in torch.
"""

from __future__ import annotations

from probe.parse import parse_reply
from probe.profile_log import record

# Room for the profile as well as the answer. Their sizing looks at the expected answers
# only, which for CRUXEval is a few tokens; the profile needs several lines, and a chatty
# model needs room to get there before being cut off.
EXTRA_TOKENS = 512


# model name and precision -> (tokenizer, model), so a session loads each model once
_LOADED: dict = {}


def free_models() -> None:
    """Drop the cached models and give the memory back, for switching model mid-session."""
    import gc

    _LOADED.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def profile_transformers_class(load_in_4bit: bool = True, ask_for_profile: bool = True):
    """Build the wrapper class. Imported lazily so torch is only loaded when needed.

    Their loader calls `from_pretrained` with no dtype, which gives full precision: an 8B
    model would need about 32 GB and will not fit a Colab T4. With `load_in_4bit` the
    weights are quantised on the way in, which is what makes a model of that size usable
    on the free and Pro tiers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_models.gpu_code_llms import TransformersCodeLLM

    class ProfileTransformersLLM(TransformersCodeLLM):
        """A local model asked for a profile as well, returning only the answer."""

        def __init__(self, model_name: str, answers=None) -> None:
            # Deliberately not calling their __init__: the model has to be loaded with a
            # dtype it can fit in. Everything else below mirrors what they do.
            self.model_name = model_name

            # One copy per model for the whole session. The tester builds a fresh wrapper
            # for every run, and loading the weights again each time fills the card: five
            # runs of an 8B model in 4-bit is enough to exhaust 22 GB, because nothing
            # releases the previous copy.
            cached = _LOADED.get((model_name, load_in_4bit))
            if cached is not None:
                self.tokenizer, self.model = cached
                print(f"{model_name} reused from this session")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                options = {"device_map": "auto"}
                if load_in_4bit and torch.cuda.is_available():
                    from transformers import BitsAndBytesConfig
                    options["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                elif torch.cuda.is_available():
                    options["torch_dtype"] = torch.float16

                self.model = AutoModelForCausalLM.from_pretrained(model_name, **options)
                _LOADED[(model_name, load_in_4bit)] = (self.tokenizer, self.model)

            if answers is not None:
                self.obtain_max_new_tokens(answers=answers)
            else:
                self.max_new_token = 512
            if ask_for_profile:
                self.max_new_token += EXTRA_TOKENS
            print(f"{model_name} loaded | max new tokens {self.max_new_token} | "
                  f"profile question {'on' if ask_for_profile else 'off'}")

        def _as_chat(self, prompt: str) -> str:
            """Wrap the prompt in the model's chat format when it has one.

            Their adapter feeds the raw prompt straight in, which makes an instruction
            model continue the text rather than answer it: it narrates its reasoning and
            runs out of tokens before reaching the answer. Applying the chat template
            fixes that, and disables the thinking block on models that have one.
            """
            if not getattr(self.tokenizer, "chat_template", None):
                return prompt
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:          # older tokenizers have no thinking switch
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)

        def invoke(self, input_variables, prompt_template):
            import torch

            prompt = self._as_chat(prompt_template.format(**input_variables))
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_token,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            reply = self.tokenizer.decode(generated, skip_special_tokens=True)

            if ask_for_profile:
                answer, profile = parse_reply(reply)
                record(input_variables, profile, reply)
            else:
                # Baseline mode: their task, their prompt, their parsing. Nothing is asked
                # for beyond the answer, so this measures the model without the probe.
                answer = reply

            # Only the answer goes back, so their parser sees what it always saw.
            # Token probabilities are not computed: they are used only by the input
            # prediction task, and scoring every step would double the generation cost.
            return {"ans": answer, "geom_mean_prob": 0.0}

    return ProfileTransformersLLM


def install(load_in_4bit: bool = True, ask_for_profile: bool = True) -> type:
    """Point the tester at the wrapper. Call once, before running a GPU experiment.

    `ask_for_profile=False` keeps the quantised loader and the chat template, which their
    adapter lacks, but asks nothing extra and logs nothing. That is the control run: the
    difference between it and a probe run is what the extra question costs this model.
    """
    import prediction_inconsistency.prediction_inconsistency_tester as tester

    wrapper = profile_transformers_class(load_in_4bit=load_in_4bit,
                                         ask_for_profile=ask_for_profile)
    tester.TransformersCodeLLM = wrapper
    print(f"tester will now build {wrapper.__name__} for GPU runs")
    return wrapper
