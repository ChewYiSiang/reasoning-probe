# Execution profiles for MuCoCo consistency testing

This is the ground-truth half of the reasoning-consistency extension to MuCoCo.

MuCoCo checks whether a code model gives the same *answer* for a program and its
semantics-preserving mutants. The extension asks a second question: does the model
believe the same things about *how the program runs*? To score that, we first need to
know what really happens, so this package runs each program and records its control-flow
profile: how many times each loop iterated and how often each branch was taken.

Nothing is inserted into the program. The recording uses `sys.monitoring` (Python 3.12
and later) or `sys.settrace` on older versions, so the code a model sees later is exactly
the benchmark's code.

## What is here

    probe/constructs.py      finds the loops and branches, and names them so the names
                             survive mutation (loop1, if2, ...)
    probe/truth.py           runs a program under the tracer and turns the trace into a
                             profile
    probe/mutants.py         two semantics-preserving operators, used until the group's
                             MuCoCo operators are wired in
    probe/dataset.py         reads CRUXEval items
    probe/coverage_check.py  optional cross-check of the branch counts against coverage.py
    probe/build_truth.py     command line entry point: builds the profiles for a slice
    tests/                   unit tests

## Setup

Python 3.12 or later. Only the optional cross-check needs a third-party package.

    pip install -r requirements.txt     # pytest and coverage, both optional for a plain run

## Running it

    python -m probe.build_truth --dataset path/to/cruxeval.jsonl --out results/truth.jsonl

Useful flags: `--limit N` for a quick slice, `--operators sequential_rename` to restrict
the mutations, `--cross-check` to verify branch counts against coverage.py.

The run over all 800 CRUXEval items takes about a second and prints a summary:

    items                                      800
    loops                                      444
    branches                                   433
    one-liners excluded                        14
    loops running more than once               285
    loops never entered                        89
    items with a branch                        368
    items where one branch side never ran      343
    sequential_rename: applied                 800
    constant_unfold_add: applied               454
    recursive programs (counts less reliable)  8

The last two lines of that block matter for the study design. Only 454 of 800 programs
contain an integer to unfold, and 343 of the 368 programs with a branch never execute one
side of it on the benchmark's own input, so scoring has to separate branches the input
exercised from branches it did not.

## Output format

One JSON object per item, with the original program and each mutant:

    {
      "id": "sample_0",
      "input": "[1, 1, 3, 1, 3, 1]",
      "call": "f([1, 1, 3, 1, 3, 1])",
      "expected_output": "[(4, 1), ...]",
      "status": "ok",
      "recursive": false,
      "versions": {
        "original": {
          "code": "def f(nums): ...",
          "constructs": [{"id": "loop1", "kind": "loop", "header_line": 3,
                          "body_line": 4, "else_line": null,
                          "header_text": "for n in nums:", "measurable": true}],
          "profile": {"output": "[(4, 1), ...]",
                      "loops": {"loop1": 6},
                      "branches": {},
                      "statement_counts": {"2": 1, "3": 7, "4": 6},
                      "arc_counts": [[2, 3, 1], [3, 4, 6], [4, 3, 6]],
                      "steps": 16}
        },
        "sequential_rename": { "...": "same shape" }
      }
    }

A mutant that changes the answer or the profile is not semantics-preserving for that
input; it is kept in the file with a `dropped` field saying why, and left out of the
analysis.

## How the counts are produced

The tracer sees one event per executed line, which is not quite the same as one event per
step:

* A statement spread over several lines fires several events. Every line is mapped back
  to the statement that owns it, and a statement only scores when execution enters it.
* Counting hits on the first statement of a loop body breaks for nested loops, because
  the inner loop's header runs once per inner iteration. What is counted instead is the
  *arc*: how many times control moved from the loop header into the body. A branch is
  counted the same way, from the condition into the branch.

Coverage tools record those arcs as a set, so they can say a branch was taken but not how
often. Counting them is the step from coverage to profiling.

## Known limits

* **Recursive programs.** Arc counting tracks the previous statement without knowing
  which frame it came from, so a self-call can undercount. Eight of the 800 CRUXEval
  items are recursive; they are flagged in the output rather than dropped.
* **One-liners.** `if x: return 1` puts the body on the condition's line, so the counts
  cannot separate "condition evaluated" from "branch taken". Those constructs are marked
  `measurable: false` and excluded. There are 14 in CRUXEval.
* **One input per program.** CRUXEval gives a single input, which is why so many branches
  never run. HumanEval items carry a test suite and will be used for wider coverage.
* **Mutation operators.** The two here are stand-ins that follow MuCoCo's sequential and
  constant-unfold-addition operators, including the part that renames the function itself
  (`f` becomes `generic_function1`). `load_operators()` already prefers MuCoCo's
  implementations when the package is importable, so swapping them in is a one-line
  change and no results move.
* **Python version.** `sys.monitoring` needs 3.12. The MuCoCo repository pins 3.11.4, so
  inside that environment the tracer uses `sys.settrace` instead; both produce the same
  counts, and the tests cover whichever one is active.

## Tests

    python -m pytest tests

Eighteen tests covering construct naming, arc counting, the runaway-program guard, and
the behaviour-preserving property of both operators, including the renamed-function case.

## Next step

Phase 2 adds the prompt that asks a model for the same profile, the parser for its reply,
and the scoring against these files.

## Using it with the MuCoCo repository

Two extra modules connect the tracer to the group's pipeline:

    probe/mucoco.py             reads their task records and applies their mutation
                                operators, returning code plus the (possibly renamed)
                                entry function
    probe/build_truth_mucoco.py command line entry point for their CRUXEval tasks

Run it from the MuCoCo repository root, with this package on the path:

    set PYTHONPATH=.;C:\path\to\reasoning-probe        # Windows
    python -m probe.build_truth_mucoco --source mongo --limit 800 \
        --mutations sequential constant_unfold_add --out results/truth/cruxeval.jsonl

It reads the same MongoDB collection their database builder writes, uses their
`CodeMutator` rather than the stand-in operators here, and calls each function exactly
the way their validity check does (`call_like_mucoco` in `truth.py` mirrors
`PredictionInconsistencyHelper.check_input_output`). No model is involved and nothing
leaves the machine.

Use `--source jsonl --jsonl path/to/cruxeval.jsonl` to run without the database.

Each mutant is checked against the original: same answer, and same counts on the
constructs both versions can measure. Anything that fails lands in `mutation_warnings`.

MuCoCo's own validity check is **not** run by default. It spawns a subprocess per mutant,
which on Windows re-imports their whole dependency chain each time and turns a three-second
pass into hours. The check here is stricter anyway, because it compares control flow as
well as the answer. Pass `--verify-with-mucoco` to run theirs as well.

A full pass over 800 CRUXEval tasks with two operators takes about **three seconds**.


## Phase 2: asking the model for the profile

Three more modules turn the ground truth into a measurement:

    probe/prompt.py     appends the profile question to MuCoCo's prompt, with a checklist
                        of the loops and branches in the version being shown
    probe/probe_llm.py  a DeepSeek wrapper that splits the reply, returns only the answer
                        to MuCoCo's pipeline, and logs the profile to its own file
    probe/parse.py      reads the two sections back out of a reply
    probe/score.py      compares logged profiles with the traced ground truth

The model is asked for MuCoCo's answer first and the profile second, in one call. Their
parser runs `ast.literal_eval` over the whole reply, so a two-section reply would break
it; the wrapper takes the reply apart before their code ever sees it, which keeps their
correctness and consistency numbers untouched.

### Wiring it in, from a notebook cell

    from utility.constants import NonReasoningModels
    from probe.probe_llm import ProfileDeepSeekLLM
    from probe.prompt import ProfilePrompt

    # Registered at runtime, so nothing in their repository changes. The -probe suffix
    # keeps the results in their own folder; the wrapper strips it before calling the API.
    NonReasoningModels.DEEPSEEK_V4_FLASH_PROBE = {
        "name": "deepseek-v4-flash-probe",
        "model_class": ProfileDeepSeekLLM,
    }

    base_prompt = PredictionInconsistencyPromptTemplate.return_appropriate_prompt(task_type, prompt_type)

    llmtester.run_code_consistency_test(
        prompt_helper=ProfilePrompt(base_prompt),
        model_name="deepseek-v4-flash-probe",
        ...                       # everything else as before
    )

Profiles are appended to `results/probe/profiles.jsonl`; set `PROBE_PROFILE_LOG` to change
that. Then:

    python -m probe.score --truth results/truth/cruxeval.jsonl --predictions results/probe/profiles.jsonl

Scores are split into constructs the input exercised and constructs it did not, because
most CRUXEval branches never run one of their sides and a model answering 0 everywhere
would otherwise look accurate.


## A note on how branches are counted

A branch is scored as **(times the condition was true, times it was false)**, not as
"times each side ran". The two differ for an `if` with no `else`: the trace shows the else
branch running zero times, while the number a reader wants is how often the condition
failed. Models answer the second question, and it is the better-defined one, so it is what
both the prompt and the ground truth use.

Ground truth for the false count comes from the statement counts already in the truth
file, so changing this needed no re-tracing, and logs written under the earlier wording
can still be read: the parser accepts both "not taken" and "else".


## Evaluating the two oracle-free detectors

`probe/detectors.py` asks how good each check is, using the tracer as the answer key. The
label is "the model misread the execution in at least one version of this pair"; the
detectors are MuCoCo's answer comparison and the same rule applied to the profile field,
neither of which needs the code to be run.

    python -m probe.detectors --truth results/truth/cruxeval.jsonl \
        --results-dir results/qwen --probe-dir results/qwen

A profile disagreement should come out at precision 1.0: the mutation is
semantics-preserving, so the execution is identical in both versions, and two different
accounts of it cannot both be right.


## Comparing a mutant against its original

Mutation does not leave the set of measurable constructs alone: `ast.unparse` expands a
one-liner, so a construct that could not be measured in the original becomes measurable in
the mutant. Comparing raw totals therefore mixes the effect of the rewriting with a change
in what is being counted.

    python -m probe.audit --truth results/truth/cruxeval.jsonl \
        --probe-dir results/qwen --compare sequential constant_unfold_add

This runs over the tasks the mutation applies to, and within those only the constructs
present in both versions, so the only difference between the two columns is the rewriting.
It also reports how many constructs were dropped for not being in both.


## Do different models fail on the same constructs?

Three error rates say how often each model is wrong, not whether they are wrong about the
same things. A construct that defeats every model is a property of the construct, which is
more useful to report than three percentages.

    python -m probe.overlap --truth results/truth/cruxeval.jsonl \
        --model deepseek results/probe/profiles_no_mutation.jsonl \
        --model qwen14b results/qwen14b/profiles_no_mutation.jsonl \
        --model qwen8b results/qwen/profiles_no_mutation.jsonl

Constructs a model did not answer are left out of its column, so a missing reply never
counts as a mistake.


## Recording what a run was

Results folders accumulate, and the conditions do not survive in anyone's memory: which
model, 4-bit or not, probe on or off, and which version of the ground truth it was scored
against. One call at the end of a run writes that down.

    from probe.manifest import write
    write(RESULTS, "sequential", model=MODEL, probe=True,
          quantisation="4-bit", max_new_tokens=576, tasks_requested=800,
          truth="/content/truth_cruxeval.json")

Or afterwards, from the command line:

    python -m probe.manifest results/qwen14b --mutation no_mutation sequential \
        --model Qwen/Qwen2.5-Coder-14B-Instruct --truth results/truth/cruxeval.jsonl \
        --quantisation 4-bit --probe

The entry records the date, the outcome counts read from the CSV, and a SHA-256 of the
truth file. That last one matters: the ground truth has already changed twice, and a
re-score against a different version would move every number with nothing to say why.


## Data flow: how many times each variable changed

The control-flow side asks which way execution went and how often. This asks what happened
to the values, at the same level of detail, so the same parsing, scoring and error
categories apply.

Build the ground truth (a second tracing pass; about a second for 800 programs):

    python -m probe.dataflow build --truth results/truth/cruxeval.jsonl \
        --source mongo --limit 800 --out results/truth/writes.jsonl

The build reports how many variables were seen, how many are worth asking about and how
many questions it will ask, so it doubles as the feasibility check: if few variables
survive, the question is not worth running.

A variable is dropped when it is written once (the answer is trivially 1) or when it is a
loop variable, whose count is that loop's iteration count by construction. Everything else
is kept, including a list built by `append` inside a loop. At most three variables are asked
about per program, to keep replies short and scoring exact.

Then run with `WritesPrompt` in place of `ProfilePrompt`, and score:

    python -m probe.dataflow score --writes results/truth/writes.jsonl \
        --log results/qwen14b/profiles_no_mutation.jsonl --version no_mutation


## The second task: input prediction

MuCoCo's input prediction gives the model a program, an input and an output, and asks
whether that input produces that output. The probe attaches unchanged, because the program
and the input are both in the prompt.

    python -m probe.input_prediction --truth results/truth/cruxeval.jsonl \
        --results-dir results/qwen14b_input --probe-dir results/qwen14b_input

Read the result with its ceiling in mind: the answer is True or False, so two wrong answers
are always identical and MuCoCo's incorrectness-based inconsistency can never fire. The task
is run to show the profile signal transfers, not to produce a large answer-inconsistency
rate. Their own figure for it is the lowest of their four tasks.

## Re-asking when the two profiles disagree

The mitigation that needs nothing executed, so it still applies without a test suite and
under mutations that change behaviour.

    python -m probe.retry --truth results/truth/cruxeval.jsonl \
        --results-dir results/qwen --probe-dir results/qwen \
        --mutation sequential --out results/retry_prompts.jsonl

That writes one prompt per flagged pair, each showing the model its own two contradictory
answers before repeating the original question. Run those prompts, then score:

    python -m probe.retry --truth results/truth/cruxeval.jsonl \
        --results-dir results/qwen --probe-dir results/qwen \
        --mutation sequential --after results/retry_results.csv

A plain re-ask at temperature 0 would return the same answer, so the prompt has to differ:
showing the model its own contradiction is what makes the second generation a different one.


## Before a full operator sweep: count it first

MuCoCo skips a task when an operator does not apply, so a sweep of all eleven costs far
less than eleven times the benchmark. This counts the valid mutants locally, without
calling a model, and prices the run from speeds measured on our own runs.

    python -m probe.dry_run --source mongo --limit 800

Add `--no-probe` to price an answers-only sweep, and `--models qwen14b llama` to change
which speeds are used. It also names the operators whose mutants cannot be compared
construct by construct: for2while and for2enumerate turn a for loop into a while, so the
ids no longer line up. Profile accuracy still works for them; profile consistency does not.


## When the mutant cannot be matched by its code

Scoring normally regenerates each mutant locally and matches it to the logged reply by
program text. That requires the two copies to be character-identical, which fails for
`random`: its renaming picks fresh names on every run, so the model saw one program and
the truth builder traced another.

The log already holds the exact program in the prompt and the exact input, so the tracer
can run those instead:

    python -m probe.truth_from_log --log results/llama_sweep/profiles_random.jsonl \
        --version random --out results/truth/random_from_log.jsonl

    python -m probe.audit --truth results/truth/random_from_log.jsonl \
        --probe-dir results/llama_sweep --versions random

The output is the same shape as `build_truth_mucoco` writes, so every scoring tool reads
it unchanged. This is also the route for mutations that change behaviour, where the
original's counts do not apply to the mutant at all.
