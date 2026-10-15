# Experiment scripts

Each one produces a table that appears on the slides. They read the sweep's output and
nothing else: no GPU, no model, no network.

## Running them

```
python scripts/inconsistency_rate.py --results <folder> --truth <traces.jsonl>
```

`--results` is the folder holding `CruxEval_zero_shot_*.csv` and `profiles_*.jsonl` for one
model. `--truth` is `traces_all_operators.jsonl`, which is model-independent and shared by
all of them.

From the repo root, with the probe package importable:

```
set PYTHONPATH=.
python scripts\inconsistency_rate.py --results C:\results\irop_trace_qwen ^
                                     --truth  C:\results\traces_all_operators.jsonl
```

## What each produces

| Script | Table |
|---|---|
| `inconsistency_rate.py` | per operator, and by mutation family, in MuCoCo's metric |
| `rq1_venn.py` | the two-by-two and the four Venn regions |
| `trace_accuracy.py` | path exact, both exact, and the recovery split |
| `error_taxonomy.py` | where a failure diverges, and what kind it was |
| `mitigation.py` | what withholding on each check does to the two metrics |
| `tracer_free_checks.py` | the six checks a developer can run with no ground truth |
| `check_ground_truth.py` | whether the two tracers agree, before any cross-method claim |

`_shared.py` holds the loading and the two comparison rules. It is imported, not run.

## Why the comparison rules live in one place

They were got wrong several times before they settled: counting every CSV row rather than
the operator's own test cases, comparing variables by name under renaming, and treating a
rejected mutant as a model failure. One copy is the only way to keep the scripts agreeing
with each other.

## What needs the tracer and what does not

The inconsistency rate, the Venn and the mitigation compare the model with itself across two
versions. They use the source but never the ground truth, so they describe what a developer
could run.

The trace accuracy and the taxonomy compare the model with the tracer. They are evaluation,
and a developer cannot reproduce them without executing the program.

`tracer_free_checks.py` is the bridge: what can still be diagnosed with no ground truth.

## All of them at once

```
python scripts/run_all.py --results <folder> --truth <traces.jsonl> --out results.txt
```

Writes every table to one file, which is what the slide figures are taken from.

## If a folder is partial

A sweep that stopped part way is fine: operators without both a CSV and a reply log are
skipped, and the scripts say when there is nothing comparable to report.
