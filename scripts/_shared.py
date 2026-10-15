"""Loading and comparison shared by every experiment script.

One copy, because the scoring rules were got wrong several times before they settled and
having them in one place is the only way to keep the scripts agreeing with each other.
"""
from __future__ import annotations

import ast
import copy
import csv
import json
import os
from collections import Counter

from probe.trace_score import align, parse_trace
from probe.trace_truth import as_text, counts_from, trace, writes_from

MARK = "### TRACE"

# Which mutation family each operator belongs to, following MuCoCo's own grouping.
CATEGORY = {
    "sequential": "Lexical", "literal_format": "Lexical", "random": "Lexical",
    "constant_unfold": "Logical", "constant_unfold_add": "Logical",
    "constant_unfold_mult": "Logical", "commutative_reorder": "Logical",
    "demorgan": "Logical", "boolean_literal": "Logical",
    "for2while": "Syntactic", "for2enumerate": "Syntactic",
}

# A loop rewrite introduces counters the original never had (nums_len, loop_var0), so only
# the control-flow half of the trace is comparable for these two.
CONTROL_ONLY = {"for2while", "for2enumerate"}

# Random renaming picks fresh names on every run, so the mutant in the ground-truth file is
# not the one the model saw. Its truth is derived from the reply log instead.
FROM_LOG = {"random"}

OPERATORS = list(CATEGORY)


def load_truth(path: str) -> dict:
    """{(task, version): entry} from the trace-building step."""
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                out[(row["task_id"], version)] = entry
    return out


def load_replies(folder: str, version: str) -> dict:
    """{program text: reply} from one operator's log."""
    path = os.path.join(folder, f"profiles_{version}.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("full_sol"):
                out[row["full_sol"]] = row
    return out


def load_csv(folder: str, operator: str) -> dict:
    """{task: (model output, whether it was correct)} from MuCoCo's own result file."""
    path = os.path.join(folder, f"CruxEval_zero_shot_{operator}.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("task_id"):
                out[row["task_id"]] = (row.get("model_output") or "",
                                       not (row.get("failure_type") or "").strip())
    return out


def account(reply, entry):
    """The counts recovered from a model's trace, comparable across versions.

    Constructs keep their names under any rewrite, being numbered in source order.
    Variables do not: renaming turns `nums` into `var1`, so their counts are compared as a
    sorted multiset. The price is that two variables swapping counts look identical.
    """
    if reply is None:
        return None
    text = reply.get("raw_reply") or ""
    if MARK not in text:
        return None
    claimed = parse_trace(text)
    if not claimed:
        return None
    return (tuple(sorted(counts_from(claimed, entry["code"]).items())),
            tuple(sorted(writes_from(claimed).values())))


def constructs_only(reply, entry):
    """The control-flow half alone, for rewrites that introduce new variables."""
    if reply is None:
        return None
    text = reply.get("raw_reply") or ""
    if MARK not in text:
        return None
    claimed = parse_trace(text)
    if not claimed:
        return None
    return tuple(sorted(counts_from(claimed, entry["code"]).items()))


def truth_from_reply(reply):
    """Trace the program the model was actually shown.

    Only needed for random renaming, whose mutant differs from the one in the truth file.
    The reply log records the code and the input, which is all the tracer needs.
    """
    code, raw = reply.get("full_sol"), reply.get("test_input")
    if not code or raw is None:
        return None
    try:
        func_name = next(n.name for n in ast.parse(code).body
                         if isinstance(n, ast.FunctionDef))
        parsed = ast.literal_eval(f"({raw},)")
        args = parsed if len(parsed) != 1 else parsed[0]
        steps = trace(code, func_name, copy.deepcopy(args))
    except Exception:
        return None
    return {"code": code, "text": as_text(steps)}


def rows_in_order(folder: str, operator: str):
    """The CSV and the reply log zipped by position.

    The CSV carries the task id and the log carries the code; both are written one row per
    task in the same order. Checked on sequential, where the pairing can be verified against
    the code: the five disagreements are all programs appearing under two task ids, and the
    CSV's id is the one the tester ran.
    """
    with open(os.path.join(folder, f"CruxEval_zero_shot_{operator}.csv"),
              encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    with open(os.path.join(folder, f"profiles_{operator}.jsonl"), encoding="utf-8") as handle:
        log_rows = [json.loads(l) for l in handle]
    return list(zip(csv_rows, log_rows))


def has_operator(folder: str, operator: str) -> bool:
    """Both files present. A sweep is often partial, so a missing operator is skipped."""
    return (os.path.exists(os.path.join(folder, f"CruxEval_zero_shot_{operator}.csv"))
            and os.path.exists(os.path.join(folder, f"profiles_{operator}.jsonl")))


def operators_in(folder: str) -> list:
    """The operators this folder actually has, in the usual order."""
    return [o for o in OPERATORS if has_operator(folder, o)]


def pairs_for(folder, truth, by_code, operator):
    """(task, csv row, reply, mutant entry) for every test case this operator produced.

    A pair needs both a result row and a reply. Where the mutation was generated but then
    rejected by MuCoCo's validity check, no model call was made and there is no reply, so
    the pair drops out here.
    """
    if not has_operator(folder, operator):
        return []
    if operator in FROM_LOG:
        out = []
        for csv_row, reply in rows_in_order(folder, operator):
            entry = truth_from_reply(reply)
            if csv_row.get("task_id") and entry:
                out.append((csv_row["task_id"], csv_row, reply, entry))
        return out

    keep = {t for (t, v) in truth if v == operator}
    mutants_csv = load_csv(folder, operator)
    reply_by_task = {}
    for code, reply in load_replies(folder, operator).items():
        found = by_code.get((operator, code))
        if found:
            reply_by_task[found[0]] = (reply, found[1])
    return [(t, {"model_output": mutants_csv[t][0],
                 "failure_type": "" if mutants_csv[t][1] else "x"},
             *reply_by_task[t])
            for t in keep if t in mutants_csv and t in reply_by_task]


def answer_flag(original_out, original_ok, mutant_out, mutant_ok):
    """MuCoCo's Table 2 rule on the two answers.

    Both correct, or both wrong in the same way, is consistent. Anything else is not. Their
    third outcome, invalidity, never occurs in these files: every kept row has an answer.
    """
    if original_ok and mutant_ok:
        return False
    if not original_ok and not mutant_ok:
        return original_out != mutant_out
    return True


def trace_flag(a, b):
    """The same rule on the two recovered accounts. No ground truth is involved."""
    return (a != b) if (a is not None and b is not None) else (a is not b)


def load_all(results, truth_path):
    """Everything the scripts need: the truth, its code index, and the originals."""
    truth = load_truth(truth_path)
    by_code = {(v, e["code"]): (t, e) for (t, v), e in truth.items()}
    return (truth, by_code,
            load_csv(results, "no_mutation"), load_replies(results, "no_mutation"))


def add_arguments(parser):
    """The two paths every script needs."""
    parser.add_argument("--results", required=True,
                        help="folder of CruxEval_zero_shot_*.csv and profiles_*.jsonl")
    parser.add_argument("--truth", required=True,
                        help="traces_all_operators.jsonl")
    return parser
