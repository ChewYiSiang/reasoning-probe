"""The reasoning check as a filter, in the shape of MuCoCo's own confidence table.

Their Appendix I keeps only answers the model was confident about, and reports what happens
to two numbers as the threshold tightens: the inconsistency rate falls and accuracy rises,
because low-confidence answers are mostly the wrong ones. Their Table 17 moves from 8.03%
inconsistency at 72.79% accuracy to 1.89% at a higher accuracy.

This builds the same table with a different signal. Instead of a token probability, a pair
is dropped when the reasoning check fires on it: the model's two accounts of the same
program disagree. That needs two replies and nothing else, so it works on a model behind an
API, where a token probability is usually not available.

Three filters are reported side by side, because the point is which signal is doing the work:

    answers      drop a pair when MuCoCo's answer check fires
    reasoning    drop a pair when either reasoning field fires
    both         drop a pair when either check fires

    python -m probe.filter --truth cruxeval_all.jsonl --writes writes_all.jsonl \
        --results-dir results/llama_both --probe-dir results/llama_both --mutation sequential
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from probe.analyse import inconsistent
from probe.reasoning import load_side
from probe.score import load_truth
from probe.stats import rate


def kept(sides: tuple[dict, dict], task: str, drop_answers: bool, drop_reasoning: bool) -> bool:
    """Does this pair survive the filter?"""
    first, second = sides[0][task], sides[1][task]
    if drop_answers:
        if inconsistent(first["answer_verdict"], second["answer_verdict"],
                        first["answer_text"], second["answer_text"]):
            return False
    if drop_reasoning:
        profiles = inconsistent(first["profile_verdict"], second["profile_verdict"],
                                first["profile_text"], second["profile_text"])
        counts = inconsistent(first["write_verdict"], second["write_verdict"],
                              first["write_text"], second["write_text"])
        if bool(profiles) or bool(counts):
            return False
    return True


def measure(sides: tuple[dict, dict], tasks: list[str],
            drop_answers: bool, drop_reasoning: bool) -> dict:
    """MuCoCo's two numbers over whatever the filter leaves behind.

    Inconsistency is their formula, flagged pairs over all pairs, counted on the pairs that
    remain. Accuracy is their formula too: correct answers over answers given, counted on
    the original version of each remaining pair, which is the version a user would be asking
    about.
    """
    remaining = [task for task in tasks
                 if kept(sides, task, drop_answers, drop_reasoning)]
    if not remaining:
        return {"kept": 0}

    flagged = correct = answered = 0
    for task in remaining:
        first, second = sides[0][task], sides[1][task]
        if inconsistent(first["answer_verdict"], second["answer_verdict"],
                        first["answer_text"], second["answer_text"]):
            flagged += 1
        if first["answer_verdict"] in ("correct", "incorrect"):
            answered += 1
            correct += first["answer_verdict"] == "correct"
    return {"kept": len(remaining), "of": len(tasks), "flagged": flagged,
            "correct": correct, "answered": answered}


def tally(truth, writes, results_dir: str, probe_dir: str, mutation: str,
          benchmark: str, prompt_type: str) -> dict:
    """The four filters' counts for one operator, so several can be summed."""
    first = load_side(truth, writes, Path(results_dir), Path(probe_dir), "no_mutation",
                      benchmark, prompt_type)
    second = load_side(truth, writes, Path(results_dir), Path(probe_dir), mutation,
                       benchmark, prompt_type)
    tasks = sorted(task for task in set(first) & set(second)
                   if inconsistent(first[task]["answer_verdict"], second[task]["answer_verdict"],
                                   first[task]["answer_text"], second[task]["answer_text"])
                   is not None)
    sides = (first, second)
    out = {"pairs": len(tasks)}
    for name, drop_a, drop_r in (("none", False, False), ("answers", True, False),
                                 ("reasoning", False, True), ("either", True, True)):
        out[name] = measure(sides, tasks, drop_a, drop_r)
    return out


def load_inputs(truth_path: str, writes_path: str):
    """The traced counts and the variable lookup, the two inputs every filter needs."""
    truth = load_truth(truth_path)
    writes: dict = {}
    with open(writes_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                if entry.get("asked"):
                    writes[(row["task_id"], version)] = entry
    return truth, writes


def show(label: str, totals: dict) -> None:
    """One table, from counts that may have been summed over several operators."""
    print(f"=== {label}: {totals['pairs']} pairs, before any filter ===\n")
    print(f"  {'filter':32s}{'kept':>12}{'inconsistency':>16}{'accuracy':>12}")
    for name, text in (("none", "none: every pair"),
                       ("answers", "withhold what the answers flag"),
                       ("reasoning", "withhold what the reasoning flags"),
                       ("either", "withhold what either flags")):
        s = totals[name]
        if not s["kept"]:
            print(f"  {text:32s}{'0':>12}")
            continue
        share = f"{s['kept']}/{totals['pairs']}"
        inc = s["flagged"] / s["kept"] * 100
        acc = s["correct"] / s["answered"] * 100 if s["answered"] else 0.0
        print(f"  {text:32s}{share:>12}{inc:>15.2f}%{acc:>11.2f}%")
    print("\n  inconsistency is their formula: flagged pairs over the pairs kept")
    print("  accuracy is correct answers over answers given, on the original version")


def report(truth_path: str, writes_path: str, results_dir: str, probe_dir: str,
           mutation: str, benchmark: str = "CruxEval", prompt_type: str = "zero_shot") -> None:
    truth = load_truth(truth_path)

    writes: dict = {}
    with open(writes_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                if entry.get("asked"):
                    writes[(row["task_id"], version)] = entry

    first = load_side(truth, writes, Path(results_dir), Path(probe_dir), "no_mutation",
                      benchmark, prompt_type)
    second = load_side(truth, writes, Path(results_dir), Path(probe_dir), mutation,
                       benchmark, prompt_type)
    # Only pairs the operator actually produced count. MuCoCo's denominator is the test
    # cases generated, so a program the mutation could not touch is not a pair at all: its
    # mutant row carries no model output, and comparing the answers returns nothing.
    tasks = sorted(task for task in set(first) & set(second)
                   if inconsistent(first[task]["answer_verdict"], second[task]["answer_verdict"],
                                   first[task]["answer_text"], second[task]["answer_text"])
                   is not None)
    sides = (first, second)

    print(f"=== {mutation}: {len(tasks)} pairs, before any filter ===\n")
    print(f"  {'filter':32s}{'kept':>12}{'inconsistency':>16}{'accuracy':>12}")
    for label, drop_answers, drop_reasoning in (
            ("none: every pair", False, False),
            ("drop what the answers flag", True, False),
            ("drop what the reasoning flags", False, True),
            ("drop what either flags", True, True)):
        stats = measure(sides, tasks, drop_answers, drop_reasoning)
        if not stats["kept"]:
            print(f"  {label:34s}{'0':>12}")
            continue
        share = f"{stats['kept']}/{len(tasks)}"
        inc = stats["flagged"] / stats["kept"] * 100
        acc = stats["correct"] / stats["answered"] * 100 if stats["answered"] else 0.0
        print(f"  {label:32s}{share:>12}{inc:>15.2f}%{acc:>11.2f}%")
    print("\n  inconsistency is their formula: flagged pairs over the pairs kept")
    print("  accuracy is correct answers over answers given, on the original version")
    print("  their Table 17 moves from 8.03% at 72.79% accuracy to 1.89% as the "
          "confidence threshold rises")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--writes", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--mutation", nargs="+", default=["sequential"],
                        help="one operator, or several to pool into a single fraction")
    args = parser.parse_args()

    if len(args.mutation) == 1:
        report(args.truth, args.writes, args.results_dir, args.probe_dir, args.mutation[0])
        return

    truth, writes = load_inputs(args.truth, args.writes)
    pooled = {"pairs": 0}
    for mutation in args.mutation:
        one = tally(truth, writes, args.results_dir, args.probe_dir, mutation,
                    "CruxEval", "zero_shot")
        print(f"  {mutation:24s}{one['pairs']:>6} pairs")
        pooled["pairs"] += one["pairs"]
        for name in ("none", "answers", "reasoning", "either"):
            into = pooled.setdefault(name, {})
            for key, value in one[name].items():
                if key != "of":
                    into[key] = into.get(key, 0) + value
    print()
    show(f"{len(args.mutation)} operators pooled", pooled)


if __name__ == "__main__":
    main()
