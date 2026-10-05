"""Re-ask the model when its two accounts of the same program disagree.

This is the mitigation that survives where the method is meant to be used. It needs two
model replies and nothing else: no test suite, no execution, and it keeps working under
mutations that change behaviour, which is where supplying the true counts would not.

The trigger is the profile disagreement itself, which fires on 157 pairs in our run and is
right about a wrong count 153 times.

Re-asking has to change something, or a model at temperature 0 simply repeats itself. The
re-ask therefore shows the model its own two answers for the same program and says plainly
that they cannot both be right, which is a different prompt and so a different generation.
What is measured is whether the answer settles, and whether it settles on the right one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from probe.analyse import inconsistent, profile_text, profile_verdict, read_results_csv, verdict
from probe.score import load_predictions, load_truth
from probe.stats import mcnemar, rate

RETRY_HEADER = """You answered the same question about two versions of one program, and the
two versions do exactly the same thing. Your two descriptions of how it runs do not match,
so at least one of them is wrong.

"""


def describe(name: str, counts: tuple[int, ...]) -> str:
    return (f"{name}: {counts[0]} iterations" if len(counts) == 1
            else f"{name}: true {counts[0]} times, false {counts[1]} times")


def retry_prompt(base_prompt: str, first_profile: dict, second_profile: dict) -> str:
    """The original question, plus the model's own contradictory answers."""
    lines = ["On the first version you said:"]
    lines += [f"  {describe(name, tuple(counts))}" for name, counts in sorted(first_profile.items())]
    lines.append("")
    lines.append("On the second version you said:")
    lines += [f"  {describe(name, tuple(counts))}" for name, counts in sorted(second_profile.items())]
    lines.append("")
    lines.append("Work out which is right, then answer the original question again in the "
                 "same format.")
    return RETRY_HEADER + "\n".join(lines) + "\n\n" + base_prompt


def flagged_pairs(truth: dict, results_dir: Path, probe_dir: Path, mutation: str,
                  benchmark: str = "CruxEval", prompt_type: str = "zero_shot") -> list[dict]:
    """Every pair where the two profiles disagree: the tasks a retry would fire on."""
    def load(version: str):
        rows = read_results_csv(Path(results_dir) / f"{benchmark}_{prompt_type}_{version}.csv")
        by_code = {entry["code"]: task for (task, ver), entry in truth.items()
                   if ver == version and entry["code"]}
        replies = {}
        log = Path(probe_dir) / f"profiles_{version}.jsonl"
        if log.exists():
            for prediction in load_predictions(log):
                task = by_code.get(prediction["full_sol"])
                if task:
                    replies[task] = prediction
        return rows, replies

    base_rows, base_replies = load("no_mutation")
    mutant_rows, mutant_replies = load(mutation)

    out = []
    for task, base_row in base_rows.items():
        mutant_row = mutant_rows.get(task)
        base, mutant = base_replies.get(task), mutant_replies.get(task)
        if mutant_row is None or base is None or mutant is None:
            continue
        real_base = truth.get((task, "no_mutation"), {}).get("counts")
        real_mutant = truth.get((task, mutation), {}).get("counts")
        if not real_base or not real_mutant:
            continue

        fired = inconsistent(profile_verdict(base, real_base),
                             profile_verdict(mutant, real_mutant),
                             profile_text(base), profile_text(mutant))
        if not fired:
            continue
        out.append({
            "task_id": task,
            "prompt": base_row.get("prompt", ""),
            "answer_before": verdict(base_row),
            "first_profile": base.get("profile") or {},
            "second_profile": mutant.get("profile") or {},
            "profile_correct_before": profile_verdict(base, real_base) == "correct",
        })
    return out


def score_retry(before: list[dict], after_csv: Path) -> Counter:
    """Did re-asking change the answer, and in which direction?"""
    rows = read_results_csv(after_csv)
    stats: Counter = Counter()
    for pair in before:
        row = rows.get(pair["task_id"])
        if row is None:
            continue
        now = verdict(row)
        was = pair["answer_before"]
        if was not in ("correct", "incorrect") or now not in ("correct", "incorrect"):
            continue
        stats["retried"] += 1
        if was == "incorrect" and now == "correct":
            stats["fixed"] += 1
        elif was == "correct" and now == "incorrect":
            stats["broken"] += 1
        elif was == "correct":
            stats["still right"] += 1
        else:
            stats["still wrong"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--mutation", default="sequential")
    parser.add_argument("--out", help="write the retry prompts here, as JSONL")
    parser.add_argument("--after", help="the result CSV of the retry run, to score it")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    pairs = flagged_pairs(truth, Path(args.results_dir), Path(args.probe_dir), args.mutation)
    wrong_before = sum(1 for p in pairs if p["answer_before"] == "incorrect")
    print(f"pairs where the two profiles disagree: {len(pairs)}")
    print(f"  of those, the answer was already wrong: {rate(wrong_before, len(pairs))}")
    print(f"  the first profile was right in: {rate(sum(1 for p in pairs if p['profile_correct_before']), len(pairs))}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps({
                    "task_id": pair["task_id"],
                    "prompt": retry_prompt(pair["prompt"], pair["first_profile"],
                                           pair["second_profile"]),
                    "answer_before": pair["answer_before"],
                }) + "\n")
        print(f"  {len(pairs)} retry prompts written to {args.out}")

    if args.after:
        stats = score_retry(pairs, Path(args.after))
        total = stats["retried"]
        if not total:
            print("\nnothing scorable in the retry run yet")
            return
        print(f"\n=== retry outcome over {total} pairs ===")
        for name in ("fixed", "broken", "still right", "still wrong"):
            print(f"  {name:12s} {rate(stats[name], total)}")
        print(f"  exact McNemar p = {mcnemar(stats['broken'], stats['fixed']):.4f}")
        if stats["still wrong"] + stats["fixed"]:
            print(f"  of the answers that were wrong, the retry fixed "
                  f"{rate(stats['fixed'], stats['fixed'] + stats['still wrong'])}")


if __name__ == "__main__":
    main()
