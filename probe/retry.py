"""Re-ask the model when its two accounts of the same program disagree.

This is the mitigation that survives where the method is meant to be used. It needs two
model replies and nothing else: no test suite, no execution, and it keeps working under
mutations that change behaviour, which is where supplying the true counts would not.

The trigger is the reasoning disagreement itself: either the loop and branch counts differ
between the two versions, or the variable updates do. On sequential renaming that fires on
168 pairs for Qwen2.5-Coder-14B and 288 for LLaMA-3.1-8B, and is right about a wrong reading
98.8% and 97.6% of the time.

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


def retry_prompt(base_prompt: str, first: dict, second: dict,
                 first_writes: dict | None = None,
                 second_writes: dict | None = None) -> str:
    """The original question, plus whichever of the model's own answers actually conflict.

    Only the fields that differ are shown. Quoting the loop counts when the variables are
    what disagree would hand the model two identical lists and call them a contradiction,
    which is not something it can resolve.
    """
    # under renaming the two versions use different variable names, and the model is shown
    # the first version's code, so quoting the second version's names would name variables
    # that do not appear in the program. The counts are quoted as a sorted list instead,
    # which is also how the check itself compares them.
    renamed = set(first_writes or {}) != set(second_writes or {})

    def side(label: str, profile: dict, writes: dict | None, show_profile: bool,
             show_writes: bool) -> list[str]:
        lines = [label]
        if show_profile:
            lines += [f"  {describe(name, tuple(counts))}"
                      for name, counts in sorted(profile.items())]
        if show_writes and writes:
            if renamed:
                counts = ", ".join(str(value) for value in sorted(writes.values(), reverse=True))
                lines.append(f"  the variables changed {counts} times")
            else:
                lines += [f"  {name} changed {count} times"
                          for name, count in sorted(writes.items())]
        return lines

    show_profile = first != second
    first_counts = sorted((first_writes or {}).values())
    second_counts = sorted((second_writes or {}).values())
    show_writes = (first_counts != second_counts if renamed
                   else (first_writes or {}) != (second_writes or {}))
    if not show_profile and not show_writes:        # nothing to quote back
        show_profile = True

    lines = side("On the first version you said:", first, first_writes,
                 show_profile, show_writes)
    lines.append("")
    lines += side("On the second version you said:", second, second_writes,
                  show_profile, show_writes)
    lines.append("")
    lines.append("Work out which is right, then answer the original question again in the "
                 "same format.")
    return RETRY_HEADER + "\n".join(lines) + "\n\n" + base_prompt


def flagged_pairs(truth: dict, results_dir: Path, probe_dir: Path, mutation: str,
                  benchmark: str = "CruxEval", prompt_type: str = "zero_shot",
                  writes: dict | None = None) -> list[dict]:
    """Every pair the reasoning check flags: the tasks a retry would fire on.

    With `writes`, a disagreement in either field fires, which is the check the rest of the
    work reports. Without it, only the loop and branch counts are compared, which is the
    narrower trigger and finds about half as many pairs.
    """
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
        # straight-line programs have no loops or branches but can still have variables worth
        # asking about, so a missing construct count is not a reason to drop the pair
        if writes is None and (not real_base or not real_mutant):
            continue

        control_fired = inconsistent(profile_verdict(base, real_base),
                                     profile_verdict(mutant, real_mutant),
                                     profile_text(base), profile_text(mutant))
        data_fired = False
        base_writes = mutant_writes = {}
        if writes is not None:
            from probe.dataflow import parse_writes
            from probe.reasoning import write_text, write_verdict
            want_base = writes.get((task, "no_mutation")) or {}
            want_mutant = writes.get((task, mutation)) or {}
            data_fired = inconsistent(
                write_verdict(base, want_base.get("asked") or [],
                              want_base.get("informative") or {}),
                write_verdict(mutant, want_mutant.get("asked") or [],
                              want_mutant.get("informative") or {}),
                write_text(base) if want_base.get("asked") else "",
                write_text(mutant) if want_mutant.get("asked") else "")
            base_writes = parse_writes(base.get("raw_reply") or "")
            mutant_writes = parse_writes(mutant.get("raw_reply") or "")
        if not (control_fired or data_fired):
            continue
        out.append({
            "task_id": task,
            "prompt": base_row.get("prompt", ""),
            "answer_before": verdict(base_row),
            "first_profile": base.get("profile") or {},
            "second_profile": mutant.get("profile") or {},
            "first_writes": base_writes,
            "second_writes": mutant_writes,
            "fired_on": ("both" if control_fired and data_fired
                         else "control flow" if control_fired else "variable updates"),
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
    parser.add_argument("--writes", help="the lookup file: fires on both fields, not just control flow")
    args = parser.parse_args()

    truth = load_truth(args.truth)

    writes = None
    if args.writes:
        writes = {}
        with open(args.writes, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for version, entry in row["versions"].items():
                    if entry.get("asked"):
                        writes[(row["task_id"], version)] = entry

    pairs = flagged_pairs(truth, Path(args.results_dir), Path(args.probe_dir), args.mutation,
                          writes=writes)
    wrong_before = sum(1 for p in pairs if p["answer_before"] == "incorrect")
    field = "the reasoning check flags" if writes else "the two profiles disagree"
    print(f"pairs where {field}: {len(pairs)}")
    print(f"  of those, the answer was already wrong: {rate(wrong_before, len(pairs))}")
    print(f"  the first profile was right in: {rate(sum(1 for p in pairs if p['profile_correct_before']), len(pairs))}")
    fired = Counter(p["fired_on"] for p in pairs)
    for field in ("control flow", "variable updates", "both"):
        if fired[field]:
            print(f"  fired on {field:18s} {fired[field]}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps({
                    "task_id": pair["task_id"],
                    "prompt": retry_prompt(pair["prompt"], pair["first_profile"],
                                           pair["second_profile"],
                                           pair.get("first_writes"),
                                           pair.get("second_writes")),
                    "fired_on": pair["fired_on"],
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
