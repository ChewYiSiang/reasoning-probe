"""One reasoning check: does the model's account of the run hold up between two versions?

MuCoCo asks whether the **answer** survives a rewrite that cannot change it. This asks
whether the **account of how the program ran** survives the same rewrite. Both the loops and
branches it reports, and how many times it says each variable changed, are part of that one
account, so they are one check rather than two: a pair is flagged when either field
disagrees between the versions.

The rule is MuCoCo's own three-way one, applied to the account instead of the answer.

Variable names do not survive a renaming mutation: `found` in the original is `var1` in the
mutant, so the two cannot be matched by name. The variable updates are therefore compared as a
sorted list of numbers rather than by name, which works under any renaming. Two variables
swapping counts would look identical, which is the price of not needing a name mapping, and
is stated rather than hidden.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from probe.analyse import inconsistent, profile_text, profile_verdict, read_results_csv, verdict
from probe.dataflow import parse_writes
from probe.score import load_predictions, load_truth
from probe.stats import rate


def write_verdict(prediction: dict | None, asked: list[str], real: dict[str, int]) -> str:
    """MuCoCo's three categories, applied to the variable updates."""
    if not asked:
        return "skipped"                                # no variable worth asking about here
    if prediction is None:
        return "skipped"
    claimed = parse_writes(prediction.get("raw_reply") or "")
    if not claimed:
        return "invalid"                                # the reply carried no variable updates
    if any(name not in claimed for name in asked):
        return "invalid"                                # it answered only some of them
    return "correct" if all(claimed[name] == real[name] for name in asked) else "incorrect"


def write_text(prediction: dict | None) -> str:
    """The counts as a sorted list, so a renamed variable still compares."""
    if prediction is None:
        return ""
    claimed = parse_writes(prediction.get("raw_reply") or "")
    return ",".join(str(value) for value in sorted(claimed.values()))


def load_side(truth: dict, writes: dict, results_dir: Path, probe_dir: Path, version: str,
              benchmark: str, prompt_type: str) -> dict[str, dict]:
    """Everything known about one version of every task, keyed by task id."""
    rows = read_results_csv(results_dir / f"{benchmark}_{prompt_type}_{version}.csv") \
        if (results_dir / f"{benchmark}_{prompt_type}_{version}.csv").exists() else {}

    by_code = {entry["code"]: task for (task, ver), entry in truth.items()
               if ver == version and entry["code"]}
    replies: dict[str, dict] = {}
    log = probe_dir / f"profiles_{version}.jsonl"
    if log.exists():
        for prediction in load_predictions(log):
            task = by_code.get(prediction["full_sol"])
            if task:
                replies[task] = prediction

    out: dict[str, dict] = {}
    for task, row in rows.items():
        entry = truth.get((task, version))
        writes_entry = writes.get((task, version), {})
        out[task] = {
            "answer_verdict": verdict(row),
            "answer_text": str(row.get("model_output", "")),
            "profile_verdict": profile_verdict(replies.get(task), entry["counts"] if entry else None),
            "profile_text": profile_text(replies.get(task)),
            "write_verdict": write_verdict(replies.get(task), writes_entry.get("asked", []),
                                           writes_entry.get("informative", {})),
            "write_text": write_text(replies.get(task)),
            "wrong_somewhere": None,
        }
        # the tracer's verdict: did either version get anything wrong
        profile_wrong = out[task]["profile_verdict"] == "incorrect"
        write_wrong = out[task]["write_verdict"] == "incorrect"
        out[task]["wrong_somewhere"] = profile_wrong or write_wrong
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--writes", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--mutation", default="sequential")
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    writes: dict[tuple[str, str], dict] = {}
    with open(args.writes, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for version, entry in row["versions"].items():
                writes[(row["task_id"], version)] = entry

    results_dir, probe_dir = Path(args.results_dir), Path(args.probe_dir)
    base = load_side(truth, writes, results_dir, probe_dir, "no_mutation",
                     args.benchmark, args.prompt_type)
    mutant = load_side(truth, writes, results_dir, probe_dir, args.mutation,
                       args.benchmark, args.prompt_type)

    stats: Counter = Counter()
    for task, first in base.items():
        second = mutant.get(task)
        if second is None:
            continue

        answers = inconsistent(first["answer_verdict"], second["answer_verdict"],
                               first["answer_text"], second["answer_text"])
        profiles = inconsistent(first["profile_verdict"], second["profile_verdict"],
                                first["profile_text"], second["profile_text"])
        counts = inconsistent(first["write_verdict"], second["write_verdict"],
                              first["write_text"], second["write_text"])
        if answers is None:
            continue

        # one reasoning check: the account of the run disagrees on either field
        reasoning = bool(profiles) or bool(counts)
        wrong = first["wrong_somewhere"] or second["wrong_somewhere"]

        stats["pairs"] += 1
        stats["wrong somewhere"] += wrong
        stats["answers disagree"] += bool(answers)
        stats["control flow disagrees"] += bool(profiles)
        stats["variable updates disagree"] += bool(counts)
        stats["reasoning disagrees"] += reasoning
        stats["either check fires"] += bool(answers) or reasoning

        if wrong:
            stats["answers disagree, and wrong"] += bool(answers)
            stats["control flow disagrees, and wrong"] += bool(profiles)
            stats["reasoning disagrees, and wrong"] += reasoning
            stats["either fires, and wrong"] += bool(answers) or reasoning
            if reasoning and not answers:
                stats["only the reasoning check"] += 1
            if counts and not profiles:
                stats["only the variable updates"] += 1
            if not (answers or reasoning):
                stats["nothing fires"] += 1

    pairs, wrong = stats["pairs"], stats["wrong somewhere"]
    if not pairs:
        print("no comparable pairs: check the folders and the mutation name")
        return

    print(f"=== {args.mutation}: {pairs} pairs, {rate(wrong, pairs)} with something wrong ===\n")
    print(f"{'check':34s}{'flags':>8}{'of those, really wrong':>26}{'of all wrong, found':>22}")
    for label, flagged, found in (
            ("answers disagree (MuCoCo)", "answers disagree", "answers disagree, and wrong"),
            ("control flow disagrees", "control flow disagrees", "control flow disagrees, and wrong"),
            ("reasoning disagrees (both fields)", "reasoning disagrees", "reasoning disagrees, and wrong"),
            ("either check fires", "either check fires", "either fires, and wrong")):
        n, hit = stats[flagged], stats[found]
        print(f"  {label:32s}{n:>8}{rate(hit, n) if n else 'n/a':>26}{rate(hit, wrong):>22}")

    print(f"\n  pairs only the reasoning check finds, that are really wrong: "
          f"{rate(stats['only the reasoning check'], wrong)}")
    print(f"  of those, found by the variable updates and not the control flow: "
          f"{stats['only the variable updates']}")
    print(f"  pairs nothing finds: {rate(stats['nothing fires'], wrong)}")
    print(f"\n  variable updates alone flag {stats['variable updates disagree']} pairs; "
          f"control flow alone flags {stats['control flow disagrees']}")


if __name__ == "__main__":
    main()
