"""The probe on MuCoCo's second prediction task.

Their input-prediction task gives the model a program, an input and an output, and asks
whether that input produces that output. The answer is True or False, so the probe attaches
without any change to the question: the program and the input are both in the prompt, and
the tracer can run them.

Three things about this task are worth knowing before reading any result from it.

  * **Two wrong answers are always identical.** With a binary answer, MuCoCo's
    incorrectness-based inconsistency can never fire: one of their three categories is
    structurally unavailable here.
  * **Agreement happens by chance half the time.** Two independent guesses match 50% of the
    time, so the measured consistency rate is floored by luck rather than by competence.
  * **Their own figure for the task is the lowest of the four**, 6.65% inconsistency at
    78.97% accuracy, for exactly these reasons.

So this task is run to show that the profile signal transfers to a second task, not to
produce a large answer-inconsistency rate. A rate near zero there is the expected result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from probe.analyse import read_results_csv, verdict
from probe.score import load_predictions, load_truth
from probe.stats import rate


def profile_prompt_for_input_prediction(base_helper):
    """The same wrapper as output prediction.

    `ProfilePrompt` keys its checklist on the program text, which their input-prediction
    template passes under the same name, so nothing needs changing. This function exists to
    make that explicit rather than leaving it as a coincidence someone has to rediscover.
    """
    from probe.prompt import ProfilePrompt

    return ProfilePrompt(base_helper)


def summarise(truth: dict, results_dir: Path, probe_dir: Path, version: str,
              benchmark: str = "CruxEval", prompt_type: str = "zero_shot") -> Counter:
    """Profile accuracy and answer outcomes for one version of the input-prediction run."""
    csv_path = results_dir / f"{benchmark}_{prompt_type}_{version}.csv"
    log_path = probe_dir / f"profiles_{version}.jsonl"

    rows = read_results_csv(csv_path) if csv_path.exists() else {}
    by_code = {entry["code"]: task for (task, ver), entry in truth.items()
               if ver == version and entry["code"]}

    replies: dict[str, dict] = {}
    if log_path.exists():
        for prediction in load_predictions(log_path):
            task = by_code.get(prediction["full_sol"])
            if task:
                replies[task] = prediction

    stats: Counter = Counter()
    for (task, ver), entry in truth.items():
        if ver != version or not entry["counts"]:
            continue
        row = rows.get(task)
        if row is not None:
            stats[f"answer {verdict(row)}"] += 1
        prediction = replies.get(task)
        for name, real in entry["counts"].items():
            if prediction is None:
                stats["profile: no reply"] += 1
                continue
            claimed = (prediction.get("profile") or {}).get(name)
            if claimed is None:
                stats["profile: not answered"] += 1
            elif tuple(claimed) == tuple(real):
                stats["profile: correct"] += 1
            else:
                stats["profile: wrong"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--versions", nargs="*", default=["no_mutation", "sequential"])
    parser.add_argument("--benchmark", default="CruxEval")
    parser.add_argument("--prompt-type", default="zero_shot")
    args = parser.parse_args()

    truth = load_truth(args.truth)
    for version in args.versions:
        stats = summarise(truth, Path(args.results_dir), Path(args.probe_dir), version,
                          args.benchmark, args.prompt_type)
        if not stats:
            print(f"\n=== {version}: nothing found, check the file names ===")
            continue

        answered = stats["answer correct"] + stats["answer incorrect"]
        profiles = (stats["profile: correct"] + stats["profile: wrong"]
                    + stats["profile: not answered"] + stats["profile: no reply"])
        print(f"\n=== {version}, input prediction ===")
        if answered:
            print(f"  answer accuracy      {rate(stats['answer correct'], answered)}")
            print(f"  answers judged invalid {stats['answer invalid']}")
        print(f"  profile accuracy     {rate(stats['profile: correct'], profiles)}")
        for name in ("profile: wrong", "profile: not answered", "profile: no reply"):
            print(f"  {name:22s} {rate(stats[name], profiles)}")
        print("  note: a True or False answer makes two wrong answers identical, so their "
              "incorrectness-based inconsistency cannot fire on this task.")


if __name__ == "__main__":
    main()
