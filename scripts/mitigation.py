"""Acting on a flag: withhold the answer rather than re-ask.

MuCoCo's Appendix I keeps only answers the model was confident about and reports what that
does to their two metrics. This is the same table with each check as the signal.

    python scripts/mitigation.py --results <folder> --truth <traces.jsonl>
"""
import argparse
from collections import Counter

from _shared import (CONTROL_ONLY, account, add_arguments, answer_flag,
                     constructs_only, load_all, operators_in, pairs_for,
                     trace_flag)


def main():
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    kept = Counter()
    for operator in operators_in(args.results):
        compare = constructs_only if operator in CONTROL_ONLY else account
        for task, csv_row, reply, mutant_entry in pairs_for(args.results, truth,
                                                            by_code, operator):
            original_entry = truth.get((task, "no_mutation"))
            if task not in originals_csv or original_entry is None:
                continue
            mutant_out = (csv_row.get("model_output") or "").strip()
            if not mutant_out:
                continue

            original_out, original_ok = originals_csv[task]
            mutant_ok = not (csv_row.get("failure_type") or "").strip()
            answers = answer_flag(original_out, original_ok, mutant_out, mutant_ok)
            traces = trace_flag(compare(originals_replies.get(original_entry["code"]),
                                        original_entry),
                                compare(reply, mutant_entry))

            for rule, survives in (("answer everything", True),
                                   ("withhold what the trace check flags", not traces),
                                   ("withhold what the answer check flags", not answers),
                                   ("withhold what either flags", not answers and not traces)):
                if survives:
                    kept[f"{rule}: kept"] += 1
                    kept[f"{rule}: inconsistent"] += answers
                    kept[f"{rule}: correct"] += original_ok

    if not kept["answer everything: kept"]:
        print(f"\n  no comparable test cases in {args.results}: the folder needs both a "
              f"CSV and a reply log for at least one operator besides no_mutation\n")
        return

    print(f"\n  MuCoCo's two metrics on what each rule leaves\n")
    print(f"  {'rule':<38}{'kept':>8}{'inconsistency':>16}{'accuracy':>11}")
    for rule in ("answer everything", "withhold what the trace check flags",
                 "withhold what the answer check flags", "withhold what either flags"):
        n = kept[f"{rule}: kept"]
        if not n:
            continue
        print(f"  {rule:<38}{n:>8}{kept[f'{rule}: inconsistent'] / n:>15.2%}"
              f"{kept[f'{rule}: correct'] / n:>10.2%}")

    print(f"\n  The answer check drives its own inconsistency to zero by construction, so")
    print(f"  that column is only meaningful for the trace rows.")


if __name__ == "__main__":
    main()
