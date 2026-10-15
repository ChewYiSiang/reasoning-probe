"""RQ1: do the two checks fire on the same test cases?

A two-by-two over every pair, and the four Venn regions it gives. Neither check runs the
program; both compare the model with itself across the two versions.

    python scripts/rq1_venn.py --results <folder> --truth <traces.jsonl>
"""
import argparse
from collections import Counter

from _shared import (CONTROL_ONLY, account, add_arguments, answer_flag,
                     constructs_only, load_all, operators_in, pairs_for,
                     trace_flag)


def main():
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    cells = Counter()
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
            cells[(answers, traces)] += 1

    total = sum(cells.values())
    if not total:
        print(f"\n  no comparable test cases in {args.results}: the folder needs both a "
              f"CSV and a reply log for at least one operator besides no_mutation\n")
        return
    print(f"\n  {total:,} test cases\n")
    print(f"  {'':24}{'trace agrees':>18}{'trace differs':>18}{'row total':>14}")
    for flag, label in ((False, "answers agree"), (True, "answers differ")):
        a, b = cells[(flag, False)], cells[(flag, True)]
        print(f"  {label:24}{a:>11} ({a / total:>5.1%}){b:>11} ({b / total:>5.1%})"
              f"{a + b:>8} ({(a + b) / total:>5.1%})")
    column_a = cells[(False, False)] + cells[(True, False)]
    column_b = cells[(False, True)] + cells[(True, True)]
    print(f"  {'column total':24}{column_a:>11} ({column_a / total:>5.1%})"
          f"{column_b:>11} ({column_b / total:>5.1%}){total:>8}")

    print(f"\n  The two circles\n")
    print(f"    answers only   {cells[(True, False)]:>6}")
    print(f"    both           {cells[(True, True)]:>6}")
    print(f"    trace only     {cells[(False, True)]:>6}"
          f"   reported consistent by MuCoCo")
    print(f"    neither        {cells[(False, False)]:>6}")

    # how much more likely a disagreement is when the traces disagree, which says the two
    # are related without claiming either contains the other
    odds_agree = cells[(True, False)] / max(cells[(False, False)], 1)
    odds_differ = cells[(True, True)] / max(cells[(False, True)], 1)
    if odds_agree:
        print(f"\n  answers differ {odds_differ / odds_agree:.2f} times as often when the "
              f"traces differ")
    else:
        print(f"\n  the answer check flagged nothing where the traces agreed, so the ratio "
              f"is undefined")


if __name__ == "__main__":
    main()
