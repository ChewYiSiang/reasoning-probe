"""Tables A and B: the inconsistency rate, per operator and by mutation family.

MuCoCo's own metric, so the figures sit beside theirs. Both checks run on the same replies,
so the only difference between them is the rule.

    python scripts/inconsistency_rate.py --results <folder> --truth <traces.jsonl>
"""
import argparse
from collections import Counter

from _shared import (CATEGORY, CONTROL_ONLY, account, add_arguments,
                     answer_flag, constructs_only, load_all, operators_in, pairs_for,
                     trace_flag)


def score(folder, truth, by_code, originals_csv, originals_replies, operator):
    """Both checks on one operator's test cases."""
    compare = constructs_only if operator in CONTROL_ONLY else account
    stats = Counter()

    for task, csv_row, reply, mutant_entry in pairs_for(folder, truth, by_code, operator):
        original_entry = truth.get((task, "no_mutation"))
        if task not in originals_csv or original_entry is None:
            continue
        mutant_out = (csv_row.get("model_output") or "").strip()
        if not mutant_out:
            continue                      # the mutant was rejected, so no answer exists

        original_out, original_ok = originals_csv[task]
        mutant_ok = not (csv_row.get("failure_type") or "").strip()

        stats["cases"] += 1
        stats["answer"] += answer_flag(original_out, original_ok, mutant_out, mutant_ok)
        stats["either"] += (answer_flag(original_out, original_ok, mutant_out, mutant_ok)
                            or trace_flag(compare(originals_replies.get(original_entry["code"]),
                                                  original_entry),
                                          compare(reply, mutant_entry)))
        stats["mutant correct"] += mutant_ok
    return stats


def main():
    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    per_operator = {}
    for operator in operators_in(args.results):
        stats = score(args.results, truth, by_code, originals_csv,
                      originals_replies, operator)
        if stats["cases"]:
            per_operator[operator] = stats

    if not per_operator:
        print(f"\n  no comparable test cases in {args.results}: the folder needs both a "
              f"CSV and a reply log for at least one operator besides no_mutation\n")
        return

    print(f"\n  Per operator, ordered by what the trace check adds\n")
    print(f"  {'operator':<22}{'test cases':>12}{'Inc. ans':>10}"
          f"{'Inc. ans+trace':>16}{'Acc.':>8}")
    ordered = sorted(per_operator.items(),
                     key=lambda kv: -(kv[1]["either"] / kv[1]["cases"]))
    for operator, s in ordered:
        mark = "*" if operator in CONTROL_ONLY else " "
        print(f"  {operator:<22}{s['cases']:>12}{s['answer'] / s['cases']:>9.2%}"
              f"{s['either'] / s['cases']:>15.2%}{mark}"
              f"{s['mutant correct'] / s['cases']:>7.2%}")

    print(f"\n\n  Table A: by mutation family\n")
    print(f"  {'Mutation Type':<16}{'Inc. (ans)':>13}{'Inc. (ans+trace)':>20}{'Acc.':>9}")
    grand = Counter()
    for family in ("Logical", "Syntactic", "Lexical"):
        c, marked = Counter(), False
        for operator, s in per_operator.items():
            if CATEGORY[operator] == family:
                c.update(s)
                marked |= operator in CONTROL_ONLY
        if not c["cases"]:
            continue
        grand.update(c)
        print(f"  {family:<16}{c['answer'] / c['cases']:>12.2%}"
              f"{c['either'] / c['cases']:>18.2%}{'*' if marked else ' '}"
              f"{c['mutant correct'] / c['cases']:>8.2%}")
    print(f"  {'All (Avg.)':<16}{grand['answer'] / grand['cases']:>12.2%}"
          f"{grand['either'] / grand['cases']:>18.2%} "
          f"{grand['mutant correct'] / grand['cases']:>8.2%}")
    print(f"\n  * only the control-flow half of the trace is comparable: the rewrite")
    print(f"    introduces counters the original never had.")

    print(f"\n\n  Table B: MuCoCo's Table 10 format\n")
    print(f"  {'Approach':<36}{'Consistency errors':>20}{'Inconsistency rate':>20}")
    answer_cell = f"{grand['answer']} of {grand['cases']}"
    either_cell = f"{grand['either']} of {grand['cases']}"
    extra = grand["either"] - grand["answer"]
    print(f"  {'MuCoCo (answer check)':<36}{answer_cell:>20}"
          f"{grand['answer'] / grand['cases']:>19.2%}")
    print(f"  {'MuCoCo (extended, answer + trace)':<36}{either_cell:>20}"
          f"{grand['either'] / grand['cases']:>19.2%}")
    relative = f"{extra / grand['answer']:.2%}" if grand["answer"] else "n/a"
    print(f"  {'Improvement':<36}{f'+{extra} errors':>20}{relative:>19}")
    print(f"\n  MuCoCo publish 18.99% on output prediction, pooled over seven models.")


if __name__ == "__main__":
    main()
