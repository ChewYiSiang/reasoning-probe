"""RQ3: how well does the model trace, and what do the recovered counts measure?

The first two figures are direct: they compare the model's trace with the tracer's for the
same version. The recovery figures are split by whether the trace itself was right, because
the overall number is a mixture of the two and reflects how often the model traces correctly
rather than how well it counts.

    python scripts/trace_accuracy.py --results <folder> --truth <traces.jsonl>
"""
import argparse
from collections import Counter

from _shared import MARK, add_arguments, align, counts_from, load_all, parse_trace, writes_from


def main():
    args = add_arguments(argparse.ArgumentParser(description=__doc__)).parse_args()
    truth, by_code, originals_csv, originals_replies = load_all(args.results, args.truth)

    stats = Counter()
    for code, reply in originals_replies.items():
        found = by_code.get(("no_mutation", code))
        if not found:
            continue
        task, entry = found
        text = reply.get("raw_reply") or ""
        if MARK not in text:
            stats["no trace section"] += 1
            continue
        claimed = parse_trace(text)
        if not claimed:
            stats["unreadable"] += 1
            continue

        real = parse_trace(entry["text"])
        result = align(claimed, real)
        stats["traces scored"] += 1
        stats["path exact"] += result["path_exact"]
        stats["both exact"] += result["path_exact"] and result["values_wrong"] == 0

        claimed_counts = counts_from(claimed, entry["code"])
        real_counts = counts_from(real, entry["code"])
        bucket = "path right" if result["path_exact"] else "path wrong"
        stats[f"{bucket}: asked"] += len(real_counts)
        stats[f"{bucket}: right"] += sum(1 for n, v in real_counts.items()
                                         if claimed_counts.get(n) == v)

        claimed_writes, real_writes = writes_from(claimed), writes_from(real)
        stats["variables asked"] += len(real_writes)
        stats["variables right"] += sum(1 for n, v in real_writes.items()
                                        if claimed_writes.get(n) == v)

    n = stats["traces scored"]
    print(f"\n  {n} traces scored, {stats['no trace section'] + stats['unreadable']} skipped\n")
    print(f"  The trace itself\n")
    print(f"    the whole execution path right   {stats['path exact']:>5}  "
          f"{stats['path exact'] / n:>6.1%}")
    print(f"    the path and every value right   {stats['both exact']:>5}  "
          f"{stats['both exact'] / n:>6.1%}")

    print(f"\n  Counts recovered from the trace, split by whether the trace was right\n")
    for bucket in ("path right", "path wrong"):
        asked, right = stats[f"{bucket}: asked"], stats[f"{bucket}: right"]
        if asked:
            print(f"    where the {bucket:<12}{right:>5}/{asked:<6}{right / asked:>7.1%}")
    total_asked = stats["path right: asked"] + stats["path wrong: asked"]
    total_right = stats["path right: right"] + stats["path wrong: right"]
    print(f"    overall             {total_right:>5}/{total_asked:<6}"
          f"{total_right / total_asked:>7.1%}")
    print(f"\n    the overall figure is a mixture, so it measures how often the model")
    print(f"    traces correctly rather than how well the counts are recovered from it")

    print(f"\n  Variable change counts   {stats['variables right']}/"
          f"{stats['variables asked']} = "
          f"{stats['variables right'] / stats['variables asked']:.1%}")


if __name__ == "__main__":
    main()
