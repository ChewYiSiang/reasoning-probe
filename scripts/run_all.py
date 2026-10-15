"""Run every experiment and write the output to a file, for presenting or for the record.

    python scripts/run_all.py --results <folder> --truth <traces.jsonl> --out results.txt

Each script is run as it would be on its own, so the output is the same either way.
"""
import argparse
import io
import os
import runpy
import sys
from contextlib import redirect_stdout

SCRIPTS = [
    ("Inconsistency rate, per operator and by family", "inconsistency_rate.py"),
    ("RQ1: do the two checks fire together", "rq1_venn.py"),
    ("RQ3: how well does the model trace", "trace_accuracy.py"),
    ("Error taxonomy", "error_taxonomy.py"),
    ("Mitigation: withhold rather than re-ask", "mitigation.py"),
    ("What a developer gets without the tracer", "tracer_free_checks.py"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--out", default="results.txt")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    collected = []
    for title, script in SCRIPTS:
        buffer = io.StringIO()
        sys.argv = [script, "--results", args.results, "--truth", args.truth]
        try:
            with redirect_stdout(buffer):
                runpy.run_path(os.path.join(here, script), run_name="__main__")
        except SystemExit:
            pass
        except Exception as exc:
            buffer.write(f"\n  failed: {type(exc).__name__}: {exc}\n")
        collected.append(f"{'=' * 78}\n{title}\n{'=' * 78}\n{buffer.getvalue()}")
        print(f"  ran {script}")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(collected))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
