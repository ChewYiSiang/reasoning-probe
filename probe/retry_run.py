"""Re-ask the flagged pairs and score whether the answer settles on the right one.

The prompts are already complete, built by `probe.retry`, so this does not go through their
tester: it calls the installed model wrapper directly with a passthrough template and writes
a result CSV in their format, which `probe.retry --after` can then score.

Correctness is decided the way their tester decides it, by comparing the model's answer with
the expected output after both are parsed as Python literals, falling back to a string
comparison when either will not parse.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


class Passthrough:
    """The wrapper formats a template; the retry prompt is already the whole prompt."""

    def format(self, **variables) -> str:
        return variables["prompt"]


def wanted_value(expected: str) -> str:
    """The value their tester compares against, out of what it writes to the CSV.

    It writes the whole output record, `{'args': '4', 'metadata': 'int'}`, while the
    comparison at line 251 of their tester uses `output_args`, which is the 'args' field.
    Comparing against the record itself never matches.
    """
    expected = (expected or "").strip()
    try:
        record = ast.literal_eval(expected)
    except Exception:
        return expected
    if isinstance(record, dict) and "args" in record:
        return str(record["args"])
    return expected


def given_value(answer: str) -> str:
    """The answer out of whatever shape the reply arrived in.

    The wrapper's `invoke` returns `{'ans': ..., 'geom_mean_prob': ...}`; their tester logs
    `(value, type)`; and a bare string is possible too. All three appear in the result files,
    so all three are unwrapped here. The value inside is itself a repr, so a list arrives
    carrying its quotes, which also come off.
    """
    answer = (answer or "").strip()
    if answer.startswith("(") and "<class" in answer:
        answer = answer[1:answer.rindex(",")].strip()
    try:
        value = ast.literal_eval(answer)
    except Exception:
        return answer
    if isinstance(value, dict) and "ans" in value:
        value = value["ans"]
    return value if isinstance(value, str) else answer


def same_answer(given: str, expected: str) -> bool:
    """Their comparison: equal as Python values where possible, else as text."""
    given, expected = given_value(given), wanted_value(expected)
    try:
        return ast.literal_eval(given) == ast.literal_eval(expected)
    except Exception:
        return given.strip() == expected.strip()


def run(prompts_path: str, expected_from: str, out_csv: str, llm) -> dict:
    """Generate an answer for every flagged pair and write a result CSV in their format."""
    prompts = [json.loads(line) for line in open(prompts_path, encoding="utf-8")]

    expected = {}
    with open(expected_from, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["task_id"]] = row.get("expected_output", "")

    template = Passthrough()
    counts = {"written": 0, "correct": 0, "incorrect": 0}
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "prompt", "model_output", "expected_output",
                         "failure_type"])
        for index, pair in enumerate(prompts, start=1):
            answer = llm.invoke({"prompt": pair["prompt"]}, template)
            want = expected.get(pair["task_id"], "")
            right = same_answer(str(answer), want)
            writer.writerow([pair["task_id"], "", answer, want,
                             "" if right else "AssertionError: "])
            counts["written"] += 1
            counts["correct" if right else "incorrect"] += 1
            if index % 50 == 0:
                print(f"  {index}/{len(prompts)}")
    return counts


def rescore(after_csv: str, expected_from: str) -> dict:
    """Re-decide correctness on a retry CSV that is already written.

    The replies are the expensive part and they are kept in the CSV, so a mistake in the
    comparison costs a re-score rather than a re-run.
    """
    expected = {}
    with open(expected_from, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[row["task_id"]] = row.get("expected_output", "")

    with open(after_csv, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts = {"rows": len(rows), "correct": 0, "incorrect": 0}
    for row in rows:
        right = same_answer(row.get("model_output", ""),
                            expected.get(row["task_id"], row.get("expected_output", "")))
        row["failure_type"] = "" if right else "AssertionError: "
        counts["correct" if right else "incorrect"] += 1

    with open(after_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "prompt", "model_output",
                                                    "expected_output", "failure_type"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Re-score a retry CSV in place.")
    parser.add_argument("--after", required=True, help="the retry result CSV")
    parser.add_argument("--expected-from", required=True,
                        help="the original result CSV, for the expected answers")
    args = parser.parse_args()
    counts = rescore(args.after, args.expected_from)
    print(f"  {counts['rows']} rows | correct {counts['correct']} | "
          f"incorrect {counts['incorrect']}")


if __name__ == "__main__":
    main()
