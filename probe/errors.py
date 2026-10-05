"""What kind of profile error was it?

Knowing that 44% of counts are wrong says nothing a user can act on. This module takes
each wrong count and describes it along five axes, all computed from numbers we already
have: no reading of the model's prose, no judge.

  kind       claimed it never ran, claimed it ran when it did not, outcomes swapped,
             or simply the wrong number
  direction  over-count or under-count
  distance   off by one, off by a small factor, or out by an order of magnitude
  shape      the construct it happened on (nested, early exit, long loop, ...)
  version    the original or the mutant

The pair-level half mirrors the three granular measures MuCoCo reports for answers in its
appendices, so our numbers sit beside theirs:

  type       correctness-based, incorrectness-based or invalidity-based (their Table 2)
  direction  which side of the pair carried the error (their Appendix E)
  distance   how much of the profile differs, rather than a yes or no (their Appendix F)
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from probe.score import load_predictions, load_truth
from probe.stats import rate


@dataclass(frozen=True)
class ProfileError:
    """One construct the model got wrong, described."""
    task_id: str
    version: str
    construct: str
    kind: str
    direction: str
    distance: str
    gap: int                       # how far out, in absolute terms
    claimed: tuple[int, ...]
    real: tuple[int, ...]


def _distance_bucket(claimed: int, real: int) -> str:
    gap = abs(claimed - real)
    if gap == 0:
        return "exact"
    if gap == 1:
        return "out by one"
    larger, smaller = max(claimed, real), max(min(claimed, real), 1)
    return "out by double or more" if larger >= 2 * smaller else "out by a few"


def classify(construct: str, claimed: tuple[int, ...], real: tuple[int, ...]) -> ProfileError | None:
    """Describe one construct's error, or return None when it is correct."""
    if tuple(claimed) == tuple(real):
        return None

    if len(real) == 1:                                   # a loop
        claimed_count, real_count = claimed[0], real[0]
        if claimed_count == 0 and real_count > 0:
            kind = "said it never ran, but it did"
        elif real_count == 0 and claimed_count > 0:
            kind = "said it ran, but it never did"
        else:
            kind = "wrong number of times round the loop"
        direction = "too high" if claimed_count > real_count else "too low"
        return ProfileError("", "", construct, kind, direction,
                            _distance_bucket(claimed_count, real_count),
                            abs(claimed_count - real_count), tuple(claimed), tuple(real))

    # a branch: (times the condition was true, times it was false)
    claimed_true, claimed_false = claimed[0], claimed[1]
    real_true, real_false = real[0], real[1]
    if (claimed_true, claimed_false) == (real_false, real_true):
        kind = "true and false the wrong way round"
    elif claimed_true == 0 and real_true > 0:
        kind = "said the condition was never true, but it was"
    elif claimed_false == 0 and real_false > 0:
        kind = "said the condition was never false, but it was"
    elif claimed_true + claimed_false != real_true + real_false:
        # the two numbers do not add up to the number of times the condition was checked,
        # which means the model lost the loop around the branch, not the branch itself
        kind = "the two numbers do not add up to how often it was checked"
    else:
        kind = "right total, wrong split between true and false"

    claimed_total, real_total = claimed_true + claimed_false, real_true + real_false
    direction = ("too high" if claimed_total > real_total
                 else "too low" if claimed_total < real_total
                 else "right total, wrong split")
    # How far out the branch is: the worse of its two numbers, so a pair that adds up to
    # the right total but splits it wrongly is still measured rather than called exact.
    worse = max(abs(claimed_true - real_true), abs(claimed_false - real_false))
    if claimed_total == real_total:
        distance = "out by one" if worse == 1 else "out by a few" if worse < 2 * max(real_true, real_false, 1) else "out by double or more"
    else:
        distance = _distance_bucket(claimed_total, real_total)
    return ProfileError("", "", construct, kind, direction, distance,
                        worse, tuple(claimed), tuple(real))


def errors_for_version(task_id: str, version: str, claimed_profile: dict,
                       real_profile: dict) -> tuple[list[ProfileError], int, int]:
    """Every error in one reply, plus how many constructs were right and unanswered."""
    found, correct, unanswered = [], 0, 0
    for construct, real in real_profile.items():
        claimed = claimed_profile.get(construct)
        if claimed is None:
            unanswered += 1
            continue
        error = classify(construct, tuple(claimed), tuple(real))
        if error is None:
            correct += 1
        else:
            found.append(ProfileError(task_id, version, error.construct, error.kind,
                                      error.direction, error.distance, error.gap,
                                      error.claimed, error.real))
    return found, correct, unanswered


def profile_distance(first: dict, second: dict) -> float:
    """How much of the profile differs between two versions, as a fraction.

    MuCoCo's inconsistency distance counts the share of test cases whose outcome differs
    (Appendix F). This is the same idea over constructs, so an inconsistency has a size
    rather than being only a yes or a no.
    """
    shared = set(first) & set(second)
    if not shared:
        return 0.0
    differing = sum(1 for name in shared if tuple(first[name]) != tuple(second[name]))
    return differing / len(shared)


def report(errors: list[ProfileError], correct: int, unanswered: int,
           shapes: dict[tuple[str, str], list[str]] | None = None) -> None:
    total = correct + len(errors) + unanswered
    print(f"\n=== constructs: {total} | correct {correct} | wrong {len(errors)} | "
          f"not answered {unanswered} ===")

    for title, key in (("kind of error", lambda e: e.kind),
                       ("direction", lambda e: e.direction),
                       ("how far out", lambda e: e.distance),
                       ("which version", lambda e: e.version)):
        counts = Counter(key(error) for error in errors)
        if not counts:
            continue
        print(f"\n  {title}")
        for name, count in counts.most_common():
            print(f"    {name:38s} {rate(count, len(errors))}")

    if shapes:
        shape_counts: Counter = Counter()
        for error in errors:
            for shape in shapes.get((error.task_id, error.construct), []):
                shape_counts[shape] += 1
        if shape_counts:
            print("\n  construct shape (one error can have several)")
            for name, count in shape_counts.most_common(8):
                print(f"    {name:38s} {rate(count, len(errors))}")

    big = [error for error in errors if error.gap >= 5]
    if big:
        print(f"\n  errors out by five or more: {rate(len(big), len(errors))}")
        for error in sorted(big, key=lambda e: -e.gap)[:5]:
            print(f"    {error.task_id} {error.construct}: claimed {error.claimed}, "
                  f"actually {error.real} ({error.kind})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--probe-log", required=True, nargs="+",
                        help="one or more profile logs")
    parser.add_argument("--version", required=True, nargs="+",
                        help="the version each log belongs to, in the same order")
    args = parser.parse_args()

    if len(args.probe_log) != len(args.version):
        parser.error("give one --version for each --probe-log, in the same order")

    truth = load_truth(args.truth)
    errors, correct, unanswered = [], 0, 0

    for log_path, version in zip(args.probe_log, args.version):
        by_code = {value["code"]: key for key, value in truth.items()
                   if value["code"] and key[1] == version}
        for prediction in load_predictions(Path(log_path)):
            key = by_code.get(prediction["full_sol"])
            if key is None:
                continue
            found, right, missing = errors_for_version(key[0], version,
                                                       prediction.get("profile") or {},
                                                       truth[key]["counts"])
            errors.extend(found)
            correct += right
            unanswered += missing

    report(errors, correct, unanswered)


if __name__ == "__main__":
    main()
