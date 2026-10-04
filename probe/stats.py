"""The three statistics the plan actually needs, written out longhand.

No SciPy: each one is a few lines of arithmetic, and keeping them visible means the
numbers in the write-up can be checked by hand.

  * a proportion with its margin of error (Wilson interval)
  * the coin-flip test for paired yes/no outcomes (exact McNemar, i.e. a sign test)
  * a 2x2 table read as two conditional percentages, with Fisher's exact test
"""

from __future__ import annotations

from math import comb, sqrt


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and its 95% interval. Returns (rate, low, high)."""
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (p, max(0.0, centre - spread), min(1.0, centre + spread))


def rate(successes: int, total: int) -> str:
    """Formatted for a report: 28/978 (2.9%, 2.0 to 4.1)."""
    p, low, high = wilson(successes, total)
    return f"{successes}/{total} ({p:.1%}, {low:.1%} to {high:.1%})"


def mcnemar(only_first: int, only_second: int) -> float:
    """Two-sided p for a paired yes/no comparison.

    Only the cases where the two conditions disagree carry information. Under no
    difference each is a fair coin, so this is the chance of a split at least this
    lopsided.
    """
    n = only_first + only_second
    if n == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided p for a 2x2 table [[a, b], [c, d]], by summing hypergeometric tails."""
    row1, row2 = a + b, c + d
    col1 = a + c
    total = row1 + row2
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return 1.0

    def probability(x: int) -> float:
        return comb(row1, x) * comb(row2, col1 - x) / comb(total, col1)

    observed = probability(a)
    low = max(0, col1 - row2)
    high = min(col1, row1)
    # Sum every table at least as unlikely as the one observed.
    return min(1.0, sum(probability(x) for x in range(low, high + 1)
                        if probability(x) <= observed * (1 + 1e-9)))


def two_by_two(a: int, b: int, c: int, d: int) -> str:
    """Read a 2x2 as two conditional rates and their ratio.

    Layout: a = both agree, b = only the second differs, c = only the first differs,
    d = both differ; where "first" is the output signal and "second" the profile signal.
    """
    with_profile_diff = c + d          # pairs where profiles differ
    without = a + b
    p_with = d / with_profile_diff if with_profile_diff else 0.0
    p_without = b / without if without else 0.0
    lift = (p_with / p_without) if p_without else float("inf")
    return (f"when profiles differ, answers differ {rate(d, with_profile_diff)}\n"
            f"when profiles agree,  answers differ {rate(b, without)}\n"
            f"lift {lift:.1f}x | Fisher p = {fisher_exact(a, b, c, d):.4f}")
