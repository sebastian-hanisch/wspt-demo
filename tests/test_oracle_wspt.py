"""Orakel-Test: WSPT gegen Smiths Regel mit exakten Brüchen (Fraction, keine Fließkomma-Quotienten), Permutations-
aufzählung mit eigener Summenbildung und die Rüstzeit-Variante gegen eine Held-Karp-DP (Gewicht der Restmenge)."""

import itertools
import random
from fractions import Fraction

import numpy as np

import wspt_algorithm as A


def _wtotal(p, w, order, fam=None, setup=None):
    t = tot = 0
    prev = None
    for j in order:
        if fam is not None and prev is not None:
            t += int(setup[prev][fam[j]])
        t += int(p[j])
        prev = fam[j] if fam is not None else None
        tot += int(w[j]) * t
    return tot


def _opt_dp(p, w, fam=None, setup=None):
    n = len(p)
    total_w = sum(int(x) for x in w)
    inf = 10 ** 15
    wsum = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = (mask & -mask).bit_length() - 1
        wsum[mask] = wsum[mask & (mask - 1)] + int(w[low])
    g = [[inf] * n for _ in range(1 << n)]
    for j in range(n):
        g[1 << j][j] = total_w * int(p[j])
    for mask in range(1, 1 << n):
        rem = total_w - wsum[mask]
        for last in range(n):
            v = g[mask][last]
            if v >= inf:
                continue
            for j in range(n):
                if mask >> j & 1:
                    continue
                s = int(setup[fam[last]][fam[j]]) if fam is not None else 0
                nm = mask | 1 << j
                g[nm][j] = min(g[nm][j], v + rem * (s + int(p[j])))
    return min(g[(1 << n) - 1])


def test_wspt_equals_smith_rule_with_fractions_and_enumerated_optimum():
    rng = random.Random(8)
    for _ in range(150):
        n = rng.randint(1, 6)
        hi = rng.choice([2, 4, 100])
        p = np.array([rng.randint(1, hi) for _ in range(n)])
        w = np.array([rng.choice([1, 2, 4]) for _ in range(n)])
        smith = sorted(range(n), key=lambda j: (-Fraction(int(w[j]), int(p[j])), j))
        res = A.wspt(p, w)
        assert res.total == _wtotal(p, w, smith) == _wtotal(p, w, res.order)
        assert min(_wtotal(p, w, perm) for perm in itertools.permutations(range(n))) == res.total
        assert A.brute_force_optimal(p, w).total == res.total
        assert A.wspt_order(p, np.ones(n, dtype=np.int64)).tolist() == A.spt_order(p).tolist()


def test_setup_variant_matches_independent_total_and_held_karp_dp():
    rng = random.Random(6)
    for _ in range(60):
        n = rng.randint(1, 6)
        p = np.array([rng.randint(1, 30) for _ in range(n)])
        w = np.array([rng.choice([1, 2, 4]) for _ in range(n)])
        fam = np.array([rng.randint(0, 2) for _ in range(n)])
        s = rng.choice([0, 10, 40])
        setup = np.array([[0 if a == b else s for b in range(3)] for a in range(3)])
        order = rng.sample(range(n), n)
        assert A.evaluate_order_with_setup(p, w, fam, setup, order).total == _wtotal(p, w, order, fam, setup)
        assert A.brute_force_optimal_with_setup(p, w, fam, setup).total == _opt_dp(p, w, fam, setup)
