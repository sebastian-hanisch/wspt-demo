"""wspt_algorithm: WSPT-Optimalität gegen unabhängige Brute-Force-Vollaufzählung (Vertauschungsargument-Beweis
empirisch geprüft, nicht nur behauptet), Regressionsschutz, Determinismus, Spezialfall SPT, Rüstzeit-Variante."""

import itertools

import numpy as np
import pytest

import wspt_algorithm as A


def _pw(seed, n):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 100, size=n).astype(np.int64)
    w = rng.choice((1, 2, 4), size=n).astype(np.int64)
    return p, w


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_wspt_matches_brute_force_for_every_seed(n):
    for seed in range(10):
        p, w = _pw(seed, n)
        assert A.wspt(p, w).total == pytest.approx(A.brute_force_optimal(p, w).total)


def test_wspt_is_the_unique_optimum_up_to_ties():
    """Unabhängige Nachrechnung: unter ALLEN Permutationen ist WSPTs Zielfunktionswert das Minimum."""
    p, w = _pw(7, 6)
    wspt_total = A.wspt(p, w).total
    all_totals = [A.evaluate_order(p, w, perm).total for perm in itertools.permutations(range(6))]
    assert wspt_total == pytest.approx(min(all_totals))


def test_completion_times_are_the_cumulative_sum_in_order():
    p = np.array([5, 2, 8, 1])
    w = np.array([1, 1, 1, 1])
    order = np.array([3, 1, 0, 2])          # sortiert nach p: 1, 2, 5, 8
    result = A.evaluate_order(p, w, order)
    assert result.completion.tolist() == [1, 3, 8, 16]
    assert result.total == pytest.approx(1 + 3 + 8 + 16)


def test_weighted_total_multiplies_weight_by_completion():
    p = np.array([2, 3])
    w = np.array([5, 1])
    order = np.array([0, 1])
    result = A.evaluate_order(p, w, order)
    assert result.completion.tolist() == [2, 5]
    assert result.total == pytest.approx(5 * 2 + 1 * 5)


def test_wspt_order_is_descending_by_weight_over_processing_time():
    p = np.array([5, 2, 8, 1, 2])
    w = np.array([1, 4, 1, 1, 2])
    order = A.wspt_order(p, w)
    ratios = (w / p)[order]
    assert list(ratios) == sorted(ratios, reverse=True)


def test_wspt_collapses_to_spt_when_all_weights_are_equal():
    """Der zentrale Spezialfall: w_j = 1 für alle j macht WSPT identisch zu SPT (Stück 1 dieser Linie)."""
    p, _ = _pw(3, 10)
    w = np.ones_like(p)
    assert A.wspt_order(p, w).tolist() == A.spt_order(p).tolist()


def test_random_order_is_deterministic_given_the_rng_state():
    n = 8
    a = A.random_order(n, np.random.default_rng(0))
    b = A.random_order(n, np.random.default_rng(0))
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == list(range(n))


def test_wspt_beats_spt_on_a_hand_picked_instance_where_weights_matter():
    """Handrechnung: ein kurzer, unwichtiger Auftrag und ein langer, sehr wichtiger Auftrag - SPT stellt den
    kurzen zuerst ein (ignoriert Gewicht), WSPT stellt den wichtigen zuerst ein."""
    p = np.array([1, 10])
    w = np.array([1, 100])
    result = A.wspt(p, w)
    assert result.order.tolist() == [1, 0]                       # wichtiger, langer Auftrag zuerst
    assert result.total == pytest.approx(100 * 10 + 1 * 11)
    spt_result = A.evaluate_order(p, w, A.spt_order(p))
    assert spt_result.order.tolist() == [0, 1]                   # SPT: kurzer zuerst, ignoriert Gewicht
    assert result.total < spt_result.total


# --- Mit Rüstzeiten (Vehikel B) --------------------------------------------------------------------------------


def test_setup_variant_matches_the_plain_variant_when_setup_is_zero():
    p, w = _pw(11, 6)
    family = np.array([0, 1, 0, 1, 0, 1])
    setup = np.zeros((2, 2))
    plain = A.wspt(p, w)
    with_setup = A.evaluate_order_with_setup(p, w, family, setup, plain.order)
    assert with_setup.total == pytest.approx(plain.total)


def test_setup_time_is_only_charged_on_a_family_change():
    p = np.array([2, 2, 2])
    w = np.array([1, 1, 1])
    family = np.array([0, 0, 1])
    setup = np.array([[0, 10], [10, 0]])
    order = np.array([0, 1, 2])                                 # kein Wechsel, dann ein Wechsel
    result = A.evaluate_order_with_setup(p, w, family, setup, order)
    assert result.completion.tolist() == [2, 4, 4 + 10 + 2]


def test_brute_force_with_setup_matches_independent_full_enumeration():
    p, w = _pw(13, 5)
    family = np.array([0, 1, 0, 1, 2])
    setup = np.array([[0, 5, 8], [5, 0, 3], [8, 3, 0]])
    best = A.brute_force_optimal_with_setup(p, w, family, setup)
    all_totals = []
    for perm in itertools.permutations(range(5)):
        order = np.array(perm)
        completion = A.completion_times_with_setup(p, family, setup, order)
        all_totals.append(A.weighted_total(w, order, completion))
    assert best.total == pytest.approx(min(all_totals))


def test_ignoring_setup_can_be_worse_than_the_true_optimum():
    """WSPT (sortiert nur nach wⱼ/pⱼ) muss nicht optimal bleiben, sobald Rüstzeiten dazukommen - genau die
    Frage, die Vehikel B stellt."""
    p = np.array([1, 1, 10, 10])
    w = np.array([1, 1, 1, 1])
    family = np.array([0, 1, 0, 1])
    setup = np.array([[0, 100], [100, 0]])
    wspt_total = A.evaluate_order_with_setup(p, w, family, setup, A.wspt_order(p, w)).total
    true_opt = A.brute_force_optimal_with_setup(p, w, family, setup).total
    assert wspt_total > true_opt + 1e-6
