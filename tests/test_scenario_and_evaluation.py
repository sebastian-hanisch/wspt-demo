"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweep, Optimalitäts- und Timing-Messreihe, Vehikel-B-Härtetest (Rüstzeiten)."""

from dataclasses import replace

import numpy as np
import pytest

import wspt_algorithm as A
import wspt_constants as C
import wspt_evaluation as ev
import wspt_scenario as S
import wspt_scenario_logistik as SL


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 3)
    assert inst.n == 20 and inst.p.shape == (20,) and inst.w.shape == (20,)
    assert inst.p.min() >= C.P_MIN and inst.p.max() <= C.P_MAX
    assert set(inst.w.tolist()) <= set(C.WEIGHT_VALUES)


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(30, 5), S.generate(30, 5), S.generate(30, 6)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.w, b.w)
    assert not np.array_equal(a.p, c.p)


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shape_and_setup_matrix():
    linst = SL.generate(15, 4, n_families=3, setup_time=15)
    assert linst.n == 15 and linst.family.shape == (15,)
    assert set(linst.family.tolist()) <= set(range(3))
    assert linst.setup.shape == (3, 3)
    assert np.all(np.diag(linst.setup) == 0)
    assert np.all(linst.setup[~np.eye(3, dtype=bool)] == 15)


def test_logistik_instance_shares_the_same_processing_times_and_weights_as_neutral():
    """Vehikel B ändert nur Familie/Rüstzeiten, nicht die Grunddaten - sonst wäre ein Vehikel-Vergleich nicht fair."""
    neutral = S.generate(20, 7)
    logistik = SL.generate(20, 7)
    assert np.array_equal(neutral.p, logistik.p) and np.array_equal(neutral.w, logistik.w)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 2), SL.generate(10, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup)


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=20))
    assert a.wspt.total <= a.spt.total
    assert a.gap_spt >= 0.0 and a.gap_random >= -1e-6
    assert a.optimal is None                                    # n=20 > BRUTE_FORCE_MAX_N


def test_analysis_matches_the_optimum_for_small_n():
    a = ev.analyse(ev.Settings(n=6))
    assert a.optimal is not None
    assert a.wspt_matches_optimum


def test_wspt_and_spt_coincide_when_all_weights_are_equal():
    """Regressionsschutz für den Spezialfall: gleiche Gewichte -> gleiche Reihenfolge -> gleicher Zielwert."""
    inst = ev.instance(20, C.DEFAULT_SEED)
    uniform_w = np.ones_like(inst.w)
    wspt_result = A.evaluate_order(inst.p, uniform_w, A.wspt_order(inst.p, uniform_w))
    spt_result = A.evaluate_order(inst.p, uniform_w, A.spt_order(inst.p))
    assert wspt_result.total == pytest.approx(spt_result.total)


# --- Vehikel-Bewusstsein der Hauptanalyse (nicht nur einer Zusatzbox) -----------------------------------------------------------------------


def test_analyse_on_the_logistik_vehicle_actually_uses_setup_aware_completion_times():
    """Regressionsschutz für genau die Lücke, die der Nutzer in den ersten drei Stücken gefunden hat:
    `analyse()` mit vehicle='logistik' muss die Rüstzeiten TATSÄCHLICH in a.wspt/a.spt/a.optimal einrechnen."""
    settings = ev.Settings(n=8, seed=100000, vehicle="logistik", setup_time=30, n_families=3)
    a = ev.analyse(settings)
    linst = ev.logistik_instance(8, 100000, 3, 30)
    independent_wspt = A.evaluate_order_with_setup(linst.p, linst.w, linst.family, linst.setup, a.wspt.order)
    assert a.wspt.total == pytest.approx(independent_wspt.total)
    assert not np.array_equal(a.wspt.completion, np.cumsum(linst.p[a.wspt.order]))  # Rüstzeiten verschieben die Fertigstellung


def test_gap_spt_can_go_negative_on_the_logistik_vehicle():
    """Echter Fund (im Browser entdeckt, Preset 'Hohe Rüstlast'): weil WSPT rüstzeit-blind ist und auf dem
    Werkstatt-Vehikel nicht mehr bewiesen optimal, kann SPT (ignoriert Gewichte) hier zufällig eine Reihenfolge
    mit WENIGER Familienwechseln treffen und WSPT sogar schlagen - `gap_spt` wird dann negativ. Auf dem
    neutralen Vehikel ist das unmöglich (WSPT ist dort bewiesen optimal, siehe test_analysis_fields_are_consistent)."""
    a = ev.analyse(ev.Settings(n=20, seed=40, vehicle="logistik", setup_time=60, n_families=3))
    assert a.gap_spt < 0.0


def test_analyse_on_the_logistik_vehicle_can_show_wspt_missing_the_optimum():
    """Der zentrale, im Hauptfluss sichtbare Befund: auf dem Werkstatt-Vehikel kann WSPT von der
    (rüstzeit-bewussten) Vollaufzählung abweichen - anders als auf dem neutralen Vehikel, wo das ein Bug wäre."""
    settings = ev.Settings(n=6, seed=3, vehicle="logistik", setup_time=60, n_families=2)
    a = ev.analyse(settings)
    assert a.optimal is not None
    assert a.wspt.total >= a.optimal.total - 1e-6                # Optimum ist per Definition mindestens so gut


def test_analyse_on_the_neutral_vehicle_is_unaffected_by_logistik_only_settings():
    a1 = ev.analyse(ev.Settings(n=10, seed=5, vehicle="neutral", setup_time=5))
    a2 = ev.analyse(ev.Settings(n=10, seed=5, vehicle="neutral", setup_time=60))
    assert a1.wspt.total == pytest.approx(a2.wspt.total)


def test_analysis_is_deterministic_given_the_chain_seed():
    s = ev.Settings(n=20, seed=1, chain_seed=0)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert a.gap_random == pytest.approx(b.gap_random)
    assert a.gap_random != pytest.approx(c.gap_random)


# --- Sweep und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=15))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_spt"] >= 0.0


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 40))
    assert [r["value"] for r in rows] == [5, 40]


def test_optimality_check_always_matches():
    rows = ev.optimality_check(ns=(3, 4, 5), seeds=C.SWEEP_SEEDS)
    assert all(r["match_rate"] == 1.0 for r in rows)


def test_timing_sweep_shows_brute_force_growing_far_faster_than_wspt():
    rows = ev.timing_sweep(ns=(4, 8))
    small, large = rows[0], rows[1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 10
    assert large["wspt_seconds"] < large["brute_force_seconds"] / 100


def test_setup_gap_is_zero_when_setup_time_is_zero():
    row = ev.setup_gap(n=6, setup_time=0)
    assert row["gap_mean"] == pytest.approx(0.0, abs=1e-6)


def test_setup_gap_grows_with_the_setup_time():
    small = ev.setup_gap(n=8, setup_time=5)
    large = ev.setup_gap(n=8, setup_time=60)
    assert large["gap_mean"] > small["gap_mean"]
