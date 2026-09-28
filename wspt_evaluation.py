"""Auswertung der WSPT-Demo: WSPT gegen SPT (die falsche Regel hier - ignoriert Gewichte) und gegen zufällige
Reihenfolgen, gegen die Brute-Force-Vollaufzählung (nur kleine n), und das Vehikel-B-Experiment (bleibt WSPT
nahe am Optimum, sobald Rüstzeiten zwischen Auftragsfamilien dazukommen)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import wspt_algorithm as A
import wspt_constants as C
import wspt_scenario as S
import wspt_scenario_logistik as SL


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    vehicle: str = C.DEFAULT_VEHICLE
    setup_time: int = C.DEFAULT_SETUP_TIME
    n_families: int = C.DEFAULT_N_FAMILIES


@lru_cache(maxsize=512)
def instance(n, seed):
    return S.generate(n, seed)


@lru_cache(maxsize=512)
def logistik_instance(n, seed, n_families, setup_time):
    return SL.generate(n, seed, n_families=n_families, setup_time=setup_time)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    wspt: object
    spt: object               # die falsche Regel hier: ignoriert Gewichte
    random_mean: float
    random_runs: int
    optimal: object            # None, wenn n > BRUTE_FORCE_MAX_N

    @property
    def gap_spt(self):
        return 100.0 * (self.spt.total - self.wspt.total) / self.wspt.total

    @property
    def gap_random(self):
        return 100.0 * (self.random_mean - self.wspt.total) / self.wspt.total

    @property
    def wspt_matches_optimum(self):
        return self.optimal is not None and abs(self.wspt.total - self.optimal.total) < 1e-6


def analyse(settings, random_draws=20):
    """Wertet WSPT auf dem gewählten Vehikel aus - Neutral oder Werkstatt/Logistik (Rüstzeit beim
    Familienwechsel zählt mit). WSPT selbst bleibt in beiden Fällen dieselbe Regel (sortiert nur nach wⱼ/pⱼ,
    kennt keine Rüstzeiten) - nur die BEWERTUNG der Reihenfolgen (und damit auch der Vollaufzählung) wechselt
    mit dem Vehikel, damit die Haupt-Kennzahlen ehrlich widerspiegeln, was auf dem gewählten Vehikel tatsächlich
    passiert."""
    if settings.vehicle == "logistik":
        inst = logistik_instance(settings.n, settings.seed, settings.n_families, settings.setup_time)
        p, w, family, setup = inst.p, inst.w, inst.family, inst.setup

        def ev(order):
            return A.evaluate_order_with_setup(p, w, family, setup, order)

        optimal = A.brute_force_optimal_with_setup(p, w, family, setup) if settings.n <= C.BRUTE_FORCE_MAX_N else None
    else:
        inst = instance(settings.n, settings.seed)
        p, w = inst.p, inst.w

        def ev(order):
            return A.evaluate_order(p, w, order)

        optimal = A.brute_force_optimal(p, w) if settings.n <= C.BRUTE_FORCE_MAX_N else None

    wspt = ev(A.wspt_order(p, w))
    spt = ev(A.spt_order(p))
    rng = np.random.default_rng(settings.chain_seed)
    random_totals = [ev(A.random_order(settings.n, rng)).total for _ in range(random_draws)]
    return Analysis(settings, inst, wspt, spt, float(np.mean(random_totals)), random_draws, optimal)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap_spt": a.gap_spt, "gap_random": a.gap_random})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 5, 10, 20, 40, 60)}
SWEEP_LABELS = {"n": "Aufträge"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def optimality_check(ns=C.BRUTE_FORCE_SWEEP_N, seeds=C.SWEEP_SEEDS):
    """WSPT gegen Brute-Force-Vollaufzählung über mehrere n und Instanzen - Anteil exakter Treffer (muss 100 %
    sein, sonst ist der Beweis oder die Implementierung falsch)."""
    rows = []
    for n in ns:
        matches = 0
        for seed in seeds:
            inst = instance(n, seed)
            wspt_total = A.wspt(inst.p, inst.w).total
            opt_total = A.brute_force_optimal(inst.p, inst.w).total
            if abs(wspt_total - opt_total) < 1e-6:
                matches += 1
        rows.append({"value": n, "match_rate": matches / len(seeds)})
    return rows


def timing_sweep(ns=C.BRUTE_FORCE_SWEEP_N, seed=C.DEFAULT_SEED):
    """Gemessene Rechenzeit: Brute-Force-Vollaufzählung (O(n!)) gegen WSPT (O(n log n))."""
    rows = []
    for n in ns:
        inst = instance(n, seed)
        t0 = time.perf_counter()
        A.brute_force_optimal(inst.p, inst.w)
        t_bf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(100):
            A.wspt(inst.p, inst.w)
        t_wspt = (time.perf_counter() - t0) / 100
        rows.append({"value": n, "brute_force_seconds": t_bf, "wspt_seconds": t_wspt})
    return rows


def setup_gap(n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME):
    """Vehikel-B-Härtetest: WSPT (sortiert nur nach wⱼ/pⱼ, ignoriert Rüstzeiten) gegen die echte Optimallösung
    MIT Rüstzeiten (Brute-Force, deshalb kleines n). Der Abstand ist eine echte Messfrage, kein behaupteter
    Befund."""
    gaps = []
    for seed in seeds:
        linst = SL.generate(n, seed, n_families=n_families, setup_time=setup_time)
        wspt_ord = A.wspt_order(linst.p, linst.w)
        wspt_total = A.evaluate_order_with_setup(linst.p, linst.w, linst.family, linst.setup, wspt_ord).total
        opt_total = A.brute_force_optimal_with_setup(linst.p, linst.w, linst.family, linst.setup).total
        gaps.append(100.0 * (wspt_total - opt_total) / opt_total)
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def setup_gap_sweep(setup_times=(0, 5, 15, 30, 60), n=8, seeds=C.SWEEP_SEEDS, n_families=C.DEFAULT_N_FAMILIES):
    return [{"value": s, **setup_gap(n=n, seeds=seeds, n_families=n_families, setup_time=s)} for s in setup_times]
