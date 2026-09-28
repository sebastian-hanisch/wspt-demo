"""WSPT (Weighted Shortest Processing Time first, Smith's Rule) für 1||ΣwⱼCⱼ: n Aufträge auf einer Maschine,
jeder mit Gewicht wⱼ, Ziel ist die GEWICHTETE Summe der Fertigstellungszeiten zu minimieren. WSPT (absteigend
nach wⱼ/pⱼ sortieren) ist dafür beweisbar optimal - dasselbe Vertauschungsargument wie SPT (siehe README/App),
nur mit Gewichten: tauscht man zwei benachbarte Aufträge i, j mit wᵢ/pᵢ < wⱼ/pⱼ, sinkt ΣwⱼCⱼ. SPT (Bearbeitungszeit
ignoriert die Gewichte komplett) ist hier die falsche Regel - genau der Kontrast, den die Messreihe zeigt.

Hier zusätzlich: Brute-Force-Vollaufzählung als unabhängige Gegenprobe (nur für kleine n praktikabel), sowie die
Rüstzeit-Variante für Vehikel B (Werkstatt/Logistik)."""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    order: np.ndarray
    completion: np.ndarray
    total: float               # ΣwⱼCⱼ (gewichtet)


def completion_times(p, order):
    return np.cumsum(np.asarray(p)[order].astype(np.float64))


def weighted_total(w, order, completion):
    return float(np.asarray(w)[order].astype(np.float64) @ completion)


def evaluate_order(p, w, order):
    order = np.asarray(order)
    completion = completion_times(p, order)
    return Result(order, completion, weighted_total(w, order, completion))


def wspt_order(p, w):
    """WSPT/Smith's Rule: absteigend nach wⱼ/pⱼ sortiert - stabile Sortierung, damit Gleichstände reproduzierbar
    sind (bei gleichem Verhältnis ist die Reihenfolge untereinander für ΣwⱼCⱼ ohnehin egal)."""
    ratio = np.asarray(w, dtype=np.float64) / np.asarray(p, dtype=np.float64)
    return np.argsort(-ratio, kind="stable")


def wspt(p, w):
    return evaluate_order(p, w, wspt_order(p, w))


def spt_order(p):
    """Die falsche Regel hier: ignoriert Gewichte komplett, sortiert nur nach Bearbeitungszeit (was für
    1||ΣCⱼ optimal war, aber hier nicht mehr ist)."""
    return np.argsort(p, kind="stable")


def random_order(n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return order


def brute_force_optimal(p, w):
    """Volle Aufzählung aller n! Reihenfolgen - unabhängige Gegenprobe, nur für kleine n (siehe
    wspt_constants.BRUTE_FORCE_MAX_N)."""
    n = len(p)
    best_order, best_total = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        completion = completion_times(p, order)
        total = weighted_total(w, order, completion)
        if total < best_total:
            best_total, best_order = total, order
    return evaluate_order(p, w, best_order)


# --- Mit Rüstzeiten (Vehikel B: Werkstatt/Logistik) ---------------------------------------------------------


def completion_times_with_setup(p, family, setup, order):
    t = 0.0
    out = np.empty(len(order), dtype=np.float64)
    prev_family = None
    for idx, j in enumerate(order):
        if prev_family is not None:
            t += float(setup[prev_family, family[j]])
        t += float(p[j])
        out[idx] = t
        prev_family = family[j]
    return out


def evaluate_order_with_setup(p, w, family, setup, order):
    order = np.asarray(order)
    completion = completion_times_with_setup(p, family, setup, order)
    return Result(order, completion, weighted_total(w, order, completion))


def brute_force_optimal_with_setup(p, w, family, setup):
    n = len(p)
    best_order, best_total = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        completion = completion_times_with_setup(p, family, setup, order)
        total = weighted_total(w, order, completion)
        if total < best_total:
            best_total, best_order = total, order
    return evaluate_order_with_setup(p, w, family, setup, best_order)
