"""Vehikel A "Neutral" der WSPT-Demo: n Aufträge mit Bearbeitungszeit pⱼ UND Gewicht wⱼ auf EINER Maschine.
Anders als in Stück 1 (SPT) ist das Gewicht hier keine Vorbereitung für später, sondern die Hauptgröße: WSPT
verallgemeinert SPT genau um dieses Gewicht (wⱼ ∈ {1,2,4}, schief verteilt - wenige sehr wichtige Aufträge,
viele normale). Keine Fälligkeiten in diesem Stück (1||ΣwⱼCⱼ kennt keine Fristen)."""

from dataclasses import dataclass

import numpy as np

import wspt_constants as C


@dataclass(frozen=True)
class Instance:
    n: int
    p: np.ndarray        # Bearbeitungszeiten
    w: np.ndarray        # Gewichte
    seed: int


def generate(n, seed, p_min=C.P_MIN, p_max=C.P_MAX, weight_values=C.WEIGHT_VALUES, weight_probs=C.WEIGHT_PROBS):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    w = rng.choice(weight_values, size=n, p=weight_probs).astype(np.int64)
    return Instance(n, p, w, seed)
