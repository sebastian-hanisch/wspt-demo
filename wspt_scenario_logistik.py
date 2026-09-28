"""Vehikel B "Werkstatt/Logistik" der WSPT-Demo: dieselben Aufträge wie Vehikel A (Bearbeitungszeit, Gewicht),
zusätzlich eine Familie je Auftrag (Werkzeug/Material) und eine sequenzabhängige Rüstzeit beim Familienwechsel
(dieselbe Idee wie in `spt-scheduling-demo`/`moore-hodgson-demo`, dort bereits Vepsalainen & Morton 1987
zitiert). Kein neues Zufallsmodell für Bearbeitungszeit/Gewicht - dieselbe Erzeugung wie `wspt_scenario.py`,
nur um Familie und Rüstmatrix ergänzt, damit ein Vergleich zwischen den Vehikeln nicht an unterschiedlich
verteilten Grunddaten hängt."""

from dataclasses import dataclass

import numpy as np

import wspt_constants as C


@dataclass(frozen=True)
class LogistikInstance:
    n: int
    p: np.ndarray
    w: np.ndarray
    family: np.ndarray          # Familien-Index je Auftrag
    setup: np.ndarray           # (F, F) Rüstzeit-Matrix, 0 auf der Diagonale
    seed: int


def generate(n, seed, n_families=C.DEFAULT_N_FAMILIES, setup_time=C.DEFAULT_SETUP_TIME,
             p_min=C.P_MIN, p_max=C.P_MAX, weight_values=C.WEIGHT_VALUES, weight_probs=C.WEIGHT_PROBS):
    rng = np.random.default_rng(seed)
    p = rng.integers(p_min, p_max + 1, size=n).astype(np.int64)
    w = rng.choice(weight_values, size=n, p=weight_probs).astype(np.int64)
    family = rng.integers(0, n_families, size=n).astype(np.int64)
    setup = np.full((n_families, n_families), setup_time, dtype=np.int64)
    np.fill_diagonal(setup, 0)
    return LogistikInstance(n, p, w, family, setup, seed)
