"""Plotly-Abbildungen der WSPT-Demo: Auftragsübersicht (Bearbeitungszeit UND Gewicht), Gantt-artiges
Balkendiagramm der Reihenfolge, gewichtete Fertigstellungs-Kurve, Sweeps, Timing. Achsen sind gesperrt
(fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

WEIGHT_COLORS = {1: "#9ecae9", 2: "#4c78a8", 4: "#e45756"}
SPT_COLOR = "#e45756"
RANDOM_COLOR = "#7f7f7f"
OPTIMAL_COLOR = "#54a24b"
SETUP_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_jobs_chart(p, w):
    """Ein Balken je Auftrag (Höhe = Bearbeitungszeit), Farbe nach Gewicht - zeigt vor dem Einplanen, dass
    Bearbeitungszeit und Gewicht unabhängig voneinander streuen (ein kurzer Auftrag kann leicht oder wichtig
    sein, ein langer ebenso)."""
    fig = go.Figure()
    for wt in sorted(set(w.tolist())):
        idx = np.where(w == wt)[0]
        fig.add_trace(go.Bar(x=idx.tolist(), y=p[idx].tolist(), marker_color=WEIGHT_COLORS.get(wt, "#4c78a8"),
                              name=f"Gewicht {wt}", hovertemplate="Auftrag %{x}<br>Dauer %{y}<extra></extra>"))
    fig.update_xaxes(title_text="Auftrag (unsortiert)")
    fig.update_yaxes(title_text="Bearbeitungszeit")
    return _base(fig, 260)


def build_schedule(p, w, order, completion, upto=None):
    """Balken je Auftrag in der gegebenen Reihenfolge (Gantt-artig, eine Maschine), Farbe nach Gewicht wie im
    ersten Schritt. `completion` sind die TATSÄCHLICHEN Fertigstellungszeiten - auf dem Werkstatt/Logistik-
    Vehikel enthalten sie Lücken durch Rüstzeiten."""
    order = np.asarray(order)
    upto = len(order) if upto is None else upto
    starts = np.asarray(completion) - p[order]
    fig = go.Figure()
    for i in range(upto):
        j = order[i]
        fig.add_trace(go.Bar(x=[float(p[j])], y=["Maschine"], base=[float(starts[i])], orientation="h",
                              marker=dict(color=WEIGHT_COLORS.get(int(w[j]), "#4c78a8"), line=dict(width=1, color="white")),
                              name=f"Auftrag {j} (Gewicht {w[j]})", hovertemplate=f"Auftrag {j}<br>Dauer {p[j]}<br>Gewicht {w[j]}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(showticklabels=False)
    return _base(fig, 180)


def build_completion_curve(w, wspt_order, wspt_completion, spt_order, spt_completion):
    """Kumulierte gewichtete Fertigstellung ΣwⱼCⱼ über die Position - WSPT gegen SPT (ignoriert Gewichte)."""
    x = np.arange(1, len(wspt_completion) + 1)
    wspt_cum = np.cumsum(w[wspt_order] * wspt_completion)
    spt_cum = np.cumsum(w[spt_order] * spt_completion)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=wspt_cum, mode="lines+markers", line=dict(color="#4c78a8", width=2.5), name="WSPT (kumulierte gewichtete Fertigstellung)"))
    fig.add_trace(go.Scatter(x=x, y=spt_cum, mode="lines+markers", line=dict(color=SPT_COLOR, width=2, dash="dash"), name="SPT (ignoriert Gewichte)"))
    fig.update_xaxes(title_text="Aufträge eingeplant")
    fig.update_yaxes(title_text="Σ wⱼCⱼ bisher")
    return _base(fig, 320)


def build_sweep(rows, param_label, value_key="value", y_keys=(("gap_spt", "WSPT gegen SPT (ignoriert Gewichte)", SPT_COLOR), ("gap_random", "WSPT gegen Zufall", RANDOM_COLOR))):
    xs = [r[value_key] for r in rows]
    fig = go.Figure()
    for key, name, color in y_keys:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zu WSPT (%)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)


def build_timing(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["brute_force_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=SPT_COLOR, width=2.5), name="Brute-Force (O(n!))"))
    fig.add_trace(go.Scatter(x=xs, y=[r["wspt_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color="#4c78a8", width=2.5), name="WSPT (O(n log n))"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Rechenzeit (ms)", type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)


def build_setup_gap(rows):
    xs = [r["value"] for r in rows]
    upper = [r["gap_max"] for r in rows]
    lower = [r["gap_min"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(245,133,24,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap_mean"] for r in rows], mode="lines+markers", line=dict(color=SETUP_COLOR, width=2.5), name="WSPT (ignoriert Rüstzeiten) über dem echten Optimum"))
    fig.update_xaxes(title_text="Rüstzeit je Familienwechsel (Minuten)")
    fig.update_yaxes(title_text="Abstand zum Optimum mit Rüstzeiten (%)")
    return _base(fig, 340)
