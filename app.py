"""WSPT / Smith's Rule - wenn nicht jeder Auftrag gleich wichtig ist - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der neuen Konzepte-Linie "Klassische Scheduling-Theorie": n Aufträge auf einer Maschine, jeder mit
einem Gewicht wⱼ, Ziel ist die GEWICHTETE Summe der Fertigstellungszeiten zu minimieren (1||Σwⱼ Cⱼ in der
α|β|γ-Notation). WSPT (absteigend nach wⱼ/pⱼ sortieren, Smith 1956) ist dafür BEWEISBAR optimal - dasselbe
Vertauschungsargument wie SPT (Stück 1), nur mit Gewichten. SPT selbst (ignoriert Gewichte) ist hier die falsche
Regel. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import wspt_constants as C
from wspt_evaluation import Settings, SWEEP_LABELS, analyse, instance, optimality_check, run_config, setup_gap, setup_gap_sweep, sweep, timing_sweep
from wspt_presets import KEPT, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_chain_seed, randomize_seed, seed_widget, sync_query_params
from wspt_visualization import build_completion_curve, build_jobs_chart, build_schedule, build_setup_gap, build_sweep, build_timing

st.set_page_config(page_title="WSPT – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _optimality():
    return optimality_check()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_sweep()


@st.cache_data(show_spinner=False)
def _setup_gap_sweep(n, n_families):
    return setup_gap_sweep(n=n, n_families=n_families)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _fmt_pct(x):
    """Vorzeichen-korrekt: `+10.4 %` (Vergleichsregel schlechter als WSPT) oder `-11.4 %` (auf dem
    Werkstatt/Logistik-Vehikel möglich - WSPT ist dort rüstzeit-blind und nicht mehr bewiesen optimal, also
    KANN eine Vergleichsregel zufällig besser abschneiden)."""
    return f"{x:+.1f} %"


st.title("⚖️ WSPT – wenn nicht jeder Auftrag gleich wichtig ist")
st.markdown(
    r"""
**n Aufträge auf einer Maschine, jeder mit einem Gewicht $w_j$, gesucht ist die Reihenfolge, die die GEWICHTETE
Summe der Fertigstellungszeiten minimiert** ($1||\sum w_j C_j$). SPT (Stück 1 dieser Linie) ignoriert Gewichte
komplett - hier ist das die falsche Regel. Die richtige Antwort ist **WSPT / Smith's Rule** (Smith 1956): absteigend
nach dem Verhältnis $w_j / p_j$ sortieren - ein wichtiger, kurzer Auftrag zuerst, ein unwichtiger, langer zuletzt.
Auch das ist **beweisbar optimal**, mit demselben Vertauschungsargument wie SPT, nur um das Gewicht erweitert.
SPT ist der Spezialfall $w_j = 1$ für alle $j$ - WSPT verallgemeinert die Wurzel dieser Linie.
"""
)
st.caption(
    "Viertes Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - verallgemeinert SPT (Stück 1) um Gewichte. "
    "Zwei Vehikel: **Neutral** (Aufträge mit Bearbeitungszeit und Gewicht) und **Werkstatt/Logistik** (dieselben "
    "Aufträge, aber in Familien mit Rüstzeit beim Wechsel) - der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert WSPT", expanded=True):
    st.markdown(
        r"""
1. **Sortieren.** Alle Aufträge absteigend nach dem Verhältnis $w_j / p_j$ ordnen - fertig. $O(n \log n)$, kein Suchverfahren nötig.
2. **Warum das optimal ist.** Vertauschungsargument: zwei benachbarte Aufträge mit $w_i/p_i < w_j/p_j$ tauschen verringert $\sum w_j C_j$ - jede nicht so sortierte Reihenfolge lässt sich also verbessern.
3. **Der Spezialfall.** Sind alle Gewichte gleich ($w_j = 1$), fällt WSPT exakt auf SPT zurück - die Wurzel dieser Linie ist ein Sonderfall.
4. **Die Grenze der Annahme.** WSPT setzt wie SPT voraus, dass die Bearbeitungszeit nicht von der Reihenfolge abhängt - keine Rüstzeiten. Das Vehikel „Werkstatt/Logistik“ prüft, was passiert, wenn das nicht mehr stimmt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.BRUTE_FORCE_MAX_N} läuft die Vollaufzählung aller n! Reihenfolgen live mit.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: nur Bearbeitungszeiten und Gewichte. Werkstatt/Logistik: dieselben Aufträge, zusätzlich in Familien mit Rüstzeit beim Wechsel.")
    if vehicle == "logistik":
        seed_widget("setup_time_slider")
        setup_time = st.slider("Rüstzeit je Familienwechsel (Minuten)", *bounds("setup_time_slider"), key="setup_time_slider",
                                help="0 Minuten kollabiert exakt zum neutralen Vehikel (siehe Test/Messreihe).")
        st.session_state[KEPT["setup_time_slider"]] = setup_time
        seed_widget("n_families_slider")
        n_families = st.slider("Auftragsfamilien", *bounds("n_families_slider"), key="n_families_slider",
                                help="Weniger Familien bei gleicher Auftragszahl bedeutet mehr Wechsel und damit mehr Rüstzeit insgesamt.")
        st.session_state[KEPT["n_families_slider"]] = n_families
    else:
        setup_time = int(st.session_state.get(KEPT["setup_time_slider"], C.DEFAULT_SETUP_TIME))
        n_families = int(st.session_state.get(KEPT["n_families_slider"], C.DEFAULT_N_FAMILIES))
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Bearbeitungszeiten und Gewichte.")
    chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                  help="Steuert nur die zufällige Vergleichs-Reihenfolge - WSPT selbst ist deterministisch (kein Zufall im Kern).")
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für die Zufalls-Vergleichsreihenfolge.")

sync_query_params({"n_slider": int(n_jobs), "seed_input": int(seed), "chain_seed_input": int(chain_seed), "vehicle_radio": vehicle,
                    "setup_time_slider": int(setup_time), "n_families_slider": int(n_families)})

settings = Settings(int(n_jobs), int(seed), int(chain_seed), vehicle=vehicle, setup_time=int(setup_time), n_families=int(n_families))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
p, w = inst.p, inst.w
data_key = settings

# --- WSPT in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 WSPT in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Einplanen", 3: "3 · Ergebnis"}
if "wspt_step" not in st.session_state or st.session_state.get("wspt_step_owner") != data_key:
    st.session_state["wspt_step"] = 1
    st.session_state["wspt_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="wspt_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        upto = st.slider("Eingeplante Aufträge", 1, int(n_jobs), value=int(n_jobs), key="wspt_upto")
else:
    upto = int(n_jobs)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (Bearbeitungszeit in Minuten, Farbe nach Gewicht)")
        st.plotly_chart(build_jobs_chart(p, w), width="stretch", key="s1_jobs")
    elif step == 2:
        st.markdown(f"**WSPT-Reihenfolge nach {upto} von {n_jobs} Aufträgen**")
        st.plotly_chart(build_schedule(p, w, a.wspt.order, a.wspt.completion, upto=upto), width="stretch", key=f"s2_sched_{upto}")
        st.caption(f"Σ wⱼCⱼ bisher: {_fmt_int((w[a.wspt.order][:upto] * a.wspt.completion[:upto]).sum())}")
    else:
        st.markdown("**WSPT gegen SPT (ignoriert Gewichte): Σ wⱼCⱼ über die Zeit**")
        st.plotly_chart(build_completion_curve(w, a.wspt.order, a.wspt.completion, a.spt.order, a.spt.completion), width="stretch", key="s3_curve")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(p.min())} und {int(p.max())} Minuten, Gewichte {sorted(set(w.tolist()))} (Seed {seed}).")
elif step == 2:
    gap_note = " Lücken zwischen Balken sind Rüstzeit bei einem Familienwechsel." if vehicle == "logistik" else ""
    st.caption(f"Jeder Balken ist ein Auftrag, Farbe = Gewicht; ein wichtiger, kurzer Auftrag steht weiter vorne als ein unwichtiger, langer.{gap_note}")
else:
    st.caption(f"WSPT: Σ wⱼCⱼ {_fmt_int(a.wspt.total)}. SPT (ignoriert Gewichte): {_fmt_int(a.spt.total)} (Differenz {_fmt_pct(a.gap_spt)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was das Gewichten bringt")
vehicle_note = " Auf dem Werkstatt/Logistik-Vehikel zählt die Rüstzeit beim Familienwechsel mit - WSPT kennt sie nicht, alle Zahlen hier berücksichtigen sie trotzdem." if vehicle == "logistik" else ""
st.caption(f"**Abstand:** Σ wⱼCⱼ einer Reihenfolge gegenüber WSPT in Prozent - positiv heißt die Vergleichsregel braucht mehr, negativ heißt sie schneidet sogar besser ab (nur auf dem Werkstatt/Logistik-Vehikel möglich, siehe unten). WSPT selbst ist deterministisch (kein Zufall im Kern) - nur die Zufalls-Vergleichsreihenfolge streut.{vehicle_note}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("WSPT (Σ wⱼCⱼ)", _fmt_int(a.wspt.total), help="Die Zielgröße: gewichtete Summe aller Fertigstellungszeiten in WSPT-Reihenfolge, auf dem gewählten Vehikel.")
m2.metric("SPT (ignoriert Gewichte)", _fmt_pct(a.gap_spt), delta_color="off", help="Die Regel der Wurzel dieser Linie - hier falsch, weil sie Gewichte ignoriert. Negativ auf dem Werkstatt-Vehikel möglich: WSPT ist rüstzeit-blind und dort nicht mehr bewiesen optimal.")
m3.metric(f"Zufällige Reihenfolge (Mittel über {a.random_runs})", _fmt_pct(a.gap_random), delta_color="off", help="Mittel über mehrere zufällige Reihenfolgen derselben Instanz.")
if a.optimal is not None:
    m4.metric("Vollaufzählung (Gegenprobe)", "trifft WSPT exakt" if a.wspt_matches_optimum else "WEICHT AB", delta_color="off",
              help=f"Alle {n_jobs}! Reihenfolgen durchprobiert (auf dem gewählten Vehikel) - unabhängige Bestätigung bzw. Gegenprobe.")
else:
    m4.metric("Vollaufzählung", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off", help="Bei dieser Größe wäre die Vollaufzählung zu langsam - siehe das Timing-Experiment unten.")

if a.optimal is not None and not a.wspt_matches_optimum:
    if vehicle == "neutral":
        st.error("⚠️ WSPT weicht von der Vollaufzählung ab - das wäre ein Fehler im Beweis oder in der Implementierung, bitte melden.")
    else:
        gap = 100.0 * (a.wspt.total - a.optimal.total) / a.optimal.total
        st.warning(f"⚠️ WSPT ist hier NICHT mehr optimal: {gap:.1f} % über dem echten Optimum MIT Rüstzeiten. Der Beweis oben setzt keine Rüstzeiten voraus - siehe 🚧 unten.")
elif vehicle == "logistik" and (a.gap_spt < 0 or a.gap_random < 0):
    worse_than = "SPT (das die Gewichte ignoriert)" if a.gap_spt < 0 else "eine zufällige Reihenfolge"
    worst_gap = min(a.gap_spt, a.gap_random)
    st.warning(f"⚠️ Auf diesem Werkstatt-Vehikel schneidet WSPT hier sogar SCHLECHTER ab als {worse_than}: {abs(worst_gap):.1f} % mehr. Kein Fehler - WSPT ist rüstzeit-blind und für diese Instanz zufällig ungünstig einsortiert; genau die Grenze aus 🚧 unten.")
else:
    tail = " (auch mit Rüstzeiten - bei dieser Instanz trifft WSPT trotzdem das Optimum, das ist nicht garantiert)" if vehicle == "logistik" and a.optimal is not None else ""
    proof = "bei dieser Zielfunktion beweisbar die beste überhaupt" if vehicle == "neutral" else "auf diesem Vehikel nicht mehr bewiesen optimal, aber hier weiterhin besser als beide Vergleichsregeln"
    st.success(f"✅ WSPT ist {a.gap_spt:.1f} % besser als SPT (das ignoriert die Gewichte) und {a.gap_random:.1f} % besser als eine zufällige Reihenfolge - {proof}{tail}.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanzgröße ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004) mit je drei Zufalls-Ketten für die Vergleichsreihenfolge.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stimmt der Beweis wirklich? Vollaufzählung gegen WSPT")
if st.button("Vollaufzählung über n = 2 bis 9 berechnen (dauert etwa 5 Sekunden)", key="opt_start"):
    st.session_state["opt_on"] = True
if st.session_state.get("opt_on"):
    rows_opt = _optimality()
    st.table({"Aufträge": [r["value"] for r in rows_opt], "Trefferquote": [f"{r['match_rate']:.0%}" for r in rows_opt]})
    st.caption("Für jede Instanzgröße 5 feste Instanzen: WSPT gegen die Vollaufzählung aller n! Reihenfolgen. Jede Abweichung von 100 % wäre ein Fehler im Beweis oder in der Implementierung.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine Vollaufzählung wirklich?")
if st.button("Rechenzeit für n = 2 bis 9 messen (dauert etwa 1 Sekunde)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing()
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht die Vollaufzählung bereits {last['brute_force_seconds']*1000:.0f} ms, WSPT {last['wspt_seconds']*1000:.3f} ms - {last['brute_force_seconds']/max(last['wspt_seconds'],1e-9):.0f}-mal langsamer.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt WSPT gut, wenn Rüstzeiten dazukommen?")
if st.button("Rüstzeit von 0 bis 60 Minuten durchfahren (dauert wenige Sekunden)", key="setup_start"):
    st.session_state["setup_on"] = True
if st.session_state.get("setup_on"):
    rows_s = _setup_gap_sweep(min(int(n_jobs), C.BRUTE_FORCE_MAX_N), int(n_families))
    st.plotly_chart(build_setup_gap(rows_s), width="stretch", key="setup_chart")
    st.caption("WSPT sortiert weiterhin nur nach wⱼ/pⱼ und ignoriert die Rüstzeit beim Familienwechsel; verglichen mit der echten Optimallösung MIT Rüstzeiten (Vollaufzählung, deshalb kleine Instanz). Bei Rüstzeit 0 fallen beide exakt zusammen.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Bearbeitungszeit hängt nicht von der Reihenfolge ab** | Sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel „Werkstatt/Logistik“), ist WSPT nicht mehr beweisbar optimal - der Abstand zum echten Optimum wächst mit der Rüstzeit (siehe Experiment oben). | Kein direkter Nachfolger in dieser Linie |
| **Es gibt keine Fristen** | Wenn zusätzlich zum Gewicht eine Frist dazukommt, wird das Problem NP-schwer - kein einfaches Sortierkriterium reicht mehr. | **Gewichtete Verspätung** (Folgestück) |
| **Es gibt nur eine Maschine** | Mit mehreren Maschinen wird aus einer Sortierfrage eine Zuordnungs- UND Reihenfolgefrage. | **Johnson-Regel (2 Maschinen), LPT (parallele Maschinen), Job Shop** (Folgestücke) |
"""
)
st.caption(
    "Viertes Stück der Linie „Klassische Scheduling-Theorie“: verallgemeinert die Wurzel (SPT, Stück 1) um "
    "Gewichte. Verwandt: `warehouse-transfer-demo` (ATCS-Regel praktisch, gewichtet UND mit Fristen)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** ($1||\sum w_j C_j$): $n$ Aufträge mit Bearbeitungszeit $p_j$ und Gewicht $w_j$ auf einer Maschine;
eine Reihenfolge $\pi$ legt die Fertigstellungszeit $C_j$ jedes Auftrags fest. Gesucht: $\pi$, das
$\sum_j w_j C_j$ minimiert.

**Satz (Smith 1956).** WSPT (absteigend nach $w_j / p_j$ sortieren) minimiert $\sum_j w_j C_j$.

**Beweis (Vertauschungsargument).** Sei $\pi$ eine beliebige Reihenfolge, in der zwei benachbarte Aufträge $i$
vor $j$ mit $w_i/p_i < w_j/p_j$ stehen (existiert, solange $\pi$ nicht WSPT-sortiert ist). Vertauscht man $i$ und
$j$, ändern sich nur ihre beiden Fertigstellungszeiten. Vorher: $C_i = t + p_i$, $C_j = t + p_i + p_j$ (mit $t$ =
Summe davor), Beitrag $w_i C_i + w_j C_j$. Nachher (j vor i): $C_j' = t + p_j$, $C_i' = t + p_j + p_i$, Beitrag
$w_j C_j' + w_i C_i'$. Die Differenz ist $w_i p_j - w_j p_i$ - negativ (also eine Verbesserung) genau dann, wenn
$w_i/p_i < w_j/p_j$. Jede nicht WSPT-sortierte Reihenfolge lässt sich also durch eine solche Vertauschung
verbessern; WSPT ist der einzige Zustand, an dem keine benachbarte Vertauschung mehr hilft.

**Spezialfall.** Für $w_j = 1$ (alle Aufträge gleich wichtig) ist $w_j/p_j = 1/p_j$ - absteigend danach sortieren
ist dasselbe wie aufsteigend nach $p_j$ sortieren: WSPT fällt exakt auf SPT (Stück 1) zurück.

**Kennzahl.** Abstand zu WSPT $= 100 \cdot (\Sigma w_jC_j - \Sigma w_jC_j^{\text{WSPT}}) / \Sigma w_jC_j^{\text{WSPT}}$.
Für $n \le 9$ zusätzlich die Vollaufzählung aller $n!$ Reihenfolgen als unabhängige Gegenprobe.

**Grenzen.** Rüstzeiten zwischen Auftragsfamilien verletzen dieselbe Voraussetzung wie bei SPT - WSPT bleibt
dann nur noch eine gute Heuristik, kein Beweis mehr (Vehikel B).

Implementiert in `wspt_algorithm.py` (WSPT, Brute-Force-Gegenprobe, Rüstzeit-Variante), `wspt_scenario.py`/
`wspt_scenario_logistik.py` (die zwei Vehikel), `wspt_evaluation.py` (Kennzahlen, Sweep, Optimalitäts- und
Timing-Messreihe, Vehikel-B-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
