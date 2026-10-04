# WSPT / Smith's Rule – wenn nicht jeder Auftrag gleich wichtig ist – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-wspt-demo.streamlit.app/)**

Viertes Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning": $n$ Aufträge mit Bearbeitungszeit $p_j$ und Gewicht $w_j$ auf
**einer** Maschine, Ziel ist die GEWICHTETE Summe der Fertigstellungszeiten zu minimieren ($1||\sum w_j C_j$ in
der α|β|γ-Notation).

**Einordnung in die Linie:** WSPT (Weighted Shortest Processing Time first, Smith 1956) verallgemeinert SPT
(Stück 1) um Gewichte – absteigend nach dem Verhältnis $w_j / p_j$ sortieren, dasselbe Vertauschungsargument wie
SPT, nur mit Gewicht. Für $w_j = 1$ (alle Aufträge gleich wichtig) fällt WSPT exakt auf SPT zurück – die Wurzel
dieser Linie ist ein Sonderfall von WSPT, nicht umgekehrt. SPT selbst (ignoriert Gewichte) ist hier die falsche
Regel und dient als Kontrast.
```
SPT (Wurzel: 1||ΣCⱼ, beweisbar optimal)                                          [Stück 1]
 ├─ EDD (1||Lmax, dasselbe Beweismuster, andere Zielfunktion)                     [Stück 2]
 ├─ Moore-Hodgson (1||ΣUⱼ, gierig + schlechtesten Verspäteten entfernen)          [Stück 3]
 ├─ WSPT / Smith's Rule (1||ΣwⱼCⱼ, verallgemeinert SPT mit Gewichten)             [dieses Stück]
 ├─ Johnson-Regel (F2||Cmax, zweite Maschine)                                     [Folgestück]
 ├─ LPT (Pm||Cmax, parallele Maschinen)                                           [Folgestück]
 └─ Job Shop (Konvergenzpunkt: Reihenfolge UND Maschinenwahl)                     [Folgestück]
```

Ergebnis in Kürze: **WSPT trifft auf jeder getesteten Instanz (n = 2 bis 9) exakt das Minimum der Vollaufzählung
– der Beweis stimmt.** Bei 20 Aufträgen mit Gewichten 1/2/4 liegt WSPT im Mittel **18.8 %** unter SPT (das die
Gewichte ignoriert) und **86.2 %** unter einer zufälligen Reihenfolge. Die Vollaufzählung wird schnell
unpraktikabel: bei 9 Aufträgen braucht sie über eine Sekunde, WSPT bleibt im Mikrosekundenbereich.
**Der ehrliche Bruch (zweifach):** sobald Rüstzeiten zwischen Auftragsfamilien dazukommen (Vehikel
Werkstatt/Logistik), setzt der Beweis nicht mehr – WSPT liegt im Mittel messbar über dem echten Optimum, und der
Abstand wächst mit der Rüstzeit (0 % bei 0 Minuten, **30.3 %** bei 60 Minuten). Auf diesem Vehikel ist WSPT sogar
nicht mehr GARANTIERT besser als SPT: bei einer der getesteten Instanzen (Seed 40, 20 Aufträge, 60 Minuten
Rüstzeit) schneidet WSPT **11.4 % schlechter** ab als die simple, gewichts-blinde Regel – ein Fund, den die App
ehrlich zeigt statt zu verstecken.

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds) |
|---|---|
| Standardfall (20 Aufträge) | ✅ WSPT liegt **18.8 %** unter SPT und **86.2 %** unter einer zufälligen Reihenfolge |
| **Beweis gegen Vollaufzählung** | ✅ **100 %** Trefferquote bei n = 2 bis 9 – kein einziger Fall, in dem WSPT nicht das Minimum trifft |
| **Rechenzeit** | ➖ Vollaufzählung bei n = 9 bereits über 1000 ms, WSPT im Mikrosekundenbereich |
| **Vehikel Werkstatt/Logistik** | ❌ Rüstzeit 0/5/15/30/60 Minuten: WSPT liegt **0/0.7/4.6/13.9/30.3 %** über dem echten Optimum – im Einzelfall kann WSPT sogar schlechter als SPT abschneiden |
| **Spezialfall** | ✅ Gleiche Gewichte ($w_j=1$ für alle $j$): WSPT-Reihenfolge ist bitidentisch mit SPT (per Test belegt) |

## Was die Demo zeigt

1. **WSPT in Aktion** (Schritt-Slider): **Aufträge** (unsortierte Bearbeitungszeiten, Farbe nach Gewicht) →
   **Einplanen** (Regler "eingeplante Aufträge", Gantt-artiges Balkendiagramm baut sich auf) → **Ergebnis**
   (Σ wⱼCⱼ-Kurve WSPT gegen SPT).
2. **Was das Gewichten bringt:** WSPT, SPT (falsche Regel hier), zufällige Reihenfolge, Vollaufzählungs-Gegenprobe
   (n ≤ 9); auf dem Werkstatt/Logistik-Vehikel zusätzlich ehrlicher Hinweis, wenn WSPT von SPT geschlagen wird.
3. **📐 Sweep** über die Anzahl der Aufträge.
4. **🔬 Experimente auf Abruf:** Vollaufzählung gegen WSPT über n = 2 bis 9 (Beweis-Check); Rechenzeit $n!$ gegen
   $n \log n$; Rüstzeit-Härtetest auf dem Werkstatt/Logistik-Vehikel.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (keine Rüstzeiten, keine Fristen, nur eine
   Maschine) mit Verweisen auf die Folgestücke.

Regler: Aufträge (2–60), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich Rüstzeit und Anzahl
Familien), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Vergleichsreihenfolge – WSPT
selbst ist deterministisch).

## Die zwei Vehikel (gelten für die ganze Linie)

- **Neutral** (`wspt_scenario.py`): $n$ Aufträge mit Bearbeitungszeit $p_j \sim U(1, 100)$ und Gewicht
  $w_j \in \{1,2,4\}$ (schief verteilt: 50/30/20 %) – anders als in Stück 1 ist das Gewicht hier keine
  Vorbereitung für später, sondern die Hauptgröße. Keine Fälligkeiten (1||Σwⱼ Cⱼ kennt keine Fristen).
- **Werkstatt/Logistik** (`wspt_scenario_logistik.py`): dieselben Bearbeitungszeiten/Gewichte, aber jeder
  Auftrag gehört zu einer Familie; ein Familienwechsel kostet eine feste Rüstzeit (dieselbe Idee wie in
  `spt-scheduling-demo`/`moore-hodgson-demo`, dort bereits Vepsalainen & Morton 1987 zitiert). Rüstzeit 0
  kollabiert exakt zum neutralen Vehikel (per Test belegt).
- Wie die ganze Linie: **zwei echte, unabhängige Generatoren** statt eines einzigen mit Sonderfall (bewusste
  Abweichung vom sonst üblichen Website-Muster, siehe `spt-scheduling-demo`).

## Modell und Verfahren

- **Instanz** (`wspt_scenario.py`, `wspt_scenario_logistik.py`): Bearbeitungszeiten, Gewichte, Familien und
  Rüstzeit-Matrix, Seed-erzeugt wie jede andere Konzepte-Linie dieser Website.
- **WSPT** (`wspt_algorithm.py`): absteigend nach $w_j/p_j$ sortieren, $O(n \log n)$. Dazu die
  Brute-Force-Vollaufzählung (nur für kleine $n$) als unabhängige Gegenprobe, und die Rüstzeit-Variante für das
  Werkstatt/Logistik-Vehikel.
- **Auswertung** (`wspt_evaluation.py`): Kennzahlen, Sweep, Optimalitäts- und Timing-Messreihe,
  Rüstzeit-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "WSPT bleibt zumindest besser als SPT, auch wenn der Beweis nicht mehr gilt"** – **widerlegt
  im Einzelfall**: bei hoher Rüstlast (Seed 40, n=20, Rüstzeit 60) schneidet WSPT 11.4 % schlechter ab als die
  einfachere, gewichts-blinde Regel SPT. Über die 5 festen Sweep-Instanzen gemittelt gewinnt WSPT zwar weiterhin
  meistens, aber "meistens" ist eben keine Garantie mehr, sobald der Beweis (der keine Rüstzeiten kennt) nicht
  mehr trägt – genau der Punkt der Vehikel-B-Idee dieser Linie.
- **Ein Streamlit-Frontend/Backend-Desync gefunden und behoben:** der Rüstzeit-Regler wird nur auf dem
  Werkstatt/Logistik-Vehikel gezeichnet; ein Wert, der in einem Lauf OHNE den Regler in dessen Widget-Zustand
  gesetzt wird, erschien beim nächsten Zeichnen als Mindestwert im Regler, während die App intern weiterhin mit
  dem echten Wert rechnete (nur im echten Browser sichtbar, nicht in AppTest). Behoben mit dem
  `seed_widget`/`KEPT`-Muster, das auch an anderer Stelle im Portfolio für genau dieses Problem verwendet wird.
- **Die Vollaufzählung ist die einzige echte Gegenprobe**, aber praktisch nur bis $n \approx 9$ nutzbar – ab dort
  vertraut die Demo dem Beweis (Vertauschungsargument) statt einer weiteren Vollaufzählung.
- **Synthetische Instanzen:** Bearbeitungszeiten gleichverteilt, Gewichte aus einer festen Verteilung, keine
  Präzedenzen, ein Auftrag = eine Operation. Zeiten hängen vom Rechner ab, nur die Größenordnung zählt.

## Verifikation

- **Beweis gegen unabhängige Vollaufzählung:** für jede getestete Instanzgröße (n = 2 bis 9) und jede der 5
  festen Instanzen trifft WSPT exakt das Minimum aller $n!$ Reihenfolgen – 100 % Trefferquote.
- **Rüstzeit-Variante gegen unabhängige Vollaufzählung** (mit Rüstzeiten statt ohne) und **Konsistenz-Test**:
  Rüstzeit 0 liefert exakt dieselbe Zielfunktion wie das neutrale Vehikel.
- **Spezialfall-Test:** gleiche Gewichte für alle Aufträge → WSPT-Reihenfolge ist bitidentisch mit SPT-Reihenfolge.
- **Handrechnung:** eine kleine, von Hand nachgerechnete Instanz (2 Aufträge, extremer Gewichtsunterschied)
  bestätigt WSPT-Reihenfolge und Zielfunktionswert exakt gegen SPT.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Optimalitäts-Trefferquote, Rechenzeit,
  Rüstzeit-Härtetest, jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative
  Aussagen inklusive des Falls, in dem WSPT schlechter als SPT abschneidet); alle 5 Presets geprüft; AppTest-
  Rauchtests (Voreinstellung, jedes Preset, jeder Schritt auf beiden Vehikeln, Würfel-Knöpfe, Permalink-Grenzen
  inkl. ungültigem Vehikel, Extremwerte, Experimente auf Abruf, Footer); eigener Test für die
  ausblendbaren Regler (kein verwaister Widget-Zustand nach Permalink/Preset).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweep, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `wspt_algorithm.py` | WSPT, Brute-Force-Gegenprobe, Rüstzeit-Variante |
| `wspt_scenario.py` | Vehikel Neutral |
| `wspt_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix) |
| `wspt_constants.py` | Konstanten, Presets |
| `wspt_evaluation.py` | Kennzahlen, Sweep, Optimalitäts- und Timing-Messreihe, Rüstzeit-Härtetest |
| `wspt_presets.py`, `wspt_visualization.py` | Permalink/Presets (inkl. `seed_widget`/`KEPT` für ausblendbare Regler), Plotly-Figuren (achsengesperrt) |
| `tests/` | Beweis gegen Vollaufzählung, Szenario und Auswertung, Aussagen der App, Presets, versteckter Widget-Zustand, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Scheduling-Theorie: SPT bis RCPSP](https://sebastianhanisch.net/konzepte-klassische-scheduling-theorie.html).
