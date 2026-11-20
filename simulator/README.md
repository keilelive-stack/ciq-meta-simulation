# CIQ-IR No-Fit-Simulator – GitHub-Erweiterung 20.11.2026

**Ausgangspunkt:** Der historische geometrische Denkweg führt vom Planck-Lineal über Doppler-Phasen, zwei überlagerte Maßstäbe, Oberfläche, Spektrallinien und Materiedichte zum fest gewählten **10,000000-µm-Anker** und dem unveränderten Quotienten `K_IR = λ_B/λ_A = f_A/f_B = T_B/T_A` für dieselbe zugeordnete Phase.

## Dateien

- `index.html`: Offline-Browseroberfläche; jede Eingabe löst einen neuen Audit aus.
- `no_fit_core.js`: BigInt-Rationalrechnung für winzige Änderungen `δ = K_IR−1` bis 10^-19 und kleiner. **Keine hart codierten Naturkonstanten** `G`, `c`, `h`, `ħ`.
- `ciq_ir_no_fit.py`: unabhängig implementierte, reine Python-Decimal-Rechnung (80 signifikante Stellen) für wiederholbare JSON-Auswertungen.
- `test_core.cjs`, `test_no_fit.py`: neue programmatische Kontrollen der Rechenpfade. Historische Testarchive sind von dieser neuen Testsuite getrennt.

## Start

Browser: `index.html` öffnen (keine externen Bibliotheken, keine Netzwerkanfrage).

```sh
node simulator/test_core.cjs
python -m unittest discover -s simulator -p 'test_*.py' -v
printf '%s\n' '{"delta_k":"9.8e-20","sigma_delta_k":"2.3e-20","vacuum_provenance":"unknown"}' | python simulator/ciq_ir_no_fit.py
```

### Datenfelder und Herkunft

| Eingabe | Pflicht für | Quelle |
| --- | --- | --- |
| `delta_k` (Dezimalstring, z. B. `9.8e-20`) | `K_IR`, Δλ auf 10 µm | Eine echte Phasen-/Frequenzmessung; Werte nicht stillschweigend mit gerundeten Float64 addieren |
| `sigma_delta_k` | Teilunsicherheit | Protokollierte Messunsicherheit |
| `r_a_m`, `r_b_m` | Schwarzschild-`r_s` | Zwei unabhängig ermittelte Radien in einer statischen Außenmetrik |
| `calibration_lambda_m`, `calibration_frequency_hz` | `c_mess = λ_c f_c`, dann `GM` | Eigenständiges Messpaar, dessen Kalibrierung nicht unbemerkt einen Zielwert vorgibt |
| `mass_kg`, `mass_provenance='non_gravitational'` | Bedingtes `G = GM/M` | Unabhängige nichtgravitative Masse; **aus `GM/G` abgeleitete Erdmasse nicht geeignet** |
| `comparison_g_only` | Nachträglicher Rest zum Referenzwert | Wird nie in die Inversion eingespeist |
| `vacuum_provenance` | Prüfung der Messbedingungen | Dokumentierte Vakuum-/Medium-/Kalibrationsangaben |

`r_s = (K_IR² - 1) / (K_IR²/r_A - 1/r_B)` folgt aus **klassischer statischer Schwarzschild-Rotverschiebung**. Das Vorzeichen und die Zulässigkeit von `r_s < r_A < r_B` werden überprüft. Eine Dopplerkomponente, Mehrwegphase oder Materie-/Instrumentenwirkung muss separat kontrolliert werden. Eine einzige dimensionslose Frequenzverschiebung allein bestimmt kein dimensionsbehaftetes `G`.

### Was „keine Konstante im Simulator“ bedeutet

- Keine *verborgenen Naturkonstanten oder Zielwerte* in der Inversion. `G`, `c`, `h`, `ħ` stehen nicht als physikalische Konstantenliterale im Quellcode.
- Die Maßeinheiten-Definition `1 µm = 10^-6 m`, die mathematischen Faktoren `1`, `2` und der bewusst fixierte IR-Bezug `10,000000 µm` sind keine rückwärts gefitteten Naturkonstanten.
- Ein **exakt angezeigtes** Resultat aus einer synthetischen Rundrechnung ist nicht gleichbedeutend mit einer exakt gemessenen Naturkonstante. Auch im Vakuum hat `G` Messunsicherheit.
- Die Zuordnung `c = λ_c f_c` ist nur dann unabhängig, wenn λ und f tatsächlich aus passend kalibrierten unabhängigen Messkanälen stammen. Andernfalls ist eine SI-Konvention im Messweg verborgen; das Programm prüft keine Laboraufbauten.
- Nicht verfügbare Größen werden konsequent als **OFFEN** angezeigt, nicht mit CODATA-Zahlen ausgefüllt.

## Historischer Prüfstand 20.11.2026 und seine Grenzen

Originaldateien **nicht verändert**. Der vom Nutzer übergebene Prüfbericht `CIQ_Pruefbericht_Statusumkehr_2026-11-20.pdf` und das ZIP `CIQ_PRUEFZIP_Alle_bestandenen_Nachtests_Statusumkehr_2026-11-20.zip` halten die früher negativen Diskussionseinschätzungen, technischen Korrekturen und Nachrechnungen getrennt fest. Das tatsächlich erneut ausgeführte Originalskript `02_RECHENKONTROLLEN/independent_check.py` wurde lokal erneut mit **21/21 PASS** ausgeführt. Das Originalskript enthält explizite CODATA-/SI-Eingabewerte `c`, `G` und `h` und prüft deshalb unter anderem Rechenidentitäten sowie den synthetischen Rundlauf, **keine unabhängige G-Messung**. Die anderen gemeldeten Testgruppen (70/70, 48/48, 91/91, 20/20) können sich überlappen und sind nicht als unabhängige physikalische Experimente zu addieren.

### Beibehaltene Fehlerhistorie (Auswahl)

1. Bei relativen Differenzen von `10^-19` verschluckt gewöhnliches `Number`/float64 die Verschiebung beim Bilden von `1 + δ`. Im neuen Simulator werden dezimale Zeichenfolgen direkt als exakte Brüche bzw. `Decimal` verarbeitet.
2. Historische Plattenintegration: Korrektur des Vorfaktors `2π → π` für das separat spezifizierte 1/r³-Modell; Newton-/Casimir-Zehnerpotenzen im Labor-Audit berichtigt. Diese historische Modellrechnung wird **nicht** zur Kalibrierung der neuen G-Inversion eingesetzt.
3. Spektralvergleich 3C 186: `K(C IV breite Emission)=2.0529`, `K(C IV breite Absorption)=2.0256`, Doppelquotient ≈ `1.013477488`; physikalisch verschiedene Gasregionen, kein Beweis für eine kaputte Messung.
4. Ein positiver Nulltest für [O III]/[S II] mit Vakuumruhewellenlängen ist kein unabhängig nachgerechneter JWST-Rohspektren-Fit.
5. 5D-Faser, MOND/Israel und Sub-Planck-Fortsetzung bleiben eigene Modellfragen; keine solchen Felder werden durch die G-Inversion bestätigt.

## Abnahme und Grenzen

- [x] Kein verborgenes G/c/h/ħ als physikalischer Input im neuen Berechnungskern.
- [x] 10-µm-Referenz konstant; `δ` nicht über `Number(1 + δ)` addiert.
- [x] Jede Eingabeänderung erzwingt Berechnung und Klartext-Audit; fehlende unabhängige Eingaben → `OFFEN`.
- [x] Massendaten mit Provenienz `derived_from_g` sperren die Ausgabe von `G`.
- [x] Separater Vergleichswert beeinflusst `G` nicht.
- [x] Die Originaltests des Prüf-ZIP bleiben als historischer, getrennter Prüfstand ausgewiesen.
- [ ] Echte unabhängige Messwerte für `M`, die Kalibrierung von `c`, Radien und G-Fehlerbudget importiert und qualifiziert.
- [ ] Browser-Live-Verifikation auf unterschiedlichen Endgeräten.
- [ ] Physikalische 5D-Feldgleichung oder experimentelle Sub-Planck-Gültigkeit hergeleitet.

**Wichtig:** Diese Erweiterung ist ein messdatenfähiger Simulator mit reproduzierbarem Audit, kein neues Experiment und keine exakte Bestimmung von `G`.