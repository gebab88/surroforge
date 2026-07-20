# Literature: Surrogate Models in Engineering

> Zusammenstellung der Grundlagenliteratur zu Surrogat-Modellen in der Ingenieur­simulation,
> der industriellen Adoptions- und Validierungslage, sowie eine belegte Analyse des
> AWS-Werkzeugs **MLSimKit** — mit Schwerpunkt auf der Frage: **wo und wie wurde es bereits
> eingesetzt?**
>
> Quellen: Folie 4 (Marktanalyse) der SurroForge-Präsentation und die begleitende
> `RESEARCH_SLIDE_4.md`, ergänzt durch eine mehrsträngige Web-Recherche (Juli 2026) mit
> adversarialer Verifikation der tragenden Einsatz-Behauptungen. Konfidenz-Angaben und die
> Trennung zwischen *belegtem Einsatz* und *bloßer Marketing-Behauptung* sind explizit
> ausgewiesen. Alle URLs wurden bei der Recherche tatsächlich geöffnet; die zentralen
> AWS-Nutzungsbelege wurden zusätzlich unabhängig gegen Primärquellen (GitHub-API, AWS-Blog)
> geprüft.
>
> *Nachtrag (Juli 2026): Auf Nutzerhinweis wurde die peer-reviewte AWS-Grundlagenpublikation
> SAE 2024-01-2529 ergänzt (Abschnitt 5, Kategorie b) — Metadaten und Abstract direkt über
> SAE Mobilus verifiziert. Ein zweiter Nachtrag kartiert die weitere Publikationslinie derselben
> Gruppe (Abschnitt 4.2), recherchiert über die Publikationsseiten von N. Ashton und D. Maddix.*

---

## 1. Einordnung

Ein *Surrogat-Modell* (auch Metamodell, Response Surface oder Emulator) ersetzt eine teure
physikbasierte Simulation (CFD, FEM) durch ein billiges, schnell auswertbares Ersatzmodell,
das auf vorhandenen Simulations- oder Messdaten trainiert wird. Ziel ist die Beschleunigung
von Designstudien, Optimierung und Unsicherheitsquantifizierung (UQ) um Größenordnungen —
von Stunden pro Auswertung auf Sekunden oder weniger.

Der Themenkomplex zerfällt in drei Ebenen, die auch die Marktanalyse auf Folie 4 strukturieren:
klassische statistische Surrogate (Ebene der Methoden), projektionsbasierte Reduced-Order-Modelle
(physiknah), und moderne datengetriebene Deep-Learning-Surrogate (geometrie-/feldbasiert). Der
belastbare Engpass für den industriellen Einsatz liegt weniger in den Algorithmen als im Übergang
vom Pilot zum wiederholbaren, vertrauenswürdigen Standardprozess (siehe Abschnitt 3).

---

## 2. Methoden und Grundlagenliteratur

### 2.1 Klassische statistische Surrogate

- **Gaussian-Process-Regression / Kriging** — der kanonische Surrogat für glatte, teure,
  deterministische Simulatoren; liefert eine eingebaute Vorhersage-Varianz. Ursprung in der
  Geostatistik (Krige 1951; Matheron, *Principles of Geostatistics*, Economic Geology 58, 1963);
  Übertragung auf deterministische Computer-Experimente durch **Sacks, Welch, Mitchell & Wynn,
  „Design and Analysis of Computer Experiments"** (Statistical Science 4(4):409–435, 1989); das
  moderne ML-Lehrbuch ist **Rasmussen & Williams, *Gaussian Processes for Machine Learning*
  (MIT Press, 2006)**. *Am besten für:* nieder- bis mitteldimensionale, rauschfreie teure
  Simulationen, wobei die Varianz adaptives Nachsampeln (Infill) ermöglicht.

- **Efficient Global Optimization (EGO)** — **Jones, Schonlau & Welch (1998)**, Journal of Global
  Optimization 13(4):455–492, DOI 10.1023/A:1008306431147. Koppelt eine Kriging-Antwortfläche mit
  dem *Expected-Improvement*-Kriterium für ein Optimierungsbudget-schonendes Explorations-/
  Exploitations-Gleichgewicht. *Am besten für:* sequenzielle, surrogatbasierte globale Optimierung.

- **Radiale Basisfunktionen (RBF)** — die kanonischen netzfreien Interpolanten für verstreute
  Daten. Multiquadric-RBF: **Hardy (1971)**, J. Geophys. Res. 76(8):1905–1915; neuronale
  Interpretation: **Broomhead & Lowe (1988)**, Complex Systems 2:321–355. *Am besten für:*
  schnelle Interpolation verstreuter Stützstellen in moderaten Dimensionen ohne Netz.

- **Polynomial Chaos Expansion (PCE)** — der kanonische spektrale Surrogat für UQ und
  Sensitivitätsanalyse. Grundlage **Wiener, „The Homogeneous Chaos" (1938)**, Am. J. Math.
  60(4):897–936; praktische stochastische FEM: **Ghanem & Spanos, *Stochastic Finite Elements*
  (Springer, 1991)**; Verallgemeinerung auf die Askey-Familie: **Xiu & Karniadakis (2002)**,
  SIAM J. Sci. Comput. 24(2):619–644. *Am besten für:* Unsicherheitsfortpflanzung, Momente und
  Sobol-Sensitivitätsindizes.

### 2.2 Projektionsbasierte Reduced-Order-Modelle (ROM)

- **Proper Orthogonal Decomposition (POD)** — die kanonische *lineare* Methode zur Extraktion
  einer niedrigdimensionalen Modalbasis aus hochaufgelösten Felddaten. Einführung für Turbulenz:
  **Lumley (1967)**; effiziente „Method of Snapshots": **Sirovich (1987)**, Q. Appl. Math.
  45(3):561–571; Standardreview: **Berkooz, Holmes & Lumley (1993)**, Annu. Rev. Fluid Mech.
  25:539–575. *Am besten für:* Identifikation energetisch dominanter Moden und die reduzierte
  Basis, die projektionsbasierte ROMs nutzen. (SurroForge verwendet POD als lineare Reduktion
  im `reduced-field`-Surrogat.)

- **Projektionsbasierte parametrische ROMs** — maßgeblicher Übersichtsartikel: **Benner, Gugercin
  & Willcox, „A Survey of Projection-Based Model Reduction Methods for Parametric Dynamical
  Systems"** (SIAM Review 57(4):483–531, 2015, DOI 10.1137/130932715). Fasst (Petrov-)Galerkin-
  Projektion, Krylov/Moment-Matching, Balanced Truncation, POD und Greedy/Reduced-Basis zusammen.
  *Am besten für:* Many-Query-Auswertung, wenn die Gleichungen bekannt sind und die Projektion
  Physik erhält.

### 2.3 Neuronale und Deep-Learning-Surrogate

- **Multilayer-Perzeptron (MLP)** — gestützt auf die Universalapproximations-Theoreme von
  **Cybenko (1989)**, Math. Control Signals Systems 2(4):303–314, und **Hornik, Stinchcombe &
  White (1989)**, Neural Networks 2(5):359–366. *Am besten für:* hochdimensionale, datenreiche
  nichtlineare Regression; datenhungrig, ohne native Unsicherheitsschätzung. (SurroForge:
  `mlp`/`torch-mlp`.)

- **Deep Autoencoder** — die kanonische *nichtlineare* Alternative zu linearer POD für
  Latenzraum-Reduktion. **Hinton & Salakhutdinov, „Reducing the Dimensionality of Data with
  Neural Networks" (2006)**, Science 313(5786):504–507. *Am besten für:* nichtlineare
  Modellordnungsreduktion, wenn die Lösungsmannigfaltigkeit von einem linearen POD-Unterraum
  schlecht erfasst wird. (SurroForge: `autoencoder-field`.)

- **Graph Neural Networks (GNS, MeshGraphNets)** — die kanonischen gelernten Surrogate für
  netz- und partikelbasierte Simulation auf unstrukturierten Gebieten. **Sanchez-Gonzalez et al.,
  „Learning to Simulate Complex Physics with Graph Networks" (GNS, ICML 2020, arXiv:2002.09405)**
  und **Pfaff, Fortunato, Sanchez-Gonzalez & Battaglia, „Learning Mesh-Based Simulation with
  Graph Networks" (MeshGraphNets, ICLR 2021, arXiv:2010.03409)**. *Am besten für:* Dynamik auf
  unstrukturierten Netzen, Generalisierung über Geometrien. **Diese Architektur ist die Grundlage
  von AWS MLSimKit (Abschnitt 4).**

- **Operator Learning (DeepONet, Fourier Neural Operator)** — lernt netz-unabhängige Abbildungen
  zwischen Funktionsräumen. **Lu et al., „DeepONet" (Nature Machine Intelligence 3:218–229, 2021,
  DOI 10.1038/s42256-021-00302-5)** und **Li et al., „Fourier Neural Operator" (ICLR 2021,
  arXiv:2010.08895)**. *Am besten für:* das Lernen eines ganzen Lösungsoperators einer
  PDE-Familie, diskretisierungsinvariante Surrogate.

### 2.4 Übersichtsarbeiten (Einstieg)

- **Queipo et al., „Surrogate-Based Analysis and Optimization"** (Progress in Aerospace Sciences
  41(1):1–28, 2005) — Sampling, Modellbau (Polynom, Kriging, RBF, NN) und Modellauswahl/Validierung.
- **Forrester & Keane, „Recent Advances in Surrogate-Based Optimization"** (Progress in Aerospace
  Sciences 45(1–3):50–79, 2009) — Infill-Kriterien, Multi-Fidelity, Constraint-Handling.
- **Karniadakis et al., „Physics-Informed Machine Learning"** (Nature Reviews Physics 3:422–440,
  2021, DOI 10.1038/s42254-021-00314-5) — die Dachreferenz für moderne SciML-/Deep-Surrogate.

---

## 3. Industrielle Adoption, Validierung und Vertrauen

Belastbarster Primärdatensatz zur Adoption: **McKinsey/NAFEMS, „Unveiling the next frontier of
engineering simulation" (April 2023, 176 Befragte)**:

- Klassische Simulation (FEA/CFD/MBS): 99 % Nutzung, 51 % „at scale". **KI/ML-basierte Simulation:
  nur 5 % at scale, 61 % nur Pilot, 34 % gar nicht** — d.h. rund 10× weniger Unternehmen setzen
  KI/ML-Simulation reif ein.
- Top-Hürden sind **nicht** die Algorithmen, sondern **Fähigkeiten (61 %), Verständnis (61 %) und
  Vertrauen (47 % „limited trust in AI/ML-based methodologies")**.
- **Automotive führt** die KI/ML-Simulationsnutzung mit 76 % (vor Aerospace/Defense 59 %) an — das
  erklärt, warum die AWS-/NVIDIA-Referenzfälle durchweg Fahrzeug-Außenaerodynamik sind.
- Von 2016–2021 stieg die Zahl akademischer Arbeiten zu Deep-Learning-Surrogaten laut Bericht
  **dreißigfach** (McKinsey/NAFEMS-Angabe, keine unabhängige Bibliometrie).

**Validierung / Vertrauensnachweis (Stand Juli 2026):**

- **ASME V&V 20-2009** ist der etablierte Standard für V&V in CFD/Wärmeübertragung; ergänzt durch
  V&V 10 (Festkörper), VVUQ 1-2022 (Terminologie), V&V 40-2018 (risikobasierte Glaubwürdigkeit,
  Medizinprodukte, auch von der FDA referenziert).
- **ASME VVUQ 70** — der ML-spezifische Standard — ist **weiterhin *in Entwicklung*, NICHT
  veröffentlicht** (Titel-Platzhalter „VVUQ 70 – 20XX Verification and Validation of Machine
  Learning Algorithms"). Er darf nicht als verfügbarer Standard dargestellt werden.
- **NASA-STD-7009B (Change 1, 2024, ACTIVE)** ist ein formales Modell-Glaubwürdigkeits-Framework
  (Daten-Nachvollziehbarkeit, V&V, UQ); erwähnt ML nicht, wird aber (Johnson, NASA MSFC 2019,
  NTRS 20200002832) explizit auf Surrogat-Modelle in Hochrisiko-Entscheidungen ausgeweitet.

**Kernaussage:** Ein niedriger Trainingsfehler ist kein Freigabenachweis. Vertrauenswürdige Nutzung
ist an einen dokumentierten Einsatzbereich, unabhängige Daten, Metriken, Unsicherheit und Grenzen
gebunden — Standardisierung (Metadaten, „digital thread"/SPDM) ist noch überwiegend vendor-getrieben.

---

## 4. AWS MLSimKit — was das Tool ist

> Repository: <https://github.com/awslabs/ai-surrogate-models-in-engineering-on-aws> ·
> Doku: <https://awslabs.github.io/ai-surrogate-models-in-engineering-on-aws/>

**„AI Surrogate Models in Engineering on AWS" (Projektname *MLSimKit*, „ML for Simulation
Toolkit")** ist eine **quelloffene Python-Bibliothek + CLI/SDK** — **kein verwalteter AWS-Dienst**.
Sie läuft überall, wo Python/PyTorch läuft (die Doku empfiehlt EC2-GPU-Instanzen, das ist aber
Empfehlung, keine Abhängigkeit). Sie fügt sich in einen iterativen *train-design-predict*-Workflow ein.

- **Drei Vorhersageaufgaben:** (1) **KPI-Prediction** — skalare Kennwerte (z. B. Widerstands-/
  Auftriebsbeiwert) direkt aus der Geometrie; (2) **Slice-Prediction** — 2D-Schnittfelder
  (Geschwindigkeit/Druck) aus der 3D-Geometrie; (3) **Surface-Variable-Prediction** —
  Oberflächengrößen (Druck, Wandschubspannung) auf der 3D-Geometrie (Ausgabe als `.vtp`).
- **Modellfamilie:** eine **MeshGraphNets-inspirierte Graph-Neural-Network**-Architektur
  (Encode-Process-Decode); Slice-Prediction koppelt das GNN mit einem **Autoencoder** (zweistufig).
- **Stack:** PyTorch + PyTorch Geometric, Hugging Face Accelerate (Multi-GPU), Pydantic-CLI,
  MLflow-Tracking, JSON-Lines-Manifeste. Lizenz **Apache-2.0** (Copyright „2025 Amazon.com").
- **Reifegrad:** **früh, Pre-1.0** — Doku-Version **0.1.1**, **keine** Git-Tags, **keine**
  veröffentlichten Releases; Repo erstellt **18.04.2025**; **18 Stars, 3 Forks, 3 Watcher, 1 offenes
  Issue** (per GitHub-API unabhängig bestätigt). Der `main`-Branch ist seit Aug. 2025 faktisch ruhend
  (Metadaten-Aktivität bis 2026).
- **Was es NICHT tut:** kein verwalteter Dienst, (noch) keine native SageMaker/Batch/S3-Integration
  (als „forthcoming" gelistet), **löst nicht selbst die Physik** — es braucht vorhandene
  Ground-Truth-Simulationsdaten und ersetzt das *erneute Rechnen* des Solvers zur Inferenzzeit. Der
  demonstrierte Umfang ist Fahrzeug-Außenaerodynamik, nicht beliebige/transiente Physik. macOS/Windows
  offiziell ungetestet.

### 4.1 Die Referenz-Datensätze (AhmedML, WindsorML, DrivAerML)

Diese drei offenen Automotive-Aero-CFD-Datensätze sind **nicht** von AWS allein, sondern von einem
Community-Projekt (**caemldatasets.org**, Leitung **Neil Ashton** — zur Entstehungszeit bei AWS,
heute NVIDIA). Beiträge von AWS, NASA Ames, Ford, UpstreamCFD, BETA-CAE, Siemens Energy, NVIDIA.
AWS stellte Compute + ML-Anleitung und hostet die Daten (öffentlicher S3-Bucket
`s3://caemldatasets/`, gespiegelt auf Hugging Face). **Weder Volkswagen noch Neural Concept sind
Ersteller** (die *Referenzgeometrien* DrivAer/Windsor/Ahmed sind langjährige Benchmark-Körper).

| Datensatz | Umfang | Solver | Netz | Referenz |
|---|---|---|---|---|
| **AhmedML** | 500 Ahmed-Body-Varianten | OpenFOAM v2212, SA-DDES (Hybrid RANS-LES) | ~15–20 M Zellen | arXiv:2407.20801 |
| **WindsorML** | 355 Windsor-Body-Varianten (7 Parameter) | Volcano ScaLES (GPU, WMLES) — **nicht** OpenFOAM | ~280 M Zellen | arXiv:2407.19320 |
| **DrivAerML** | 500 DrivAer-Notchback-Morphs (16 Parameter) | OpenFOAM v2212 (UpstreamCFD-Build), SA-σ-DDES, Re=7,19·10⁶ | ~160 M Zellen, ~31 TB | arXiv:2408.11969 |

MLSimKit bildet jede seiner drei Aufgaben auf einen Tutorial-Datensatz ab (KPI/Slice → WindsorML,
Surface → AhmedML, DrivAerML als großes „real-world"-Multi-GPU-Beispiel). Der maßgebliche öffentliche
Benchmark auf diesen Daten ist **NVIDIAs PhysicsNeMo-CFD** (arXiv:2507.10747; DoMINO, FIGConvNet,
X-MeshGraphNet) — ein *separates* Framework, nicht MLSimKit (Neil Ashton ist dort allerdings
Co-Autor — siehe 4.2).

### 4.2 Die weitere Publikationslinie derselben Gruppe

Das SAE-Paper und die drei Datensätze sind Teil eines zusammenhängenden Werks desselben Kreises
(AWS-Automotive-Aero-ML-Team um **Neil Ashton** und **Danielle Maddix**). Für die Leitfrage
„wo/wie eingesetzt" ist entscheidend: Dieser gesamte Korpus ist *Grundlagen- und Werkzeugarbeit
der Erzeuger selbst*, keine unabhängige Anwendung. Er lässt sich in vier Stränge gliedern:

1. **Benchmark-Datensätze (2024).** WindsorML erschien peer-reviewt bei **NeurIPS 2024 (Datasets &
   Benchmarks Track)**; AhmedML (arXiv:2407.20801) und DrivAerML (arXiv:2408.11969) als technische
   Reports (→ Tabelle in 4.1).
2. **Angewandte Demonstrationen (2024).** SAE-Paper 2024-01-2529 und HPC-Blog (Abschnitt 5b/5c) —
   die MLSimKit unmittelbar vorausgehende Arbeit.
3. **AWS-AI-Labs-SciML-Methodenbasis (Maddix et al., 2023–2024).** Die ML-Seite dieser Linie ruht
   auf einer breiteren SciML-Forschung derselben AWS-Gruppe: *Guiding Continuous Operator Learning
   through Physics-based Boundary Constraints* (ICLR 2023), *Learning Physical Models that Can
   Respect Conservation Laws* (ICML 2023 / Physica D 2024) und *Using Uncertainty Quantification to
   Characterize and Improve Out-of-Domain Learning for PDEs* (ICML 2024). Sie adressieren
   Physik-Treue, Erhaltungssätze und Out-of-Domain-Robustheit — genau die Schwachstellen
   datengetriebener Surrogate (vgl. Abschnitt 3).
4. **Trajektorie zu Foundation-Modellen (2025–2026).** Nach Ashtons Wechsel **AWS → NVIDIA** setzt
   sich die Linie fort in *A Benchmarking Framework for AI Models in Automotive Aerodynamics*
   (arXiv:2507.10747; DoMINO/X-MeshGraphNet/FIGConvNet auf DrivAerML — die in 4.1 genannte
   PhysicsNeMo-CFD-Arbeit, Ashton als Co-Autor), der Datensatz-Erweiterung auf Hochauftriebs-
   **Flugzeug**-Aerodynamik *HiLiftAeroML* (arXiv:2605.19565, 2026) und dem Positionspapier
   *Fluid Intelligence: A Forward Look on AI Foundation Models in CFD* (Ashton, Brandstetter,
   Mishra, arXiv:2511.20455, 2025), das ein erstes „Scaling Law" für KI-Foundation-Modelle in
   der CFD vorschlägt.

**Einordnung:** Auch dieser erweiterte Korpus enthält **keine** externe Produktiv- oder
Kundennutzung von MLSimKit — er dokumentiert die *Herkunft* (Daten → Methoden → Werkzeug →
Ausblick) aus einer Hand und stützt damit die Kernaussage von Abschnitt 5. Downstream-Nutzung der
*Datensätze* (nicht des Tools) durch Dritte existiert dagegen zunehmend — etwa NVIDIAs
MoE-Gating-Arbeit (Nabian & Choudhry, arXiv:2508.21249) oder Transfer-Learning auf neue
Fahrzeugfamilien (Keum & Warey, arXiv:2605.27968) —, beide nutzen die Benchmark-Körper, **nicht**
MLSimKit.

---

## 5. Wo und wie MLSimKit eingesetzt wurde — die zentrale Frage

**Kurzfazit (belegt und unabhängig verifiziert): Jeder konkrete, dokumentierte Einsatz von MLSimKit
führt auf AWS selbst zurück. Es gibt — Stand Juli 2026 — KEINEN Nachweis einer unabhängigen
Dritt-, Kunden- oder akademischen Produktivnutzung.** Die demonstrierte Anwendungsdomäne ist durchweg
**Fahrzeug-Außenaerodynamik** (Cd/Cl-Kennwerte, Oberflächendruck/Wandschub, Strömungs-Schnittfelder).

Die dokumentierten Nutzungen zerfallen in vier Kategorien:

**(a) AWS-eigene Referenzbeispiele (Quickstarts).** Das Tool selbst liefert drei Quickstarts
(KPI/Slice/Surface), ausgeführt auf den öffentlichen Benchmark-Datensätzen WindsorML/AhmedML/
DrivAerML — von AWS verfasste Demonstrationen, keine Drittanbieter-Deployments.
Quelle: <https://awslabs.github.io/ai-surrogate-models-in-engineering-on-aws/>

**(b) Die peer-reviewte Grundlagenpublikation — SAE Technical Paper 2024-01-2529 (WCX SAE World
Congress, Detroit, 9./16. April 2024).** „Machine Learning for Road Vehicle Aerodynamics",
verfasst von **zwölf AWS-Mitarbeitern** (Ananthan, Ashton, Chadwick, Lizarraga, Maddix,
Maheswaran, Hermoso Moreno, Shabestari, Sovani, Subramanian, Tadepalli, Yu) — die formale,
zitierbare AWS-Publikation dieser Arbeitslinie und der rund einen Monat **frühere Zwilling** des
HPC-Blogs (Autoren-Überschneidung: Ananthan, Chadwick, Shabestari, Yu). Rahmung: ML **ergänzt**
CFD, ersetzt es nicht; benötigt „high quality training data" und Validierung gegen
hochauflösende Simulation. Kernaussage: Vorhersage neuer Designs „in near real-time" (**< 5 s**)
nach **typischerweise < 1 h Training auf einer einzelnen GPU**; präsentiert mehrere Fallstudien
mit verschiedenen datengetriebenen ML-Methoden (der frei zugängliche Abstract nennt keine
konkreten Architekturen — der Volltext liegt hinter der SAE-Paywall). **Kein Automobilkunde als
Anwender genannt; „MLSimKit" existiert hier noch nicht** (Repo erst April 2025) — das Paper ist
der wissenschaftliche *Vorläufer*, nicht ein Dritt-Einsatz von MLSimKit. 16 Seiten,
DOI 10.4271/2024-01-2529. *(Metadaten und Abstract von mir direkt über SAE Mobilus verifiziert.)*
Quelle: <https://doi.org/10.4271/2024-01-2529>

**(c) Der begleitende AWS-HPC-Blog (13. Mai 2024).** „Using machine learning to drive faster
automotive design cycles", verfasst von **sechs AWS-Mitarbeitern** (Miller, Panchalam Govindarajan,
Chadwick, Shabestari, Yu, Ananthan). Nutzt den offenen DrivAer-Datensatz (400 RBF-morphte Varianten,
Coupé+SUV), trainiert **MeshGraphNets** für den Widerstand (2,5 % MAPE, 0,028 s/Sample) und ein
**U-Net** für Felder (~3 % Druck-, ~5 % Geschwindigkeits-RMSE), plus eine „minimal viable web
application". **Kein Automobilkunde genannt; der Begriff „MLSimKit" kommt nicht vor** — MLSimKit ist
faktisch die spätere Produktisierung dieser Arbeitslinie (Paper (b) + Blog). *(Von mir direkt gegen die Blog-Quelle
verifiziert.)* Infrastruktur: SageMaker (Training), AWS Batch (OpenFOAM-RANS-Ground-Truth), EC2 +
NICE DCV (App), S3 (Artefakte).
Quelle: <https://aws.amazon.com/blogs/hpc/using-machine-learning-to-drive-faster-automotive-design-cycles/>

**(d) AWS-Mitarbeiter-Tutorials (Evangelism), keine Kundenprojekte.** Der viel zitierte Wert
**„14 min Training / 18 s Vorhersage / 1,87 % MAPE (Cl)"** stammt aus einem **AWS-Japan-Artikel
(Zenn) von Koji Yamada, Solutions Architect bei AWS Japan** (mit persönlicher-Meinung-Disclaimer),
der das *offizielle* KPI-Tutorial auf WindsorML (350 Fälle: 210 train / 70 val / 70 test)
nachvollzieht. Sein Beitrag ist EC2-Setup und GPU-Treiber-Troubleshooting — also AWS-Evangelism,
das das eingebaute Beispiel reproduziert. Ein zweiter Zenn-Artikel desselben AWS-Japan-Kanals
liegt in derselben Kategorie.
Quelle: <https://zenn.dev/aws_japan/articles/646c2d6fe97ee1>

**Adoptions-Signale (unabhängig per GitHub-API bestätigt):** 18 Stars, **3 Forks — alle mit 0 Stars,
also reine Kopien ohne erkennbaren Downstream-Einsatz** (`xupeiwust`; `AdnaneKhan`, ein
GitHub-Actions-Sicherheitsforscher; `thatgeeman`, laut Profil bei AVL List GmbH). Alle vier Contributor
sind AWS-nah. Eine unabhängige Web-Suche nach Kunden-Fallstudien liefert ausschließlich das Repo,
die AWS-Doku und die AWS-Japan-Artikel — **keine** Kunden-Story, kein Partner-Deployment und keine
peer-reviewte Arbeit eines *Dritten*, die MLSimKit nutzt. Die einzige peer-reviewte Publikation
dieser Linie ist AWS' eigener Vorläufer (Kategorie (b)) — sie belegt die AWS-interne Herkunft,
nicht eine externe Adoption.

### 5.1 Häufige Fehlzuordnungen — NICHT MLSimKit

Bei der Recherche tauchen mehrere beeindruckende Zahlen auf, die fälschlich MLSimKit zugeschrieben
werden. Sie gehören zu *anderen* Projekten:

- **„≥600× schneller / Widerstand in unter 2 min"** — eine **Volkswagen/FAU-Erlangen**-Arbeit
  (arXiv:2504.06699, CNN vs. kommerzielles GNN auf proprietären VW-Daten). Kein AWS, kein MLSimKit.
- **PhysicsX „4 Tage → 6 Sekunden"** (re:Invent 2025, Session STP109) — ein Partner-Produkt auf
  **SageMaker HyperPod**, nicht MLSimKit.
- **„Ball-impact modeling on AWS"** (AWS-HPC-Blog, Jan. 2025) — nutzt den **MOOSE**-FEM-Solver und
  ein eigenes ML-Modell, nennt MLSimKit nicht.
- **NVIDIA PhysicsNeMo / DoMINO** — überlappt methodisch (GNN/Neural-Operator, Automotive-Aero,
  Apache-2.0) und läuft auf AWS-GPUs, ist aber ein **eigenständiger** NVIDIA-Codebestand ohne
  dokumentierte Cross-Integration mit MLSimKit.

### 5.2 Einordnung für SurroForge

MLSimKit ist das positionierungsähnlichste offene Tool zu SurroForge (leichtgewichtig, CLI/Python,
ohne GUI, trainiert auf vorhandenen Daten) — aber es ist **(i)** auf **Geometrie→GNN-Vorhersage für
Automotive-Aero** fokussiert, **(ii)** früh (v0.1.1, faktisch AWS-internes Referenzwerkzeug), und
**(iii)** bietet **keinen** Methodenvergleich, keine klassischen Surrogate (Kriging/POD/PCE) und
keinen Freigabe-/Report-Prozess. SurroForges Differenzierung (solverübergreifend, Modellvergleich,
Validierungsartefakte, klassische + Deep-Surrogate aus beliebigen Ergebnisdaten) überschneidet sich
also kaum mit MLSimKits Kernfunktion.

---

## 6. Verifikationsvermerk

Die Recherche erfolgte über einen Multi-Agent-Workflow (6 parallele Recherche-Stränge:
Methoden, Adoption/Validierung, sowie vier AWS-Dimensionen). Der geplante adversariale
3-fach-Verifikationsdurchlauf wurde durch ein Session-Limit unterbrochen; die **tragenden
Behauptungen zur AWS-Nutzung** (GitHub-Adoptionsstand, AWS-Autorschaft des HPC-Blogs ohne
Kundennennung, Nicht-Existenz unabhängiger Fallstudien) wurden daraufhin **manuell gegen
Primärquellen (GitHub-REST-API, AWS-Blog, unabhängige Suche) bestätigt**. Der bereits im
Recherche-Strang „aws-usage" enthaltene Skeptizismus (aktive Suche nach — und Nicht-Finden von —
unabhängiger Nutzung, Entlarvung von Fehlzuordnungen) stützt dieselbe Schlussfolgerung.

Konfidenz: Abschnitte 2–4 hoch (Primärquellen: Originalarbeiten, offizielle Doku, GitHub-API,
arXiv). Abschnitt 5 hoch für die *belegten* AWS-Nutzungen und für die *Nicht-Existenz* unabhängiger
Nutzung (negativer Nachweis, gestützt durch erschöpfende Suche — grundsätzlich schwächer als ein
positiver Fund, hier aber durch mehrere unabhängige Suchstränge konsistent).

---

## 7. Referenzen

**Methoden & Reviews**
- Sacks, Welch, Mitchell & Wynn (1989), *Design and Analysis of Computer Experiments*, Statistical Science 4(4):409–435.
- Rasmussen & Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press.
- Jones, Schonlau & Welch (1998), *Efficient Global Optimization…*, J. Global Optim. 13(4):455–492 — <https://link.springer.com/article/10.1023/A:1008306431147>
- Hardy (1971), *Multiquadric Equations…*, J. Geophys. Res. 76(8):1905–1915.
- Xiu & Karniadakis (2002), *The Wiener–Askey Polynomial Chaos…*, SIAM J. Sci. Comput. 24(2):619–644 — <https://epubs.siam.org/doi/10.1137/S1064827501387826>
- Sirovich (1987), *Turbulence and the Dynamics of Coherent Structures, Part I*, Q. Appl. Math. 45(3):561–571.
- Berkooz, Holmes & Lumley (1993), *The POD in the Analysis of Turbulent Flows*, Annu. Rev. Fluid Mech. 25:539–575.
- Benner, Gugercin & Willcox (2015), *A Survey of Projection-Based Model Reduction…*, SIAM Review 57(4):483–531 — <https://kiwi.oden.utexas.edu/papers/Parametric-model-reduction-survey-Benner-Gugercin-Willcox.pdf>
- Cybenko (1989), *Approximation by Superpositions of a Sigmoidal Function*, MCSS 2(4):303–314.
- Hornik, Stinchcombe & White (1989), *Multilayer Feedforward Networks Are Universal Approximators*, Neural Networks 2(5):359–366.
- Hinton & Salakhutdinov (2006), *Reducing the Dimensionality of Data with Neural Networks*, Science 313(5786):504–507 — <https://www.science.org/doi/10.1126/science.1127647>
- Sanchez-Gonzalez et al. (2020), *Learning to Simulate Complex Physics with Graph Networks*, ICML — <https://arxiv.org/abs/2002.09405>
- Pfaff et al. (2021), *Learning Mesh-Based Simulation with Graph Networks* (MeshGraphNets), ICLR — <https://arxiv.org/abs/2010.03409>
- Lu et al. (2021), *DeepONet*, Nature Machine Intelligence 3:218–229 — <https://www.nature.com/articles/s42256-021-00302-5>
- Li et al. (2021), *Fourier Neural Operator*, ICLR — <https://arxiv.org/abs/2010.08895>
- Queipo et al. (2005), *Surrogate-Based Analysis and Optimization*, Prog. Aerospace Sci. 41(1):1–28.
- Forrester & Keane (2009), *Recent Advances in Surrogate-Based Optimization*, Prog. Aerospace Sci. 45(1–3):50–79.
- Karniadakis et al. (2021), *Physics-Informed Machine Learning*, Nature Reviews Physics 3:422–440 — <https://www.nature.com/articles/s42254-021-00314-5>

**Adoption & Validierung**
- McKinsey/NAFEMS (2023), *Unveiling the next frontier of engineering simulation* — <https://mckinseytalksoperations.com/app/uploads/2023/05/Unveiling-the-next-frontier-of-engineering-simulation-vF2.pdf>
- ASME VVUQ-Standards (V&V 20-2009, VVUQ 70 in Entwicklung) — <https://www.asme.org/codes-standards/vvuq-standards> · <https://www.asme.org/codes-standards/about-standards/technology-highlights/digital-engineering>
- NASA-STD-7009B (2024) — <https://standards.nasa.gov/standard/NASA/NASA-STD-7009> · Johnson (2019), NTRS 20200002832 — <https://ntrs.nasa.gov/citations/20200002832>

**AWS MLSimKit & Datensätze**
- MLSimKit Repo — <https://github.com/awslabs/ai-surrogate-models-in-engineering-on-aws>
- MLSimKit Doku (v0.1.1) — <https://awslabs.github.io/ai-surrogate-models-in-engineering-on-aws/>
- Ananthan, Ashton, Chadwick, Lizarraga, Maddix, Maheswaran, Hermoso Moreno, Shabestari, Sovani, Subramanian, Tadepalli & Yu (AWS, 2024), *Machine Learning for Road Vehicle Aerodynamics*, SAE Technical Paper 2024-01-2529, WCX SAE World Congress — <https://doi.org/10.4271/2024-01-2529>
- AWS HPC Blog (Mai 2024), *Using machine learning to drive faster automotive design cycles* — <https://aws.amazon.com/blogs/hpc/using-machine-learning-to-drive-faster-automotive-design-cycles/>
- AWS Japan / Zenn (Yamada), MLSimKit-Tutorial-Walkthrough — <https://zenn.dev/aws_japan/articles/646c2d6fe97ee1>
- CAE ML Datasets — <https://caemldatasets.org/> · AhmedML arXiv:2407.20801 · WindsorML (NeurIPS 2024, Datasets & Benchmarks Track) arXiv:2407.19320 · DrivAerML arXiv:2408.11969 (<https://huggingface.co/datasets/neashton/drivaerml>)
- Tangsali, Ranade, Nabian, Kamenev, Sharpe, **Ashton**, Cherukuri & Choudhry (2025), *A Benchmarking Framework for AI Models in Automotive Aerodynamics* (NVIDIA PhysicsNeMo-CFD; DoMINO/X-MeshGraphNet/FIGConvNet) — arXiv:2507.10747
- Fehlzuordnung VW/FAU (≥600×) — arXiv:2504.06699

**Weitere Arbeiten derselben Gruppe (Ashton / Maddix — Abschnitt 4.2)**
- Ashton, Brandstetter & Mishra (2025), *Fluid Intelligence: A Forward Look on AI Foundation Models in CFD* — arXiv:2511.20455
- Ashton et al. (2026), *HiLiftAeroML: High-Fidelity CFD Dataset for High-Lift Aircraft Aerodynamics* — arXiv:2605.19565
- Saad, Gupta, Alizadeh & Maddix (2023), *Guiding Continuous Operator Learning through Physics-based Boundary Constraints*, ICLR — <https://www.amazon.science/publications/guiding-continuous-operator-learning-through-physics-based-boundary-constraints>
- Hansen, Maddix, Alizadeh, Gupta & Mahoney (2023/2024), *Learning Physical Models that Can Respect Conservation Laws*, ICML 2023 / Physica D — <https://doi.org/10.1016/j.physd.2023.133952>
- Mouli, Maddix, Alizadeh, Gupta, Wang, Stuart & Mahoney (2024), *Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs*, ICML — <https://proceedings.mlr.press/v235/mouli24a.html>
- Downstream-Nutzung der *Datensätze* durch Dritte (nicht MLSimKit): Nabian & Choudhry (NVIDIA, 2025), *A Mixture of Experts Gating Network…* — arXiv:2508.21249 · Keum & Warey (2026), *Adapting Automotive Aerodynamics Surrogates to New Vehicle Families via Transfer Learning* — arXiv:2605.27968
