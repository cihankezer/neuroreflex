# Evaluation and reproducibility

Run `python scripts/run_benchmarks.py` from the checkout root. The runner produces eight matched conditions using six constant-velocity trajectory types, four examples per type, seed 42, 0.3?2.0 m/s speeds, and a 0.5 s observation tail. P with legacy TTC, memoryless geometry, and hysteretic geometry is compared with a frozen conventional PD plus hysteresis. Clean and noisy conditions share the same physical paths and per-trial seeds.

The runner stores command arguments, source SHA-256 hashes, per-condition CSVs, and console summaries in `benchmarks/results/portable`. Each CSV contains trial, summary, and configuration rows. Dynamic trial distances/times use m/s; summary rows explicitly label mm/ms/percent. The human-readable configuration is `configs/portable-benchmark.json`; the runner manifest records the actual executed parameters.

**Collision:** true sphere/capsule signed surface clearance <= 0 (tolerance 1e-12 m). Evaluate the initial state, 2 ms geometry substeps, and relevant analytic times. Touching counts as contact. Finite sampling of a moving arm can miss arbitrarily brief grazing contact.

**False motor trigger:** qualifying motor-command deviation in an encounter that would not contact a stationary neutral arm. Denominator: non-threatening encounters only. State-activation false triggers are additionally reported and distinguish active intent from actual motor deviation.

**Reaction latency:** first qualifying motor-command deviation minus intrusion onset within 1.5 m surface clearance of the neutral arm. The command threshold is 1 deg/s. Only nonnegative responses to nominal-arm threats contribute to the summary median/p95. Other statuses remain explicit. In the table, clean conditions and geometric noisy conditions have eight eligible responses; noisy TTC-only has six. Do not compare this conditional median as a hardware performance measurement.

**Clearance:** worst/median/p05 of encounter minima, including collisions and non-threats. Negative values remain negative. The quoted table covers 192 encounter evaluations, with 24 per condition; it is not 192 independent physical paths.

Historical fixtures under `benchmark_results` preserve P behavior, static traces, gate results, and release-policy fingerprints required by tests. Extensive historical studies, duplicate CSVs, private recordings, and archived analysis are intentionally omitted from the public release. The original source and a private audit snapshot remain intact locally.
