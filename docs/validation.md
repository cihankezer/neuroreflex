# Validation record

Validated locally on 18 October 2026. Public CI is separately reported by GitHub Actions; a workflow file is not evidence of a completed CI run.

| Check | Outcome |
| --- | --- |
| Clean editable install, `python -m pip install -e ".[test,visualization]"` | Passed on Windows CPython 3.13.6 |
| `python -m pytest -p no:cacheprovider -q tests --tb=short` | 153 passed, 16 optional Lava tests skipped; final run 22.94 s |
| Python 3.10.22/Lava 0.10.0: `python -m unittest discover -s tests -p test_spiking.py -v` | 20 passed, including 16 actual CPU graph tests |
| `python scripts/run_benchmarks.py` | Eight conditions, 24 encounters each, 192 evaluations; every runner exited successfully |
| `python scripts/run_demo.py` | Passed; one 1.2 m/s collision-course encounter, no contact, minimum clearance 452.048 mm |
| `python scripts/generate_plots.py` | Passed; 59-frame GIF rendered from actual recorded control ticks and visually reviewed |
| `python examples/reflex_pipeline.py` | Passed; `contact=False, clearance=0.452048 m` |
| `python -m pip check` in clean portable environment | No broken requirements |
| Build wheel, install in second clean environment, import from outside checkout | Passed; imports resolved to installed site-packages |
| Installed wheel: six-trial dynamic CLI plus both JSON presets | Passed; 0/6 contacts and 0/3 false triggers |
| Installed wheel: historical 16-case static workbook CLI | Executed successfully; 15/16 safe, one contact, 6.25% collision rate |

The static workbook runner is a different legacy experiment: it treats every row as an incoming straight-Z obstacle, even labels that imply no threat. Its 100% non-threat false-trigger result and 20 ms median reaction are not comparable to the matched dynamic table. The static outcome is retained here rather than presented as an improved result. The packaged workbook is needed by original regression tests.

The clean environment used NumPy 2.5.4, pytest 9.1.1, Matplotlib 3.11.2, and Pillow 13.0.0. Exact versions are in `requirements-validated.txt`. A second clean wheel environment installed only the wheel and NumPy. Existing Lava dependencies were used read-only; a fresh Lava installation was not verified.

## Existing failures and release scope

Before changing the candidate, the original complete test collection was executed in a private local snapshot, with no writes in the source repository. It produced **258 passed, 4 failed, 4 skipped, and 16 errors**. Four preservation failures concern stale historical hashes in the visual/event-perception suites; the 16 runtime errors came from the original `lava/` directory being detected as an installed SDK. The original files, expected hashes, and failure log are preserved privately.

The public repository releases the author's portable controller/safety/evaluation contribution. Six recording, visual integration, and historical event-study test modules remain in the private archive with their excluded dependencies. They are not counted as passing public tests. The conflicting teammate `lava/` directory is excluded from this release; the optional motor SDK is therefore detected correctly. An intermediate candidate run also found four omitted regression fixtures; restoring the exact original fixtures fixed those failures without changing expected outcomes.

## Not executed

Physical robot trials, event-camera recordings, end-to-end teammate visual perception, Loihi 2 deployment, energy measurements, fresh Lava dependency installation, and GitHub-hosted CI before publication. No success is claimed for these paths.

## Isolation and public-file review

A fingerprint check covered **1,199 pre-existing source files**, including tracked files and untracked presentation assets: no content changes. Git HEAD, config, index, packed refs, and working-tree status also matched the initial snapshot. No source history or remote was copied into the independent repository. The staged release is separately scanned for credentials, local paths, internal URLs, and excluded assets before committing.

Only local path metadata was redacted from the copied historical static CSV; all numerical trial and summary values are unchanged. Fresh benchmark runs explicitly record relative preset and output paths. Original unredacted artifacts remain privately preserved. Text files use LF across platforms to preserve byte-based regression checks.

The exact staged 97-file release was exported without private files or Git history: 153 tests passed, 16 Lava tests skipped, and the headless demo succeeded. CI uses Windows runners because historical exact-fingerprint fixtures were recorded on Windows; other operating systems remain unverified.

## Cross-version CI finding and correction

The first hosted CI run passed on Python 3.13 but Python 3.10 rejected raw historical trace hashes. A local paired run had identical encounter metrics and controller/state decisions; ten passive diagnostic float values in a 92-tick example differed at roughly 1e-16. Runtime-dependent floating-point summation makes byte-identical JSON an inappropriate cross-version requirement.

Original raw fixtures and hashes remain unchanged and are still strictly checked on their CPython 3.13/Windows reference family. A supplementary fixture holds 324 trace fingerprints with finite floats rounded to nine decimal places. Branch decisions, labels, counts, structure, and original encounter/diagnostic summaries remain exact; seeded within-runtime determinism tests remain active. The supplementary fingerprints were derived only after matching every original raw trace on the reference runtime. `python scripts/generate_trace_fingerprints.py` reproduces that derivation on the reference family and refuses changed raw outcomes.

The affected Python 3.10 robustness and release suites passed locally (11 and 25 tests respectively). This changes validation portability, not runtime control or published benchmark values. The workflow also uses current Node 24-compatible action versions.
