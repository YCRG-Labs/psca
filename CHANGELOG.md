# Changelog

All notable changes to P-SCA are documented here. Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning follows [Semantic Versioning](https://semver.org/) with `0.x` pre-stable semantics: breaking changes may appear in minor versions until `1.0.0`.

## [Unreleased]

## [0.1.1] - 2026-05-16

### Fixed
- Stale absolute imports inside `cli.py`, `analysis.py`, and `runner.py` that caused `ModuleNotFoundError` for users installing from PyPI. Affected commands: `psca analyze`, `psca sobol`, `psca system_decomp`, `psca hierarchical_decomp`.
- README Python API example: `generate_specifications(n_samples=...)` (was incorrectly documented as `n_specs=`).

## [0.1.0] - 2026-05-22

### Added
- Initial public release coinciding with the Yale FDS AI for Social Science workshop lightning talk.
- `psca.sampler`: Latin Hypercube and Saltelli specification generators over the prompt multiverse.
- `psca.prompts`: prompt construction from specification, demographic profile, and ANES item.
- `psca.runner`: async multi-provider API runner with per-provider concurrency limits and exponential-backoff retries.
- `psca.analysis`: variance decomposition (eta-squared), Fisher r-to-z dominance test, permutation inference, bootstrap CIs, Saltelli/Sobol indices, profile jackknife, flipped-specification analysis, ANES benchmark, empirical coverage threshold derivation.
- `psca.config`: six prompt dimensions, 20 battleground-state demographic profiles, three 2024 ANES items, cost tables.
- CLI: `psca lhs`, `psca saltelli`, `psca analyze`, `psca permutation`, `psca threshold`, `psca anes`, `psca bootstrap`, `psca flipped`, `psca sobol`, `psca fisher`, `psca system_decomp`, `psca hierarchical_decomp`, `psca profile_sensitivity`.

[Unreleased]: https://github.com/YCRG-Labs/psca/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YCRG-Labs/psca/releases/tag/v0.1.0
