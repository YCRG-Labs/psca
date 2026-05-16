# P-SCA: Prompt Specification Curve Analysis

[![PyPI](https://img.shields.io/pypi/v/psca.svg)](https://pypi.org/project/psca/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)

A specification curve analysis framework for evaluating the robustness of LLM-simulated public opinion across prompt design choices. P-SCA systematically varies six prompt dimensions, **model**, **persona format**, **question framing**, **system prompt**, **temperature**, and **few-shot examples**, to measure how sensitive LLM partisan-gap estimates are to arbitrary researcher decisions. Benchmarked against ANES 2024 ground-truth survey data.

## Install

```bash
pip install psca
```

Or from source:

```bash
git clone https://github.com/YCRG-Labs/psca && cd psca
pip install -e .
```

Set API keys in `.env`:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Quickstart

Latin Hypercube sampling, the main run:

```bash
psca lhs --n_specs 600 --output full_lhs.json
```

Reproduce Gemini-excluded headline numbers:

```bash
psca analyze --output full_lhs.json --exclude_models gemini-2.5-flash
```

Derive empirical coverage thresholds (10k permutations):

```bash
psca threshold --output full_lhs.json --exclude_models gemini-2.5-flash --n_permutations 10000
```

Permutation inference for the partisan signal:

```bash
psca permutation --output full_lhs.json
```

Variance decomposition and Fisher r-to-z dominance:

```bash
psca analyze --output full_lhs.json
psca fisher --output full_lhs.json
```

Bootstrap CIs on eta-squared (5,000 resamples):

```bash
psca bootstrap --output full_lhs.json
```

ANES benchmark comparison (amplification factor):

```bash
psca anes --output full_lhs.json
```

Flipped specification analysis:

```bash
psca flipped --output full_lhs.json
```

Saltelli sampling for Sobol sensitivity indices:

```bash
psca saltelli --items gun_control --output saltelli_gun.json
psca sobol --output saltelli_gun.json
```

Or use the Python API:

```python
import psca

specs = psca.generate_specifications(n_samples=600, seed=42)
df = psca.load_results("full_lhs.json", exclude_models=["gemini-2.5-flash"])
psca.variance_decomposition(df)
psca.derive_coverage_threshold(df, n_permutations=10000)
```

## Models supported

GPT-5.4, GPT-5.4-nano, Claude Sonnet 4.6, Llama 3.3 70B, Mistral Small. Gemini 2.5 Flash is queried in the same multiverse design but excluded from primary analyses on parse-rate grounds (see paper §4.1).

## Project structure

| Path | Purpose |
|---|---|
| `src/psca/config.py` | Six prompt dimensions, 20 battleground-state profiles, ANES items, cost tables |
| `src/psca/sampler.py` | Latin Hypercube and Saltelli specification generators |
| `src/psca/prompts.py` | Prompt construction from spec, profile, and item |
| `src/psca/runner.py` | Async multi-provider API runner with retries |
| `src/psca/analysis.py` | Partisan gaps, eta-squared, bootstrap CIs, Sobol, permutation tests, ANES benchmarks, threshold derivation |
| `src/psca/cli.py` | CLI entry point (`psca ...`) |
| `ordering_test.py` | Position bias test for forced-choice framing |
| `patch_run.py` | Reruns failed specifications from a previous run |
| `download_anes.py` | ANES 2024 data download and processing |

## Data and logs

Results files in `results/*.json` are the API call logs. Each record includes the model's raw text reply in the `raw_response` field alongside the parsed `score` and full specification metadata (model, persona, framing, system prompt, temperature, few-shot count, profile, item, repeat).

## Citation

If you use P-SCA in academic work, please cite both the methodology paper and the software:

```bibtex
@software{crainic_psca_2026,
  author  = {Crainic, Jacob and Yee, Brandon and Koh, Pairie},
  title   = {{P-SCA}: Prompt Specification Curve Analysis},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.PENDING},
  url     = {https://github.com/YCRG-Labs/psca}
}
```

See `CITATION.cff` for machine-readable metadata.

## License

MIT. See [LICENSE](LICENSE).
