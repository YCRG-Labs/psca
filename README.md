# Prompt Specification Curves (P-SCA)

A specification curve analysis framework for evaluating the robustness of LLM-simulated public opinion across prompt design choices.

## What it does

P-SCA systematically varies six prompt dimensions — **model**, **persona format**, **question framing**, **system prompt**, **temperature**, and **few-shot examples** — to measure how sensitive LLM partisan gap estimates are to arbitrary researcher decisions. We benchmark against ANES 2024 ground-truth survey data.

## Models

GPT-5.4, GPT-5.4-nano, Claude Sonnet 4.6, Gemini 2.5 Flash, Llama 3.3 70B, Mistral Small

## Setup

```bash
pip install -r requirements.txt
```

Set API keys in `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=...
```

## Usage

```bash
Latin Hypercube sampling (main run)
python pilot.py lhs --n_specs 600 --output full_lhs.json

Analyze results
python pilot.py analyze --output full_lhs.json

Permutation inference
python pilot.py permutation --output full_lhs.json

Flipped specification analysis
python pilot.py flipped --output full_lhs.json

Bootstrap CIs on eta-squared (default 5,000 resamples)
python pilot.py bootstrap --output full_lhs.json

ANES benchmark comparison
python pilot.py anes --output full_lhs.json

Saltelli sampling for Sobol indices
python pilot.py saltelli --items gun_control --output saltelli_gun.json

Sobol analysis on existing Saltelli results
python pilot.py sobol --output saltelli_gun.json

Patch failed specs from a previous run
python patch_run.py --input saltelli_gun.json --items gun_control

Forced-choice option ordering test
python ordering_test.py
```

## Project structure

| File | Purpose |
|---|---|
| `pilot.py` | CLI entrypoint |
| `config.py` | Dimensions, profiles, items, cost tables |
| `sampler.py` | LHS and Saltelli specification generators |
| `prompts.py` | Prompt construction from spec + profile + item |
| `runner.py` | Async multi-provider API runner with retries |
| `analysis.py` | Partisan gaps, eta-squared, bootstrap CIs, Sobol, permutation tests, ANES benchmarks |
| `ordering_test.py` | Position bias test for forced-choice framing |
| `patch_run.py` | Reruns failed specifications from a previous run |
| `download_anes.py` | ANES 2024 data download and processing |
