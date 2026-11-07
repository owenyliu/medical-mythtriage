# Medical MythTriage — CS329Z Final Project (Midway Prototype)

**Can a fast decision model be trusted to pre-screen medical claims?**

A two-tier delegation system for medical-claim screening. **Tier 1 (the
glancer)** sees each claim exactly once — no search, no tools — and returns a
verdict plus a confidence. A **gate** (confidence threshold) settles claims
above it and escalates the rest to **Tier 2**, a slow verification agent that
checks the claim against the published literature and answers with validated
citations. The research question is not whether the glancer is accurate, but
how much it may safely settle: the headline metric is **coverage at 95%
precision**.

Midway report: [`midway-report.md`](midway-report.md) ·
[`midway-report.pdf`](midway-report.pdf)

## Status at midway (November 6, 2026)

- Two project datasets built and citation-verified: 100 Chinese claims and
  100 US claims, 3 verified citations each (`rumors/`).
- Quiz/browse web prototypes for both editions (demoed in the midway video).
- Tier 2 agent prototype built and smoke-tested live (`agent/`).
- **First systematic evaluation done**: the glancer was run, unmodified, on
  two published external benchmarks — HealthFC (327 decisive claims) and a
  balanced MEDFACT sample (92 claims). Headline result:

  | | HealthFC | MEDFACT (balanced) |
  |---|---|---|
  | Gate @0.90 precision | 1.000 | 0.712 |
  | **Coverage @95% precision** | **0.165** | **0.000** |

  The gate is safe on obscure, expert-phrased claims and unsafe on fluent,
  LLM-style claims — glancer confidence tracks fluency, not truth. Full
  protocol, numbers, and per-claim outputs: [`experiments/`](experiments/).
- Final phase: glancer sweep on the project's own datasets + wording twins,
  K1′ label-agreement check, Tier 2 rescue rate, live PubMed retrieval.

## Repository layout

```
midway-report.md / .pdf   Midway report (source + rendered)
rumors/                   Project datasets: rumors-100.json (Chinese edition),
                          rumors-us-100.json (US edition) — 3 verified
                          citations per claim
agent/                    Tier 1 glancer + gate + Tier 2 verifier prototype
                          (see agent/AGENT_README.md); 22 tests
experiments/              External-validation eval scripts, RESULTS.md,
                          and full per-claim outputs (results/)
data/external/            External benchmarks used by experiments/,
                          each with its license and attribution
```

## Running the agent prototype

```bash
cd agent
# the OpenRouter API key is read from an env file you point to:
export MYTHTRIAGE_ENV_FILE=/path/to/your/.env   # contains OPENROUTER_API_KEY
python3 -m mythtriage.demo "Drinking warm salt water cures high blood pressure"
python3 -m pytest tests/                          # offline test suite
```

The demo runs the full pipeline — glancer → gate (0.90) → verifier — over a
535-document corpus built from the datasets' cited sources. Citation IDs in
a verdict are validated against what the verifier actually retrieved;
unretrieved citations are dropped, and a verdict whose citations are all
dropped becomes an abstention.

## Running the external validation

```bash
cd experiments
export MYTHTRIAGE_ENV_FILE=/path/to/your/.env
python3 healthfc_eval.py    # HealthFC decisive subset (327 claims)
python3 medfact_eval.py     # MEDFACT balanced sample (92 claims)
```

Both scripts are resumable (per-claim outputs accumulate in `results/`),
use only the Python standard library, and call the project's own glancer
from `../agent` unmodified.

## Datasets and licenses

- **Project datasets** (`rumors/`): built for this project; every citation
  verified against PubMed or the publisher before the claim was admitted.
- **HealthFC** (`data/external/healthfc/`): Vladika, Schneider, & Matthes
  (LREC-COLING 2024), from Medizin Transparent (Cochrane Austria).
  **CC BY-NC-ND 4.0** — included verbatim and unmodified, with the required
  attribution in `data/external/healthfc/ATTRIBUTION.md`.
- **MEDFACT** (`data/external/medfact/`): CCL 2026 evaluation task dataset
  of Chinese claims from LLM-generated medical answers. **Apache 2.0** —
  only the derived 92-claim balanced sample is included; see
  `data/external/medfact/README.md` for the derivation.

## Disclaimer

Verdicts summarize published evidence for a course project. This is not
medical advice.
