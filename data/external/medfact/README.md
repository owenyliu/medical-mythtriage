# MEDFACT — Balanced Evaluation Sample

`medfact_claims.json` is the 92-claim balanced sample used by
`experiments/medfact_eval.py`, derived from the MEDFACT dataset:

- **Dataset:** MedFact: A Large-scale Chinese Dataset for Evidence-based
  Medical Fact-checking of LLM Responses (CCL 2026 Evaluation Task).
  https://github.com/AshleyChenNLP/MedFact
- **License:** Apache License 2.0.
- **Derivation:** From the released training split (5,064 claims: 3,386
  supported / 1,020 partially supported / 522 uncertain / 90 not applicable /
  46 refuted), take all 46 `refuted` claims (→ gold `myth`) and 46
  `supported` claims drawn with seed 42 (→ gold `true`); deduplicated by
  claim text (no label conflicts found). The remaining labels describe an
  evidence relation rather than a verdict on the claim itself and are
  excluded; the full dataset is available at the source above.
