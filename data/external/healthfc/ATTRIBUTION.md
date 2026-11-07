# HealthFC — Attribution and License

`Datensatz.csv` in this folder is the HealthFC dataset, included **verbatim
and unmodified** for reproducibility of the external-validation experiment in
`experiments/`.

- **Paper:** Vladika, J., Schneider, P., & Matthes, F. (2024). HealthFC:
  Verifying Health Claims with Evidence-Based Medical Fact-Checking.
  Proceedings of LREC-COLING 2024, 8093–8105.
- **Source:** https://github.com/jvladika/HealthFC
- **License:** CC BY-NC-ND 4.0 (see `LICENSE-CC-BY-NC-ND` in this folder).
  Non-commercial research use; no modified versions are distributed here or
  anywhere in this repository — the balanced/decisive subsets used by the
  evaluation are produced at runtime by `experiments/healthfc_eval.py`,
  not stored as modified data.

**Attribution (as required by the dataset authors):** The claims were
fact-checked by the Medizin Transparent team at the University for
Continuing Education Krems (Danube University Krems): Bernd Kerschner,
Jana Meixner, Teresa König, Iris Hinneburg, Julia Harlfinger,
Claudia Christof, Jörg Wipplinger, Iris Mair, Verena Ahne, Tanja Wolf,
Björn Bernitt, and Lilith Teusch.

In this project the dataset is used only as an external validation set:
of its 750 claims, the 327 with decisive verdicts (202 Supported,
125 Refuted) are scored; the 423 Not-Enough-Information claims are excluded
because they carry no decisive ground truth.
