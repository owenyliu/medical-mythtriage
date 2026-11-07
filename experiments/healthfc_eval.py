"""External validation: run the project's OWN glancer (Tier 1, untouched)
on the HealthFC benchmark (Vladika et al., LREC-COLING 2024).

Design (mirrors the project's evaluation protocol):
  - Claims: English claims from Datensatz.csv with a decisive gold label:
    label 0 (Supported) -> gold "true"; label 2 (Refuted) -> gold "myth".
    The 423 NEI claims are excluded (no decisive gold; reported as such).
  - The glancer is used exactly as shipped in agent/mythtriage/glancer.py:
    one look, no tools, verdict + confidence. Nothing is re-prompted.
  - A glancer answer of "mixed"/"unknown" on this binary set counts as
    NOT settled (it is not a decisive verdict), never as silently correct.
  - Reported: strict accuracy, accuracy per gold class, the 0.90-gate
    behaviour (settled share + precision among settled), and coverage at
    95% precision from a confidence sweep over decisive answers.

Key handling is inherited from mythtriage.openrouter_lm: set
MYTHTRIAGE_ENV_FILE to the .env that already holds OPENROUTER_API_KEY
(e.g. the HW1 .env). The key is never printed or written anywhere.

Usage:
  python3 healthfc_eval.py [--limit N] [--csv Datensatz.csv]
  Results append to healthfc_results.json next to this script; rerunning
  resumes where a previous run stopped.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_DIR = next((c for c in (HERE / "agent", HERE.parent / "agent")
                  if (c / "mythtriage").is_dir()), HERE / "agent")
sys.path.insert(0, str(AGENT_DIR))

from mythtriage.glancer import run_glancer          # noqa: E402
from mythtriage.openrouter_lm import (get_api_key,  # noqa: E402
                                       make_openrouter_lm, shape_check)

RESULTS_PATH = (HERE / "results" / "healthfc_results.json"
                if (HERE / "results").is_dir() else HERE / "healthfc_results.json")
GOLD = {"0": "true", "2": "myth"}
DECISIVE = ("myth", "true")


def load_claims(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    claims = []
    for row in rows:
        gold = GOLD.get(str(row.get("label", "")).strip())
        claim = (row.get("en_claim") or "").strip()
        if gold and claim:
            claims.append({"claim": claim, "gold": gold})
    return claims


def load_done():
    if RESULTS_PATH.is_file():
        try:
            done = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            return {r["claim"]: r for r in done}
        except Exception:
            return {}
    return {}


def coverage_at_precision(records, target=0.95):
    """Largest share of ALL claims settleable at >= target precision by a
    single confidence threshold (the gate): sweep the threshold over the
    observed confidence values of decisive answers; a threshold settles
    every decisive answer at or above it. This is the coverage a real
    gate can achieve -- unlike a sorted-prefix walk, it cannot split a
    group of answers that share one confidence value."""
    decisive = [r for r in records if r["pred"] in DECISIVE]
    total = len(records)
    best = 0.0
    for t in sorted({r["confidence"] for r in decisive}):
        settled = [r for r in decisive if r["confidence"] >= t]
        if settled and sum(1 for r in settled if r["correct"]) / len(settled) >= target:
            best = max(best, len(settled) / total)
    return best


def summarize(records):
    n = len(records)
    decisive = [r for r in records if r["pred"] in DECISIVE]
    strict_correct = sum(1 for r in records if r["correct"])
    dec_correct = sum(1 for r in decisive if r["correct"])
    per_class = {}
    for gold in ("true", "myth"):
        sub = [r for r in decisive if r["gold"] == gold]
        per_class[gold] = (sum(1 for r in sub if r["correct"]), len(sub))
    settled90 = [r for r in decisive if r["confidence"] >= 0.90]
    prec90 = (sum(1 for r in settled90 if r["correct"]) / len(settled90)
              if settled90 else None)
    hedged = n - len(decisive)
    print("\n================ HealthFC external validation ================")
    print(f"claims scored            : {n}")
    print(f"strict accuracy          : {strict_correct}/{n} = {strict_correct/n:.3f}"
          "  (mixed/unknown counted as not correct)")
    print(f"decisive answers         : {len(decisive)}  (hedged/abstained: {hedged})")
    if decisive:
        print(f"accuracy on decisive     : {dec_correct}/{len(decisive)} = {dec_correct/len(decisive):.3f}")
    for gold, (c, m) in per_class.items():
        if m:
            print(f"  gold={gold:<5} accuracy : {c}/{m} = {c/m:.3f}")
    if settled90:
        print(f"gate @0.90               : settles {len(settled90)}/{n} = {len(settled90)/n:.3f}"
              f" of claims, precision {prec90:.3f}")
    cov = coverage_at_precision(records)
    print(f"coverage @95% precision  : {cov:.3f}")
    print("===============================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only N claims (smoke test)")
    ap.add_argument("--csv", default=next(
        (c for c in (str(HERE / "Datensatz.csv"), str(HERE / "../data/external/healthfc/Datensatz.csv")) if Path(c).is_file()), str(HERE / "Datensatz.csv")))
    args = ap.parse_args()

    _length, _prefix_ok = shape_check(get_api_key())
    print(f"key check: found (length {_length}, prefix ok: {_prefix_ok})")
    claims = load_claims(args.csv)
    random.Random(42).shuffle(claims)
    if args.limit:
        claims = claims[: args.limit]
    print(f"HealthFC decisive claims in scope: {len(claims)} "
          f"(gold true/myth; NEI excluded)")

    done = load_done()
    todo = [c for c in claims if c["claim"] not in done]
    print(f"already scored: {len(claims) - len(todo)}; to run now: {len(todo)}")

    lm = make_openrouter_lm()
    records = [done[c["claim"]] for c in claims if c["claim"] in done]
    for i, item in enumerate(todo, 1):
        t0 = time.time()
        out = run_glancer(item["claim"], lm)
        rec = {"claim": item["claim"], "gold": item["gold"],
               "pred": out["verdict"], "confidence": out["confidence"]}
        rec["correct"] = rec["pred"] == rec["gold"]
        records.append(rec)
        done[rec["claim"]] = rec
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(list(done.values()), ensure_ascii=False, indent=1),
                                encoding="utf-8")
        if i % 25 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] last: pred={rec['pred']} conf={rec['confidence']:.2f} "
                  f"gold={rec['gold']} ({time.time()-t0:.1f}s)")

    summarize(records)


if __name__ == "__main__":
    main()
