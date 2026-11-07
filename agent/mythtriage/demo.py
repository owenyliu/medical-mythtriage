"""End-to-end demo — offline (FakeLM) by default, or live via OpenRouter.

Run from the agent/ folder:

    python -m mythtriage.demo                                   # offline
    python -m mythtriage.demo "Turkey makes you sleepy because of tryptophan"
    python -m mythtriage.demo "打流感疫苗反而会让人得流感"

    # Live, with a real model (key read from your local .env — never printed):
    python -m mythtriage.demo --live "The flu shot can give you the flu"
    python -m mythtriage.demo --live --model openai/gpt-4o-mini "..."

Prints the full trajectory: glancer -> gate -> each tool call -> final
verdict with validated citations (or the abstain reason).
"""

import argparse
import sys

from .corpus import Corpus
from .fake_lm import FakeLM
from .verifier import verify_claim

DEFAULT_CLAIM = "Getting a flu shot can actually give you the flu"


def _print_result(result):
    print("Trajectory:")
    for step in result["trajectory"]:
        print(f"  [{step['step']}] {step['detail']}")
    print(f"\nVerdict: {result['verdict']}  (source: {result['source']}, "
          f"confidence: {result['confidence']:.2f})")
    print(f"Citations (validated): {result['citations'] or 'none'}")
    if result.get("dropped_citations"):
        print(f"Citations dropped as fabricated/unretrieved: {result['dropped_citations']}")
    if result["abstain_reason"]:
        print(f"Abstain reason: {result['abstain_reason']}")
    print(f"Disclaimer: {result['disclaimer']}")


def _live_lm(model):
    """Build the OpenRouter LM, or exit loudly if no key is configured.

    The pre-flight check matters: the agent loop treats a failed model
    call as an abstention, so without it a missing key would look like a
    quiet "unknown" instead of the configuration error it is. Only the
    key's SHAPE (length + prefix) is printed — never the key itself.
    """
    from .openrouter_lm import (DEFAULT_MODEL, make_openrouter_lm,
                                require_api_key, shape_check)

    try:
        key = require_api_key()
    except RuntimeError as exc:
        print(f"Cannot run --live: {exc}", file=sys.stderr)
        raise SystemExit(2)
    length, prefix_ok = shape_check(key)
    print(f"OpenRouter key: found (length {length}, "
          f"prefix \"sk-or-\" {'ok' if prefix_ok else 'NOT matched — check the key'})")
    print(f"Model: {model or DEFAULT_MODEL} (live via OpenRouter)")
    return make_openrouter_lm(model=model) if model else make_openrouter_lm()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mythtriage.demo",
        description="Run the Medical MythTriage verifier agent on one claim.")
    parser.add_argument("claim", nargs="*",
                        help="the claim to check (default: a built-in example)")
    parser.add_argument("--live", action="store_true",
                        help="use a real LM via OpenRouter instead of the "
                             "offline FakeLM (needs OPENROUTER_API_KEY in "
                             "your environment or a local .env)")
    parser.add_argument("--model", default=None,
                        help="OpenRouter model id for --live "
                             "(default: openai/gpt-4o-mini)")
    args = parser.parse_args(argv)

    claim = " ".join(args.claim) if args.claim else DEFAULT_CLAIM
    corpus = Corpus()
    print(f"Corpus: {len(corpus.claims)} known claims, "
          f"{len(corpus.papers)} literature docs (offline, from the datasets)")
    print(f"Claim: {claim}\n")

    lm = _live_lm(args.model) if args.live else FakeLM()
    result = verify_claim(claim, lm, corpus)
    _print_result(result)
    return result


if __name__ == "__main__":
    main()
