"""Tier 1 — the glancer. A *pipeline*, not an agent (like HW1 `priority.py`).

One model call, no tools, no search. The model sees the claim exactly once
and must answer verdict + confidence. Everything defensive is copied from
HW1 priority.py's parse_reply: try JSON first, then a labelled line, and if
nothing parses, say "unknown" — never guess, never crash.
"""

import json
import re

VERDICTS = ("myth", "true", "mixed", "unknown")

GLANCER_SYSTEM = (
    "TASK: GLANCE. You are a fast pre-screening model. You see one health "
    "claim exactly once, with no search and no tools. Reply with ONLY one "
    "JSON object: {\"verdict\": \"myth\"|\"true\"|\"mixed\"|\"unknown\", "
    "\"confidence\": <0.0-1.0>}. Confidence is how sure you are; if you are "
    "not sure, say so with a low number — a slow verifier will check the rest."
)


def parse_glancer_reply(reply):
    """Model text -> {"verdict": ..., "confidence": ...}. Never raises.

    Layer 1: a JSON object (whole reply, or the {...} slice inside prose).
    Layer 2: a `Verdict: myth ... Confidence: 0.8` labelled line.
    Fallback: unknown / 0.0 — "say you don't know" beats guessing (HW1 Q7).
    """
    text = str(reply or "").strip()
    if not text:
        return {"verdict": "unknown", "confidence": 0.0}

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict):
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in VERDICTS:
                try:
                    confidence = float(obj.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                return {"verdict": verdict,
                        "confidence": min(1.0, max(0.0, confidence))}

    match = re.search(r"verdict\s*[:\-]\s*(myth|true|mixed|unknown)", text, re.IGNORECASE)
    if match:
        conf = re.search(r"confidence\s*[:\-]\s*(0?\.\d+|1(?:\.0)?)", text, re.IGNORECASE)
        return {"verdict": match.group(1).lower(),
                "confidence": float(conf.group(1)) if conf else 0.0}
    return {"verdict": "unknown", "confidence": 0.0}


def run_glancer(claim, lm):
    """Call `lm` once. `lm` is any callable: messages -> reply text (HW1 style)."""
    messages = [
        {"role": "system", "content": GLANCER_SYSTEM},
        {"role": "user", "content": f"Claim: {claim}"},
    ]
    try:
        reply = lm(messages)
    except Exception:
        return {"verdict": "unknown", "confidence": 0.0}
    return parse_glancer_reply(reply)


def gate_decision(glancer_result, threshold):
    """The gate is CODE, not a prompt (HW1 lesson: code enforces, prompts ask).

    Settle only a real verdict at or above the confidence threshold;
    everything else escalates to the Tier-2 verifier agent.
    """
    if glancer_result["verdict"] != "unknown" and glancer_result["confidence"] >= threshold:
        return "settle"
    return "escalate"
