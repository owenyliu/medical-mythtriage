"""A deterministic, offline stand-in for a real LLM — no API key, no network.

Same callable shape as HW1's `lm`: messages -> reply text. It tells the two
call sites apart by the TASK marker in the system prompt:

- GLANCE: replies with a low-confidence JSON verdict (0.40 by default), so
  the gate escalates — that is the honest behaviour for a script pretending
  to glance, and it exercises the interesting half of the system.
- VERIFY: follows a fixed, readable strategy a beginner can explain:
    turn 1: search_claims     (have we already verified this claim?)
    turn 2: search_literature (what does the literature corpus say?)
    turn 3: read_paper        (read the top paper properly)
    turn 4: FINAL             (verdict borrowed from the top known-claim
                               match; citations = doc_ids really retrieved)
  If no known claim matched, it finishes "unknown" — it abstains rather
  than inventing a verdict or a citation.

Swap this class for any real `messages -> str` callable and nothing else
in the package changes.
"""

import re

from .bm25 import tokenize

STOPWORDS = {
    "the", "a", "an", "can", "you", "your", "does", "do", "is", "are", "it",
    "to", "of", "and", "or", "in", "on", "for", "with", "that", "this",
    "actually", "really", "will", "from", "because", "called", "make",
    "makes", "get", "getting", "give", "gives", "cause", "causes",
}


def keywords(claim, limit=6):
    """Content words of the claim, first-seen order — the fake's 'query plan'."""
    seen = []
    for tok in tokenize(claim):
        if tok.isascii() and len(tok) >= 3 and tok not in STOPWORDS and tok not in seen:
            seen.append(tok)
        if len(seen) >= limit:
            break
    return " ".join(seen) or claim


class FakeLM:
    def __init__(self, glancer_confidence=0.40, glancer_verdict="myth"):
        self.glancer_confidence = glancer_confidence
        self.glancer_verdict = glancer_verdict

    def __call__(self, messages):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "TASK: GLANCE" in system:
            return (f'{{"verdict": "{self.glancer_verdict}", '
                    f'"confidence": {self.glancer_confidence}}}')
        if "TASK: VERIFY" not in system:
            return '{"verdict": "unknown", "confidence": 0.0}'

        claim = re.search(r"Claim: (.*)", user)
        query = keywords(claim.group(1)) if claim else ""
        searched_claims = "search_claims -> ok" in user or "[search_claims" in user
        searched_lit = "[search_literature" in user
        read_done = "[read_paper" in user

        if not searched_claims:
            return f'TOOL: search_claims {{"query": "{query}"}}'
        if not searched_lit:
            return f'TOOL: search_literature {{"query": "{query}"}}'
        doc_ids = re.findall(r"DOC (\S+?)[ |]", user)
        if doc_ids and not read_done:
            return f'TOOL: read_paper {{"doc_id": "{doc_ids[0]}", "start_line": 1}}'

        verdict = re.search(r"Top known-claim match verdict: (\w+)", user)
        if verdict and doc_ids:
            return ('FINAL: {"verdict": "%s", "confidence": 0.97, '
                    '"citations": ["%s"], "reason": "Top retrieved paper backs '
                    'the matched verified claim."}' % (verdict.group(1), doc_ids[0]))
        return ('FINAL: {"verdict": "unknown", "confidence": 0.0, '
                '"citations": [], "reason": "No verified claim or paper matched '
                'well enough; abstaining."}')
