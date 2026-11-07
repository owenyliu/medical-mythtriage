"""Medical MythTriage Tier-2 verifier agent (CS329Z final project).

Public surface: verify_claim(claim, lm, corpus) — see verifier.py.
`lm` is any callable taking a messages list and returning reply text,
exactly the HW1 harness convention, so FakeLM can be swapped for a real
LLM without touching the agent code.
"""

from .corpus import Corpus
from .fake_lm import FakeLM
from .verifier import verify_claim

__all__ = ["Corpus", "FakeLM", "verify_claim"]
