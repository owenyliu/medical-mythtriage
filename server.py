"""Medical MythTriage — tiny local web server for the demo UI.

Serves mythtriage-web.html and exposes POST /check, which runs the REAL
two-tier agent (glancer -> gate -> verifier) from the mythtriage package.

Run it from the folder that contains the agent/ package (~/rumor-crusher):

    # LIVE (real model via OpenRouter; key read from your local .env,
    # never printed, never sent anywhere except OpenRouter):
    MYTHTRIAGE_ENV_FILE=~/assignment1-harness/.env python3 server.py --live

    # OFFLINE (deterministic stub model, no key needed):
    python3 server.py

Then open http://127.0.0.1:8765 in a browser.
"""

import argparse
import importlib
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(HERE, "agent") if os.path.isdir(
    os.path.join(HERE, "agent")) else HERE
sys.path.insert(0, AGENT_DIR)
os.chdir(AGENT_DIR)

# The agent package was renamed mythtriage (repo) from rumor_agent (the
# copy staged on Owen's Mac). Import whichever one is actually present.
PKG = None
for _name in ("mythtriage", "rumor_agent"):
    try:
        importlib.import_module(_name + ".corpus")
        PKG = _name
        break
    except ImportError:
        continue
if PKG is None:
    raise SystemExit("Cannot find the agent package (looked for "
                     "'mythtriage' and 'rumor_agent' under " + AGENT_DIR + ")")
print(f"Agent package: {PKG}")
CORPUS_MOD = importlib.import_module(PKG + ".corpus")
Corpus = CORPUS_MOD.Corpus
FakeLM = importlib.import_module(PKG + ".fake_lm").FakeLM
verify_claim = importlib.import_module(PKG + ".verifier").verify_claim

CORPUS = Corpus()
STATE = {"live": False, "model": "offline stub"}


def make_lm(live, model=None):
    if not live:
        return FakeLM()
    orlm = importlib.import_module(PKG + ".openrouter_lm")
    DEFAULT_MODEL = orlm.DEFAULT_MODEL
    make_openrouter_lm = orlm.make_openrouter_lm
    require_api_key = orlm.require_api_key
    shape_check = orlm.shape_check
    key = require_api_key()
    length, ok = shape_check(key)
    print(f"OpenRouter key: found (length {length}, "
          f"prefix ok: {ok}) — the key itself is never printed.")
    STATE["model"] = model or DEFAULT_MODEL
    return make_openrouter_lm(model=model) if model else make_openrouter_lm()


LM = None  # built in main()


def dataset_match(claim_text):
    """If the claim closely matches a claim in the project datasets,
    return that entry's verdict and written explanation (the 'why'),
    so the UI can show a dataset conclusion like a fact-check page."""
    hits = CORPUS.search_claims(claim_text, k=1)
    if not hits:
        return None
    top = hits[0]
    a = set(re.findall(r"[a-z0-9]+", claim_text.lower()))
    b = set(re.findall(r"[a-z0-9]+", top["claim_text"].lower()))
    if not a or not b or len(a & b) / len(a | b) < 0.45:
        return None
    full = next((c for c in CORPUS.claims
                 if c["claim_id"] == top["claim_id"]), {})
    if not full.get("why"):
        return None
    # The claim's own source-verified citations, in dataset order.
    doc_id_of = getattr(CORPUS_MOD, "_citation_doc_id", None)
    papers, seen = [], set()
    for cit in full.get("citations", []):
        doc_id = doc_id_of(cit) if doc_id_of else None
        p = CORPUS.papers.get(doc_id) if doc_id else None
        if p and p["doc_id"] not in seen:
            seen.add(p["doc_id"])
            papers.append(paper_view(p))
    return {"claim_text": top["claim_text"], "why": full["why"],
            "verdict": top["verdict"], "papers": papers}


def paper_view(p):
    return {
        "doc_id": p.get("doc_id", ""),
        "title": p.get("title", ""),
        "journal": p.get("journal", ""),
        "year": p.get("year", ""),
        "key_finding": p.get("key_finding", ""),
        "url": p.get("url") or p.get("link", ""),
    }


def enrich(result):
    out = dict(result)
    out["dataset_match"] = dataset_match(result.get("claim", ""))
    papers = []
    for doc_id in result.get("citations") or []:
        p = CORPUS.papers.get(doc_id, {})
        papers.append(paper_view({**p, "doc_id": doc_id}))
    out["citation_papers"] = papers
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "mythtriage-web.html"),
                      encoding="utf-8") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/mode":
            self._send(200, json.dumps(STATE), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/check":
            self._send(404, "not found", "text/plain")
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            claim = json.loads(self.rfile.read(n) or b"{}").get("claim", "")
            claim = str(claim).strip()
            if not claim:
                raise ValueError("empty claim")
            result = verify_claim(claim, LM, CORPUS)
            self._send(200, json.dumps(enrich(result)), "application/json")
        except Exception as exc:  # surface the error to the UI honestly
            self._send(500, json.dumps({"error": str(exc)}),
                       "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use a real model via OpenRouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    global LM
    LM = make_lm(args.live, args.model)
    STATE["live"] = bool(args.live)
    print(f"Corpus: {len(CORPUS.claims)} claims, "
          f"{len(CORPUS.papers)} literature docs")
    print(f"Mode: {'LIVE via OpenRouter (' + STATE['model'] + ')' if args.live else 'OFFLINE stub'}")
    print(f"Open http://127.0.0.1:{args.port} in your browser")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
