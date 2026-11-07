"""Load the two Medical MythTriage datasets and build two offline corpora.

Corpus 1 — known claims: the 200 claims themselves (for `search_claims`).
Corpus 2 — literature: one document per *citation* across both datasets
(for `search_literature` / `read_paper`). A literature doc's id is its PMID
when we have one (Chinese edition stores it; US edition hides it inside the
PubMed link), otherwise a stable `DOC-xxxxxxxx` hash of the source link —
never an invented PMID.

No network, no API keys: everything comes from the dataset JSON files.
"""

import hashlib
import json
import re
from pathlib import Path

from .bm25 import build_index

# agent/mythtriage/corpus.py -> files/rumors/*.json
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "rumors"

VERDICT_NORM = {
    "谣言": "myth", "myth": "myth",
    "属实": "true", "true": "true", "supported": "true",
    "半真半假": "mixed", "mixed": "mixed", "partly true": "mixed",
}

_PMID_IN_LINK = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


def normalize_verdict(raw):
    text = str(raw).strip()
    return VERDICT_NORM.get(text) or VERDICT_NORM.get(text.lower(), "unknown")


def _citation_doc_id(citation):
    """PMID if the dataset gives us one (directly or via link), else a hash id."""
    pmid = citation.get("pmid")
    if pmid:
        return str(pmid)
    link = citation.get("link") or ""
    match = _PMID_IN_LINK.search(link)
    if match:
        return match.group(1)
    digest = hashlib.md5((link or citation.get("title", "")).encode()).hexdigest()[:8]
    return f"DOC-{digest}"


def load_datasets(data_dir=None):
    """Return the unified list of claim dicts from both editions."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    claims = []
    for filename, edition in (("rumors-100.json", "CN"), ("rumors-us-100.json", "US")):
        raw = json.loads((data_dir / filename).read_text(encoding="utf-8"))
        for item in raw["claims"]:
            claim_text = item.get("claim") or item.get("claim_en") or ""
            if item.get("claim_zh"):
                claim_text = f"{claim_text} {item['claim_zh']}".strip()
            claims.append({
                "claim_id": f"{edition}-{item['id']}",
                "edition": edition,
                "category": item.get("category", ""),
                "claim_text": claim_text,
                "verdict_raw": item.get("verdict", ""),
                "verdict": normalize_verdict(item.get("verdict", "")),
                "why": item.get("why") or item.get("why_zh") or "",
                "citations": item.get("citations", []),
            })
    return claims


class Corpus:
    """Both BM25 indexes plus the lookup tables the tools need."""

    def __init__(self, data_dir=None):
        self.claims = load_datasets(data_dir)
        self.claim_index = build_index(
            [f"{c['claim_text']} {c['category']}" for c in self.claims])

        # Literature docs: deduplicated by doc_id (a paper can back 2 claims).
        self.papers = {}  # doc_id -> paper dict
        for claim in self.claims:
            for cit in claim["citations"]:
                doc_id = _citation_doc_id(cit)
                paper = self.papers.setdefault(doc_id, {
                    "doc_id": doc_id,
                    "title": cit.get("title", ""),
                    "authors": cit.get("authors", ""),
                    "journal": cit.get("journal") or cit.get("source") or "",
                    "year": cit.get("year", ""),
                    "key_finding": cit.get("key_finding", ""),
                    "link": cit.get("link") or (f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/"
                                                if doc_id.isdigit() else ""),
                    "claim_ids": [],
                })
                if claim["claim_id"] not in paper["claim_ids"]:
                    paper["claim_ids"].append(claim["claim_id"])
        self.paper_ids = sorted(self.papers)  # stable order for the index
        self.paper_index = build_index([
            f"{self.papers[d]['title']} {self.papers[d]['key_finding']} "
            f"{self.papers[d]['journal']} "
            f"{' '.join(self._claim_text(cid) for cid in self.papers[d]['claim_ids'])}"
            for d in self.paper_ids
        ])

    def _claim_text(self, claim_id):
        for claim in self.claims:
            if claim["claim_id"] == claim_id:
                return claim["claim_text"]
        return ""

    # -- two-stage retrieval, HW1 doc_search style: cards first, full text later --

    def search_claims(self, query, k=5):
        """Stage 1 for known claims: small cards, not full records."""
        return [{
            "claim_id": self.claims[i]["claim_id"],
            "verdict": self.claims[i]["verdict"],
            "verdict_raw": self.claims[i]["verdict_raw"],
            "claim_text": self.claims[i]["claim_text"],
            "citation_ids": [_citation_doc_id(c) for c in self.claims[i]["citations"]],
            "score": round(score, 3),
        } for i, score in self.claim_index.search(query, k)]

    def search_literature(self, query, k=5):
        """Stage 1 for papers: directory cards (doc_id + title + one line)."""
        cards = []
        for i, score in self.paper_index.search(query, k):
            paper = self.papers[self.paper_ids[i]]
            cards.append({
                "doc_id": paper["doc_id"],
                "title": paper["title"],
                "journal": paper["journal"],
                "year": paper["year"],
                "snippet": paper["key_finding"][:160],
                "score": round(score, 3),
            })
        return cards

    def read_paper(self, doc_id, start_line=1, lines_per_page=3):
        """Stage 2: one window of a paper, with the map (start/end/total).

        HW1 `read_doc` contract, copied: 1-based lines, at least one line,
        and the caller learns exactly where the next window starts.
        Raises ValueError for an unknown doc_id or bad start_line — the
        registry turns that into a tool_error the model can recover from.
        """
        paper = self.papers.get(str(doc_id))
        if paper is None:
            raise ValueError(f"unknown doc_id: {doc_id!r}")
        lines = [
            f"Title: {paper['title']}",
            f"Authors: {paper['authors']}",
            f"Journal: {paper['journal']} ({paper['year']})",
            f"Key finding: {paper['key_finding']}",
            f"Backs claim(s): {', '.join(paper['claim_ids'])}",
            f"Source: {paper['link']}",
        ]
        total = len(lines)
        if start_line < 1 or start_line > total:
            raise ValueError(f"start_line must be 1..{total}, got {start_line}")
        picked = lines[start_line - 1:start_line - 1 + lines_per_page]
        return {
            "doc_id": paper["doc_id"],
            "start_line": start_line,
            "end_line": start_line - 1 + len(picked),
            "total_lines": total,
            "text": "\n".join(picked),
        }
