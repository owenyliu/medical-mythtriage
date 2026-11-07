"""BM25 search — adapted from Owen's CS329Z HW1 `pipelines/bm25.py`.

Same formula, same constants (K1, B), same inverted-index idea:
a word -> {which docs contain it, how many times}. One change: the HW1
course tokenizer only kept a-z0-9, which deletes every Chinese character.
Our claims are half Chinese, so `tokenize` here also emits each CJK
character plus CJK bigrams (a cheap stand-in for Chinese word splitting).
"""

import math
import re

K1 = 1.5
B = 0.75

_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text):
    """Lowercase `text`; return latin words + CJK unigrams/bigrams."""
    text = str(text).lower()
    tokens = _LATIN.findall(text)
    for run in _CJK.findall(text):
        tokens.extend(run)  # single characters
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))  # bigrams
    return tokens


class BM25Index:
    """An inverted index over a list of document strings (HW1, unchanged logic)."""

    def __init__(self, docs):
        self.n = len(docs)
        self.doc_len = []
        self.postings = {}  # term -> {doc_index: term frequency}
        total = 0
        for doc_id, doc in enumerate(docs):
            tokens = tokenize(doc)
            self.doc_len.append(len(tokens))
            total += len(tokens)
            counts = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                self.postings.setdefault(tok, {})[doc_id] = tf
        self.avgdl = total / self.n if self.n else 0.0

    def search(self, query, k):
        """Return top-k [(doc_index, score)], best first; ties -> lower index."""
        if k <= 0 or self.n == 0:
            return []
        terms = set(tokenize(query))
        if not terms:
            return []
        scores = {}
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in posting.items():
                dl = self.doc_len[doc_id]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl) if self.avgdl else tf + K1
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
        return [(int(doc_id), float(score)) for doc_id, score in ranked[:k]]


def build_index(docs):
    return BM25Index(list(docs))
