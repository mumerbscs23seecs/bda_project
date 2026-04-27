"""
SimHash retriever (from-scratch implementation).
- Each chunk -> 64-bit fingerprint via weighted token hashing
- Retrieval = chunks within Hamming distance <= threshold from the query fingerprint
"""
import re
import time
import hashlib
from collections import Counter


def tokenize(text: str):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


def _hash_token(token: str, bits: int = 64) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h, 16) & ((1 << bits) - 1)


def simhash(text: str, bits: int = 64) -> int:
    tokens = tokenize(text)
    if not tokens:
        return 0
    weights = Counter(tokens)
    v = [0] * bits
    for token, w in weights.items():
        h = _hash_token(token, bits)
        for i in range(bits):
            if h & (1 << i):
                v[i] += w
            else:
                v[i] -= w
    fp = 0
    for i in range(bits):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class SimHashRetriever:
    """
    Linear-scan SimHash retrieval. For very large corpora, a banded
    LSH-on-fingerprints could be added, but at typical handbook sizes
    (hundreds–thousands of chunks) a linear scan is sub-millisecond.
    """

    def __init__(self, bits: int = 64, threshold: int = 28):
        self.bits = bits
        self.threshold = threshold
        self.fingerprints = {}
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        t0 = time.time()
        for c in chunks:
            self.fingerprints[c["id"]] = simhash(c["text"], self.bits)
        return time.time() - t0

    def query(self, q_text: str, top_k: int = 5):
        qf = simhash(q_text, self.bits)
        scored = []
        for c in self.chunks:
            d = hamming(qf, self.fingerprints[c["id"]])
            if d <= self.threshold:
                scored.append((c["id"], 1 - d / self.bits, d))
        scored.sort(key=lambda x: -x[1])
        results = []
        for cid, sim, d in scored[:top_k]:
            results.append({**self.chunks[cid], "score": float(sim), "hamming": d})
        return results
