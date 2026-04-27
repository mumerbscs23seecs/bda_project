"""
MinHash + LSH retriever.
- Document = set of word-level k-shingles (default k=5)
- MinHash signature with `num_perm` permutations
- Bands auto-configured by datasketch.MinHashLSH from threshold + num_perm
"""
import re
import time
from datasketch import MinHash, MinHashLSH


def shingle(text: str, k: int = 3):  # was 5, now 3
    """Word-level k-shingles. For short queries, also adds character 4-grams."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    words = text.split()
    word_shingles = set()
    if len(words) >= k:
        word_shingles = {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}
    elif words:
        word_shingles = {" ".join(words)}
    # Always add character 4-grams as fallback for short text
    clean = re.sub(r"\s+", " ", text)
    char_shingles = {clean[i:i+4] for i in range(len(clean) - 3)} if len(clean) >= 4 else set()
    return word_shingles | char_shingles


class MinHashLSHRetriever:
    def __init__(self, num_perm: int = 128, threshold: float = 0.1, k_shingle: int = 3):
        self.num_perm = num_perm
        self.threshold = threshold
        self.k_shingle = k_shingle
        self.lsh = None
        self.minhashes = {}
        self.chunks = []

    def _mh(self, text):
        m = MinHash(num_perm=self.num_perm)
        for s in shingle(text, self.k_shingle):
            m.update(s.encode("utf-8"))
        return m

    def build(self, chunks):
        self.chunks = chunks
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        t0 = time.time()
        for c in chunks:
            m = self._mh(c["text"])
            self.minhashes[c["id"]] = m
            self.lsh.insert(str(c["id"]), m)
        return time.time() - t0

    def query(self, q_text: str, top_k: int = 5):
        qm = self._mh(q_text)
        candidates = self.lsh.query(qm)
        if not candidates:
            # Backoff: temporarily widen threshold by re-querying via direct estimate
            # (in practice, queries are short; if no band collision, return empty)
            return []
        scored = []
        for cid in candidates:
            cid_int = int(cid)
            sim = qm.jaccard(self.minhashes[cid_int])
            scored.append((cid_int, sim))
        scored.sort(key=lambda x: -x[1])
        results = []
        for cid, sim in scored[:top_k]:
            ch = self.chunks[cid]
            results.append({**ch, "score": float(sim)})
        return results
