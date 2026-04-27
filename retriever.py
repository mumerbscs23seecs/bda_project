"""
Hybrid LSH retriever: MinHash+LSH (recall) UNION SimHash (precision),
combined with weighted score. This is the 'core' retrieval system per spec.
"""
from minhash_lsh import MinHashLSHRetriever
from simhash_module import SimHashRetriever


class HybridRetriever:
    def __init__(
        self,
        num_perm: int = 128,
        lsh_threshold: float = 0.2,
        simhash_bits: int = 64,
        simhash_threshold: int = 28,
        mh_weight: float = 0.6,
        sh_weight: float = 0.4,
    ):
        self.mh = MinHashLSHRetriever(num_perm=num_perm, threshold=lsh_threshold)
        self.sh = SimHashRetriever(bits=simhash_bits, threshold=simhash_threshold)
        self.mh_weight = mh_weight
        self.sh_weight = sh_weight

    def build(self, chunks):
        t1 = self.mh.build(chunks)
        t2 = self.sh.build(chunks)
        return {"minhash_build_s": t1, "simhash_build_s": t2}

    def query(self, q_text: str, top_k: int = 5):
        mh_res = self.mh.query(q_text, top_k=top_k * 3)
        sh_res = self.sh.query(q_text, top_k=top_k * 3)

        bag = {}
        for r in mh_res:
            bag[r["id"]] = {"chunk": r, "mh": r["score"], "sh": 0.0}
        for r in sh_res:
            if r["id"] in bag:
                bag[r["id"]]["sh"] = r["score"]
            else:
                bag[r["id"]] = {"chunk": r, "mh": 0.0, "sh": r["score"]}

        merged = []
        for cid, v in bag.items():
            score = self.mh_weight * v["mh"] + self.sh_weight * v["sh"]
            merged.append({
                **v["chunk"],
                "score": score,
                "mh_score": v["mh"],
                "sh_score": v["sh"],
            })
        merged.sort(key=lambda x: -x["score"])
        return merged[:top_k]
