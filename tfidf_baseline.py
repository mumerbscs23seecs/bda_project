"""
Baseline retriever: TF-IDF + cosine similarity.
This is the *exact* retrieval method we compare LSH against.
"""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFRetriever:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        t0 = time.time()
        self.matrix = self.vectorizer.fit_transform(texts)
        return time.time() - t0

    def query(self, q_text: str, top_k: int = 5):
        qv = self.vectorizer.transform([q_text])
        sims = cosine_similarity(qv, self.matrix).flatten()
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            results.append({**self.chunks[int(i)], "score": float(sims[i])})
        return results
