"""
Extension: Frequent Itemset Mining (Apriori) over the query log.
Identifies common patterns in user questions — e.g., "{gpa, minimum, requirement}"
or "{attendance, policy}". This is a Big Data course topic per spec.
"""
import re
from collections import Counter

STOP = {
    "the","a","an","is","are","what","how","do","does","can","i","of","for","to",
    "in","on","and","or","if","this","my","your","there","be","with","at","it","as",
    "by","that","not","will","would","should","could","have","has","had","you","me",
    "we","they","them","us","but","so","than","then","also","just","about","when",
    "where","which","who","why",
}


def normalize_query(q: str):
    q = re.sub(r"[^\w\s]", " ", q.lower())
    return [w for w in q.split() if w not in STOP and len(w) > 2]


def apriori(transactions, min_support: int = 2, max_k: int = 3):
    """Vanilla Apriori. Returns dict: frozenset -> support count."""
    counts = Counter()
    for t in transactions:
        for item in set(t):
            counts[item] += 1
    L1 = {frozenset([i]): c for i, c in counts.items() if c >= min_support}

    all_freq = dict(L1)
    Lk = L1
    for k in range(2, max_k + 1):
        Ck = set()
        keys = list(Lk.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                u = keys[i] | keys[j]
                if len(u) == k:
                    Ck.add(u)
        Lk = {}
        for cand in Ck:
            c = sum(1 for t in transactions if cand.issubset(set(t)))
            if c >= min_support:
                Lk[cand] = c
        all_freq.update(Lk)
        if not Lk:
            break
    return all_freq


def find_query_patterns(query_log, min_support: int = 2, max_k: int = 3):
    transactions = [normalize_query(q) for q in query_log]
    transactions = [t for t in transactions if t]
    if not transactions:
        return []
    freq = apriori(transactions, min_support=min_support, max_k=max_k)
    out = sorted(freq.items(), key=lambda x: (-len(x[0]), -x[1]))
    return [(set(k), v) for k, v in out]


if __name__ == "__main__":
    demo = [
        "What is the minimum GPA requirement?",
        "What is the GPA scale?",
        "What is the attendance policy?",
        "What is the policy on attendance?",
        "What is the minimum attendance?",
        "How many courses can I repeat?",
        "What happens if I fail a course?",
    ]
    for items, sup in find_query_patterns(demo, min_support=2, max_k=3):
        if len(items) >= 2:
            print(f"support={sup}: {sorted(items)}")
