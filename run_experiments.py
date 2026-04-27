"""
Run all required experiments:
  1. Exact (TF-IDF) vs Approximate (MinHash-LSH, SimHash, Hybrid)
  2. Parameter sensitivity (num_perm, LSH threshold, SimHash threshold)
  3. Scalability (corpus duplication 1x, 2x, 5x, 10x)

Saves: experiments_results.json + plots/*.png
"""
import sys
import time
import json
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingestion import ingest
from minhash_lsh import MinHashLSHRetriever
from simhash_module import SimHashRetriever
from tfidf_baseline import TFIDFRetriever
from retriever import HybridRetriever


# 12 test queries with expected keywords (manual ground truth)
TEST_QUERIES = [
    {"q": "What is the minimum GPA requirement?",
     "keywords": ["gpa", "cgpa", "2.0", "minimum", "grade point"]},
    {"q": "What happens if a student fails a course?",
     "keywords": ["fail", "repeat", "retake", "f grade", "failure"]},
    {"q": "What is the attendance policy?",
     "keywords": ["attendance", "absent", "75%", "absence"]},
    {"q": "How many times can a course be repeated?",
     "keywords": ["repeat", "course", "twice", "two", "retake"]},
    {"q": "What is the policy on academic dishonesty?",
     "keywords": ["plagiarism", "cheat", "dishonest", "misconduct", "honesty"]},
    {"q": "What is the grading scale?",
     "keywords": ["grade", "gpa", "letter", "scale", "marks"]},
    {"q": "What are the requirements for graduation?",
     "keywords": ["graduation", "degree", "credit", "requirement"]},
    {"q": "What is the procedure for appealing a grade?",
     "keywords": ["appeal", "grade", "procedure", "review"]},
    {"q": "What are the rules about leave of absence?",
     "keywords": ["leave", "absence", "deferment", "semester freeze"]},
    {"q": "How is the final grade calculated?",
     "keywords": ["final", "grade", "marks", "weight", "calculate"]},
    {"q": "What are the admission requirements?",
     "keywords": ["admission", "requirement", "eligibility", "criteria"]},
    {"q": "What is the fee refund policy?",
     "keywords": ["fee", "refund", "withdraw", "tuition"]},
]


def is_relevant(text: str, keywords) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def precision_at_k(results, keywords, k):
    if not results:
        return 0.0
    rel = sum(1 for r in results[:k] if is_relevant(r["text"], keywords))
    return rel / k


def recall_at_k(results, keywords, all_chunks, k):
    """Recall vs ground truth = chunks containing any keyword (best-effort)."""
    relevant_total = sum(1 for c in all_chunks if is_relevant(c["text"], keywords))
    if relevant_total == 0:
        return 0.0
    retrieved_rel = sum(1 for r in results[:k] if is_relevant(r["text"], keywords))
    return retrieved_rel / relevant_total


def evaluate(retriever, queries, k=5, all_chunks=None):
    Ps, Rs, Ls = [], [], []
    for q in queries:
        t0 = time.time()
        res = retriever.query(q["q"], top_k=k)
        Ls.append((time.time() - t0) * 1000)
        Ps.append(precision_at_k(res, q["keywords"], k))
        if all_chunks is not None:
            Rs.append(recall_at_k(res, q["keywords"], all_chunks, k))
    return {
        "mean_precision": sum(Ps) / len(Ps),
        "precisions": Ps,
        "mean_recall": (sum(Rs) / len(Rs)) if Rs else None,
        "recalls": Rs,
        "mean_latency_ms": sum(Ls) / len(Ls),
        "latencies_ms": Ls,
    }


def measure_memory(build_fn):
    tracemalloc.start()
    out = build_fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1024 / 1024  # MB


# ---------------------------------------------------------------- EXPERIMENT 1
def experiment_1(chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Exact (TF-IDF) vs Approximate (MinHash-LSH, SimHash, Hybrid)")
    print("=" * 70)
    out = {}

    def run(name, builder):
        ret, peak_mb = measure_memory(builder)
        ev = evaluate(ret, TEST_QUERIES, k=5, all_chunks=chunks)
        ev["peak_memory_mb"] = peak_mb
        out[name] = ev
        print(f"\n{name}")
        print(f"  P@5             : {ev['mean_precision']:.3f}")
        print(f"  R@5             : {ev['mean_recall']:.3f}" if ev['mean_recall'] is not None else "")
        print(f"  Latency (ms)    : {ev['mean_latency_ms']:.2f}")
        print(f"  Peak memory (MB): {peak_mb:.2f}")

    def build_tfidf():
        r = TFIDFRetriever(); r.build(chunks); return r
    def build_mh():
        r = MinHashLSHRetriever(num_perm=128, threshold=0.2); r.build(chunks); return r
    def build_sh():
        r = SimHashRetriever(bits=64, threshold=28); r.build(chunks); return r
    def build_hyb():
        r = HybridRetriever(); r.build(chunks); return r

    run("tfidf_exact", build_tfidf)
    run("minhash_lsh", build_mh)
    run("simhash", build_sh)
    run("hybrid_lsh", build_hyb)
    return out


# ---------------------------------------------------------------- EXPERIMENT 2
def experiment_2(chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Parameter sensitivity")
    print("=" * 70)
    out = {"minhash_num_perm": {}, "lsh_threshold": {}, "simhash_threshold": {}}

    print("\n[2a] MinHash num_perm (LSH threshold=0.2):")
    for n in [32, 64, 128, 256]:
        r = MinHashLSHRetriever(num_perm=n, threshold=0.2); r.build(chunks)
        ev = evaluate(r, TEST_QUERIES, k=5, all_chunks=chunks)
        out["minhash_num_perm"][n] = ev
        print(f"  num_perm={n:>3}: P@5={ev['mean_precision']:.3f} "
              f"R@5={ev['mean_recall']:.3f} lat={ev['mean_latency_ms']:.2f}ms")

    print("\n[2b] LSH Jaccard threshold (num_perm=128):")
    for th in [0.1, 0.15, 0.2, 0.3, 0.5]:
        r = MinHashLSHRetriever(num_perm=128, threshold=th); r.build(chunks)
        ev = evaluate(r, TEST_QUERIES, k=5, all_chunks=chunks)
        out["lsh_threshold"][th] = ev
        print(f"  threshold={th:.2f}: P@5={ev['mean_precision']:.3f} "
              f"R@5={ev['mean_recall']:.3f} lat={ev['mean_latency_ms']:.2f}ms")

    print("\n[2c] SimHash Hamming threshold (bits=64):")
    for th in [12, 18, 24, 28, 32, 36]:
        r = SimHashRetriever(bits=64, threshold=th); r.build(chunks)
        ev = evaluate(r, TEST_QUERIES, k=5, all_chunks=chunks)
        out["simhash_threshold"][th] = ev
        print(f"  threshold={th:>2}: P@5={ev['mean_precision']:.3f} "
              f"R@5={ev['mean_recall']:.3f} lat={ev['mean_latency_ms']:.2f}ms")
    return out


# ---------------------------------------------------------------- EXPERIMENT 3
def experiment_3(chunks):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Scalability (corpus duplication 1x, 2x, 5x, 10x)")
    print("=" * 70)
    out = {}
    queries = TEST_QUERIES[:6]  # smaller probe
    for mult in [1, 2, 5, 10]:
        big = []
        for _ in range(mult):
            for c in chunks:
                big.append({**c, "id": len(big)})
        print(f"\nMultiplier {mult}x  →  {len(big)} chunks")
        for name, builder in [
            ("tfidf_exact",
             lambda: (lambda r: (r.build(big), r)[1])(TFIDFRetriever())),
            ("minhash_lsh",
             lambda: (lambda r: (r.build(big), r)[1])(MinHashLSHRetriever(num_perm=128, threshold=0.2))),
            ("simhash",
             lambda: (lambda r: (r.build(big), r)[1])(SimHashRetriever(bits=64, threshold=28))),
        ]:
            t0 = time.time()
            r, peak_mb = measure_memory(builder)
            bt = time.time() - t0
            ev = evaluate(r, queries, k=5)
            entry = {
                "n_chunks": len(big),
                "build_time_s": bt,
                "mean_latency_ms": ev["mean_latency_ms"],
                "peak_memory_mb": peak_mb,
            }
            out.setdefault(name, {})[mult] = entry
            print(f"  {name:<14}: build={bt:5.2f}s  lat={ev['mean_latency_ms']:6.2f}ms  "
                  f"mem={peak_mb:5.1f}MB")
    return out


# ---------------------------------------------------------------- PLOTS
def make_plots(e1, e2, e3, out_dir="plots"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    Path(out_dir).mkdir(exist_ok=True)

    # E1 bar charts: P@5 and latency
    names = list(e1.keys())
    p5 = [e1[n]["mean_precision"] for n in names]
    lat = [e1[n]["mean_latency_ms"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, p5, color=["#444", "#2A6FDB", "#D87C00", "#0BA37F"])
    axes[0].set_title("Precision@5 — Exact vs Approximate")
    axes[0].set_ylabel("P@5"); axes[0].set_ylim(0, 1)
    for i, v in enumerate(p5):
        axes[0].text(i, v + 0.01, f"{v:.2f}", ha="center")
    axes[1].bar(names, lat, color=["#444", "#2A6FDB", "#D87C00", "#0BA37F"])
    axes[1].set_title("Mean query latency (ms)")
    axes[1].set_ylabel("ms")
    for i, v in enumerate(lat):
        axes[1].text(i, v + 0.05, f"{v:.1f}", ha="center")
    plt.tight_layout(); plt.savefig(f"{out_dir}/exp1_exact_vs_approx.png", dpi=140); plt.close()

    # E2: param sensitivity (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    np_vals = sorted(e2["minhash_num_perm"].keys())
    axes[0].plot(np_vals, [e2["minhash_num_perm"][k]["mean_precision"] for k in np_vals],
                 marker="o", label="P@5")
    axes[0].plot(np_vals, [e2["minhash_num_perm"][k]["mean_recall"] for k in np_vals],
                 marker="s", label="R@5")
    axes[0].set_title("MinHash: num_perm sensitivity")
    axes[0].set_xlabel("num_perm"); axes[0].legend()

    th_vals = sorted(e2["lsh_threshold"].keys())
    axes[1].plot(th_vals, [e2["lsh_threshold"][k]["mean_precision"] for k in th_vals],
                 marker="o", label="P@5")
    axes[1].plot(th_vals, [e2["lsh_threshold"][k]["mean_recall"] for k in th_vals],
                 marker="s", label="R@5")
    axes[1].set_title("LSH: Jaccard threshold")
    axes[1].set_xlabel("threshold"); axes[1].legend()

    sh_vals = sorted(e2["simhash_threshold"].keys())
    axes[2].plot(sh_vals, [e2["simhash_threshold"][k]["mean_precision"] for k in sh_vals],
                 marker="o", label="P@5")
    axes[2].plot(sh_vals, [e2["simhash_threshold"][k]["mean_recall"] for k in sh_vals],
                 marker="s", label="R@5")
    axes[2].set_title("SimHash: Hamming threshold")
    axes[2].set_xlabel("threshold"); axes[2].legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/exp2_param_sensitivity.png", dpi=140); plt.close()

    # E3 scalability
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    mults = sorted(next(iter(e3.values())).keys())
    for name, color in [("tfidf_exact", "#444"), ("minhash_lsh", "#2A6FDB"),
                        ("simhash", "#D87C00")]:
        ns = [e3[name][m]["n_chunks"] for m in mults]
        bt = [e3[name][m]["build_time_s"] for m in mults]
        lt = [e3[name][m]["mean_latency_ms"] for m in mults]
        axes[0].plot(ns, bt, marker="o", label=name, color=color)
        axes[1].plot(ns, lt, marker="o", label=name, color=color)
    axes[0].set_title("Build time vs corpus size"); axes[0].set_xlabel("# chunks")
    axes[0].set_ylabel("seconds"); axes[0].legend()
    axes[1].set_title("Query latency vs corpus size"); axes[1].set_xlabel("# chunks")
    axes[1].set_ylabel("ms"); axes[1].legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/exp3_scalability.png", dpi=140); plt.close()
    print(f"\n📊 Plots saved to {out_dir}/")


# ---------------------------------------------------------------- MAIN
if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"
    print(f"Ingesting {pdf} ...")
    chunks = ingest(pdf)
    print(f"-> {len(chunks)} chunks across {len(set(c['page'] for c in chunks))} pages")

    e1 = experiment_1(chunks)
    e2 = experiment_2(chunks)
    e3 = experiment_3(chunks)

    results = {"exact_vs_approx": e1, "param_sensitivity": e2, "scalability": e3}
    with open("experiments_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n✅ experiments_results.json written.")

    make_plots(e1, e2, e3)
    print("\nDone.")
