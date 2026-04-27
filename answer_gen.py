"""
Answer generation.
- Primary: LLM via OpenAI-compatible endpoint (works with OpenAI, Groq, Together, OpenRouter).
- Fallback: extractive (top sentences by TF-IDF similarity to the query).

Constraint per spec: answers are grounded in retrieved chunks and we display
supporting evidence (chunk + page).
"""
import os
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extractive_answer(query, chunks, max_sentences: int = 3):
    """Pick the top-N sentences from retrieved chunks most similar to the query."""
    sents = []
    for c in chunks:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            if len(s.split()) > 5:
                sents.append({"text": s.strip(), "page": c["page"], "chunk_id": c["id"]})
    if not sents:
        return "No relevant content found in the handbook."

    vec = TfidfVectorizer(stop_words="english").fit([s["text"] for s in sents] + [query])
    M = vec.transform([s["text"] for s in sents])
    q = vec.transform([query])
    sims = cosine_similarity(q, M).flatten()
    top = np.argsort(-sims)[:max_sentences]
    chosen = sorted(top.tolist())
    return " ".join(sents[i]["text"] for i in chosen)


def llm_answer(
    query,
    chunks,
    api_key=None,
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
):
    """Generate a grounded answer with an OpenAI-compatible LLM. Returns None on failure."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(
        f"[Chunk {i+1} | Page {c['page']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"""You answer questions about a university student handbook.

RULES:
- Use ONLY the context below.
- If the context does not contain the answer, reply: "Not found in the handbook."
- Cite chunks inline like [Chunk 1] or [Chunk 1, 3].
- Be concise (3-5 sentences max).

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 400,
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"[LLM] HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[LLM] error: {e}")
    return None


def generate_answer(query, chunks, use_llm=True, api_key=None, model="gpt-4o-mini",
                    base_url="https://api.openai.com/v1"):
    if use_llm:
        a = llm_answer(query, chunks, api_key=api_key, model=model, base_url=base_url)
        if a:
            return {"answer": a, "method": "llm"}
    return {"answer": extractive_answer(query, chunks), "method": "extractive"}
