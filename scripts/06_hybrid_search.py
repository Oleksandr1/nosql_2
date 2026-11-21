# scripts/06_hybrid_search.py
# Запуск: python scripts/06_hybrid_search.py
import re

import pandas as pd
from rank_bm25 import BM25Okapi

from common import (PARQUET_PATH, get_matches, load_model, pinecone_client,
                    row_from_id, short)

INDEX_NAME = "arxiv-papers"
TOP_K = 10        # беремо ширше, щоб RRF міг переранжувати
SHOW = 5
RRF_K = 60        # стандартне значення з оригінальної статті про RRF (Cormack et al., 2009)
# Автори додані в BM25-корпус: інакше запит за ім'ям автора для BM25 безглуздий
# (імені немає ні в заголовку, ні в анотації).
BM25_FIELDS = ("title", "abstract", "authors")

QUERIES = [
    ("Точний термін", "BERT fine-tuning"),
    ("Ім'я автора", "Yann LeCun convolutional networks"),
    ("Перефразування", "making computers understand human emotions from text"),
]

pc = pinecone_client()
index = pc.Index(INDEX_NAME)
model = load_model()
df = pd.read_parquet(PARQUET_PATH).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 1. BM25-індекс
# ---------------------------------------------------------------------------
STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or that
the this to was were with we our which can these using based""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    # Дефіс розбиває «fine-tuning» → fine, tuning, і так само в запиті — симетрично
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


corpus = [tokenize(" ".join(str(r[f]) for f in BM25_FIELDS)) for _, r in df.iterrows()]
bm25 = BM25Okapi(corpus)
print(f"BM25-індекс: {len(corpus)} документів, поля: {', '.join(BM25_FIELDS)}")


# ---------------------------------------------------------------------------
# 4. Три методи пошуку → список (row, score), відсортований за спаданням
# ---------------------------------------------------------------------------
def bm25_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    top = scores.argsort()[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    res = index.query(vector=q.tolist(), top_k=top_k, include_metadata=False)
    return [(row_from_id(vec_id), score) for vec_id, score, _ in get_matches(res)]


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion
#    RRF(d) = Σ_m 1 / (k + rank_m(d)),  rank починається з 1.
#    Використовує лише РАНГИ: BM25-скор (0..∞) і cosine (−1..1) непорівнянні напряму.
# ---------------------------------------------------------------------------
def rrf(ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for results in ranked_lists:
        for rank, (doc, _) in enumerate(results, 1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k: int = TOP_K, k: int = RRF_K):
    b, v = bm25_search(query, top_k), vector_search(query, top_k)
    return rrf([b, v], k=k)[:top_k], b, v


# ---------------------------------------------------------------------------
# 5–6. Демонстрація
# ---------------------------------------------------------------------------
def fmt(row: int) -> str:
    r = df.iloc[row]
    return f"{short(r['title'], 70)} ({r['category']}, {r['year']})"


summary = []
for label, query in QUERIES:
    print("\n" + "=" * 78)
    print(f"{label}: «{query}»")
    print("=" * 78)

    fused, b, v = hybrid_search(query)
    b_rows = {d for d, _ in b[:SHOW]}
    v_rows = {d for d, _ in v[:SHOW]}
    b_rank = {d: i for i, (d, _) in enumerate(b, 1)}
    v_rank = {d: i for i, (d, _) in enumerate(v, 1)}

    print("\n[BM25] топ-5")
    for i, (d, s) in enumerate(b[:SHOW], 1):
        print(f"  {i}. {s:7.3f} | {fmt(d)}")
    if not b:
        print("  (жодного документа зі словами запиту)")

    print("\n[Vector / Pinecone] топ-5")
    for i, (d, s) in enumerate(v[:SHOW], 1):
        print(f"  {i}. {s:7.4f} | {fmt(d)}")

    print(f"\n[Hybrid RRF, k={RRF_K}] топ-5")
    for i, (d, s) in enumerate(fused[:SHOW], 1):
        tag = "" if d in b_rows or d in v_rows else "  ← немає в топ-5 жодного методу"
        print(f"  {i}. {s:.5f} (BM25 #{b_rank.get(d, '—')}, vec #{v_rank.get(d, '—')}) | {fmt(d)}{tag}")

    # Чутливість до k
    fused_k1 = rrf([b, v], k=1)[:SHOW]
    print("\n[Hybrid RRF, k=1] топ-5")
    for i, (d, s) in enumerate(fused_k1, 1):
        print(f"  {i}. {s:.5f} (BM25 #{b_rank.get(d, '—')}, vec #{v_rank.get(d, '—')}) | {fmt(d)}")
    same = [d for d, _ in fused[:SHOW]] == [d for d, _ in fused_k1]
    print(f"  Порядок топ-5 при k=60 і k=1 однаковий: {same}")

    summary.append((label, query,
                    fmt(b[0][0]) if b else "—",
                    fmt(v[0][0]) if v else "—",
                    fmt(fused[0][0]) if fused else "—",
                    len(b_rows & v_rows)))

print("\n\nПорівняльна таблиця для README (топ-1 кожного методу):\n")
print("| Тип запиту | Запит | BM25 топ-1 | Vector топ-1 | Hybrid топ-1 | Спільних у топ-5 BM25∩Vector |")
print("|---|---|---|---|---|---|")
for row in summary:
    print("| " + " | ".join(str(x).replace("|", "/") for x in row) + " |")
