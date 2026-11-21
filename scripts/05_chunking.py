# scripts/05_chunking.py
# Запуск: python scripts/05_chunking.py
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (PARQUET_PATH, VECTOR_DIM, get_matches, get_or_create_index,
                    load_model, pinecone_client, short, wait_for_count)

N_PAPERS = 30
FIXED_WORDS = 60          # розмір fixed-чанка, слів
FIXED_OVERLAP = 15        # перекриття, слів (25%)
SEMANTIC_MAX_WORDS = 80   # максимум слів у semantic-чанку
BATCH_SIZE = 100
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
# Тестові запити. Після першого запуску варто підігнати їх під теми 30 статей,
# які скрипт друкує на початку (інакше топ-5 буде просто «найменш нерелевантним»).
QUERIES = [
    "extending a programming language with custom syntax and operators",
    "distributed algorithms for matching and independent sets in graphs",
    "security and reliability of cryptocurrency payments",
]

pc = pinecone_client()
model = load_model()
df = pd.read_parquet(PARQUET_PATH)


def header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. 30 статей із найдовшими анотаціями
# ---------------------------------------------------------------------------
df["n_words"] = df["abstract"].str.split().str.len()
papers = df.sort_values("n_words", ascending=False).head(N_PAPERS).copy()
papers["row"] = papers.index

header(f"1. {N_PAPERS} статей із найдовшими анотаціями")
for r in papers.itertuples():
    print(f"  {r.n_words:4d} слів | {r.category:12s} | {short(r.title, 70)}")


# ---------------------------------------------------------------------------
# 2. Стратегії chunking
# ---------------------------------------------------------------------------
def fixed_chunks(text: str, size: int = FIXED_WORDS, overlap: int = FIXED_OVERLAP) -> list[str]:
    """Вікно фіксованої довжини у словах з перекриттям; межі речень ігноруються."""
    assert 0 <= overlap < size
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


# Кінець речення: . ! ? перед пробілом і великою літерою/цифрою/дужкою.
# Не ріжемо на «e.g. », «i.e. », «et al. », «Fig. » тощо.
_ABBREV = r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bal\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bvs\.)(?<!\bcf\.)"
_SENT_SPLIT = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z0-9(\[$])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def semantic_chunks(text: str, max_words: int = SEMANTIC_MAX_WORDS) -> list[str]:
    """
    Жадібно склеюємо цілі речення, поки не перевищимо max_words.
    Речення, довше за max_words, ріжемо по словах (рідкісний випадок).
    """
    chunks, current, current_len = [], [], 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if n > max_words:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(fixed_chunks(sent, size=max_words, overlap=0))
            continue
        if current_len + n > max_words and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def ends_mid_sentence(chunk: str) -> bool:
    return not chunk.rstrip().endswith((".", "!", "?", ".)", ".]"))


def build_chunks(strategy) -> list[dict]:
    out = []
    for p in papers.itertuples():
        for n, text in enumerate(strategy(p.abstract)):
            out.append({
                "id": f"paper_{p.row}_chunk_{n}",
                "text": text,
                "metadata": {
                    "arxiv_id": str(p.id),
                    "title": p.title,
                    "chunk_text": text,
                    "chunk_num": n,
                    "n_words": len(text.split()),
                    "year": int(p.year),
                    "category": p.category,
                },
            })
    return out


chunks_fixed = build_chunks(fixed_chunks)
chunks_semantic = build_chunks(semantic_chunks)

header("2. Статистика стратегій")
for name, ch in (("fixed", chunks_fixed), ("semantic", chunks_semantic)):
    lens = np.array([c["metadata"]["n_words"] for c in ch])
    cut = sum(ends_mid_sentence(c["text"]) for c in ch)
    print(f"  {name:9s}: чанків={len(ch):4d} | слів у чанку: середнє={lens.mean():.1f}, "
          f"min={lens.min()}, max={lens.max()} | обірвані на півреченні: {cut} ({cut / len(ch):.0%})")

example = papers.iloc[0]
print(f"\nПриклад розбиття: «{short(example.title, 70)}»")
for name, fn in (("FIXED", fixed_chunks), ("SEMANTIC", semantic_chunks)):
    print(f"\n  [{name}]")
    for i, c in enumerate(fn(example.abstract)):
        print(f"   #{i} ({len(c.split())} сл.): {short(c, 160)}")

# Вплив overlap на кількість чанків і покриття
header("2b. Вплив overlap (fixed, size=60) на кількість чанків")
total_words = int(papers["n_words"].sum())
print(f"  Усього слів у {N_PAPERS} анотаціях: {total_words}")
print(f"  {'overlap':>8} | {'чанків':>6} | {'слів у чанках':>13} | {'надлишок':>8}")
for ov in (0, 10, 15, 20, 30, 45):
    ch = [c for a in papers["abstract"] for c in fixed_chunks(a, FIXED_WORDS, ov)]
    stored = sum(len(c.split()) for c in ch)
    print(f"  {ov:>8} | {len(ch):>6} | {stored:>13} | {stored / total_words - 1:>7.0%}")


# ---------------------------------------------------------------------------
# 3–5. Індекси, ембеддинги, батчеве завантаження
# ---------------------------------------------------------------------------
def upload(index_name: str, chunks: list[dict]):
    index = get_or_create_index(pc, index_name, dim=VECTOR_DIM, metric="cosine")
    vectors = model.encode([c["text"] for c in chunks], batch_size=64,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=True)
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Upsert → {index_name}"):
        batch = [
            {"id": c["id"], "values": vectors[i].tolist(), "metadata": c["metadata"]}
            for i, c in enumerate(chunks[start:start + BATCH_SIZE], start)
        ]
        index.upsert(vectors=batch)
    total = wait_for_count(index, expected=len(chunks))
    print(f"Векторів в '{index_name}': {total}")
    return index


header("3–5. Завантаження чанків у Pinecone")
index_fixed = upload(INDEX_FIXED, chunks_fixed)
index_semantic = upload(INDEX_SEMANTIC, chunks_semantic)


# ---------------------------------------------------------------------------
# 6. Пошук по чанках
# ---------------------------------------------------------------------------
def search_chunks(index, query: str, top_k: int = 5):
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    return get_matches(index.query(vector=q.tolist(), top_k=top_k, include_metadata=True))


header("6. Пошук по чанках")
for query in QUERIES:
    print(f"\n>>> Запит: «{query}»")
    for name, idx in (("FIXED", index_fixed), ("SEMANTIC", index_semantic)):
        results = search_chunks(idx, query)
        papers_hit = len({md["arxiv_id"] for _, _, md in results})
        print(f"\n  [{name}] (унікальних статей у топ-5: {papers_hit})")
        for rank, (_, score, md) in enumerate(results, 1):
            print(f"   {rank}. [{score:.4f}] {short(md['title'], 60)} | чанк #{int(md['chunk_num'])}")
            print(f"      {short(md['chunk_text'], 150)}")
