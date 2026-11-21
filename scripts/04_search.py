# scripts/04_search.py
# Запуск: python scripts/04_search.py
import numpy as np
import pandas as pd

from common import (EMBEDDINGS_PATH, PARQUET_PATH, get_matches, load_model,
                    paper_text, pinecone_client, row_from_id, short)

INDEX_NAME = "arxiv-papers"
TOP_K = 5

pc = pinecone_client()
index = pc.Index(INDEX_NAME)
model = load_model()
df = pd.read_parquet(PARQUET_PATH)  # для отримання повного abstract


def header(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 2. Кодування запиту — тією самою моделлю й тією самою нормалізацією, що й документи
# ---------------------------------------------------------------------------
def encode_query(text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]


def search(query: str, top_k: int = TOP_K, flt: dict | None = None):
    res = index.query(
        vector=encode_query(query).tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=flt,
    )
    return get_matches(res)


def print_results(results):
    if not results:
        print("  (порожньо)")
    for rank, (vec_id, score, md) in enumerate(results, 1):
        full_abstract = df.iloc[row_from_id(vec_id)]["abstract"]   # повний текст — з parquet
        print(f"{rank}. [{score:.4f}] {md['title']}")
        print(f"   {md['category']} | {md['year']} | arXiv:{md['arxiv_id']}")
        print(f"   {short(full_abstract, 220)}")


# ---------------------------------------------------------------------------
# 3. Чистий семантичний пошук
# ---------------------------------------------------------------------------
QUERY_SEMANTIC = "teaching machines to recognize objects in pictures"
header(f"3. Семантичний пошук: «{QUERY_SEMANTIC}»")
print_results(search(QUERY_SEMANTIC))

# ---------------------------------------------------------------------------
# 4. Пошук із фільтрацією метаданих
# ---------------------------------------------------------------------------
QUERY_RL = "reinforcement learning agents learning policies from rewards"
max_year = int(df["year"].max())
year_from = max_year - 4          # «останні 5 років» відносно найсвіжішої статті у вибірці

filter_a = {"category": {"$eq": "cs.LG"}, "year": {"$gte": year_from}}
filter_b = {"year": {"$lt": 2015}}

header(f"4A. «{QUERY_RL}» | cs.LG, {year_from}–{max_year}")
res_a = search(QUERY_RL, flt=filter_a)
print_results(res_a)

header(f"4B. «{QUERY_RL}» | до 2015 року, будь-яка категорія")
res_b = search(QUERY_RL, flt=filter_b)
print_results(res_b)

header("4C. Той самий запит без фільтра (для порівняння)")
res_c = search(QUERY_RL)
print_results(res_c)

ids_a, ids_b, ids_c = ({r[0] for r in x} for x in (res_a, res_b, res_c))
print("\nПорівняння:")
print(f"  Перетин A∩B: {len(ids_a & ids_b)} | A∩без_фільтра: {len(ids_a & ids_c)} | "
      f"B∩без_фільтра: {len(ids_b & ids_c)}")
for name, res in (("A", res_a), ("B", res_b), ("без фільтра", res_c)):
    if res:
        scores = [r[1] for r in res]
        cats = pd.Series([r[2]["category"] for r in res]).value_counts().to_dict()
        print(f"  {name}: score {min(scores):.4f}–{max(scores):.4f}, категорії {cats}")
pool_a = df[(df["category"] == "cs.LG") & (df["year"] >= year_from)]
pool_b = df[df["year"] < 2015]
print(f"  Розмір пулу кандидатів: A={len(pool_a)}, B={len(pool_b)}, усього={len(df)}")

# ---------------------------------------------------------------------------
# 5. Порівняння метрик на локальних ембеддингах
# ---------------------------------------------------------------------------
header(f"5. Порівняння метрик (локально): «{QUERY_SEMANTIC}»")
E = np.load(EMBEDDINGS_PATH)            # (N, 768), нормалізовані
q = encode_query(QUERY_SEMANTIC)        # (768,), нормалізований


def top_k_by(scores: np.ndarray, k: int = TOP_K, ascending: bool = False) -> np.ndarray:
    order = np.argsort(scores)
    return order[:k] if ascending else order[::-1][:k]


def compare_metrics(E: np.ndarray, q: np.ndarray, label: str, row_ids=None):
    cosine = (E @ q) / (np.linalg.norm(E, axis=1) * np.linalg.norm(q))
    dot = E @ q
    l2 = np.linalg.norm(E - q, axis=1)

    tops = {
        "cosine (↓ більше = ближче)": top_k_by(cosine),
        "dot product (↓ більше = ближче)": top_k_by(dot),
        "L2 distance (↑ менше = ближче)": top_k_by(l2, ascending=True),
    }
    values = {"cosine": cosine, "dot": dot, "l2": l2}
    print(f"\n--- {label} ---")
    for name, idx in tops.items():
        print(f"\n{name}:")
        for rank, i in enumerate(idx, 1):
            row = row_ids[i] if row_ids is not None else i
            print(f"  {rank}. cos={cosine[i]:.4f} dot={dot[i]:.4f} L2={l2[i]:.4f} | "
                  f"{short(df.iloc[row]['title'], 80)}")
    lists = [list(v) for v in tops.values()]
    print(f"\nОднаковий порядок топ-{TOP_K}: cosine==dot → {lists[0] == lists[1]}, "
          f"cosine==L2 → {lists[0] == lists[2]}")
    return values


vals = compare_metrics(E, q, "Нормалізовані ембеддинги (як в індексі)")
# Чисельна перевірка тотожності ||a-b||² = 2 - 2·cos(a,b) для одиничних векторів
max_err = np.max(np.abs(vals["l2"] ** 2 - (2 - 2 * vals["cosine"])))
print(f"max | L2² − (2 − 2·cos) | = {max_err:.2e}")

# Що було б без нормалізації: кодуємо кандидатів заново БЕЗ normalize_embeddings
# (топ-300 за cosine — щоб не перераховувати всі 10 000 текстів).
POOL = 300
pool_rows = top_k_by(vals["cosine"], k=POOL)
pool_texts = [paper_text(df.iloc[i]["title"], df.iloc[i]["abstract"]) for i in pool_rows]
E_raw = model.encode(pool_texts, batch_size=64, normalize_embeddings=False,
                     convert_to_numpy=True, show_progress_bar=True)
q_raw = model.encode([QUERY_SEMANTIC], normalize_embeddings=False, convert_to_numpy=True)[0]
raw_norms = np.linalg.norm(E_raw, axis=1)
print(f"\nНорми ненормалізованих векторів у пулі: min={raw_norms.min():.2f}, "
      f"max={raw_norms.max():.2f}, норма запиту={np.linalg.norm(q_raw):.2f}")
compare_metrics(E_raw, q_raw, f"БЕЗ нормалізації (пул топ-{POOL} за cosine)", row_ids=pool_rows)
