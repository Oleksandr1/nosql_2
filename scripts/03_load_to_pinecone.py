# scripts/03_load_to_pinecone.py
# Запуск: python scripts/03_load_to_pinecone.py
# Ідемпотентний: upsert з тими самими id перезаписує вектори, а не дублює їх.
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (EMBEDDINGS_PATH, PARQUET_PATH, VECTOR_DIM,
                    get_or_create_index, pinecone_client, wait_for_count)

INDEX_NAME = "arxiv-papers"
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів (ліміт запиту — 2 МБ)
# Ембеддинги нормалізовані → cosine == dotproduct за ранжуванням.
# Беремо cosine: коректний навіть якщо колись завантажимо ненормалізовані вектори.
METRIC = "cosine"

pc = pinecone_client()

# 1. Індекс
index = get_or_create_index(pc, INDEX_NAME, dim=VECTOR_DIM, metric=METRIC)

# 2. Дані
df = pd.read_parquet(PARQUET_PATH)
embeddings = np.load(EMBEDDINGS_PATH)
assert len(df) == len(embeddings), "Кількість рядків parquet і ембеддингів не збігається"
assert embeddings.shape[1] == VECTOR_DIM, f"Очікується розмірність {VECTOR_DIM}"
print(f"Статей: {len(df)}, ембеддинги: {embeddings.shape}")


# 3. Підготовка записів
def make_vector(i: int, row) -> dict:
    return {
        "id": f"paper_{i}",                       # номер рядка в parquet
        "values": embeddings[i].tolist(),
        "metadata": {
            "arxiv_id": str(row.id),
            "title": row.title,
            "abstract": row.abstract[:500],       # ліміт метаданих — 40 KB на вектор
            "authors": row.authors[:200],
            "year": int(row.year),                # число → працюють фільтри $gte/$lt
            "category": row.category,
        },
    }


# 4. Батчеве завантаження з прогресом
rows = list(df.itertuples(index=False))
for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Upsert у Pinecone"):
    batch = [make_vector(i, rows[i]) for i in range(start, min(start + BATCH_SIZE, len(rows)))]
    index.upsert(vectors=batch)

# 5. Перевірка
total = wait_for_count(index, expected=len(df))
print(f"\nВекторів в індексі '{INDEX_NAME}': {total}")
print(index.describe_index_stats())
