# scripts/02_embed.py
# Запуск: python scripts/02_embed.py
# На CPU ~10–25 хв для 10 000 статей; на GPU — 1–2 хв.
import os

import numpy as np
import pandas as pd

from common import EMBEDDINGS_PATH, PARQUET_PATH, load_model, paper_text

BATCH_SIZE = 64

# 1. Дані
df = pd.read_parquet(PARQUET_PATH)

# 2. Тексти у форматі SPECTER2: title [SEP] abstract
texts = [paper_text(t, a) for t, a in zip(df["title"], df["abstract"])]

# 3. Модель (CLS pooling, див. common.load_model)
model = load_model()
print(f"Модель: {model}")
print(f"Пристрій: {model.device}")

# 4. Кодування: батчі, прогрес-бар, L2-нормалізація
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
).astype(np.float32)

# 5. Перевірки
print(f"\nОброблено текстів: {len(texts)}")
print(f"Розмірність ембеддингів: {embeddings.shape[1]}")
print(f"Норма першого ембеддингу: {np.linalg.norm(embeddings[0]):.6f}")
norms = np.linalg.norm(embeddings, axis=1)
print(f"Норми всіх векторів: min={norms.min():.6f}, max={norms.max():.6f}")

# Скільки текстів довші за 512 токенів (будуть обрізані моделлю) — аргумент для Частини 4
tok = model.tokenizer
lengths = [len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts]
over = sum(l > 512 for l in lengths)
print(f"Текстів довших за 512 токенів (обрізані): {over} ({over / len(texts):.1%}), "
      f"медіана довжини: {int(np.median(lengths))} токенів")

# 6–7. Збереження
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
np.save(EMBEDDINGS_PATH, embeddings)
print(f"\nЗбережено в {EMBEDDINGS_PATH} ({embeddings.nbytes / 1e6:.1f} МБ)")
