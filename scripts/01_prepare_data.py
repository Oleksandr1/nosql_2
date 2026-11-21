# scripts/01_prepare_data.py
# Запуск: python scripts/01_prepare_data.py
#
# ЗМІНИ ВІДНОСНО ШАБЛОНУ З УМОВИ (пояснення — README, розділ «Відхилення від шаблону»):
# 1. Шаблон брав ПЕРШІ 10 000 рядків файлу. Снепшот відсортований за arXiv ID,
#    тому це статті квітня–травня 2007 року, майже всі з фізики/математики.
#    Фільтр «останні 5 років + cs.LG» на такій вибірці повертає порожній результат.
#    Тут — випадкова стратифікована вибірка по всьому файлу (reservoir sampling).
# 2. Шаблон робив .replace("\\n", " ") — тобто шукав ДВА символи «\» і «n».
#    Після json.loads у тексті справжні переводи рядка, тож заміна нічого не робила.
#    Тут — нормалізація всіх пробільних символів через " ".join(s.split()).
# 3. Між прізвищем та ініціалами автора додано пробіл ("Balázs C.", а не "BalázsC.").
# 4. Відкидаються відкликані статті (анотація «This paper has been withdrawn…»).
import json
import os
import random

import pandas as pd
from tqdm import tqdm

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"
MAX_RECORDS = 10_000
CS_SHARE = 0.8                      # частка статей із CS / stat.ML
SEED = 42
# Гарантовано додаємо статті цих авторів (для демонстрації пошуку за автором у Частині 5)
MUST_INCLUDE_AUTHORS = ["LeCun"]
MUST_INCLUDE_LIMIT = 30

os.makedirs("data", exist_ok=True)
rng = random.Random(SEED)


def clean(text: str) -> str:
    return " ".join(str(text).split())


def extract_year(paper: dict) -> int:
    """
    Рік першої версії статті — дата публікації на arXiv.
    update_date — дата останнього оновлення, може бути на роки пізніше.
    Формат created: "Mon, 2 Apr 2007 19:18:42 GMT"
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            return int(versions[0]["created"].split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """authors_parsed → "Прізвище І., Прізвище І." (не більше 10 авторів)."""
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:
            last = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return clean(paper.get("authors", ""))


def is_cs(category: str) -> bool:
    return category.startswith("cs.") or category == "stat.ML"


def to_record(paper: dict, title: str, abstract: str, category: str) -> dict:
    return {
        "id": paper["id"],
        "title": title,
        "abstract": abstract,
        "authors": format_authors(paper),
        "year": extract_year(paper),
        "category": category,
    }


n_cs_target = int(MAX_RECORDS * CS_SHARE)
n_other_target = MAX_RECORDS - n_cs_target

must_include = []
reservoir = {"cs": [], "other": []}
seen = {"cs": 0, "other": 0}
targets = {"cs": n_cs_target, "other": n_other_target}
skipped = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Читаємо датасет", unit=" рядків"):
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = clean(paper.get("abstract", ""))
        title = clean(paper.get("title", ""))
        if not abstract or not title or "paper has been withdrawn" in abstract.lower():
            skipped += 1
            continue

        # categories може містити кілька категорій через пробіл: "cs.LG cs.AI" — беремо першу
        category = (paper.get("categories") or "unknown").split()[0]

        if len(must_include) < MUST_INCLUDE_LIMIT and any(
            a in paper.get("authors", "") for a in MUST_INCLUDE_AUTHORS
        ):
            must_include.append(to_record(paper, title, abstract, category))
            continue

        # Reservoir sampling (Algorithm R): рівноймовірна вибірка без читання файлу в пам'ять
        bucket = "cs" if is_cs(category) else "other"
        seen[bucket] += 1
        res, k = reservoir[bucket], targets[bucket]
        if len(res) < k:
            res.append(to_record(paper, title, abstract, category))
        else:
            j = rng.randrange(seen[bucket])
            if j < k:
                res[j] = to_record(paper, title, abstract, category)

# must_include займають місце випадкових CS-статей, щоб загальний розмір лишився MAX_RECORDS
cs_sample = reservoir["cs"][: max(0, n_cs_target - len(must_include))]
records = must_include + cs_sample + reservoir["other"]
rng.shuffle(records)

df = pd.DataFrame(records).drop_duplicates(subset="id").reset_index(drop=True)

print(f"\nПереглянуто статей: CS={seen['cs']:,}, інші={seen['other']:,}, пропущено={skipped:,}")
print(f"Завантажено статей: {len(df)} (з них за MUST_INCLUDE_AUTHORS: {len(must_include)})")
print("\nРозподіл за категоріями (топ-10):")
print(df["category"].value_counts().head(10))
print("\nРозподіл за роками (останні 10):")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nДовжина анотації, слів: медіана={df['abstract'].str.split().str.len().median():.0f}, "
      f"макс={df['abstract'].str.split().str.len().max()}")
print("\nПриклад запису:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nЗбережено в {OUTPUT_FILE}")
