# scripts/common.py
# Спільні налаштування й хелпери для скриптів 02–06.
import os
import time

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
PARQUET_PATH = "data/arxiv_subset.parquet"
EMBEDDINGS_PATH = "embeddings/embeddings.npy"
SEP = " [SEP] "

# Pinecone Starter: усі serverless-індекси лише в AWS us-east-1
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


def load_model():
    """
    allenai/specter2_base — «сирий» BERT без конфігурації sentence-transformers.
    Якщо завантажити його як SentenceTransformer(MODEL_NAME), бібліотека
    мовчки додасть MEAN pooling. Картка моделі ж бере ембеддинг першого токена
    ([CLS]: last_hidden_state[:, 0, :]), тому збираємо модель явно з CLS pooling.
    Однакова функція для документів і запитів — інакше вектори несумісні.
    """
    from sentence_transformers import SentenceTransformer, models

    transformer = models.Transformer(MODEL_NAME, max_seq_length=512)
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode="cls",
    )
    return SentenceTransformer(modules=[transformer, pooling])


def paper_text(title: str, abstract: str) -> str:
    """Формат входу, на якому навчена SPECTER2: title [SEP] abstract."""
    return f"{title}{SEP}{abstract}"


def get_or_create_index(pc, name: str, dim: int = VECTOR_DIM, metric: str = "cosine"):
    """Створює serverless-індекс, якщо його немає, чекає готовності, повертає Index."""
    from pinecone import ServerlessSpec

    existing = pc.list_indexes().names()
    if name not in existing:
        print(f"Створюємо індекс '{name}' (dim={dim}, metric={metric})...")
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        print(f"Індекс '{name}' вже існує — використовуємо його.")

    # Індекс стає доступним не миттєво
    for _ in range(60):
        if _is_ready(pc.describe_index(name)):
            break
        time.sleep(2)

    desc = pc.describe_index(name)
    print(f"Індекс '{name}': metric={desc.metric}, dimension={desc.dimension}")
    return pc.Index(name)


def _is_ready(desc) -> bool:
    """pinecone-client 4.x повертає status як dict, новіші SDK — як об'єкт."""
    status = desc.status
    try:
        return bool(status["ready"])
    except (TypeError, KeyError):
        return bool(getattr(status, "ready", False))


def _stat(stats, key):
    try:
        return stats[key]
    except (TypeError, KeyError):
        return getattr(stats, key)


def wait_for_count(index, expected: int, timeout_s: int = 90) -> int:
    """
    Pinecone serverless — eventually consistent: після upsert статистика
    оновлюється із затримкою. Чекаємо, поки total_vector_count дожене очікуване.
    """
    count = 0
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        count = _stat(index.describe_index_stats(), "total_vector_count")
        if count >= expected:
            break
        time.sleep(3)
    return count


def short(text: str, n: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def row_from_id(vec_id: str) -> int:
    """'paper_123' → 123 (номер рядка в parquet)."""
    return int(vec_id.split("_")[1])


def pinecone_client():
    from dotenv import load_dotenv
    from pinecone import Pinecone

    load_dotenv()
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_matches(res):
    """Уніфікований доступ до результатів query: [(id, score, metadata), ...]."""
    matches = _stat(res, "matches")
    out = []
    for m in matches:
        # При include_metadata=False pinecone-client 4.x взагалі не має поля metadata
        # і кидає PineconeApiAttributeError (підклас AttributeError)
        try:
            md = _stat(m, "metadata")
        except AttributeError:
            md = None
        out.append((_stat(m, "id"), float(_stat(m, "score")), dict(md or {})))
    return out
