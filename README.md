# Прізвище_nosql_2 — семантичний пошук за науковими статтями (arXiv + SPECTER2 + Pinecone)

Пайплайн від сирого JSONL arXiv до семантичного пошуку з фільтрами, порівняння метрик, chunking і гібридний пошук BM25 + вектори через RRF.

> Усі блоки виводу — з реального прогону 21.11.2026 (Windows 11, CPU, Python 3.14). Довгі анотації у виводі скорочено («…»).

---

## 1. Налаштування й запуск

Потрібно: Python 3.11+, акаунт [Pinecone](https://www.pinecone.io/) (Starter, безкоштовний), Kaggle API-токен (`~/.kaggle/kaggle.json`), ~6 ГБ вільного місця під датасет.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt     # тягне також torch (~200 МБ CPU-збірка на Windows)

cp .env.example .env                # вписати PINECONE_API_KEY

kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip                     # → arxiv-metadata-oai-snapshot.json (~5.7 ГБ, снепшот 2026 р.)
```

Порядок запуску (строго послідовно, з кореня репозиторію):

| Крок | Команда | Час (CPU-ноутбук) |
|---|---|---|
| 1 | `python scripts/01_prepare_data.py` | ~3–5 хв (читає весь файл) |
| 2 | `python scripts/02_embed.py` | ~10–25 хв |
| 3 | `python scripts/03_load_to_pinecone.py` | ~1–2 хв |
| 4 | `python scripts/04_search.py` | ~1 хв |
| 5 | `python scripts/05_chunking.py` | ~1–2 хв |
| 6 | `python scripts/06_hybrid_search.py` | ~1 хв |

Усі скрипти ідемпотентні: індекси створюються лише якщо їх немає, `upsert` з тими самими `id` перезаписує вектори.

**Сумісність бібліотек.** Якщо `pinecone-client==4.1.0` відмовиться працювати з поточною версією API Pinecone, встановіть новий SDK: `pip uninstall -y pinecone-client && pip install pinecone`. Код написаний так, що працює з обома (див. `_is_ready`, `_stat` у `scripts/common.py`). Якщо `sentence-transformers==3.0.1` падає на імпорті через надто новий `transformers`, допоможе `pip install "transformers==4.41.2"`. У нашому прогоні (Python 3.14) жодне з цього не знадобилось: `pip install -r requirements.txt` підтягнув `transformers` 4.57.6 і `torch` 2.15 (CPU), імпорти й усі скрипти працюють з `pinecone-client==4.1.0`.

### Структура
```
.
├── .env.example            # шаблон; .env не комітиться
├── .gitignore              # .env, data/, embeddings/, датасет
├── requirements.txt
├── data/arxiv_subset.parquet        (генерується, не в git)
├── embeddings/embeddings.npy        (генерується, не в git)
├── scripts/
│   ├── common.py           # модель, формат тексту, хелпери Pinecone
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
└── README.md
```

---

## 2. Відхилення від шаблонів з умови і чому

1. **Вибірка даних.** Шаблон `01_prepare_data.py` бере *перші* 10 000 рядків. Снепшот відсортований за arXiv ID, тому це статті квітня–червня 2007 року, переважно з фізики та математики. На такій вибірці фільтр «cs.LG за останні 5 років» з Частини 3 повертає порожню видачу, а запити про розпізнавання об'єктів і BERT — нерелевантне. Замість цього — **стратифікована випадкова вибірка по всьому файлу** (reservoir sampling, `seed=42`): 80 % CS + stat.ML, 20 % решта категорій. Плюс до 30 статей авторів із `MUST_INCLUDE_AUTHORS = ["LeCun"]`, щоб запит за автором у Частині 5 мав що знаходити.

   **Перевірка на числах** (перші 10 000 рядків JSON, тобто те, що взяв би шаблон):

   | Показник | Значення |
   |---|---|
   | Діапазон arXiv ID | `0704.0001` → `0706.1313` |
   | Роки (перша версія) | **2007 — усі 10 000** |
   | Верхньорівневі галузі (перша категорія) | math 1 955 · astro-ph 1 838 · cond-mat 1 756 · hep-th 680 · hep-ph 671 · physics 663 · quant-ph 564 · **cs лише 490** |
   | Топ-3 категорії | astro-ph 1 838, hep-th 680, hep-ph 671 |
   | Основна категорія cs.LG і рік ≥ 2021 | **0** |

   Тобто фільтр Частини 3 на шаблонній вибірці гарантовано повернув би порожню видачу.
2. **Очищення тексту.** Шаблон робить `.replace("\\n", " ")`, тобто шукає два символи `\` + `n`. Але після `json.loads` у рядку стоять справжні переводи рядка, і заміна нічого не робить. Тут використано `" ".join(text.split())`: нормалізуються всі пробільні символи. Між прізвищем та ініціалами автора додано пробіл. Відкидаються відкликані статті («This paper has been withdrawn»).
3. **CLS pooling.** `SentenceTransformer("allenai/specter2_base")` — це «сирий» BERT, і бібліотека мовчки додає до нього **mean** pooling. У картці моделі ж ембеддинг береться з першого токена (`last_hidden_state[:, 0, :]`, тобто [CLS]). Тому модель збирається явно: `Transformer` + `Pooling(pooling_mode="cls")`, див. `common.load_model()`. Ця функція спільна для документів і запитів, бо інакше вектори несумісні.
4. **BM25 по `title + abstract + authors`.** Без поля авторів запит «Yann LeCun …» для BM25 безглуздий: імені немає ні в назві, ні в анотації.
5. **«Останні 5 років»** рахуються від найсвіжішого року у вибірці (`max_year − 4 … max_year`), а не від системної дати.
6. **Тестові запити Частини 4** (`QUERIES` у `05_chunking.py`) підібрано під теми 30 статей, які скрипт реально відбирає (мови програмування, графові алгоритми, криптоплатежі). Початкові запити (зокрема про квантову заплутаність) не мали жодної цільової статті серед цих 30.
7. **Виправлення в `common.get_matches()`.** `06_hybrid_search.py` робить запит із `include_metadata=False`, і в `pinecone-client` 4.1.0 такий результат узагалі не має поля `metadata` — клієнт кидає `PineconeApiAttributeError` (підклас `AttributeError`), який `_stat()` не перехоплював. Тепер відсутні метадані замінюються на `{}`; логіка пошуку не змінилась.

---

## 3. Частина 1 — дані та вибір інструментів

### Вивід `01_prepare_data.py`
```
Переглянуто статей: CS=877,224, інші=2,373,952, пропущено=1,246
Завантажено статей: 10000 (з них за MUST_INCLUDE_AUTHORS: 30)

Розподіл за категоріями (топ-10):
cs.CV      1563
cs.LG      1369
cs.CL       816
cs.AI       483
cs.RO       449
cs.IT       387
cs.CR       341
cs.HC       212
stat.ML     203
cs.SE       197

Розподіл за роками (останні 10):
2017     296
2018     428
2019     499
2020     600
2021     685
2022     741
2023     824
2024    1111
2025    1322
2026    1592

Довжина анотації, слів: медіана=165, макс=499

Приклад запису:
{'id': '2109.13711', 'title': 'One to rule them all: Towards Joint Indic Language Hate Speech Detection',
 'abstract': 'This paper is a contribution to the Hate Speech and Offensive Content Identification in
 Indo-European Languages (HASOC) 2021 shared task. …', 'authors': 'Bhatia Mehar, Bhotia Tenzin Singhay,
 Agarwal Akshat, Ramesh Prakash, Gupta Shubham, Shridhar Kumar, Laumann Felix, Dash Ayushman',
 'year': 2021, 'category': 'cs.CL'}

Збережено в data/arxiv_subset.parquet
```

Снепшот свіжий, тому найбільша група — статті 2025–2026 років, а «останні 5 років» для фільтра — це 2022–2026.

### 3.1. Pinecone vs Qdrant vs Chroma

| | Pinecone | Qdrant | Chroma |
|---|---|---|---|
| Розгортання | Тільки керований хмарний сервіс (serverless); self-host немає (є лише локальний емулятор для тестів) | Self-host (Docker, Kubernetes, бінарник) або Qdrant Cloud | Вбудований у процес (Python/JS, як SQLite для векторів), client-server або Chroma Cloud |
| Ліцензія | Пропрієтарна, закритий код | Apache 2.0 | Apache 2.0 |
| Ядро | Закрите; розділені сховище й обчислення | Rust, HNSW з фільтрацією по payload, квантування (scalar/binary/product), sparse-вектори | Python/Rust, HNSW; фокус на простоті API |

**Продуктивність.** Універсального переможця немає: результат залежить від розміру даних, фільтрів, вимог до recall і заліза. Публічні бенчмарки здебільшого публікують самі вендори. У Pinecone serverless до кожного запиту додається мережева затримка до хмари, зате масштабування й шардінг не треба налаштовувати. Qdrant на власному залізі дає найбільше контролю: параметри HNSW (`m`, `ef`), квантування, розміщення в RAM чи на диску. Chroma для одного вузла й до кількох мільйонів векторів цілком швидка, але це не розподілена система.

**Коли що обирати:**
- **Pinecone** — команда не хоче супроводжувати інфраструктуру, потрібно швидко вийти в прод і рости без перебудови, а бюджет дозволяє платити за використання.
- **Qdrant** — потрібен контроль над даними (on-prem, вимоги щодо резидентності), складні фільтри разом з векторним пошуком, оптимізація вартості на великих обсягах або open source без vendor lock-in.
- **Chroma** — прототипи, ноутбуки, локальний RAG, юніт-тести пайплайну: `pip install` і жодного сервера.

**Власна думка.** Мій досвід з Elasticsearch опосередкований: сам кластер я не налаштовував, але працював на продукті, де він використовувався, і писав на Python тести, які шукали причину його помилок. Виявилось, що ES падав, бо в нього приходило забагато даних, і використовувався він не зовсім за призначенням. Висновок, який я переношу на векторні БД: інструмент треба обирати під реальний характер навантаження, а обсяг даних на вході контролювати ще до бази. Managed-сервіс на кшталт Pinecone не скасовує обмежень (2 МБ на upsert-запит, 40 KB метаданих на вектор), він лише робить їх явними.

### 3.2. Чому `specter2_base`, а не `all-MiniLM-L6-v2`

`all-MiniLM-L6-v2` — універсальна модель для речень і коротких абзаців. Вона дає 384-вимірні вектори, навчена на загальних парах речень і за замовчуванням обрізає вхід до 256 word pieces. Для наукових анотацій це означає дві проблеми. По-перше, довгі анотації обрізаються. По-друге, модель нічого не знає про те, що дві статті «близькі», бо одна цитує іншу, навіть коли вони вживають різну термінологію.

SPECTER2 зроблена саме під наукові статті. Згідно з карткою моделі, базова модель навчена на понад 6 млн триплетів, побудованих із **цитувань** між статтями: якір → стаття, яку він цитує (позитив) → нецитована стаття (негатив, з тієї самої або іншої галузі). Тобто модель вчилась ставити поруч статті, пов'язані за змістом у науковому сенсі. Вхід — `title [SEP] abstract` до 512 токенів, тобто рівно той формат, який ми подаємо. У картці призначення proximity-ембеддингів описано так: *«Encode papers as queries and candidates eg. Link Prediction, Nearest Neighbor Search.»* Окрім цього, родина SPECTER2 має адаптери під класифікацію, регресію і пошук за короткими запитами (`allenai/specter2_adhoc_query`).

Застереження: наші запити — короткі фрази природною мовою, а не статті. Для такого сценарію картка радить кодувати запити адаптером `adhoc_query`, а документи — proximity-адаптером. Базова модель без адаптерів (як вимагає умова) теж працює, але якість на коротких запитах нижча. **Прогін це підтвердив:** у Частині 3 модель знайшла радше «приблизно схоже», ніж точні влучання, а всі score топ-5 скупчились у вузькому діапазоні 0.907–0.919 (див. 5.1). Найгостріше це видно в Частині 5: на запит «BERT fine-tuning» векторний пошук повернув 5 статей про reinforcement learning і жодної про NLP (див. розділ 7).

### 3.3. Рекомендована метрика схожості

**У картці `specter2_base` метрика схожості явно не вказана.** Є лише код отримання ембеддингів (CLS-токен, без нормалізації). Висновок доводиться робити опосередковано. Оригінальний SPECTER навчався з triplet margin loss на **евклідовій (L2)** відстані між ембеддингами, SPECTER2 успадковує цей підхід. Тобто «рідний» простір моделі — L2.

Чому це важливо при створенні індексу:
- **Метрика індексу Pinecone фіксується при створенні** (`metric="cosine" | "dotproduct" | "euclidean"`) і потім не змінюється. Помилка означає перестворити індекс і перезавантажити всі вектори.
- Якщо метрика не збігається з тим, як модель вчилася розташовувати вектори, ранжування буде гіршим. Наприклад, `dotproduct` на ненормалізованих векторах почне віддавати перевагу векторам з великою нормою, а не близьким за змістом.
- **Наше рішення:** ембеддинги нормалізуються (`normalize_embeddings=True`). Для одиничних векторів cosine, dot product і L2 дають **однакове ранжування** (доведення — у розділі 3.4 і Частині 6). Тому вибір між ними стає питанням зручності, а не якості. Обрано `cosine`: він коректний, навіть якщо колись завантажити ненормалізовані вектори.

### 3.4. Чому для нормалізованих векторів cosine = dot product

За визначенням
$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}.$$
Після L2-нормалізації $\lVert a \rVert = \lVert b \rVert = 1$, тож знаменник дорівнює 1 і $\cos(a,b) = a \cdot b$. Косинус — це скалярний добуток, поділений на добуток довжин. Коли довжини одиничні, ділити нема на що: лишається чистий кут між векторами. Практичний бонус: dot product дешевший, бо не потрібно рахувати дві норми. Тому багато векторних БД для нормалізованих даних рекомендують `dotproduct` як швидший еквівалент.

### Вивід `02_embed.py`
```
Пристрій: cpu
Оброблено текстів: 10000
Розмірність ембеддингів: 768
Норма першого ембеддингу: 1.000000
Норми всіх векторів: min=1.000000, max=1.000000
Текстів довших за 512 токенів (обрізані): 42 (0.4%), медіана довжини: 235 токенів

Збережено в embeddings/embeddings.npy (30.7 МБ)
```

Обрізання на 512 токенах зачіпає лише 0.4 % статей, тож для повних анотацій це не проблема. Chunking (Частина 4) тут потрібен для точнішого пошуку фрагментів, а не через ліміт моделі.

---

## 4. Частина 2 — завантаження в Pinecone

- Індекс `arxiv-papers`: serverless, AWS `us-east-1` (на Starter інших регіонів немає), `dimension=768`, `metric=cosine`.
- Запис: `id = "paper_<номер рядка parquet>"`, `values` = ембеддинг, `metadata` = `arxiv_id`, `title`, `abstract[:500]`, `authors[:200]`, `year` (int, щоб працювали `$gte`/`$lt`), `category`.
- Батчі по 200 векторів (ліміт запиту upsert — 2 МБ; 200 × 768 float ≈ 0.6 МБ + метадані).
- Повний `abstract` лишається в parquet і підтягується за номером рядка з `id` після пошуку. Метадані Pinecone обмежені 40 KB на вектор, і зберігати там повні тексти — марна трата сховища.
- Після upsert скрипт чекає, поки `total_vector_count` дожене очікуване значення: serverless-індекс eventually consistent, і статистика оновлюється із затримкою.

```
Створюємо індекс 'arxiv-papers' (dim=768, metric=cosine)...
Індекс 'arxiv-papers': metric=cosine, dimension=768
Статей: 10000, ембеддинги: (10000, 768)
Upsert у Pinecone: 100%|██████████| 50/50 [00:33<00:00,  1.48it/s]

Векторів в індексі 'arxiv-papers': 10000
{'dimension': 768,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 10000}},
 'total_vector_count': 10000}
```

50 батчів по 200 векторів — 33 с. Затримки eventual consistency в цьому прогоні не було: лічильник одразу показав 10 000.

![Консоль Pinecone: індекс arxiv-papers](screenshots/pinecone_index.png)

*Консоль Pinecone: індекс `arxiv-papers` — 10 000 записів, AWS `us-east-1`, 3 індекси в проєкті. Пошук за ID `paper_0` (вкладка Browser) показує метадані запису й найближчих сусідів: статтю про hate speech (score 1.0000 — сама себе) оточують інші статті про hate speech (0.9654, 0.9640).*

---

## 5. Частина 3 — пошукові запити

### 5.1. Семантичний пошук
Запит: «teaching machines to recognize objects in pictures». У запиті немає жодного з термінів «image classification», «object detection», «CNN».
```
1. [0.9189] Learning as the Unsupervised Alignment of Conceptual Systems
   cs.LG | 2019 | arXiv:1906.09012
2. [0.9125] Unlimited Knowledge Distillation for Action Recognition in the Dark
   cs.CV | 2023 | arXiv:2308.09327
3. [0.9122] Learning Visually Grounded Domain Ontologies via Embodied Conversation and Explanation
   cs.AI | 2024 | arXiv:2412.09770
4. [0.9097] Do Vision Models Develop Human-Like Progressive Difficulty Understanding?
   cs.CV | 2025 | arXiv:2503.13058
5. [0.9070] Building the information kernel and the problem of recognition
   cs.CV | 2009 | arXiv:0903.4513
```
(Pinecone зберігає числові метадані як float, тому скрипт друкує рік як `2019.0`; тут округлено.)

Коментар: так, 3 з 5 статей — cs.CV, хоча в запиті немає жодного їхнього терміна. Але я вважаю, що модель знайшла радше **«приблизно схоже»**, ніж справді релевантне: розпізнавання дій на темних відео, «складність» задач для vision-моделей, стаття 2009 року про «information kernel». Жодної статті саме про класифікацію чи детекцію об'єктів, а топ-1 узагалі cs.LG про узгодження концептів. До того ж score майже не розрізняють результати (0.907–0.919). Семантичний пошук «вловив напрям» (зір, розпізнавання, навчання), але без адаптера `adhoc_query` базова SPECTER2 погано зіставляє коротку фразу з анотаціями (див. 3.2).

### 5.2. Пошук із фільтрами
- **A:** `{"category": {"$eq": "cs.LG"}, "year": {"$gte": max_year-4}}`
- **B:** `{"year": {"$lt": 2015}}`
- **C:** без фільтра (для порівняння)

```
4A. «reinforcement learning agents learning policies from rewards» | cs.LG, 2022–2026
1. [0.9579] Policy Gradient Methods for Non-Markovian Reinforcement Learning      cs.LG | 2026
2. [0.9561] BlendRL: A Framework for Merging Symbolic and Neural Policy Learning  cs.LG | 2024
3. [0.9542] Which Rewards Matter? Reward Selection for RL under Limited Feedback  cs.LG | 2025
4. [0.9506] DGPO: Discovering Multiple Strategies with Diversity-Guided Policy…  cs.LG | 2022
5. [0.9503] Proto Successor Measure: Representing the Behavior Space of an RL Agent  cs.LG | 2024

4B. той самий запит | до 2015 року, будь-яка категорія
1. [0.9463] Reinforcement Learning by Value Gradients                              cs.NE   | 2008
2. [0.9129] Temporal-Difference Learning to Assist Human Decision Making during…  cs.AI   | 2013
3. [0.9063] Towards A Theory-Of-Mind-Inspired Generic Decision-Making Framework    cs.AI   | 2014
4. [0.9025] Estimating or Propagating Gradients Through Stochastic Neurons         cs.LG   | 2013
5. [0.8971] No More Pesky Learning Rates                                           stat.ML | 2012

4C. без фільтра — ідентично 4A (ті самі 5 статей у тому самому порядку)

Порівняння:
  Перетин A∩B: 0 | A∩без_фільтра: 5 | B∩без_фільтра: 0
  A: score 0.9503–0.9579, категорії {'cs.LG': 5}
  B: score 0.8971–0.9463, категорії {'cs.AI': 2, 'cs.NE': 1, 'cs.LG': 1, 'stat.ML': 1}
  без фільтра: score 0.9503–0.9579, категорії {'cs.LG': 5}
  Розмір пулу кандидатів: A=943, B=1437, усього=10000
```

Як пояснювати відмінності:
- Фільтр у Pinecone застосовується **під час** пошуку, а не після нього. Повертається топ-5 *серед документів, що пройшли фільтр*, тому видача A і B майже не перетинається: пули кандидатів не перетинаються за роками.
- Score у B зазвичай нижчий, ніж у C. Найближчі за змістом статті часто свіжі, а фільтр їх відсікає, тож лишаються «найкращі з доступних». **Підтверджено:** B — 0.897–0.946, C — 0.950–0.958; навіть найкращий результат B нижчий за найгірший у C. Тематично B теж «розпливається»: №4–5 (stochastic neurons, learning rates) — уже не про RL, а про оптимізацію в цілому.
- У B з'являться статті не з cs.LG, бо до 2015 року RL-тематика частіше потрапляла в інші категорії. **Підтверджено:** 4 з 5 — не cs.LG (cs.AI ×2, cs.NE, stat.ML). Очікуваної math.OC у топ-5 немає, натомість є cs.NE (нейронні й еволюційні обчислення).
- **Фільтр A виявився зайвим для цього запиту:** видача A повністю збіглася з C (A∩C = 5). Найсвіжіші статті cs.LG водночас найближчі за змістом, тож без фільтра топ-5 і так складається з них. Фільтр звузив пул з 10 000 до 943 кандидатів, але ранжування не змінив.

### 5.3. Метрики на локальних ембеддингах
```
--- Нормалізовані ембеддинги (як в індексі) ---
cosine / dot product / L2 — однаковий топ-5:
  1. cos=0.9189 dot=0.9189 L2=0.4027 | Learning as the Unsupervised Alignment of Conceptual Systems
  2. cos=0.9125 dot=0.9125 L2=0.4184 | Unlimited Knowledge Distillation for Action Recognition in the Dark
  3. cos=0.9121 dot=0.9121 L2=0.4192 | Learning Visually Grounded Domain Ontologies via Embodied Conversation…
  4. cos=0.9098 dot=0.9098 L2=0.4248 | Do Vision Models Develop Human-Like Progressive Difficulty Understanding?
  5. cos=0.9070 dot=0.9070 L2=0.4312 | Building the information kernel and the problem of recognition
Однаковий порядок топ-5: cosine==dot → True, cosine==L2 → True
max | L2² − (2 − 2·cos) | = 9.24e-07

Норми ненормалізованих векторів у пулі: min=21.30, max=22.06, норма запиту=21.71

--- БЕЗ нормалізації (пул топ-300 за cosine) ---
cosine:                                          dot product:
  1. Learning as the Unsupervised Alignment…       1. Learning as the Unsupervised Alignment…   dot=433.22
  2. Unlimited Knowledge Distillation…             2. Learning Visually Grounded Ontologies…    dot=432.45
  3. Learning Visually Grounded Ontologies…        3. Unlimited Knowledge Distillation…         dot=429.58
  4. Do Vision Models Develop…                     4. Do Vision Models Develop…                 dot=429.04
  5. Building the information kernel…              5. Perception for Autonomous Systems (PAZ)   dot=428.12  (cos=0.9062)
L2 distance: той самий порядок, що й cosine (L2 = 8.74, 9.08, 9.13, 9.23, 9.31)
Однаковий порядок топ-5: cosine==dot → False, cosine==L2 → True
```

**Чи збігаються топ-5 для cosine і dot product?** Так, повністю, разом зі значеннями (з точністю до float32). Для одиничних векторів $\cos(a,b) = a\cdot b$, див. 3.4.

**Чи відрізняються результати для L2?** Значення відрізняються: це відстань, тож менше — краще, і діапазон $[0, 2]$ замість $[-1, 1]$. Але **порядок той самий**, бо для одиничних векторів
$$\lVert a-b \rVert^2 = 2 - 2\cos(a,b),$$
і L2 є строго спадною функцією від cosine. Скрипт перевіряє цю тотожність чисельно: максимальна похибка **9.24e-07**, тобто шум float32, як і очікувалось.

**Що сталося б без нормалізації?** Скрипт це перевіряє на реальних даних: перекодовує топ-300 кандидатів без `normalize_embeddings` і рахує метрики заново.
- **cosine** не змінюється: він сам ділить на норми.
- **dot product** починає залежати від **довжини вектора**: $a\cdot b = \lVert a\rVert\lVert b\rVert\cos\theta$. Статті з більшою нормою ембеддингу піднімаються вгору незалежно від змісту. Норма CLS-ембеддингу залежить від довжини й «типовості» тексту, а не від релевантності.
- **L2** штрафує вектори, норма яких далека від норми запиту, навіть якщо напрямок ідеальний.
- **Результат прогону:** норми в пулі 21.30–22.06, норма запиту 21.71. **Розійшовся лише dot product:** №2 і №3 помінялися місцями (у «Learning Visually Grounded…» більша норма), а на №5 піднялась «Perception for Autonomous Systems (PAZ)» з нижчим cosine (0.9062), витіснивши статтю з cosine 0.9070. **L2, всупереч очікуванню, порядок не змінив.** Причина: норми в пулі майже однакові (розкид ≈ 3.5 %), і норма запиту лежить посередині цього діапазону, тому відмінності в довжинах малі порівняно з відмінностями в напрямку. Для SPECTER2 на однорідних анотаціях L2 без нормалізації практично повторює cosine. Проблема L2 проявилась би на векторах із суттєво різними нормами (наприклад, дуже короткі й дуже довгі тексти або суміш моделей).

---

## 6. Частина 4 — chunking

Налаштування: 30 статей із найдовшими анотаціями. **Fixed** — 60 слів, overlap 15. **Semantic** — жадібне склеювання цілих речень до 80 слів; речення довше за ліміт ріжеться по словах. Розбиття на речення не ріже на `e.g.`, `i.e.`, `et al.`, `Fig.`, `Eq.`. Окремі індекси: `arxiv-chunks-fixed`, `arxiv-chunks-semantic`. Метадані: `arxiv_id`, `title`, `chunk_text`, `chunk_num`, `n_words`, `year`, `category`.

30 статей із найдовшими анотаціями (296–499 слів) — строката суміш: 23 з CS (мови програмування, алгоритми, криптографія, мережі, HCI, ML) і 1 stat.ML, решта — математика, астрофізика, q-bio.

```
2. Статистика стратегій
  fixed    : чанків= 216 | слів у чанку: середнє=56.5, min=26, max=60 | обірвані на півреченні: 179 (83%)
  semantic : чанків= 154 | слів у чанку: середнє=61.1, min=15, max=80 | обірвані на півреченні: 1 (1%)

Приклад розбиття: «User-Defined Operators Including Name Binding for New Language Constr…» (499 слів)
  [FIXED]
   #0 (60 сл.): User-defined syntax extensions are useful to implement an embedded domain specific language (EDSL)…
   #1 (60 сл.): promising approach uses user-defined operators. A user-defined operator is a function with…
   #2 (60 сл.): the definition of an operator before the operator is used. In addition, the compiler can…
   #3 (60 сл.): locations). Our inquiry is whether we can design a system for user-defined operators…
   … (усього 11 чанків, останній 49 сл.)
  [SEMANTIC]
   #0 (70 сл.): User-defined syntax extensions are useful to implement an embedded domain specific language (EDSL)…
   #1 (66 сл.): An advantage of user-defined operators is that an operator can be statically typed…
   #2 (69 сл.): Our inquiry is whether we can design a system for user-defined operators involving…
   … (усього 8 чанків)
   #7 (15 сл.): This paper shows programmers can define the former two: their own syntax and name binding.

2b. Вплив overlap (fixed, size=60) на кількість чанків
  Усього слів у 30 анотаціях: 9409
   overlap | чанків | слів у чанках | надлишок
         0 |    174 |          9409 |      0%
        10 |    190 |         11009 |     17%
        15 |    216 |         12199 |     30%
        20 |    237 |         13549 |     44%
        30 |    299 |         17479 |     86%
        45 |    552 |         32899 |    250%

Векторів в 'arxiv-chunks-fixed': 216
Векторів в 'arxiv-chunks-semantic': 154

6. Пошук по чанках
>>> «extending a programming language with custom syntax and operators»
  [FIXED]    (унікальних статей у топ-5: 1) — усі 5: User-Defined Operators…, чанки #1, #0, #10, #4, #2 (0.9542–0.9325)
  [SEMANTIC] (унікальних статей у топ-5: 1) — усі 5: User-Defined Operators…, чанки #7, #0, #1, #2, #6 (0.9470–0.9232)

>>> «distributed algorithms for matching and independent sets in graphs»
  [FIXED] (унікальних статей у топ-5: 2)
   1–4. [0.9448–0.8927] Distributed Maximal Matching and MIS on Hypergraphs | чанки #0, #2, #1, #5
   5.   [0.8911] Optimally Reliable & Cheap Payment Flows on the Lightning Network | чанк #6
        «out to be a hard problem in general as well - even in the single path case - it appears to be surprisingly tractable…»
  [SEMANTIC] (унікальних статей у топ-5: 3)
   1–3. [0.9441–0.8944] Distributed Maximal Matching and MIS on Hypergraphs | чанки #0, #1, #4
   4.   [0.8876] Pure Nash Equilibria in Resource Graph Games | чанк #5
   5.   [0.8865] Longest Common Extensions in Trees | чанк #2

>>> «security and reliability of cryptocurrency payments»
  [FIXED] (унікальних статей у топ-5: 2)
   1. [0.9090] Optimally Reliable & Cheap Payment Flows on the Lightning N… | чанк #4
   2. [0.9076] On the (in)security of Proofs-of-Space based Longest-Chain … | чанк #0
   3. [0.9005] On the (in)security of Proofs-of-Space based Longest-Chain … | чанк #1
   4. [0.8911] Optimally Reliable & Cheap Payment Flows on the Lightning N… | чанк #0
   5. [0.8849] Optimally Reliable & Cheap Payment Flows on the Lightning N… | чанк #3
  [SEMANTIC] (унікальних статей у топ-5: 3)
   1. [0.9099] On the (in)security of Proofs-of-Space based Longest-Chain … | чанк #0
   2. [0.8993] Optimally Reliable & Cheap Payment Flows on the Lightning N… | чанк #2
   3. [0.8948] Optimally Reliable & Cheap Payment Flows on the Lightning N… | чанк #0
   4. [0.8841] On the (in)security of Proofs-of-Space based Longest-Chain … | чанк #1
   5. [0.8754] FAST CLOUD: Pushing the Envelope on Delay Performance of Cl… | чанк #0
```

Усі три запити знайшли цільові статті в обох індексах, зокрема третій — перефразування без слів «blockchain» і «Lightning». Semantic у двох запитах із трьох дав більше унікальних статей у топ-5 (3 проти 2). Перший запит — наочна ілюстрація проблеми дублікатів: 5 чанків однієї статті витіснили другу статтю про мови програмування (Programming Education in Graphics).

**Яка стратегія дає більш осмислені чанки?** Semantic. Кожен чанк складається з цілих речень і зазвичай відповідає одній думці анотації: постановці задачі, методу або результату. Fixed-чанк починається і закінчується де завгодно, часто посеред речення (у прогоні **83 %** fixed-чанків обірвані, у semantic — **1 %**). Ціна semantic — нерівна довжина чанків (**15–80** слів): короткий хвостовий чанк із одного речення *може* бути надто бідним на контекст. У нашому прогоні це не підтвердилось: хвостовий чанк #7 із 15 слів («This paper shows programmers can define the former two: their own syntax and name binding.») став **топ-1** у semantic-індексі на запит про мови програмування, бо це одне речення, що концентровано підсумовує статтю. Короткий чанк погано працює, коли він — уривок без змісту, а не тоді, коли він просто короткий. Застереження: наш «semantic» — це sentence-aware chunking. Повноцінний semantic chunking розрізає там, де різко падає схожість ембеддингів сусідніх речень, і сюди не входить.

**Чи є розрізані речення і як це впливає на ембеддинги?** У fixed — так, системно. Обрізане речення втрачає присудок або об'єкт («…we propose a novel» | «method for …»). Ембеддинг такого фрагмента представляє «половину думки», а модель SPECTER2 навчена на цілих `title [SEP] abstract`, тож уривок ще далі від її розподілу. На практиці: (1) частина чанка «шумить» і розмиває вектор, (2) релевантне речення, розрізане навпіл, гірше знаходиться обома половинами. Overlap частково це лікує: розрізане речення повністю потрапляє в сусідній чанк. **Приклад із виводу:** fixed-чанк #1 починається з «promising approach uses user-defined operators…» (початок речення «A promising approach…» лишився в #0), а #3 — з «locations). Our inquiry is…», тобто з хвоста дужки попереднього речення. Ще наочніше — запит про графові алгоритми: fixed-індекс підняв на №5 уривок «out to be a hard problem in general as well… surprisingly tractable in practice» зі статті про Lightning Network. Цей фрагмент без початку речення нічого не каже про платежі й «схожий» на алгоритмічний текст лише загальними словами; semantic-індекс натомість дав на тих позиціях дві справді графові статті.

**Як overlap впливає на кількість чанків і покриття?** Кількість чанків для тексту з $N$ слів:
$$n \approx \left\lceil \frac{N - o}{s - o} \right\rceil,$$
де $s$ — розмір, $o$ — overlap. Зі зростанням overlap крок вікна $s-o$ зменшується, і чанків стає більше, причому нелінійно: при $o \to s$ кількість різко зростає. Покриття тексту повне за будь-якого overlap, але росте **дублювання**: ті самі слова зберігаються і ембеддяться кілька разів (таблиця: **30 %** надлишку при overlap 15, **86 %** при 30, а при 45 із 60 — уже 250 % і 552 чанки замість 174). Більше overlap — менше втрачених на межах думок, але більше векторів (вартість сховища й upsert) і більше майже дублікатів у топ-5, які витісняють інші статті.

---

## 7. Частина 5 — гібридний пошук (додатково)

BM25 (`rank_bm25.BM25Okapi`, токенізація lowercase + `[a-z0-9]+`, короткий стоп-лист) по `title + abstract + authors`, векторний пошук через Pinecone, злиття через RRF:
$$\text{RRF}(d) = \sum_{m \in \{\text{BM25},\,\text{vec}\}} \frac{1}{k + \text{rank}_m(d)}, \quad k = 60.$$
Кожен метод повертає топ-10, RRF переранжовує об'єднання. RRF використовує лише **ранги**, бо BM25-скор ($0..\infty$) і cosine ($-1..1$) у різних шкалах і напряму не порівнюються.

```
BM25-індекс: 10000 документів, поля: title, abstract, authors

Точний термін: «BERT fine-tuning»
[BM25] топ-5
  1.  20.478 | Memorization of Named Entities in Fine-tuned BERT Models (cs.CL, 2022)
  2.  17.781 | IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization (cs.CL, 2020)
  3.  17.108 | BERT4SO: Neural Sentence Ordering by Fine-tuning BERT (cs.CL, 2021)
  4.  15.523 | Fine-Tuning Models for Biomedical Relation Extraction (cs.CL, 2026)
  5.  12.905 | Boosting the Performance of Transformer Architectures for Semantic Te… (cs.CL, 2023)
[Vector / Pinecone] топ-5
  1.  0.9331 | Explainable Reinforcement Learning for Formula One Race Strategy (cs.LG, 2025)
  2.  0.9315 | Robust Reinforcement Learning via Adversarial training with Langevin … (cs.LG, 2020)
  3.  0.9302 | Resolving Implicit Coordination in Multi-Agent Deep Reinforcement Lea… (cs.MA, 2020)
  4.  0.9294 | Action Shapley: A Training Data Selection Metric for World Model in R… (cs.LG, 2026)
  5.  0.9287 | A Large-Scale Empirical Evaluation of MMAO Under Fair-Budget Continuo… (cs.NE, 2026)
[Hybrid RRF, k=60] топ-5
  1. 0.01639 (BM25 #1, vec #—) | Memorization of Named Entities in Fine-tuned BERT Models
  2. 0.01639 (BM25 #—, vec #1) | Explainable Reinforcement Learning for Formula One Race Strategy
  3. 0.01613 (BM25 #2, vec #—) | IsoBN: Fine-Tuning BERT with Isotropic Batch Normalization
  4. 0.01613 (BM25 #—, vec #2) | Robust Reinforcement Learning via Adversarial training with Langevin …
  5. 0.01587 (BM25 #3, vec #—) | BERT4SO: Neural Sentence Ordering by Fine-tuning BERT
[Hybrid RRF, k=1] — той самий порядок (0.500, 0.500, 0.333, 0.333, 0.250)
  Порядок топ-5 при k=60 і k=1 однаковий: True

Ім'я автора: «Yann LeCun convolutional networks»
[BM25] топ-5                       (усі 5 — статті за участю LeCun)
  1.  23.554 | Convolutional Matching Pursuit and Dictionary Training (cs.CV, 2010)
  2.  22.637 | Spectral Networks and Locally Connected Networks on Graphs (cs.LG, 2013)
  3.  22.542 | Exploiting Linear Structure Within Convolutional Networks for Efficie… (cs.CV, 2014)
  4.  22.046 | Convolutional Neural Networks Applied to House Numbers Digit Classifi… (cs.CV, 2012)
  5.  21.546 | Fast Training of Convolutional Networks through FFTs (cs.CV, 2013)
[Vector / Pinecone] топ-5          (LeCun — лише №4)
  1.  0.9453 | Deep Neural Nets with Interpolating Function as Output Activation (cs.LG, 2018)
  2.  0.9421 | Diving deeper into mentee networks (cs.LG, 2016)
  3.  0.9409 | SMPConv: Self-moving Point Representations for Continuous Convolution (cs.CV, 2023)
  4.  0.9392 | Exploiting Linear Structure Within Convolutional Networks for Efficie… (cs.CV, 2014)
  5.  0.9388 | Deep Neural Networks for Pattern Recognition (cs.CV, 2018)
[Hybrid RRF, k=60] топ-5
  1. 0.03150 (BM25 #3, vec #4) | Exploiting Linear Structure Within Convolutional Networks for Efficie…
  2. 0.01639 (BM25 #1, vec #—) | Convolutional Matching Pursuit and Dictionary Training
  3. 0.01639 (BM25 #—, vec #1) | Deep Neural Nets with Interpolating Function as Output Activation
  4. 0.01613 (BM25 #2, vec #—) | Spectral Networks and Locally Connected Networks on Graphs
  5. 0.01613 (BM25 #—, vec #2) | Diving deeper into mentee networks
[Hybrid RRF, k=1] топ-5
  1. 0.50000 (BM25 #1, vec #—) | Convolutional Matching Pursuit and Dictionary Training
  2. 0.50000 (BM25 #—, vec #1) | Deep Neural Nets with Interpolating Function as Output Activation
  3. 0.45000 (BM25 #3, vec #4) | Exploiting Linear Structure Within Convolutional Networks for Efficie…
  4. 0.33333 (BM25 #2, vec #—) | Spectral Networks and Locally Connected Networks on Graphs
  5. 0.33333 (BM25 #—, vec #2) | Diving deeper into mentee networks
  Порядок топ-5 при k=60 і k=1 однаковий: False

Перефразування: «making computers understand human emotions from text»
[BM25] топ-5
  1.  15.854 | Engaging with AI: How Interface Design Shapes Human-AI Collaboration … (cs.HC, 2025)
  2.  15.470 | Can Computers overcome Humans? Consciousness interaction and its impl… (cs.AI, 2017)
  3.  14.608 | Multi-Task Learning Framework for Extracting Emotion Cause Span and E… (cs.CL, 2022)
  4.  14.032 | Enhancing Student Engagement in Online Learning through Facial Expres… (cs.CV, 2023)
  5.  13.575 | Conversational Machine Reading Comprehension for Vietnamese Healthcar… (cs.CL, 2021)
[Vector / Pinecone] топ-5
  1.  0.9391 | A Comparative Evaluation of Large Language Models for Persian Sentime… (cs.CL, 2025)
  2.  0.9373 | AdCOFE: Advanced Contextual Feature Extraction in Conversations for e… (cs.CL, 2021)
  3.  0.9323 | Towards Label-Agnostic Emotion Embeddings (cs.CL, 2020)
  4.  0.9307 | Multi-Task Learning Framework for Extracting Emotion Cause Span and E… (cs.CL, 2022)
  5.  0.9260 | EmotionIC: emotional inertia and contagion-driven dependency modeling… (cs.CL, 2023)
[Hybrid RRF, k=60] топ-5
  1. 0.03150 (BM25 #3, vec #4) | Multi-Task Learning Framework for Extracting Emotion Cause Span and E…
  2. 0.01639 (BM25 #1, vec #—) | Engaging with AI: How Interface Design Shapes Human-AI Collaboration …
  3. 0.01639 (BM25 #—, vec #1) | A Comparative Evaluation of Large Language Models for Persian Sentime…
  4. 0.01613 (BM25 #2, vec #—) | Can Computers overcome Humans? Consciousness interaction and its impl…
  5. 0.01613 (BM25 #—, vec #2) | AdCOFE: Advanced Contextual Feature Extraction in Conversations for e…
[Hybrid RRF, k=1] топ-5
  1. 0.50000 (BM25 #1, vec #—) | Engaging with AI: How Interface Design Shapes Human-AI Collaboration …
  2. 0.50000 (BM25 #—, vec #1) | A Comparative Evaluation of Large Language Models for Persian Sentime…
  3. 0.45000 (BM25 #3, vec #4) | Multi-Task Learning Framework for Extracting Emotion Cause Span and E…
  4. 0.33333 (BM25 #2, vec #—) | Can Computers overcome Humans? Consciousness interaction and its impl…
  5. 0.33333 (BM25 #—, vec #2) | AdCOFE: Advanced Contextual Feature Extraction in Conversations for e…
  Порядок топ-5 при k=60 і k=1 однаковий: False
```

| Тип запиту | Запит | BM25 топ-1 | Vector топ-1 | Hybrid топ-1 | Спільних у топ-5 |
|---|---|---|---|---|---|
| Точний термін | BERT fine-tuning | Memorization of Named Entities in Fine-tuned BERT Models (cs.CL, 2022) | Explainable Reinforcement Learning for Formula One Race Strategy (cs.LG, 2025) | Memorization of Named Entities in Fine-tuned BERT Models (cs.CL, 2022) | 0 |
| Ім'я автора | Yann LeCun convolutional networks | Convolutional Matching Pursuit and Dictionary Training (cs.CV, 2010) | Deep Neural Nets with Interpolating Function as Output Activation (cs.LG, 2018) | Exploiting Linear Structure Within Convolutional Networks… (cs.CV, 2014) | 1 |
| Перефразування | making computers understand human emotions from text | Engaging with AI: How Interface Design Shapes Human-AI Collaboration… (cs.HC, 2025) | A Comparative Evaluation of LLMs for Persian Sentiment… (cs.CL, 2025) | Multi-Task Learning Framework for Extracting Emotion Cause Span… (cs.CL, 2022) | 1 |

Релевантних у топ-5 (оцінка вручну за заголовками; для автора — перевірено поле `authors`):

| Запит | BM25 | Vector | Hybrid k=60 |
|---|---|---|---|
| BERT fine-tuning | **5/5** | 0/5 | 3/5 |
| Yann LeCun convolutional networks | **5/5** | 1/5 | 3/5 |
| making computers understand human emotions | 1/5 | **5/5** | 3/5 |

**Який метод дав кращий результат і чому?** Очікування підтвердились за напрямом, але виявились різкішими, ніж я припускав.
- **«BERT fine-tuning» — BM25, 5/5.** Векторний пошук тут не просто «гірший», він **повністю провалився**: 5 статей про reinforcement learning і жодної про NLP. Я вважаю це слабкістю моделі: базова SPECTER2 без адаптера `adhoc_query` не вміє зіставити коротку абревіатуру з анотаціями, і запит із двох токенів потрапляє в довільну область простору. **Гібрид тут гірший за чистий BM25**: списки не перетинаються, тож RRF просто чергує їх і підмішує 2 RL-статті на №2 і №4.
- **«Yann LeCun convolutional networks» — BM25, 5/5.** Прізвище — точний токен із поля `authors`, якого вектор не бачить узагалі (ембеддинг будується з `title [SEP] abstract`). Вектор знайшов загальні статті про згорткові мережі, і лише одна з них (№4) — LeCun.
- **«making computers understand human emotions from text» — вектор, 5/5.** Усі статті про emotion recognition і sentiment analysis, хоча слів запиту в них немає. BM25 знайшов лише одну релевантну (Emotion Cause), решту — за загальними словами «computers», «human», «understand» (HCI, свідомість, онлайн-навчання).
- **Гібрид** справді ніколи не провалився повністю (3/5 на кожному запиті), але й ніде не виграв. Його цінність — у страхуванні, коли тип запиту невідомий, а ціна — у «розбавленні» видачі, коли один метод явно неправий.

**Чи є в топ-5 гібрида документи, яких немає в топ-5 окремих методів?** Можуть бути, і це очікувана властивість RRF. Документ на 7-му місці в *обох* списках отримує $2/(60+7) \approx 0.0299$, а документ на 1-му місці лише в одному — $1/61 \approx 0.0164$. Консенсус двох методів важить більше, ніж перемога в одному. Скрипт позначає такі документи «← немає в топ-5 жодного методу». **У прогоні цього не трапилось:** у кожному з трьох запитів спільним для обох топ-10 був щонайбільше один документ, і він уже був у топ-5 обох методів. Зате ефект консенсусу видно чітко: у запитах про LeCun і про емоції топ-1 гібрида — документ, який був лише №3 у BM25 і №4 у векторному пошуку ($1/63 + 1/64 \approx 0.0315$), і він обійшов лідерів обох списків ($1/61 \approx 0.0164$).

**Як k впливає на видачу (k=60 vs k=1)?** $k$ задає, наскільки різко падає внесок із рангом.
- При $k=1$: 1-ше місце дає $0.5$, 2-ге — $0.333$, 10-те — $0.09$. Перші позиції домінують, гібрид поводиться як «чергування лідерів» двох списків, а консенсус на середніх позиціях майже не враховується.
- При $k=60$: $1/61 \approx 0.0164$ проти $1/70 \approx 0.0143$ для 10-го місця. Шкала пласка, вирішальним стає **присутність в обох списках**.
- Якщо списки не перетинаються, порядок топ-5 при обох $k$ однаковий: чергування #1 BM25, #1 vec, #2 BM25… Різниця з'являється саме тоді, коли документи є в обох списках. **Підтверджено прогоном:** для «BERT fine-tuning» (0 спільних) порядок однаковий — `True`; для двох інших запитів (по 1 спільному) — `False`. При $k=1$ консенсусний документ опускається з №1 на №3 ($1/4 + 1/5 = 0.45 < 0.5$), а вгору виходять лідери окремих списків.

---

## 8. Частина 6 — аналіз і висновки

### 8.1. Семантичний пошук vs ключові слова
Семантичний пошук сильний там, де **користувач і автор описують те саме різними словами**: запити природною мовою, перефразування, синоніми, «пошук за ідеєю» (у прогоні найкращий приклад — «making computers understand human emotions from text» → 5/5 статей про emotion recognition і sentiment analysis без жодного слова запиту; «teaching machines to recognize objects in pictures» дав лише тематично близьке, див. 5.1). Він також толерантний до помилок у формулюваннях і знаходить суміжні роботи. Слабкий там, де значення має **точний токен**: абревіатури й назви моделей (BERT, ResNet), імена авторів, ідентифікатори, формули, рідкісні терміни. Ембеддинг стискає текст у «загальний зміст», і конкретний рідкісний токен у ньому розчиняється. BM25 навпаки: чесно знаходить точний збіг, але не бачить нічого поза ним.

**Загальне правило:** якщо запит — *опис потреби*, обирайте вектори; якщо запит — *ідентифікатор* (назва, ім'я, код, термін), обирайте BM25. Якщо тип запиту наперед невідомий, а в реальному продукті це майже завжди так, обирайте гібрид. Я з цим правилом погоджуюсь, і в реальному продукті обрав би гібрид за замовчуванням. **Приклади з Частини 5:** «BERT fine-tuning» (ідентифікатор) — BM25 5/5, вектор 0/5; «Yann LeCun …» (ім'я) — BM25 5/5, вектор 1/5; «making computers understand human emotions from text» (опис потреби) — вектор 5/5, BM25 1/5. Гібрид на всіх трьох дав 3/5: ніде не найкращий, але й ніде не нульовий. Застереження з прогону: коли списки не перетинаються (BERT), гібрид гірший за правильний окремий метод, тому якщо тип запиту можна визначити (наприклад, запит з одного-двох токенів або з лапками → BM25), краще маршрутизувати, а гібрид лишати для решти.

### 8.2. Вплив розміру чанка
- **Занадто малий (10–15 слів):** чанк — це уривок речення без контексту. Вектор стає «шумним» і неоднозначним: «the proposed method outperforms baselines» підходить до тисячі статей. Пошук знаходить формально схожі фрази з нерелевантних статей, точність падає. Векторів стає в рази більше: дорожче зберігати, повільніше шукати, більше дублікатів однієї статті в топі.
- **Занадто великий (500+ слів):** перевищує 512-токенне вікно SPECTER2 (≈380–400 англійських слів), і хвіст тексту **мовчки обрізається**. Навіть у межах вікна один вектор усереднює кілька тем, і конкретна деталь розчиняється в «загальному змісті» (recall на вузьких запитах падає). Для RAG великий чанк ще й засмічує контекст LLM.
- **Оптимум залежить від задачі:** для пошуку конкретних фактів і RAG — 100–300 токенів з overlap 10–20 %; для пошуку «схожих документів» — якомога ближче до одиниці змісту (для анотацій це вся анотація, тобто чанкати взагалі не треба). Практичне правило: чанк має бути найменшою одиницею, яка має сенс без сусідів, і не довшою за вікно моделі. Перевіряти це варто на власних запитах, а не теоретично.

### 8.3. Невідповідна метрика: індекс `euclidean` + нормалізовані вектори
Для одиничних векторів $\lVert a\rVert = \lVert b\rVert = 1$:
$$\lVert a-b\rVert^2 = (a-b)\cdot(a-b) = \lVert a\rVert^2 - 2\,a\cdot b + \lVert b\rVert^2 = 2 - 2\,a\cdot b = 2 - 2\cos(a,b).$$
Отже $\lVert a-b\rVert = \sqrt{2 - 2\cos(a,b)}$ — строго спадна функція від $\cos$ на $[-1,1]$. Звідси:
- **Ранжування не зміниться:** найменша L2-відстань завжди відповідає найбільшому cosine. Ми це перевірили чисельно в `04_search.py` (порядок той самий, похибка тотожності 9.24e-07).
- Зміняться **значення score**: Pinecone для euclidean повертає відстань (менше — краще, діапазон $[0, 2]$). Будь-які пороги на кшталт «score > 0.8 = релевантно», задані для cosine, стануть неправильними. Ламається і код, який сортує чи фільтрує за score, вважаючи, що «більше = краще».
- Справжня проблема виникає, коли вектори **не** нормалізовані: тоді dot і L2 *можуть* ранжувати інакше, ніж cosine. У нашому прогоні (5.3) розійшовся dot product, а L2 — ні, бо норми SPECTER2 на анотаціях майже однакові; на даних із сильно різними нормами розійдеться і L2.
- **Висновок:** для нормалізованих векторів euclidean не помилка якості, а помилка інтерпретації score. Але для ненормалізованих векторів метрика має відповідати тій, з якою модель навчалась.

### 8.4. Обмеження Pinecone Starter і масштаб у 10 млн статей
Поточні ліміти Starter: 1 проєкт, до 5 serverless-індексів, усі тільки в AWS `us-east-1`, 2 ГБ сховища на організацію, 100 namespaces на індекс, місячні квоти на read/write units. У цій роботі використано 3 індекси (ліміт 5). Сховище: 10 000 × 768 × 4 Б ≈ 31 МБ векторів плюс метадані, тобто запас великий. Найреальніші проблеми: (1) регіон — затримка з Києва до `us-east-1` у кожному запиті, (2) eventual consistency — статистика індексу оновлюється не одразу після upsert. **Насправді:** у ліміти ми не вперлися (консоль після всіх прогонів: Storage 0.04 ГБ / 2 ГБ, write units 40K / 2M — див. скріншот у розділі 4) — 3 індекси з 5 (`arxiv-papers` 10 000 векторів, `arxiv-chunks-fixed` 216, `arxiv-chunks-semantic` 154), upsert 10 000 векторів зайняв 33 с, а затримки eventual consistency не було: лічильник одразу показував очікувану кількість. Єдина реальна проблема — сумісність SDK: `pinecone-client` 4.1.0 при `include_metadata=False` повертає збіги взагалі без поля `metadata`, через що `06_hybrid_search.py` падав (виправлено в `common.get_matches()`, див. розділ 2).

**10 млн статей:**
- **Сховище:** $10^7 \times 768 \times 4$ Б ≈ **30.7 ГБ** лише векторів, у 15 разів більше за ліміт Starter → платний план, або зменшення розмірності / квантування (float16, int8 ≈ 7.7 ГБ), або self-hosted Qdrant з квантуванням на диску.
- **Ембеддинги:** на CPU (~10 статей/с) це ~11–12 днів. Потрібні GPU-батчі (кілька сотень–тисяча статей/с → кілька годин) і стрімінгова обробка JSONL без завантаження у пам'ять, з чекпойнтами.
- **Завантаження:** не 50 000 upsert-запитів по 200, а bulk import з об'єктного сховища (Parquet у S3) — платна функція Pinecone.
- **Дані:** у метаданих лише те, що потрібне для фільтрів і відображення; повні тексти — в Parquet/S3 або Postgres, підтягувати за `id`.
- **Організація:** namespaces за великими категоріями, якщо запити зазвичай у межах однієї галузі (менший обсяг пошуку → нижча вартість і затримка).
- **BM25:** `rank_bm25` тримає весь корпус у пам'яті Python і не масштабується. Потрібні OpenSearch/Elasticsearch або sparse-вектори (BM25/SPLADE) в тому ж Pinecone-індексі (`dotproduct`) для гібридного пошуку одним запитом.

**Власна думка.** Шардами й репліками Elasticsearch я сам не керував, мій досвід — з боку тестів: на продукті, де ES використовувався, я на Python шукав причину його падінь, і нею виявився надмірний обсяг даних, що в нього приходив, плюс використання не зовсім за призначенням. Тому для 10 млн статей я б насамперед думав не про те, як масштабувати базу, а про те, що й скільки в неї відправляти: тільки потрібні для пошуку поля в метаданих, батчі в межах лімітів, тексти — окремо. Managed serverless (Pinecone) знімає з команди налаштування шардів і реплік, але надлишкові дані там не «валять» сервіс, а перетворюються на рахунок за сховище й read/write units.
