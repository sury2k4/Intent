# Product pgvector semantic category matcher

The production path uses one pinned model:

```text
BAAI/bge-m3
revision `5617a9f61b028005a4858fdac845db406aefb181`
1024-dimensional normalized dense embeddings
```

This project routes search text to a **category/place-type intent**, not to an individual POI. It is designed for the search surface of ride-hailing and food apps; it does not implement a chatbot or conversational clarification loop.

## Hierarchy-aware category profiles

Each selectable leaf category is rendered into one bounded semantic profile per language:

```text
Category: Food & drink > Coffee (Ăn uống > Quán cà phê)
Description: A café serving coffee and related drinks.
Alternative names: coffee shop; cafe
```

The profile contains only the complete parent-to-leaf hierarchy, a rich localized description, and aliases/synonyms. The description is the place to express category meaning, typical user goals, and useful facilities; there are no separate purpose, amenity, intent, or example fields in the production source or database text. The current taxonomy has 238 nodes and 222 leaves, so the bilingual corpus contains 444 aggregate texts. Vietnamese accent folding remains a query-side variant.

## Local product stack

Copy environment settings and start PostgreSQL/pgvector:

```bash
copy .env.example .env
docker compose up -d db
docker compose ps
```

Install dependencies:

```bash
.venv/Scripts/python.exe -m pip install -e ".[all]"
```

Set the connection string in the shell if `.env` is not loaded by your terminal:

```text
DATABASE_URL=postgresql://model_intent:model_intent_dev@localhost:5432/model_intent
```

Application code only reads `DATABASE_URL`; moving to an authorized PostgreSQL/pgvector service requires changing this URL and running the same migration/sync commands.

## Migrate and synchronize profiles

Migrations are checksum-protected. The original variant, v1/v2, and E5-base v3 migrations remain unchanged for rollback. Migration `006_category_search_profiles_bge_m3.sql` adds a separate BGE-M3 serving table containing `place_type_id`, language, one aggregate `search_text`, and a 1024-dimensional embedding in the same row.

```bash
.venv/Scripts/python.exe -m model_intent db migrate
.venv/Scripts/python.exe -m model_intent db status
.venv/Scripts/python.exe -m model_intent embeddings sync --local-files-only
.venv/Scripts/python.exe -m model_intent embeddings status
```

The first profile sync builds and activates an immutable generation. Repeating the command with unchanged taxonomy, enrichment, renderer, and model inputs performs no embedding inference and reuses the ready generation.

Expected production cardinality:

```text
category profiles:       444
indexed leaf categories: 222
vector dimension:        1024
profile renderer:        category-profile-v3
storage:                 category_search_profiles_bge_m3
```

Generation management:

```bash
.venv/Scripts/python.exe -m model_intent embeddings activate <generation-id>
.venv/Scripts/python.exe -m model_intent embeddings cleanup --keep 3
```

The active BGE-M3 pointer is separate from the E5-base v3, v2, v1, and legacy pointers, so older applications can still roll back without reading the BGE table.

## Query through pgvector

```bash
.venv/Scripts/python.exe -m model_intent query \
  --local-files-only \
  --query "chỗ uống cà phê làm việc có wifi"
```

Default mode is exact pgvector cosine retrieval. The application embeds the original and accent-folded query forms, PostgreSQL searches the pinned profile generation, and results are grouped to one score per category before the deterministic `accept / clarify / reject` policy. `clarify` is a fixed UI choice among categories, not a generated chatbot response.

At the current size (444 vectors), exact scan is intentionally the serving default. The optional IVFFlat cosine index is available for future scale, but serving should switch to it only after measuring recall/action parity. The E5-base v3, v2/v1, and legacy tables remain available for rollback and are not used by the BGE-M3 runtime.

## Product UI

```bash
.venv/Scripts/python.exe -m streamlit run src/model_intent/ui.py
```

Open `http://localhost:8501`. The UI shows:

- active profile generation, profile set, model revision, and taxonomy fingerprint;
- indexed profile/category counts;
- pgvector exact search and query/DB/decision latency;
- ranked category profiles and deterministic clarification options.

No keyword, dictionary, substring, fuzzy, edit-distance, trigram, or lexical score is mixed into this semantic tier.

## Benchmark

```bash
.venv/Scripts/python.exe -m model_intent benchmark pgvector --local-files-only
```

The report is written to `artifacts/pgvector/benchmark.json` and separates:

- query embedding latency;
- connection-pool wait;
- PostgreSQL vector search;
- decision latency;
- total latency;
- profile/category cardinality and generation provenance.

The exact NumPy and pgvector paths are expected to return identical category rankings for the same pinned profile generation, with score differences below `1e-5`.

## Tests

Network-free unit tests:

```bash
.venv/Scripts/python.exe -m pytest -q -m "not pgvector and not real_model"
```

Database integration/parity tests:

```bash
.venv/Scripts/python.exe -m pytest -q -m pgvector
```

The database uses a named Docker volume, so restarting the application or database container does not require re-embedding an unchanged profile set.
