# HCMC place-type and POI search (E5-base 768D)

The active POI search path uses pinned `intfloat/multilingual-e5-base` (revision `d128750597153bb5987e10b1c3493a34e5a4502a`, 768-dimensional normalized embeddings). It routes search text to a **category/place-type intent**, then searches HCMC POIs when the category is accepted. The category-only `query` command does not return POIs. Neither path implements a chatbot or conversational clarification loop; this is a POC, not a production-quality or latency guarantee.

**Live snapshot update (2026-10-02):** the unified DB at `127.0.0.1:5433/model_intent_unified` now serves taxonomy `place-types-poc-v3` (**384 leaves, 768 E5 category profiles**, active generation **6**) and **40,639 OSM-backed POI rows** (active generation **2**). Default CLI `query`/`search` and Streamlit load `configs/benchmark.poi.pg.v3.yaml` and taxonomy v3. A new provisional E5 calibration was refit with max sequence length **192** because v3 profiles exceeded the old 128-token limit. This is **not** all real-world HCMC venues: the pinned snapshot registry records 41,212 in-boundary candidates, of which 245 clear-but-unmodelled, 263 ambiguous and 65 open-way objects are explicitly excluded; labels and source data are not independent ground truth. Missing attributes remain unknown; **no rating/review/amenity mock** is generated in the active POI generation. The previous v2 pair (452 profiles/28,548 mostly mock POIs) is retained for rollback. A fresh unified backup and promotion rollback record are under ignored `artifacts/db-backups/2026-10-02-pre-osm-v3-rollout/`. The v4/v5 resource drafts are **not active**; some v4 tag values are names or composite types and must not be promoted as categories. The older v2 instructions/counts below describe a historical release unless explicitly labeled current.

## Hierarchy-aware category profiles

Each selectable leaf category is rendered into one bounded semantic profile per language:

```text
Category: Food & drink > Coffee (Ăn uống > Quán cà phê)
Description: A café serving coffee and related drinks.
Alternative names: coffee shop; cafe
```

A profile contains the complete parent-to-leaf hierarchy, rich localized description, and aliases/synonyms; there are no separate purpose, amenity, intent, or example fields in the profile text. The **active v3 POI taxonomy** has 384 leaves and **768 bilingual E5 profiles**. The older v2 POI taxonomy had 226 leaves / 452 profiles; v1 had 238 nodes / 222 leaves / 444 profiles. Vietnamese accent folding is a query-side variant.

## Local product stack and database safety

The default `docker-compose.yml` and compatibility `docker-compose.unified.yml` resolve to the **same** Compose project `model-intent-unified`, service `db`, named volume `model-intent-unified_model_intent_unified_pgdata`, PostgreSQL database `model_intent_unified`, and host binding `127.0.0.1:5433`. They are two entry points to one existing DB, not two independent databases. Inspect their resolved configuration and the running service without starting or stopping it:

```bash
docker compose config
docker compose ps
# Optional: docker compose -f docker-compose.unified.yml config
```

Do not use `docker compose down -v` or remove the named volume: it holds the verified E5 and POI generations. Check any existing `.env` before copying `.env.example`; the old `POSTGRES_DB`/`POSTGRES_PORT`/`DATABASE_URL` values may still point to the separate legacy 5432 database. Compose reads `.env` for **Compose interpolation only**; the Python CLI, Streamlit, and your terminal do **not** automatically load it. The unified PostgreSQL config requires an explicit URL in the shell (Git Bash example):

```bash
export DATABASE_URL='postgresql://model_intent:model_intent_dev@127.0.0.1:5433/model_intent_unified'
.venv/Scripts/python.exe -m pip install -e ".[all]"
```

`configs/benchmark.poi.pg.v3.yaml` is the **default for CLI `query`/`search` and the Streamlit UI**; it requires the explicit unified URL and the v3 taxonomy. For explicit **CLI-only** offline v3 search without a DB, pass `--config configs/benchmark.poi.v3.yaml --taxonomy data/taxonomies/place_types.poc.v3.json` to `search`. `configs/benchmark.poi.pg.yaml`/`configs/benchmark.poi.v2.yaml` and the old mock-backed artifact are rollback/historical modes, not current defaults. Streamlit requires PostgreSQL and never silently falls back to an artifact. Do not migrate or sync the legacy volume on port 5432.

## E5 category generations (unified database)

Migrations are checksum-protected. The original variant, v1/v2, E5-base v3, BGE-M3 1024D, and POI generation tables remain in the migration history; they must not be dropped to switch configurations. The BGE table/pointer from `006_category_search_profiles_bge_m3.sql` and older 444-profile taxonomy are **historical, retained but inert**: the current sync writer builds/activates only E5-base 768D category generations. Merely using `configs/benchmark.cpu.yaml` (which lists BGE) or running `embeddings sync` does not create an active BGE generation. Do not present BGE as the serving default.

The unified DB now has a **768-profile / 384-category** active ready E5 generation and a **40,639-row OSM-only** active POI generation; the older **452/28,548 v2 pair** remains ready for rollback. Check status without writing to the DB:

```bash
.venv/Scripts/python.exe -m model_intent db status
.venv/Scripts/python.exe -m model_intent embeddings status
.venv/Scripts/python.exe -m model_intent pois status
```

The unified v3 category stage was synced with `--no-activate` and the v3 POI artifact with `--no-activate`; both were verified before a **single paired promotion**, recording prior IDs in `artifacts/db-backups/2026-10-02-pre-osm-v3-rollout/promotion-v3.json`. `db migrate`, `embeddings sync`, and `pois sync` write data and must not be run reflexively against the live volume; CLI sync now stages by default and unilateral activation/cleanup are disabled. The previous pair and the separate legacy 5432 migration ledger/POI volume have not been modified or deleted.

## Category-only query through pgvector

```bash
.venv/Scripts/python.exe -m model_intent query \
  --config configs/benchmark.poi.pg.v3.yaml \
  --taxonomy data/taxonomies/place_types.poc.v3.json \
  --model e5-base-multilingual --local-files-only \
  --query "chỗ uống cà phê làm việc có wifi"
```

With this **explicit** config, category retrieval uses exact pgvector cosine search over the active E5 768D generation. The application embeds original and accent-folded query variants, groups results to one score per category, and applies a deterministic `accept / clarify / reject` policy. `clarify` offers fixed category choices, not a generated response. At 768 vectors exact scan is the serving mode; measure recall/action parity before considering an approximate index. POI retrieval is a separate stage of `search`, not `query`.

## HCMC POI search POC

The category `query` command remains category-only. A separate `search` path adds HCMC POIs from OpenStreetMap (OSM), structured query properties, and POI ranking. OSM's Vietnam PBF is clipped offline against the pinned Ho Chi Minh City administrative boundary (OSM relation 1973756, version 288; see `data/osm/sources/hcmc.poc.v1.json`). The mapping file covers categories with defensible OSM tags and explicitly reports unsupported taxonomy leaves rather than guessing.

The POC preserves properties actually tagged in OSM and deterministically fills absent, category-applicable properties plus synthetic `rating`/`review_count`. Every synthetic field is marked `source=mock`; ratings and amenities are demo values, not facts or recommendations. Missing OSM coverage is not evidence that a place/property does not exist. Search filters required properties before ranking; the ranking combines preferred property match, Bayesian-adjusted rating/review volume, and optional distance when an origin is supplied. No PostGIS is required.

Install the optional OSM/geospatial reader dependencies with `.[poi]` or `.[all]`. The following **historical v1** example builds an artifact offline; it is not the active v2 serving artifact (see the v2/v4 reconstruction steps below):

```bash
python -m model_intent pois fetch --output data/osm/raw/vietnam-latest.osm.pbf
python -m model_intent pois validate
python -m model_intent pois build \
  --pbf data/osm/raw/vietnam-latest.osm.pbf \
  --output artifacts/osm/hcmc-poc-v1
```

The build writes canonical `pois.jsonl`, `manifest.json`, category coverage, unmapped-tag, and rejection reports under the ignored `artifacts/` directory. The POC pins the fetched Vietnam snapshot SHA-256 in `data/osm/sources/hcmc.poc.v1.json`; when using a later Geofabrik snapshot, update the pinned checksum after download. For small local tests, `pois build --elements-jsonl <file>` accepts normalized OSM-like records.

The current **versioned offline artifact config** is `configs/benchmark.poi.v2.yaml` with taxonomy `data/taxonomies/place_types.poc.v2.json`, reviewed enrichment v2, OSM mapping v2 and property catalog v4. Its ignored artifact `artifacts/osm/hcmc-poc-v4/` derives from the same pinned Geofabrik Vietnam PBF and HCMC boundary; the original v1/v2 artifacts remain untouched. It contains 28,548 mapped HCMC POIs including 16 generic EV chargers, 772 bank branches, 1,608 general schools, 44 motorcycle workshops and seven branded VinFast chargers. Each has at least seven known category-relevant assertions, mostly **mock**; neither rating nor synthetic amenities are verified facts. The default serving config is `configs/benchmark.poi.pg.yaml`, using that dataset in the unified DB.

To reconstruct it from a fresh checkout with the pinned PBF locally available, run the new mapping build then the power-evidence enrichment in fresh directories:

```bash
python scripts/prepare_poi_v2_resources.py
python -m model_intent pois build --pbf data/osm/raw/vietnam-latest.osm.pbf \
  --taxonomy data/taxonomies/place_types.poc.v2.json \
  --mapping data/osm/mappings/place_types.poc.v2.json \
  --properties data/properties/poi_properties.poc.v3.json \
  --output artifacts/osm/hcmc-poc-v3
python -m model_intent pois enrich --artifact artifacts/osm/hcmc-poc-v3/manifest.json \
  --taxonomy data/taxonomies/place_types.poc.v2.json \
  --mapping data/osm/mappings/place_types.poc.v2.json \
  --properties data/properties/poi_properties.poc.v4.json \
  --output artifacts/osm/hcmc-poc-v4
```

Never overwrite an existing artifact. This POC uses `>=50 kW` with explicit OSM output-power tags as evidence for fast charging; a connector count alone is **not** verified fast charging. Generic charging does not guarantee vehicle compatibility: missing compatibility evidence remains unknown/mock.

For E5-base 768D category retrieval with artifact-backed POI search (**explicit CLI-only offline configuration**, no DB required), install the model extras and run the pinned model locally. The CLI `search` default is the **unified PostgreSQL config**; pass `--config configs/benchmark.poi.v2.yaml` to opt into artifact mode:

```bash
.venv/Scripts/python.exe -m pip install -e ".[models]"
.venv/Scripts/python.exe -m model_intent search --config configs/benchmark.poi.v2.yaml \
  --taxonomy data/taxonomies/place_types.poc.v2.json \
  --model e5-base-multilingual --local-files-only \
  --query "quán cà phê có máy lạnh, ưu tiên thoáng mát"
```

The artifact `configs/benchmark.poi.v2.yaml` pins `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a`, `768D`, `query: ` / `passage: ` prefixes, and a taxonomy-v2-specific confidence calibration. The category stage uses exact NumPy cosine search over 452 bilingual profiles; its first run may encode and cache those profiles. The POI stage reads a local JSONL artifact. The **Postgres-only UI** uses the same pinned E5 model and fails visibly rather than falling back to fake/alias-only classification or the local artifact.

Default POC filtering permits mock properties to satisfy requirements but reports how many matches have verified evidence; `--verified-only` requires OSM evidence for every hard requirement or exclusion. All ratings in the current HCMC artifact are synthetic, so a verified-only minimum-rating request reports `no_verified_rating_data`. The v2 ranking policy discounts synthetic ratings/review volume and uses distance only when coordinates are supplied. A radius phrase without origin is marked ignored and warned about. Generic `restaurant` retrieves cuisine-specific restaurant leaves; `japanese_restaurant` stays specific. The UI displays property sources, accepts optional latitude/longitude, and has a verified-filter-evidence toggle.

The 5,698-query synthetic backtest is useful for diagnosis, **not** evidence of a production-ready 85% Top-1 search. Its stored JSONL rounds probabilities and keeps only Top-3 candidates, and uses wrong-category proxy gold for Bank/EV/tire queries. Run the audited baseline and (once available) a full-precision E5 trace/replay:

```bash
python scripts/audit_backtest_5698.py
.venv/Scripts/python.exe scripts/capture_e5_trace.py --config configs/benchmark.poi.v2.yaml \
  --taxonomy data/taxonomies/place_types.poc.v2.json \
  --output artifacts/backtest_port_8502/trace_v4_final_5698.jsonl
python scripts/replay_e5_trace.py --trace artifacts/backtest_port_8502/trace_v4_final_5698.jsonl \
  --properties data/properties/poi_properties.poc.v4.json
```

The replay verifies exact parity against captured decisions, reports validation-only acceptance-threshold sensitivity, and **retains the baseline policy** because independent OOD validation/adjudicated gold are insufficient. The taxonomy-v2 E5 calibration was fit from 23 legacy cases only; treat its probabilities as POC confidence. No production-latency SLA is claimed: a separate warm local CPU check on 45 curated queries × 2 passes measured mean **186 ms**, p95 **269 ms** and **70/90** calls above 150 ms. This is a single-machine artifact-backed smoke benchmark, not a production load test.

```bash
.venv/Scripts/python.exe scripts/evaluate_hcmc_poi_queries.py \
  --model e5-base-multilingual
```

The versioned regression corpus in `data/eval/hcmc_poi_queries.v3.jsonl` checks category action/ID, extraction, category/property overlaps and hard-filter invariants for 45 authored examples. The default fake path runs 35 legacy examples for parser regression; neither is a blind population-quality score. On the new E5 taxonomy, exact replay of all 5,698 captured queries reproduced the recorded decisions with zero mismatches. Among 1,936 validation examples without known-invalid proxy labels, lowering the top-correct gate from 0.615 to 0.55 raised correct accepted cases from 487 to 542, but **reduced accepted precision from 84.70% to 82.87%**; no independent OOD-validation split is present. Therefore the pinned calibrated policy was *not* loosened. The blind synthetic supported test baseline has 592 correct accepts among 1,936 examples and five OOD false accepts among 400; these are synthetic-proxy metrics, not a production guarantee.

Do not run POI sync against the separate legacy 5432 volume: it has an older, incompatible migration ledger and a conflicting `pois` table. The verified unified 5433 database is already populated; use the artifact config for offline search or the PostgreSQL config with an explicit unified `DATABASE_URL`.

`search` first interprets category and property spans together: it protects complete place-type phrases, marks overlapping nouns as category components, and removes only safe property constraints and cues. E5 evaluates the focused text and original query when they differ; conflicting calibrated category decisions trigger clarification instead of guessing. With E5, no alias can turn a rejected decision into a POI match. In v2 the generic `ev_charging` category and its OSM mapping are distinct from VinFast-only charging; “trạm sạc nhanh, có nhà vệ sinh” now filters mapped generic charging stations, but most amenity/rating values in this POC are **mock**, and charger-to-vehicle compatibility is not verified. The fake-model path retains a lexical alias fallback only for deterministic logic tests. POI no-match does not change an accepted category decision. OSM data is © OpenStreetMap contributors and distributed under ODbL 1.0; retain attribution when redistributing derived extracts.

## Unified E5-base PostgreSQL (one existing database)

`configs/benchmark.poi.pg.yaml` serves **both** E5-base `vector(768)` category profiles and canonical HCMC POIs from one PostgreSQL/pgvector database. Both Compose files now name the **existing** project `model-intent-unified` and its existing volume `model-intent-unified_model_intent_unified_pgdata` on `127.0.0.1:5433`; using the default `docker compose` commands does not select the legacy 5432 DB. The separate old `model_intent` database/volume contains only 15 legacy POI rows and an older migration ledger incompatible with the canonical `pois` table. Before the unified service was created, both old databases were exported under `artifacts/db-backups/2026-10-01-unified-preflight/` (ignored by Git); `pg_restore -l` read both archives, but a full restore drill has not been run. **With explicit user confirmation, only the legacy container `model-intent-db-1` was stopped; its volume and both dumps were preserved.** Keep the old volume and backups until an explicit cleanup decision; do not run migration/sync against it.

For a **read-only** search against the already-populated unified database (Git Bash):

```bash
docker compose ps
export DATABASE_URL='postgresql://model_intent:model_intent_dev@127.0.0.1:5433/model_intent_unified'
.venv/Scripts/python.exe -m model_intent search --config configs/benchmark.poi.pg.yaml \
  --taxonomy data/taxonomies/place_types.poc.v2.json --local-files-only \
  --query 'trạm sạc nhanh, có nhà vệ sinh'
```

The unified database was verified with one active ready E5 generation (**452 vectors / 226 categories**) and one active ready POI generation (**28,548 records**); all per-row ID/content hashes matched the v4 artifact, and eight representative searches returned identical category decisions, POI IDs and rounded ranking scores from NumPy/artifact and pgvector/Postgres. The unified runtime requires an explicit `DATABASE_URL` targeting `127.0.0.1:5433/model_intent_unified`; it never silently connects to the legacy default. Streamlit uses `configs/benchmark.poi.pg.yaml` by default; export that `DATABASE_URL` **before launching** it, since `.env` alone will not do this and the UI does not support an artifact fallback. For **CLI-only** offline use or rollback, pass `--config configs/benchmark.poi.v2.yaml` to `search` without touching either database; the old DB/volume remains intact.

The v4 artifact is ignored by Git and requires the pinned PBF or a separately transferred verified artifact on a fresh machine. Backups, source extracts, mock rating/amenity values, and the ~400K external POI estimate are not committed or treated as verified ground truth.

## Product UI

```bash
.venv/Scripts/python.exe -m streamlit run src/model_intent/ui.py
```

Open `http://localhost:8501` (or `http://localhost:8502` when 8501 is occupied). Launch this with `.venv/Scripts/python.exe`; the system Python used by an older 8501 server may not contain `sentence-transformers`. The HCMC POI tab runs pinned E5-base 768D category inference, shows its model/revision/calibration metadata, and accepts one query. The **Xem giải thích từng bước** checkbox reveals E5's category decision and focused semantic text, extracted/ignored requirements, filter counts, and the per-POI score components and provenance. The explanation uses recorded search metadata and does not convert mock attributes into verified claims. The separate category diagnostics tab shows:

- active profile generation, profile set, model revision, and taxonomy fingerprint;
- indexed profile/category counts;
- pgvector exact search and query/DB/decision latency;
- ranked category profiles and deterministic clarification options.

No keyword, dictionary, substring, fuzzy, edit-distance, trigram, or lexical score is mixed into this semantic tier.

## Benchmark

For a **category-only** pgvector benchmark against the unified E5 serving database, supply the explicit v2 config and taxonomy (the CLI defaults are older v1 resources, not this serving path):

```bash
.venv/Scripts/python.exe -m model_intent benchmark pgvector \
  --config configs/benchmark.poi.pg.yaml \
  --taxonomy data/taxonomies/place_types.poc.v2.json --local-files-only
```

This command reads the DB but writes the local benchmark report under `artifacts/pgvector/benchmark.json`; its default evaluation corpus is historical v1, not an independent POI search quality assessment. The report separates:

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

Read-only parity against the already-populated unified serving DB is covered by `tests/integration/test_unified_poi_db.py` with an **explicit** unified `DATABASE_URL` (see above). It must not migrate, sync, or activate. Do **not** run the whole `-m pgvector` suite against that database: other integration tests write migrations/profile generations. Run DB-writing integration tests only with an explicitly provisioned **disposable scratch DB**, opt in via `MODEL_INTENT_WRITE_TEST_DATABASE_URL`, and verify its database name is `model_intent_write_test` and its host port is **neither 5432 nor 5433**. Never reuse the serving or legacy volume for these tests.

The serving database uses a named Docker volume, so restarting the application or database container does not require re-embedding an unchanged profile set.
