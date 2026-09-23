-- Product schema for the E5-small (384-dimensional) semantic index.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE IF NOT EXISTS embedding_model_configs (
    id BIGSERIAL PRIMARY KEY,
    model_key TEXT NOT NULL,
    model_id TEXT NOT NULL,
    revision TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension = 384),
    query_prefix TEXT NOT NULL,
    document_prefix TEXT NOT NULL,
    normalize_embeddings BOOLEAN NOT NULL,
    max_seq_length INTEGER NOT NULL,
    config_hash CHAR(64) NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'retired')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS taxonomy_snapshots (
    id BIGSERIAL PRIMARY KEY,
    taxonomy_version TEXT NOT NULL,
    fingerprint CHAR(64) NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL,
    source TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS category_variants (
    id BIGSERIAL PRIMARY KEY,
    taxonomy_snapshot_id BIGINT NOT NULL REFERENCES taxonomy_snapshots(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    language TEXT NOT NULL,
    variant_kind TEXT NOT NULL,
    text TEXT NOT NULL,
    source_text TEXT NOT NULL,
    path_ids TEXT[] NOT NULL,
    renderer_version TEXT NOT NULL,
    variant_hash CHAR(64) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (taxonomy_snapshot_id, variant_hash)
);

CREATE TABLE IF NOT EXISTS embedding_generations (
    id BIGSERIAL PRIMARY KEY,
    model_config_id BIGINT NOT NULL REFERENCES embedding_model_configs(id),
    taxonomy_snapshot_id BIGINT NOT NULL REFERENCES taxonomy_snapshots(id),
    renderer_version TEXT NOT NULL,
    content_set_hash CHAR(64) NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('building', 'ready', 'failed', 'retired')),
    expected_count INTEGER NOT NULL,
    completed_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (model_config_id, taxonomy_snapshot_id, renderer_version, content_set_hash)
);

CREATE TABLE IF NOT EXISTS variant_embeddings (
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    variant_id BIGINT NOT NULL REFERENCES category_variants(id) ON DELETE CASCADE,
    content_hash CHAR(64) NOT NULL,
    embedding vector(384) NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, variant_id),
    CHECK (vector_dims(embedding) = 384),
    CHECK (vector_norm(embedding) > 0)
);

CREATE TABLE IF NOT EXISTS active_embedding_generation (
    model_key TEXT PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS category_variants_snapshot_item_idx
    ON category_variants (taxonomy_snapshot_id, item_id);
CREATE INDEX IF NOT EXISTS variant_embeddings_generation_idx
    ON variant_embeddings (generation_id);
CREATE INDEX IF NOT EXISTS embedding_generations_state_idx
    ON embedding_generations (state, model_config_id);
