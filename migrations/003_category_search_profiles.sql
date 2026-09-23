-- Additive schema for hierarchy-aware category search profiles.
-- Keep 001_core.sql and 002_hnsw.sql unchanged: legacy variant generations
-- remain available for rollback while profile generations are rolled out.

ALTER TABLE embedding_generations
    ADD COLUMN IF NOT EXISTS artifact_kind TEXT NOT NULL DEFAULT 'variant_v1';

ALTER TABLE embedding_generations
    ADD COLUMN IF NOT EXISTS profile_set_id BIGINT;

CREATE TABLE IF NOT EXISTS category_profile_sets (
    id BIGSERIAL PRIMARY KEY,
    taxonomy_snapshot_id BIGINT NOT NULL REFERENCES taxonomy_snapshots(id) ON DELETE CASCADE,
    profile_schema_version INTEGER NOT NULL CHECK (profile_schema_version > 0),
    source_fingerprint CHAR(64) NOT NULL,
    renderer_version TEXT NOT NULL,
    content_set_hash CHAR(64) NOT NULL,
    category_count INTEGER NOT NULL CHECK (category_count >= 0),
    profile_count INTEGER NOT NULL CHECK (profile_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        taxonomy_snapshot_id,
        profile_schema_version,
        source_fingerprint,
        renderer_version,
        content_set_hash
    )
);

ALTER TABLE embedding_generations
    DROP CONSTRAINT IF EXISTS embedding_generations_artifact_profile_ck;

ALTER TABLE embedding_generations
    ADD CONSTRAINT embedding_generations_artifact_profile_ck
    CHECK (
        (artifact_kind = 'variant_v1' AND profile_set_id IS NULL)
        OR (artifact_kind = 'category_profile_v1' AND profile_set_id IS NOT NULL)
    );

ALTER TABLE embedding_generations
    ADD CONSTRAINT embedding_generations_profile_set_fk
    FOREIGN KEY (profile_set_id) REFERENCES category_profile_sets(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS embedding_generations_artifact_state_idx
    ON embedding_generations (artifact_kind, state, model_config_id);

CREATE TABLE IF NOT EXISTS category_search_profiles (
    id BIGSERIAL PRIMARY KEY,
    profile_set_id BIGINT NOT NULL REFERENCES category_profile_sets(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('vi', 'en')),
    profile_kind TEXT NOT NULL DEFAULT 'primary' CHECK (profile_kind = 'primary'),
    hierarchy_text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    source_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    path_ids TEXT[] NOT NULL CHECK (cardinality(path_ids) > 0),
    renderer_version TEXT NOT NULL,
    profile_hash CHAR(64) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(trim(hierarchy_text)) > 0),
    CHECK (length(trim(search_text)) > 0),
    UNIQUE (profile_set_id, item_id, language, profile_kind),
    UNIQUE (profile_set_id, profile_hash)
);

CREATE INDEX IF NOT EXISTS category_search_profiles_set_item_idx
    ON category_search_profiles (profile_set_id, item_id);

CREATE TABLE IF NOT EXISTS category_profile_embeddings (
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    profile_id BIGINT NOT NULL REFERENCES category_search_profiles(id) ON DELETE CASCADE,
    content_hash CHAR(64) NOT NULL,
    embedding vector(384) NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, profile_id),
    CHECK (vector_dims(embedding) = 384),
    CHECK (vector_norm(embedding) > 0)
);

CREATE INDEX IF NOT EXISTS category_profile_embeddings_generation_idx
    ON category_profile_embeddings (generation_id);

CREATE TABLE IF NOT EXISTS active_category_profile_generation (
    model_key TEXT PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Existing rows are legacy by default. New application code only activates
-- generations whose artifact_kind is category_profile_v1.
