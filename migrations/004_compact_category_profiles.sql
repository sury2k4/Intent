-- Compact v2 category profile artifact.
-- 003 remains untouched so the active v1 profile generation can be rolled back.

ALTER TABLE embedding_generations
    DROP CONSTRAINT IF EXISTS embedding_generations_artifact_profile_ck;

ALTER TABLE embedding_generations
    ADD CONSTRAINT embedding_generations_artifact_profile_ck
    CHECK (
        (artifact_kind = 'variant_v1' AND profile_set_id IS NULL)
        OR (artifact_kind IN ('category_profile_v1', 'category_profile_v2') AND profile_set_id IS NOT NULL)
    );

CREATE TABLE IF NOT EXISTS category_search_profiles_v2 (
    id BIGSERIAL PRIMARY KEY,
    profile_set_id BIGINT NOT NULL REFERENCES category_profile_sets(id) ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    language TEXT NOT NULL CHECK (language IN ('vi', 'en')),
    search_text TEXT NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (length(trim(search_text)) > 0),
    UNIQUE (profile_set_id, item_id, language)
);

CREATE INDEX IF NOT EXISTS category_search_profiles_v2_set_item_idx
    ON category_search_profiles_v2 (profile_set_id, item_id);

CREATE TABLE IF NOT EXISTS category_profile_embeddings_v2 (
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    profile_id BIGINT NOT NULL REFERENCES category_search_profiles_v2(id) ON DELETE CASCADE,
    content_hash CHAR(64) NOT NULL,
    embedding vector(384) NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, profile_id),
    CHECK (vector_dims(embedding) = 384),
    CHECK (vector_norm(embedding) > 0)
);

CREATE INDEX IF NOT EXISTS category_profile_embeddings_v2_generation_idx
    ON category_profile_embeddings_v2 (generation_id);

CREATE TABLE IF NOT EXISTS active_category_profile_generation_v2 (
    model_key TEXT PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
