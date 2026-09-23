-- Additive v3 serving contract: one row contains text and its 768D embedding.
-- Migrations 001-004 remain unchanged for rollback.

ALTER TABLE embedding_model_configs
    DROP CONSTRAINT IF EXISTS embedding_model_configs_dimension_check;

ALTER TABLE embedding_model_configs
    ADD CONSTRAINT embedding_model_configs_dimension_check
    CHECK (dimension IN (384, 768));

ALTER TABLE embedding_generations
    DROP CONSTRAINT IF EXISTS embedding_generations_artifact_profile_ck;

ALTER TABLE embedding_generations
    ADD CONSTRAINT embedding_generations_artifact_profile_ck
    CHECK (
        (artifact_kind = 'variant_v1' AND profile_set_id IS NULL)
        OR (artifact_kind IN ('category_profile_v1', 'category_profile_v2', 'category_profile_v3')
            AND profile_set_id IS NOT NULL)
    );

CREATE TABLE IF NOT EXISTS category_search_profiles_v3 (
    id SERIAL PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id) ON DELETE CASCADE,
    place_type_id INT NOT NULL,
    language VARCHAR(10) NOT NULL DEFAULT 'vi',
    search_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    content_hash CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (language IN ('vi', 'en')),
    CHECK (length(trim(search_text)) > 0),
    CHECK (vector_dims(embedding) = 768),
    CHECK (vector_norm(embedding) > 0),
    UNIQUE (generation_id, place_type_id, language)
);

CREATE INDEX IF NOT EXISTS category_search_profiles_v3_generation_idx
    ON category_search_profiles_v3 (generation_id);

CREATE INDEX IF NOT EXISTS category_search_profiles_v3_generation_place_idx
    ON category_search_profiles_v3 (generation_id, place_type_id);

CREATE INDEX IF NOT EXISTS category_search_profiles_v3_embedding_ivfflat_idx
    ON category_search_profiles_v3
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);

CREATE TABLE IF NOT EXISTS active_category_profile_generation_v3 (
    model_key TEXT PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES embedding_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
