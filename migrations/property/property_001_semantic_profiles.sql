-- Apply only with apply_migrations(database, "migrations/property") after 001-007.
-- Property generations are independent of category embeddings and POI generations.
-- Staging never changes the active pointer.

CREATE TABLE property_profile_sets (
    id BIGSERIAL PRIMARY KEY,
    catalog_hash CHAR(64) NOT NULL CHECK (catalog_hash ~ '^[0-9a-f]{64}$'),
    source_hash CHAR(64) NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    taxonomy_fingerprint CHAR(64) NOT NULL CHECK (taxonomy_fingerprint ~ '^[0-9a-f]{64}$'),
    renderer_version TEXT NOT NULL CHECK (length(trim(renderer_version)) > 0),
    content_set_hash CHAR(64) NOT NULL CHECK (content_set_hash ~ '^[0-9a-f]{64}$'),
    property_count INTEGER NOT NULL CHECK (property_count > 0),
    profile_count INTEGER NOT NULL CHECK (profile_count >= property_count),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (catalog_hash, source_hash, taxonomy_fingerprint, renderer_version, content_set_hash)
);

CREATE TABLE property_embedding_generations (
    id BIGSERIAL PRIMARY KEY,
    profile_set_id BIGINT NOT NULL REFERENCES property_profile_sets(id),
    model_config_id BIGINT NOT NULL REFERENCES embedding_model_configs(id),
    content_set_hash CHAR(64) NOT NULL CHECK (content_set_hash ~ '^[0-9a-f]{64}$'),
    state TEXT NOT NULL DEFAULT 'building' CHECK (state IN ('building', 'ready', 'failed', 'retired')),
    expected_count INTEGER NOT NULL CHECK (expected_count > 0),
    completed_count INTEGER NOT NULL DEFAULT 0 CHECK (completed_count >= 0 AND completed_count <= expected_count),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (profile_set_id, model_config_id),
    CHECK (state <> 'ready' OR (completed_count = expected_count AND finished_at IS NOT NULL))
);

CREATE TABLE property_semantic_profiles (
    id BIGSERIAL PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES property_embedding_generations(id) ON DELETE CASCADE,
    property_id TEXT NOT NULL CHECK (length(trim(property_id)) > 0),
    language TEXT NOT NULL CHECK (language IN ('vi', 'en')),
    profile_key TEXT NOT NULL CHECK (length(trim(profile_key)) > 0),
    search_text TEXT NOT NULL CHECK (length(trim(search_text)) > 0),
    content_hash CHAR(64) NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    embedding vector(768) NOT NULL CHECK (vector_dims(embedding) = 768 AND vector_norm(embedding) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (generation_id, property_id, language, profile_key)
);

CREATE INDEX property_semantic_profiles_generation_idx
    ON property_semantic_profiles (generation_id);
CREATE INDEX property_embedding_generations_state_idx
    ON property_embedding_generations (state, model_config_id);

CREATE TABLE active_property_embedding_generation (
    model_key TEXT PRIMARY KEY CHECK (model_key = 'e5-base-multilingual'),
    generation_id BIGINT NOT NULL REFERENCES property_embedding_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
