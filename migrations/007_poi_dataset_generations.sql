-- Immutable HCMC OSM POI dataset generations for the POC.
-- Keep category/profile migrations unchanged; this layer is independently activated.

CREATE TABLE IF NOT EXISTS poi_dataset_generations (
    id BIGSERIAL PRIMARY KEY,
    dataset_key TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'building',
    source_manifest_hash CHAR(64),
    artifact_sha256 CHAR(64),
    content_set_hash CHAR(64),
    taxonomy_fingerprint CHAR(64) NOT NULL,
    mapping_version TEXT NOT NULL,
    mapping_sha256 CHAR(64) NOT NULL,
    property_taxonomy_version TEXT NOT NULL,
    property_taxonomy_sha256 CHAR(64) NOT NULL,
    mock_schema_version TEXT NOT NULL,
    mock_seed INTEGER NOT NULL,
    expected_count INTEGER NOT NULL CHECK (expected_count >= 0),
    completed_count INTEGER NOT NULL DEFAULT 0 CHECK (completed_count >= 0),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    error TEXT,
    UNIQUE (dataset_key, artifact_id),
    CHECK (state IN ('building', 'ready', 'failed', 'retired')),
    CHECK (completed_count <= expected_count OR state = 'failed')
);

CREATE TABLE IF NOT EXISTS pois (
    generation_id BIGINT NOT NULL REFERENCES poi_dataset_generations(id) ON DELETE CASCADE,
    poi_id TEXT NOT NULL,
    osm_type TEXT NOT NULL,
    osm_id BIGINT NOT NULL,
    place_type_id INT NOT NULL,
    name TEXT,
    display_name TEXT NOT NULL,
    names JSONB NOT NULL DEFAULT '{}'::jsonb,
    latitude DOUBLE PRECISION NOT NULL CHECK (latitude BETWEEN -90 AND 90),
    longitude DOUBLE PRECISION NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    location_method TEXT NOT NULL,
    mapping_rule_id TEXT NOT NULL,
    osm_tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    properties_osm JSONB NOT NULL DEFAULT '{}'::jsonb,
    properties_mock JSONB NOT NULL DEFAULT '{}'::jsonb,
    effective_properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    rating NUMERIC(2,1) CHECK (rating IS NULL OR rating BETWEEN 1.0 AND 5.0),
    review_count INTEGER NOT NULL DEFAULT 0 CHECK (review_count >= 0),
    rating_source TEXT NOT NULL DEFAULT 'mock',
    rating_mock_schema_version TEXT,
    rating_mock_seed INTEGER,
    content_hash CHAR(64) NOT NULL,
    source_version INTEGER,
    source_timestamp TIMESTAMPTZ,
    PRIMARY KEY (generation_id, poi_id),
    UNIQUE (generation_id, osm_type, osm_id),
    CHECK (rating_source IN ('osm', 'mock', 'unknown'))
);

CREATE INDEX IF NOT EXISTS pois_generation_place_type_idx
    ON pois (generation_id, place_type_id);

CREATE INDEX IF NOT EXISTS pois_generation_lat_lon_idx
    ON pois (generation_id, latitude, longitude);

CREATE INDEX IF NOT EXISTS pois_effective_properties_gin_idx
    ON pois USING gin (effective_properties jsonb_path_ops);

CREATE INDEX IF NOT EXISTS pois_generation_rating_idx
    ON pois (generation_id, rating DESC NULLS LAST, review_count DESC);

CREATE TABLE IF NOT EXISTS active_poi_dataset_generation (
    dataset_key TEXT PRIMARY KEY,
    generation_id BIGINT NOT NULL REFERENCES poi_dataset_generations(id),
    activated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
