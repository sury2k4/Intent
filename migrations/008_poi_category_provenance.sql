-- Optional per-POI category provenance. Existing generations and rows remain unchanged.
ALTER TABLE poi_dataset_generations
    ADD COLUMN evidence_mode TEXT DEFAULT NULL
    CHECK (evidence_mode IN ('osm_only', 'mixed_osm_mock'));

ALTER TABLE pois
    ADD COLUMN category_source TEXT DEFAULT NULL
    CHECK (category_source IN ('osm', 'osm_generic', 'mock')),
    ADD COLUMN category_evidence JSONB DEFAULT NULL
    CHECK (category_evidence IS NULL OR jsonb_typeof(category_evidence) = 'object');
