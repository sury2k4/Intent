-- ANN is optional; exact cosine remains the correctness ground truth.
CREATE INDEX IF NOT EXISTS variant_embeddings_embedding_hnsw_idx
ON variant_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
