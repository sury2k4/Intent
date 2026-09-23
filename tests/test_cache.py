import numpy as np

from model_intent.cache import load_matrix, save_matrix


def test_matrix_cache_round_trip_and_metadata_invalidation(tmp_path):
    path = tmp_path / "vectors.npz"
    matrix = np.ones((2, 3), dtype=np.float32)
    metadata = {"model": "fake", "rows": 2}
    save_matrix(path, matrix, metadata)
    loaded = load_matrix(path, metadata)
    assert np.array_equal(loaded, matrix)
    assert load_matrix(path, {"model": "other", "rows": 2}) is None
