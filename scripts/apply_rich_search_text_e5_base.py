# -*- coding: utf-8 -*-
"""Update search_text and embeddings for all categories in category_search_profiles_v3 (E5-Base 768D)."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import psycopg
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INTENTS_PATH = ROOT / "data/taxonomies/place_type_intent_keywords.v1.json"
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://model_intent:model_intent_dev@localhost:5432/model_intent",
)
MODEL_ID = "intfloat/multilingual-e5-base"
DOCUMENT_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def compute_content_hash(generation_id: int, place_type_id: int, language: str, text: str) -> str:
    raw = f"{generation_id}:{place_type_id}:{language}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main():
    print(f"Connecting to database: {DATABASE_URL}...")
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    
    with conn.cursor() as cur:
        # 1. Get active generation for e5-base-multilingual
        cur.execute("SELECT generation_id FROM active_category_profile_generation_v3 WHERE model_key = 'e5-base-multilingual';")
        row = cur.fetchone()
        if not row:
            cur.execute("SELECT id FROM embedding_generations WHERE artifact_kind = 'category_profile_v3' AND state = 'ready' ORDER BY id DESC LIMIT 1;")
            row = cur.fetchone()
        generation_id = row[0]
        print(f"Active E5-base generation_id = {generation_id}")

        # 2. Load intents
        print(f"Loading intent keywords from {INTENTS_PATH}...")
        payload = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
        items = payload["items"]
        print(f"Loaded {len(items)} category intents.")

        # 3. Load model
        print(f"Loading {MODEL_ID} model...")
        model_start = time.perf_counter()
        embedder = SentenceTransformer(MODEL_ID, device="cpu")
        print(f"Model loaded in {time.perf_counter() - model_start:.2f}s")

        # 4. Prepare passages with prefix
        place_type_ids = sorted([int(k) for k in items.keys()])
        raw_texts = [items[str(pid)]["search_text_vi"] for pid in place_type_ids]
        passages = [f"{DOCUMENT_PREFIX}{text}" for text in raw_texts]

        print(f"Encoding {len(passages)} passages with E5-base (768D)...")
        encode_start = time.perf_counter()
        embeddings = embedder.encode(
            passages,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print(f"Encoded {len(passages)} passages in {time.perf_counter() - encode_start:.2f}s. Shape: {embeddings.shape}")

        # 5. Update database table category_search_profiles_v3
        print("Updating database table category_search_profiles_v3...")
        update_start = time.perf_counter()
        updated_count = 0
        for pid, raw_text, emb in zip(place_type_ids, raw_texts, embeddings):
            c_hash = compute_content_hash(generation_id, pid, "vi", raw_text)
            emb_list = emb.tolist()
            cur.execute(
                """
                UPDATE category_search_profiles_v3
                SET search_text = %s,
                    embedding = %s::vector,
                    content_hash = %s,
                    created_at = now()
                WHERE generation_id = %s AND place_type_id = %s AND language = 'vi';
                """,
                (raw_text, emb_list, c_hash, generation_id, pid),
            )
            updated_count += cur.rowcount

        conn.commit()
        print(f"Updated {updated_count} rows in {time.perf_counter() - update_start:.2f}s.")

        # 6. Reindex table
        print("Reindexing category_search_profiles_v3...")
        cur.execute("REINDEX TABLE category_search_profiles_v3;")
        conn.commit()
        print("Reindex complete.")

    # 7. Verify with test queries
    print("\n" + "=" * 80)
    print("TESTING RETRIEVAL QUALITY FOR E5-BASE (768D) VIA PGVECTOR")
    print("=" * 80)

    test_queries = [
        "quán cà phê muối wifi mạnh ngồi học bài",
        "đau đầu sổ mũi mua thuốc hạ sốt panadol",
        "xe máy bị thủng lốp cán đinh cần vá",
        "mua bánh kem sinh nhật cắm nến",
        "cần đổi pin xe máy điện vinfast",
        "đi nhậu bia hơi vỉa hè với đồng nghiệp",
        "cần tìm cây rút tiền mặt gần đây",
        "tìm quán cơm bình dân ăn trưa",
        "đau răng buốt cần nhổ răng khôn",
    ]

    prefixed_queries = [f"{QUERY_PREFIX}{q}" for q in test_queries]
    query_embeddings = embedder.encode(prefixed_queries, normalize_embeddings=True)

    with conn.cursor() as cur:
        for query, q_emb in zip(test_queries, query_embeddings):
            q_vec_str = str(q_emb.tolist())
            cur.execute(
                """
                SELECT place_type_id, 1 - (embedding <=> %s::vector) AS similarity, search_text
                FROM category_search_profiles_v3
                WHERE generation_id = %s AND language = 'vi'
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1;
                """,
                (q_vec_str, generation_id, q_vec_str),
            )
            top_hit = cur.fetchone()
            print(f"\nQuery: '{query}'")
            print(f"  -> Top 1: [place_type_id={top_hit[0]}] Similarity: {top_hit[1]:.4f}")
            print(f"  -> Snippet: {top_hit[2][:100]}...")

    conn.close()
    print("\nE5-base update and verification completed successfully!")


if __name__ == "__main__":
    main()
