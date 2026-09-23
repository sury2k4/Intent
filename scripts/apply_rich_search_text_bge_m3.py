# -*- coding: utf-8 -*-
"""Update search_text and embeddings for all categories in category_search_profiles_bge_m3."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
INTENTS_PATH = ROOT / "data/taxonomies/place_type_intent_keywords.v1.json"
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://model_intent:model_intent_dev@localhost:5432/model_intent",
)


def compute_content_hash(generation_id: int, place_type_id: int, language: str, text: str) -> str:
    raw = f"{generation_id}:{place_type_id}:{language}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def main():
    print(f"Connecting to database: {DATABASE_URL}...")
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    
    with conn.cursor() as cur:
        # 1. Get active generation for bge-m3
        cur.execute("SELECT generation_id FROM active_category_profile_generation_bge_m3 WHERE model_key = 'bge-m3';")
        row = cur.fetchone()
        if not row:
            # Fallback check
            cur.execute("SELECT id FROM embedding_generations WHERE artifact_kind = 'category_profile_bge_m3_v1' ORDER BY id DESC LIMIT 1;")
            row = cur.fetchone()
        generation_id = row[0]
        print(f"Active BGE-M3 generation_id = {generation_id}")

        # 2. Load intents
        print(f"Loading intent keywords from {INTENTS_PATH}...")
        payload = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
        items = payload["items"]
        print(f"Loaded {len(items)} category intents.")

        # 3. Load model
        print("Loading BAAI/bge-m3 model...")
        model_start = time.perf_counter()
        embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")
        print(f"Model loaded in {time.perf_counter() - model_start:.2f}s")

        # 4. Prepare batches
        place_type_ids = sorted([int(k) for k in items.keys()])
        texts = [items[str(pid)]["search_text_vi"] for pid in place_type_ids]

        print(f"Encoding {len(texts)} search_texts with BAAI/bge-m3...")
        encode_start = time.perf_counter()
        embeddings = embedder.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        print(f"Encoded {len(texts)} texts in {time.perf_counter() - encode_start:.2f}s. Shape: {embeddings.shape}")

        # 5. Update database
        print("Updating database table category_search_profiles_bge_m3...")
        update_start = time.perf_counter()
        updated_count = 0
        for pid, text, emb in zip(place_type_ids, texts, embeddings):
            c_hash = compute_content_hash(generation_id, pid, "vi", text)
            emb_list = emb.tolist()
            cur.execute(
                """
                UPDATE category_search_profiles_bge_m3
                SET search_text = %s,
                    embedding = %s::vector,
                    content_hash = %s,
                    created_at = now()
                WHERE generation_id = %s AND place_type_id = %s AND language = 'vi';
                """,
                (text, emb_list, c_hash, generation_id, pid),
            )
            updated_count += cur.rowcount

        conn.commit()
        print(f"Updated {updated_count} rows in {time.perf_counter() - update_start:.2f}s.")

    # 6. Verify with test queries
    print("\n" + "=" * 60)
    print("TESTING RETRIEVAL QUALITY WITH PGVECTOR COSINE SIMILARITY")
    print("=" * 60)

    test_queries = [
        "quán cà phê muối wifi mạnh ngồi học bài",
        "đau đầu sổ mũi mua thuốc hạ sốt panadol",
        "xe máy bị thủng lốp cán đinh cần vá",
        "mua bánh kem sinh nhật cắm nến",
        "cần đổi pin xe máy điện vinfast",
        "đi nhậu bia hơi vỉa hè với đồng nghiệp",
        "chụp ảnh thẻ căn cước lấy ngay",
        "rút tiền mặt thẻ atm",
    ]

    query_embeddings = embedder.encode(test_queries, normalize_embeddings=True)

    with conn.cursor() as cur:
        for query, q_emb in zip(test_queries, query_embeddings):
            q_vec = q_emb.tolist()
            cur.execute(
                """
                SELECT place_type_id, search_text, 1 - (embedding <=> %s::vector) AS similarity
                FROM category_search_profiles_bge_m3
                WHERE generation_id = %s AND language = 'vi'
                ORDER BY embedding <=> %s::vector ASC
                LIMIT 1;
                """,
                (q_vec, generation_id, q_vec),
            )
            top_hit = cur.fetchone()
            print(f"\nQuery: '{query}'")
            print(f"  -> Top 1: [place_type_id={top_hit[0]}] Similarity: {top_hit[2]:.4f}")
            print(f"  -> Snippet: {top_hit[1][:100]}...")

    conn.close()
    print("\nUpdate and verification completed successfully!")


if __name__ == "__main__":
    main()
