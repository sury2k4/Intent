# Phân tích ý định loại địa điểm và thuộc tính (HCMC)

Ứng dụng **Streamlit-only** này tách một câu hỏi thành loại địa điểm (*category/place type*) và yêu cầu thuộc tính (*property intent*). Category được truy xuất bằng pgvector trên PostgreSQL; thuộc tính được nhận diện theo catalog và, với cụm chưa khớp, có thể được **gợi ý** bằng E5. Giao diện có các câu minh họa bấm nhanh, hiển thị Category và Properties thành hai cột để đối chiếu, cùng thời gian xử lý theo bước; các ứng viên chưa rõ chỉ để tham khảo và **không có thao tác xác nhận ứng viên trong UI**. Kết quả là cấu trúc ý định để xem, **không phải danh sách POI, bộ lọc đã chạy trên địa điểm thật, chatbot hay lời khuyên y tế**. Đây là POC; không có cam kết độ chính xác hoặc độ trễ production.

## Cấu hình đang dùng

- Giao diện dùng `configs/benchmark.poi.pg.v3.yaml`, taxonomy `data/taxonomies/place_types.poc.v3.json` (384 category lá, 768 profile song ngữ) và model `intfloat/multilingual-e5-base` đã ghim revision `d128750597153bb5987e10b1c3493a34e5a4502a` (768D). Category retrieval là cosine **exact** trong pgvector, không phải tìm kiếm từ khóa; các ngưỡng confidence thuộc hiệu chỉnh v3 tạm thời.
- Chính **cấu hình v3 này** dùng mục `property_intent`, catalog thuộc tính OSM-only `data/properties/poi_properties.poc.v8.json`, corpus E5 `data/properties/property_semantic_profiles.poc.v2.json` và `property_generation_id: 4` với backend `pg_exact_v1`. Property profile thuộc bảng/generation riêng trong cùng DB; ID ghim phải ở trạng thái `ready` và khớp model/catalog/hash. Nếu backend property lỗi, UI hiển thị cảnh báo khi cần gợi ý cho cụm chưa nhận diện, không tự tạo filter hoặc ngầm dùng artifact. Category generation đang active được xác minh độc lập với property generation.
- Category alias do taxonomy và `data/taxonomies/category_query_aliases.poc.v3.json` cung cấp. Cụm category khớp duy nhất có thể kích hoạt **lexical alias → canonical E5 probe** có điều kiện; kết quả chỉ được dùng khi E5 và các kiểm tra xung đột/phạm vi xác nhận. Alias không thay thế cosine, không nâng ngưỡng confidence, không tự gán một cụm mơ hồ. Một số cách viết gần đúng chỉ được **hiển thị như gợi ý tham khảo**, không tự chấp nhận. Câu phủ định/ngoài phạm vi hoặc hỏi thông tin thay vì tìm nơi không được ép thành category.
- Cụm thuộc tính có alias trực tiếp được phân tích với dấu vết và chế độ bắt buộc/ưu tiên/loại trừ; cụm chưa rõ giữ trạng thái chưa giải quyết. E5 so sánh cụm chưa rõ với property semantic profiles và hiển thị tối đa ba ứng viên tham khảo, **không tự đưa gợi ý vào điều kiện lọc**. UI hiện tại không có thao tác chọn/xác nhận ứng viên; khi category chưa rõ, giao diện cũng chỉ liệt kê tối đa năm khả năng, không ép chọn một. Điểm cosine không phải xác suất đúng, và `applied` chỉ có nghĩa là thuộc tính nhận diện trực tiếp được hiểu/phù hợp với category, **không chứng minh POI nào có thuộc tính đó**. Thiếu thẻ OSM là *chưa biết*, không phải bằng chứng thuộc tính không tồn tại; không suy diễn rating/review/tiện nghi giả.

## Cài và mở giao diện (Git Bash, Windows)

Yêu cầu Python >=3.12, PostgreSQL/pgvector thống nhất đã được cấp sẵn và model E5 được tải/có trong cache. Nếu chưa có virtual environment, tạo một lần; không thay thế `.env` hiện có:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[models,db,ui]"
```

Dùng URL của **DB thống nhất** `model_intent_unified` tại `127.0.0.1:5433`, thay `USER`/`PASSWORD` bằng tài khoản đã cấp (mã hóa ký tự đặc biệt trong URL nếu cần):

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:5433/model_intent_unified'
.venv/Scripts/python.exe -m streamlit run src/model_intent/ui.py
```

Mở `http://localhost:8501` (hoặc cổng Streamlit báo khi 8501 bận). UI có thể đọc `DATABASE_URL` từ `.env` ở gốc dự án nếu process chưa có biến môi trường; **CLI không tự nạp `.env`**. Để thử config khác trong repo, đặt `MODEL_INTENT_CONFIG` tới một file YAML đã pin/kiểm chứng; mặc định là v3. Compose cũng chỉ đọc `.env` để thay biến của Compose. Kiểm tra `.env` đang có trước khi dùng `.env.example`: cấu hình cũ có thể trỏ tới DB legacy ở cổng 5432. Không ghi URL/mật khẩu thật vào tài liệu hoặc commit. Khởi động UI không cần một POI generation đang active và không chạy truy vấn/xếp hạng POI.

Các lệnh CLI quản trị **đọc trạng thái** (không phải CLI query/search) cần `DATABASE_URL` đúng như trên:

```bash
.venv/Scripts/python.exe -m model_intent db status
.venv/Scripts/python.exe -m model_intent embeddings status
.venv/Scripts/python.exe -m model_intent properties status
```

Chẩn đoán category/property đọc DB nhưng không đồng bộ dữ liệu: `scripts/diagnose_exact_categories.py --runtime intent --item-id 23`; `scripts/evaluate_property_semantic_profiles.py --generation-id 4`. `scripts/evaluate_category_confidence.py` chỉ dùng với bộ query người dùng đã được cấp phép, ẩn danh và gán nhãn độc lập; nhãn do taxonomy/nhóm phát triển tự tạo không phải gold chất lượng thực tế. Các script không truy vấn POI.

## An toàn dữ liệu và migration

`docker-compose.yml` và `docker-compose.unified.yml` cùng trỏ project `model-intent-unified`, service `db` và **cùng named volume** `model-intent-unified_model_intent_unified_pgdata` (host `127.0.0.1:5433`). Chúng không tạo hai DB riêng. **Không dùng `docker compose down -v`, không xóa volume, không reset/ghi đè dữ liệu hay backup**. Volume legacy ở cổng 5432 có schema và migration ledger khác: không migrate/sync nó, không xóa nó. Các bản sao lưu trước đây ở `artifacts/db-backups/2026-10-01-unified-preflight/` và `artifacts/db-backups/2026-10-02-pre-osm-v3-rollout/` là dữ liệu cục bộ bị Git ignore; giữ lại cho đến khi có quyết định xử lý riêng. Đừng coi backup là đã kiểm tra phục hồi đầy đủ.

Giữ nguyên lịch sử migration gốc có kiểm checksum **`migrations/001_core.sql` đến `migrations/007_poi_dataset_generations.sql`** (bao gồm `006_category_search_profiles_bge_m3.sql` mang tính lịch sử, không phải model phục vụ hiện tại). `migrations/008_poi_category_provenance.sql` nằm ngoài phạm vi property và không được áp dụng chỉ để chạy UI. Schema property là migration **riêng** `migrations/property/property_001_semantic_profiles.sql`, không được gộp vào ledger gốc 001–007; trạng thái/việc áp dụng phải được xem xét riêng trên đúng DB sau preflight và backup. **Không chạy `db migrate`, `properties migrate`, bất kỳ lệnh sync/activate/promote/rollback nào, hay test ghi DB trên volume đang phục vụ** nếu chưa có kế hoạch và phê duyệt DB-write riêng. Các lệnh ngoài hai lệnh status ở trên chưa được hướng dẫn sử dụng trong README này.

## Giới hạn diễn giải

E5 top-hit cosine **không phải xác suất**. `calibrated_confidence` v3 là điểm proxy tạm thời dựa trên nhãn được tác giả/taxonomy biên soạn, **chưa được kiểm định độc lập bằng truy vấn người dùng**; nó mô tả semantic top-1 (sau canonical probe có thể là câu canonical), không phải độ chắc chắn của category cuối cùng do alias hoặc lựa chọn của người dùng. Các trường quyết định/nguồn chứng cứ phải được đọc tách biệt. Nếu đánh giá lại confidence, chỉ dùng query người dùng đã được cho phép và rà soát riêng tư, ẩn định danh, gán nhãn độc lập trước khi xem dự đoán, tách train/validation/test theo người dùng/nhóm diễn đạt; không ghi log hay commit query thô. Tập kiểm thử tự biên soạn hoặc số liệu benchmark cũ không chứng minh độ chính xác thực tế hay SLA. OSM data © OpenStreetMap contributors, ODbL 1.0; giữ ghi công khi phân phối lại trích xuất bắt nguồn từ OSM.
