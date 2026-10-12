# Phân tích ý định tìm địa điểm tại TP.HCM

POC này phân tích một câu hỏi tiếng Việt hoặc tiếng Anh thành **loại địa điểm** (*category/place type*) và **yêu cầu về thuộc tính** (*property intent*). Ứng dụng dùng rule/alias để nhận diện và tách các cụm trong câu, sau đó dùng mô hình embedding E5 cùng PostgreSQL/pgvector để đối chiếu category và gợi ý thuộc tính chưa rõ.

> **Phạm vi hiện tại:** Giao diện Streamlit trả về **cách hiểu câu hỏi**, không trả danh sách POI, không tìm theo tên POI, không áp dụng bộ lọc lên địa điểm thật và không đưa ra lời khuyên y tế. Các ca minh họa và điểm số trong POC không phải bằng chứng về độ chính xác hay độ trễ production.

## Mục lục

- [Chức năng và giới hạn](#chức-năng-và-giới-hạn)
- [Cách pipeline hoạt động](#cách-pipeline-hoạt-động)
- [Chạy giao diện](#chạy-giao-diện)
- [Đọc kết quả](#đọc-kết-quả)
- [Cấu hình và dữ liệu](#cấu-hình-và-dữ-liệu)
- [Kiểm tra và xử lý sự cố](#kiểm-tra-và-xử-lý-sự-cố)
- [An toàn cơ sở dữ liệu](#an-toàn-cơ-sở-dữ-liệu)
- [Kiểm thử và đánh giá](#kiểm-thử-và-đánh-giá)
- [Nguồn dữ liệu và hướng phát triển](#nguồn-dữ-liệu-và-hướng-phát-triển)

## Chức năng và giới hạn

| Đã có trong bản hiện tại | Chưa có trong luồng phục vụ hiện tại |
| --- | --- |
| Tách category và thuộc tính được nhắc đến; phân biệt yêu cầu bắt buộc, ưu tiên, loại trừ hoặc không cần. | Tìm POI theo tên hoặc kết hợp tên POI với `place_type`. |
| Tìm ứng viên category bằng E5/pgvector; khi chưa phân định, hiển thị các khả năng để tham khảo. | Truy vấn, lọc, xác minh thuộc tính hoặc xếp hạng POI thực tế. |
| Gợi ý property cho cụm chưa nhận diện; hiển thị latency của các bước phân tích. | Tự chấp nhận ứng viên category/property chưa rõ hoặc tạo filter từ gợi ý. |

Streamlit là **giao diện người dùng**. CLI `model_intent` phục vụ kiểm tra trạng thái và quản trị generation, **không phải** CLI hỏi đáp/tìm kiếm POI.

## Cách pipeline hoạt động

```text
Câu hỏi
  ├─ Dò alias thuộc tính + ngữ cảnh yêu cầu → property nhận diện trực tiếp
  ├─ Diễn giải phần còn lại → cụm category
  │    └─ E5 mã hóa → pgvector so cosine exact → ứng viên và quyết định category
  └─ Cụm thuộc tính chưa hiểu → E5/pgvector riêng → gợi ý, không tự áp dụng
                                       ↓
                       Cấu trúc ý định + thời gian từng bước
```

1. **Tách câu:** Catalog và các quy tắc xử lý vị trí, dấu nối, phủ định giúp phân vai cụm thuộc tính/category. Không cần có từ “có” để nhận diện một property đã có alias; từ “ưu tiên”, “không có”, “không cần” giúp xác định vai trò của nó. Phần không khớp được giữ là *chưa rõ*, không tự biến thành điều kiện lọc.
2. **Đối chiếu category:** E5-base mã hóa phần câu mô tả loại địa điểm. PostgreSQL/pgvector xếp hạng các profile song ngữ theo cosine **exact**. Policy và kiểm tra alias, xung đột, ngữ cảnh ngoài phạm vi quyết định chấp nhận, giữ nhiều ứng viên hoặc từ chối; một từ khớp alias không tự động thắng kết quả semantic.
3. **Xử lý property:** Thuộc tính nhận diện trực tiếp được kiểm tra khả năng áp dụng nếu category đã được chấp nhận. Khi có cụm chưa hiểu, hệ thống mới thử chỉ mục semantic property riêng và hiển thị tối đa ba gợi ý đạt ngưỡng. Lỗi backend được báo ra thay vì âm thầm thêm filter. Giao diện hiện **không có thao tác chọn/xác nhận gợi ý**.

Ví dụ minh họa cách tách: `quán cà phê có máy lạnh` có cụm category `quán cà phê` và thuộc tính trực tiếp `máy lạnh` ở chế độ bắt buộc. Đây là ví dụ về **cấu trúc đầu ra mong đợi**, không bảo đảm một lần chạy trên DB bất kỳ sẽ chấp nhận category hay tìm được POI có máy lạnh.

## Chạy giao diện

### Điều kiện tiên quyết

- Python **≥ 3.12**; hướng dẫn lệnh dưới đây dành cho **Git Bash trên Windows**, chạy từ thư mục gốc dự án.
- PostgreSQL có extension pgvector trên **DB thống nhất đã được chuẩn bị và cấp quyền**, tại `127.0.0.1:5433/model_intent_unified`. Category cần generation active phù hợp với tài nguyên đang phục vụ. Không tạo/migrate DB chỉ để thử các lệnh bên dưới.
- Snapshot của `intfloat/multilingual-e5-base` ở revision ghim phải **đã có trong cache cục bộ**; repository không chứa model weights và UI tải với `local_files_only=True`. Cài Python packages không đồng nghĩa model đã sẵn có.

Tạo virtual environment nếu chưa có và cài phụ thuộc:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[models,db,ui]"
```

Dùng tài khoản **đã được cấp** cho DB thống nhất; `USER` và `PASSWORD` là placeholder, cần mã hóa ký tự đặc biệt trong URL:

```bash
export DATABASE_URL='postgresql://USER:PASSWORD@127.0.0.1:5433/model_intent_unified'
.venv/Scripts/python.exe -m streamlit run src/model_intent/ui.py
```

Mở `http://localhost:8501` hoặc cổng Streamlit thông báo. **Không đưa URL/mật khẩu thật vào tài liệu, log hay commit.** Kiểm tra `.env` hiện có trước khi tham khảo `.env.example`: cấu hình cũ có thể trỏ vào DB legacy ở cổng 5432. UI có thể lấy `DATABASE_URL` từ `.env` ở gốc repo khi process chưa có biến này; **CLI không tự nạp `.env`**.

### Thao tác trên UI

- Chọn ngôn ngữ, nhập câu hỏi rồi bấm **Phân tích query**; hoặc bấm một nút ví dụ để điền câu và chạy phân tích ngay.
- Có **9 câu test nhanh** trong nhóm “Đã xử lý được” và **4 câu** trong nhóm “Chưa xử lý trọn vẹn” cho mỗi ngôn ngữ. Các nhãn này mô tả tình huống minh họa, **không phải kết quả kiểm định trên dữ liệu người dùng**.
- Cột **Category** cho biết loại địa điểm được chấp nhận hoặc danh sách ứng viên chưa chắc chắn. Cột **Properties** phân biệt thuộc tính đã nhận diện, vai trò yêu cầu và các cụm chưa hiểu. Bên dưới là tổng latency trên server, thời gian theo giai đoạn, các bước chi tiết và JSON kỹ thuật.

## Đọc kết quả

- `filter_candidates` (bắt buộc/loại trừ) và `preference_candidates` (ưu tiên) là **điều kiện được suy ra từ câu hỏi**, chưa chạy trên POI. `applied` nghĩa là một property được hiểu và, nếu category đã được chọn, vượt kiểm tra phù hợp; nó **không chứng minh có địa điểm thật đáp ứng**. Khi category chưa chốt, khả năng áp dụng vẫn có thể đang chờ xác định.
- Ứng viên category khi hệ thống chưa phân định và gợi ý property semantic là **tham khảo**, không tự biến thành category đã chọn hay filter. Câu chỉ có property có thể không có category để tìm vector; câu hỏi thông tin/ngoài phạm vi không bị ép thành loại địa điểm.
- Cosine E5 chỉ biểu diễn mức tương đồng, **không phải xác suất đúng**. `calibrated_confidence` v3 là điểm bằng chứng *proxy* tạm thời dựa trên nhãn tác giả/taxonomy tự xây dựng; nó mô tả top-1 semantic (có thể là canonical probe), **không phải xác suất category cuối cùng đúng**. Đọc nguồn quyết định và chứng cứ riêng trong JSON kỹ thuật.
- Latency là thời gian **request phân tích đã hoàn tất** trên server, không phải thời gian tải trang hay hoạt ảnh. Bước con nằm trong thời gian của bước cha, không cộng hai lần; bước bị bỏ qua/không đo không phải `0 ms`.

## Cấu hình và dữ liệu

Cấu hình mặc định là [`configs/benchmark.poi.pg.v3.yaml`](configs/benchmark.poi.pg.v3.yaml); biến `MODEL_INTENT_CONFIG` chỉ nên trỏ tới một YAML khác đã được ghim và kiểm chứng. Một số tài nguyên chính:

| Thành phần | Tài nguyên và cơ chế |
| --- | --- |
| Taxonomy | [`data/taxonomies/place_types.poc.v3.json`](data/taxonomies/place_types.poc.v3.json): **384 category lá**, **768 profile Việt–Anh**; enrichment và alias ở các đường dẫn do YAML quy định. |
| Embedding/category retrieval | `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`, **768 chiều**; pgvector cosine exact, generation category **active** được xác minh khi khởi tạo runtime. |
| Property | [`data/properties/poi_properties.poc.v8.json`](data/properties/poi_properties.poc.v8.json): **107 định nghĩa, 106 queryable**. Corpus gợi ý [`data/properties/property_semantic_profiles.poc.v2.json`](data/properties/property_semantic_profiles.poc.v2.json), backend `pg_exact_v1`, **property generation ID 4** được ghim trong YAML. |

Category generation active và property generation ID ghim được xác minh **độc lập**. Property generation chỉ được kiểm tra trạng thái `ready`/model/catalog/hash khi có cụm cần gợi ý; **ID 4 trong config không chứng minh DB thực tế hiện đã có generation này**. UI không yêu cầu POI generation đang active.

Tên `osm-only` của catalog **không có nghĩa mọi định nghĩa đều có thể kiểm chứng từ thẻ OSM**: một số property có `osm_tags` rỗng; trường `mock_*` còn trong dữ liệu là metadata, không phải bằng chứng về POI thật. Thiếu tag OSM là *chưa biết*, không đồng nghĩa thuộc tính không tồn tại. Không suy diễn rating/review/tiện nghi giả từ những trường đó.

## Kiểm tra và xử lý sự cố

Sau khi đã đặt `DATABASE_URL` đúng DB, có thể buộc kết nối ở chế độ đọc để xem trạng thái (không chạy sync hay migration):

```bash
export PGOPTIONS='-c default_transaction_read_only=on'
.venv/Scripts/python.exe -m model_intent db status
.venv/Scripts/python.exe -m model_intent embeddings status
.venv/Scripts/python.exe -m model_intent properties status
```

`db status` kiểm tra migration ở thư mục gốc và có thể hiển thị `008_poi_category_provenance` chưa áp dụng; điều này **không có nghĩa phải áp dụng 008** để chạy UI. Nếu `properties status` lỗi, trước tiên kiểm tra DB đích/schema property và checksum migration; không tự động chạy migrate. Với lỗi UI, kiểm tra lần lượt `DATABASE_URL`, generation category active, snapshot E5 trong cache và — chỉ khi cần gợi ý cụm property lạ — property generation ID 4. Backend property không khả dụng không được thay bằng kết quả giả.

Hai script chẩn đoán chỉ đọc, xuất ra stdout khi không truyền `--output`:

```bash
.venv/Scripts/python.exe scripts/diagnose_exact_categories.py --runtime intent --item-id 23
.venv/Scripts/python.exe scripts/evaluate_property_semantic_profiles.py --generation-id 4
```

Không lấy kết quả các script này làm chứng cứ rằng POI thật đã được truy vấn hoặc chất lượng người dùng đã được kiểm định.

## An toàn cơ sở dữ liệu

- [`docker-compose.yml`](docker-compose.yml) và [`docker-compose.unified.yml`](docker-compose.unified.yml) theo cấu hình dự án cùng dùng project `model-intent-unified`, service `db` và named volume **`model-intent-unified_model_intent_unified_pgdata`**, công bố tại `127.0.0.1:5433`. Đây **không phải hai DB độc lập**. DB legacy ở cổng 5432 có schema/lịch sử migration khác: không migrate, sync, reset hoặc xóa nó theo hướng dẫn của README này.
- Giữ nguyên checksum/lịch sử của `migrations/001_core.sql` đến `migrations/007_poi_dataset_generations.sql`; `006_category_search_profiles_bge_m3.sql` là lịch sử, không phải model phục vụ hiện nay. `migrations/008_poi_category_provenance.sql` ngoài phạm vi UI/property. Migration property ở [`migrations/property/property_001_semantic_profiles.sql`](migrations/property/property_001_semantic_profiles.sql) có **file và lệnh áp dụng riêng**, nhưng vẫn ghi vào **cùng bảng `schema_migrations`** của DB; không gộp file vào chuỗi root 001–007.
- **Không dùng `docker compose down -v`, không xóa/ghi đè volume, dữ liệu hay backup.** Không chạy migration property, sync embeddings/property, promote/rollback hoặc test ghi DB trên volume phục vụ nếu chưa có preflight, backup và phê duyệt **DB-write riêng**. Những bản backup cục bộ trong `artifacts/db-backups/` bị Git ignore, không phải phần của repo và không được mặc nhiên coi là đã thử khôi phục thành công.

## Kiểm thử và đánh giá

Cài thêm phụ thuộc kiểm thử vào môi trường đang dùng nếu cần, sau đó chạy **unit tests không đòi DB phục vụ**:

```bash
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest tests --ignore=tests/integration -q
```

Chỉ khi có quyền đọc DB thống nhất và đúng snapshot model/generation, có thể **tùy chọn** chạy smoke test tích hợp đọc DB (không chạy mặc định):

```bash
export PGOPTIONS='-c default_transaction_read_only=on'
MODEL_INTENT_READONLY_DB_TEST=1 .venv/Scripts/python.exe -m pytest tests/integration/test_pgvector.py -q
```

Không chạy test có ghi DB trên volume phục vụ; test đánh dấu `pgvector_write` chỉ dùng DB scratch được cho phép riêng. Các case stress, probe và benchmark do nhóm/taxonomy biên soạn chỉ giúp phát hiện regression, **không ước lượng độ chính xác trên người dùng thật hay SLA**. Đánh giá tiếp cần truy vấn được cấp phép, rà soát quyền riêng tư, ẩn danh và gán nhãn độc lập *trước* khi xem dự đoán; tách train/validation/test theo người dùng hoặc nhóm cách diễn đạt và không commit query thô.

## Nguồn dữ liệu và hướng phát triển

Đây là POC phân tích ý định, không phải chatbot hay hệ thống tìm POI đã hoàn chỉnh. Để đáp ứng bài toán tìm kiếm kết hợp `place_type` **và tên POI**, cần bổ sung nhận diện cụm tên, lập chỉ mục/truy vấn tên trên POI phù hợp category, xử lý tên trùng và kiểm chứng bằng tập truy vấn có nhãn độc lập. Các bước này là **định hướng, chưa được triển khai trong luồng phục vụ hiện tại**; không dùng tên file hoặc schema POI lịch sử để suy ra tính năng đã hoạt động.

Dữ liệu trích xuất có nguồn gốc OpenStreetMap: **© OpenStreetMap contributors**, giấy phép **ODbL 1.0**; cần giữ ghi công và tuân thủ điều kiện giấy phép khi phân phối lại dữ liệu bắt nguồn từ OSM. Ghi công dữ liệu không đồng nghĩa mã nguồn repository đã có cùng giấy phép.
