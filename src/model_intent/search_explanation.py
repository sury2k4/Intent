from __future__ import annotations

from typing import Any, Mapping

_CATEGORY_REASONS = {
    "explicit_place_type_phrase": "nhận ra tên loại địa điểm viết trực tiếp trong query",
    "high_confidence_leaf": "bộ semantic đủ tự tin ở một category cụ thể",
    "explicit_user_selection": "bạn đã chọn loại địa điểm sau bước hỏi lại",
    "multiple_explicit_place_types": "query có nhiều loại địa điểm nên cần hỏi lại",
    "uncertain_unrelated_candidates": "bộ semantic không đủ tin cậy để chọn một category",
    "informational_not_place_search": "đây là câu hỏi thông tin, không phải yêu cầu tìm POI",
    "unsupported_branded_category": "taxonomy chỉ có loại địa điểm gắn thương hiệu, còn query không nêu thương hiệu đó",
    "semantic_variants_disagree": "hai biến thể query cho E5 ra category khác nhau",
}
_MODE_LABELS = {"required": "bắt buộc", "preferred": "ưu tiên", "excluded": "loại trừ"}
_OPERATOR_LABELS = {"eq": "bằng", "gte": "từ", "gt": "lớn hơn", "lte": "tối đa", "lt": "nhỏ hơn", "contains": "chứa"}


def _value_text(value: Any) -> str:
    if isinstance(value, bool):
        return "có" if value else "không"
    if value is None:
        return "chưa rõ"
    return str(value)


def search_steps(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Explain recorded decisions; never infer a POI fact from its category."""
    intent = result.get("intent") or {}
    decision = intent.get("decision") or {}
    prediction = intent.get("prediction") or {}
    resolution = intent.get("category_resolution") or {}
    extraction = result.get("property_extraction") or {}
    search = result.get("poi_search") or {}
    osm_only = int(search.get("ranking_schema_version") or 0) >= 3
    steps: list[dict[str, Any]] = []

    action = decision.get("action", "unknown")
    category = decision.get("accepted_code") or decision.get("accepted_item_id") or "chưa chọn"
    source = resolution.get("source") or decision.get("reason") or "không rõ"
    reason = decision.get("reason", "không rõ")
    readable_reason = _CATEGORY_REASONS.get(str(reason), str(reason).replace("_", " "))
    action_label = {"accept": "đã chọn", "clarify": "cần hỏi lại", "reject": "chưa xác định"}.get(action, action)
    category_detail = [f"Kết luận: {action_label} → {category}; lý do: {readable_reason}."]
    explicit = resolution.get("explicit_matches") or []
    if source in {"explicit_phrase", "semantic_explicit_conflict", "ambiguous_explicit_phrases"} and explicit:
        category_detail.append("Cụm category tìm thấy: " + ", ".join(
            f"‘{row['evidence']}’ (ID {row['item_id']})" for row in explicit
        ) + ".")
    elif source in {"semantic_model", "semantic_guard_clarify", "semantic_variants_disagree", "unsupported_category"}:
        category_detail.append("E5-base 768D đánh giá category cho từng biến thể query; alias chỉ hỗ trợ phát hiện xung đột, không ép E5 accept.")
        interpretation = intent.get("query_interpretation") or {}
        roles = interpretation.get("property_roles") or []
        if roles:
            category_detail.append("Phân vai cụm từ: " + ", ".join(
                f"‘{row['evidence']}’ → {row['role']}" for row in roles
            ) + ".")
        for variant in resolution.get("semantic_variants") or []:
            category_detail.append(
                f"E5 trên ‘{variant.get('query', '?')}’ → {variant.get('action', '?')} "
                f"{variant.get('accepted_item_id') or ''}".strip() + "."
            )
        if interpretation.get("focused_reason") == "focused_text_has_no_meaningful_place_target":
            category_detail.append("Bản tách thuộc tính mất nghĩa category; chỉ dùng query gốc cho E5.")
        if source == "unsupported_category":
            category_detail.append("Category E5 gợi ý là loại gắn thương hiệu, nhưng query không nhắc thương hiệu; không trả POI sai loại. Cần bổ sung category và dữ liệu POI chung để hỗ trợ query này.")
        elif source == "semantic_guard_clarify":
            category_detail.append("Query nhắc nhiều loại địa điểm hoặc khác category E5; cần chọn trước khi tìm POI.")
    elif source == "informational_veto":
        category_detail.append("E5 đã chạy nhưng câu này có dạng hỏi thông tin; không tìm POI từ một câu hỏi không phải search.")
    elif source == "semantic_accept":
        category_detail.append("Bộ semantic đã chấp nhận category; không dùng fallback alias.")
    elif decision.get("reason") == "explicit_user_selection":
        category_detail.append("Category được chọn ở bước hỏi lại người dùng.")
    else:
        category_detail.append("Thử bộ phân loại semantic trên nguyên query; không tìm được một category tường minh duy nhất.")
    top_hits = prediction.get("hits") or []
    if top_hits:
        first = top_hits[0]
        category_detail.append(
            f"Category semantic đứng đầu: {first.get('code', first.get('item_id'))}, "
            f"cosine {float(first.get('score', 0)):.3f}; đây không phải điểm ranking POI."
        )
    steps.append({"title": "1. Xác định loại địa điểm", "details": category_detail})

    constraints = extraction.get("properties") or []
    if constraints:
        property_detail = []
        for row in constraints:
            status = "áp dụng" if row.get("applied") else f"bỏ qua ({row.get('ignored_reason') or 'không rõ'})"
            mode = _MODE_LABELS.get(str(row.get("mode")), str(row.get("mode", "required")))
            operator = _OPERATOR_LABELS.get(str(row.get("operator")), str(row.get("operator", "eq")))
            property_detail.append(
                f"‘{row.get('evidence_text') or '?'}’ → {row.get('property_id')} "
                f"{operator} {_value_text(row.get('value'))} "
                f"[{mode}, {status}]."
            )
    else:
        property_detail = ["Không phát hiện yêu cầu thuộc tính có cấu trúc trong query."]
    property_detail.extend(str(warning) for warning in extraction.get("warnings") or [])
    steps.append({"title": "2. Tách yêu cầu thuộc tính", "details": property_detail})

    if action == "accept":
        candidates = int(search.get("category_candidate_count", 0))
        eligible = int(search.get("hard_filter_match_count", 0))
        filter_detail = [
            f"Bắt đầu với {candidates:,} POI thuộc category đã chọn; "
            f"còn {eligible:,} POI sau các điều kiện bắt buộc/loại trừ và bán kính (nếu áp dụng)."
        ]
        if search.get("verified_only"):
            filter_detail.append("Chỉ dùng bằng chứng OSM cho điều kiện cứng; mock/chưa rõ không tự được coi là xác minh.")
        else:
            filter_detail.append("Chế độ POC: thuộc tính mock có thể giúp POI qua filter; không phải dữ kiện được xác minh.")
        verified = search.get("verified_required_count")
        if search.get("effective_radius_km") is not None:
            filter_detail.append(f"Bán kính thực sự áp dụng: {float(search['effective_radius_km']):g} km.")
        if verified is not None:
            filter_detail.append(
                f"Trong số POI qua filter: {int(verified):,} có bằng chứng OSM cho điều kiện cứng; "
                f"{int(search.get('mock_supported_count') or 0):,} chưa đủ bằng chứng OSM."
            )
        filter_detail.extend(str(warning) for warning in search.get("warnings") or [])
    else:
        filter_detail = [f"Không truy vấn POI vì bước 1 có kết quả ‘{action}’."]
    steps.append({"title": "3. Lọc POI", "details": filter_detail})

    if action == "accept" and search.get("status") == "ok":
        ranking_detail = [
            f"Xếp hạng {int(search.get('hard_filter_match_count', 0)):,} POI hợp lệ, "
            f"trả {int(search.get('returned_count', 0)):,} kết quả.",
        ]
        if osm_only:
            ranking_detail.extend((
                "Chỉ dùng thuộc tính OSM đã biết cho ưu tiên và khoảng cách khi có tọa độ gốc; "
                "rating và lượng review không tham gia ranking, không dùng điểm giả lập hay điểm rating tiên nghiệm.",
                "Khi bằng điểm thì xét tên hiển thị rồi POI ID để thứ tự ổn định; "
                "tên không có trong OSM chỉ là nhãn được tạo, không phải tên địa điểm.",
            ))
        else:
            ranking_detail.extend((
                "Thuộc tính ưu tiên tăng điểm; rating Bayesian và lượng review tham gia ranking. "
                "Điểm rating/review mock bị giảm ảnh hưởng; khoảng cách chỉ được tính khi có tọa độ gốc.",
                "Cùng điểm thì xét adjusted rating, số review, tên rồi POI ID để thứ tự ổn định.",
            ))
    elif action == "accept":
        ranking_detail = [f"Không xếp hạng POI vì trạng thái: {search.get('status', 'không có kết quả')}."]
    else:
        ranking_detail = ["Chưa có POI để xếp hạng."]
    steps.append({"title": "4. Xếp hạng và trả kết quả", "details": ranking_detail})
    return steps


def hit_steps(
    hit: Mapping[str, Any],
    extraction: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    verified_only: bool = False,
) -> list[str]:
    """Explain why one returned POI passed its filters and the active score terms."""
    osm_only = int(policy.get("schema_version", 1)) >= 3
    verified_only = verified_only or osm_only
    properties = (hit.get("properties") or {}).get("effective") or {}
    constraints = [row for row in extraction.get("properties") or [] if row.get("applied")]
    details: list[str] = []
    for row in constraints:
        property_id = str(row.get("property_id"))
        mode = row.get("mode")
        if property_id == "radius_km":
            continue
        if property_id == "rating" and mode == "preferred" and osm_only:
            details.append("Ưu tiên rating không tham gia xếp hạng trong chế độ OSM-only.")
            continue
        assertion = (hit.get("rating") or {}) if property_id == "rating" else (properties.get(property_id) or {})
        source = assertion.get("source", "không rõ")
        value = assertion.get("value")
        if mode == "required":
            verified = "OSM" if source == "osm" else "mock (chưa xác minh)"
            details.append(f"Qua yêu cầu bắt buộc {property_id}: {_value_text(value)}; nguồn {verified}.")
        elif mode == "excluded":
            if source == "osm" and assertion.get("status") == "known":
                details.append(f"Không có giá trị bị loại cho {property_id}: {_value_text(value)}; OSM.")
            else:
                details.append(f"Không ghi nhận giá trị bị loại cho {property_id}; dữ liệu {source}, không phải xác nhận không có.")
        elif mode == "preferred":
            matched = property_id in (hit.get("matched_properties") or [])
            if matched:
                factor = float(policy.get("mock_preferred_source_factor", 0.65)) if source == "mock" else 1.0
                details.append(f"Ưu tiên {property_id} khớp từ {source} (hệ số bằng chứng {factor:.2f}).")
            else:
                reason = "mock không được tính trong chế độ chỉ xác minh" if verified_only and source == "mock" else "không khớp hoặc chưa rõ"
                details.append(f"Ưu tiên {property_id} không tăng điểm ({reason}).")
    if not details:
        details.append("Không có điều kiện thuộc tính cứng; POI thuộc category đã chọn.")
    if hit.get("name") is None:
        details.append("Tên OSM chưa có; nhãn hiển thị được tạo từ loại địa điểm và POI ID, không phải tên thực.")

    score = hit.get("score") or {}
    if osm_only:
        details.append("Rating và lượt review không tham gia xếp hạng; không ước lượng adjusted rating khi không có bằng chứng.")
    else:
        rating = hit.get("rating") or {}
        details.append(
            f"Rating {_value_text(rating.get('value'))}/5 ({rating.get('review_count', 0)} lượt, "
            f"nguồn {rating.get('source', 'không rõ')}) → adjusted rating {float(score.get('adjusted_rating', 0)):.3f}; "
            f"hệ số rating {float(score.get('rating_source_factor', 1)):.2f}, "
            f"review {float(score.get('review_source_factor', 1)):.2f}."
        )
    weights = policy.get("weights") or {}
    v2 = int(policy.get("schema_version", 1)) >= 2
    components: list[tuple[str, float, float]] = []
    has_preference = any(row.get("mode") == "preferred" and (not osm_only or row.get("property_id") != "rating")
                         for row in constraints)
    for key, field, active in (
        ("preferred_property", "preferred_property_score", has_preference),
        ("rating", "rating_score", True),
        ("review_volume", "review_volume_score", True),
        ("distance", "distance_score", hit.get("distance_km") is not None),
    ):
        if (active or not v2) and (not osm_only or float(weights.get(key, 0)) > 0):
            components.append((key, float(weights.get(key, 0)), float(score.get(field, 0))))
    numerator = sum(weight * value for _, weight, value in components)
    denominator = sum(weight for _, weight, _ in components) if v2 else 1.0
    total = numerator / (denominator or 1.0)
    if components:
        terms = " + ".join(f"{weight:.2f}×{value:.3f} ({name})" for name, weight, value in components)
        details.append(f"Điểm = ({terms}) / {denominator:.2f} = {total:.3f}; điểm đã trả: {float(score.get('final_score', 0)):.3f}.")
    else:
        details.append(f"Không có thành phần điểm đang hoạt động; điểm đã trả: {float(score.get('final_score', 0)):.3f}; "
                       "thứ tự được quyết định bởi nhãn hiển thị và POI ID.")
    if hit.get("distance_km") is None:
        details.append("Không có tọa độ gốc nên khoảng cách không tham gia điểm.")
    else:
        details.append(f"Khoảng cách từ tọa độ gốc: {float(hit['distance_km']):.2f} km.")
    return details
