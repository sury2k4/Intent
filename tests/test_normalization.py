from model_intent.normalization import fold_vietnamese, normalize_text, text_variants


def test_vietnamese_accent_folding_and_d():
    assert fold_vietnamese("Đau đầu cần mua thuốc") == "Dau dau can mua thuoc"
    assert normalize_text("  Cần,   mua THUỐC! ") == "cần mua thuốc"


def test_query_variants_keep_original_and_folded_once():
    variants = text_variants("xe sắp hết xăng")
    assert variants == (("original", "xe sắp hết xăng"), ("accent_folded", "xe sap het xang"))
    assert text_variants("gas station") == (("original", "gas station"),)
