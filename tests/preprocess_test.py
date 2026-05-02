from src.preprocess import normalize_arabic_text, preprocess_data


def test_normalize_arabic_text_removes_diacritics_and_punctuation():
    text = "أَهْلًا!!! بِكُمْ"
    assert normalize_arabic_text(text) == "اهلا بكم"


def test_preprocess_data_returns_clean_copy():
    raw = [{"text": "إختبارٌ", "label": "ok"}]
    processed = preprocess_data(raw)

    assert processed[0]["text"] == "اختبار"
    assert processed[0]["label"] == "ok"
    assert raw[0]["text"] == "إختبارٌ"
