from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def fold_vietnamese(text: str) -> str:
    text = str(text).replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return unicodedata.normalize("NFKC", without_marks)


def text_variants(text: str) -> tuple[tuple[str, str], ...]:
    original = unicodedata.normalize("NFKC", str(text)).strip()
    folded = fold_vietnamese(original)
    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, value in (("original", original), ("accent_folded", folded)):
        key = normalize_text(value)
        if key and key not in seen:
            seen.add(key)
            output.append((kind, value))
    return tuple(output)
