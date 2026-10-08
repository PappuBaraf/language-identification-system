"""Unicode-safe normalization and tokenization for multilingual text."""
import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", flags=re.UNICODE)


def preprocess_text(text: str) -> str:
    """Normalize text while retaining letters/marks from Indic scripts."""
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.casefold()
    # Keep letters, combining marks, and digits. Punctuation becomes spacing.
    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        chars.append(char if category[0] in {"L", "M", "N"} else " ")
    return " ".join("".join(chars).split())


def tokenize(text: str) -> list[str]:
    """Return Unicode letter sequences and numbers as tokens."""
    # Splitting on normalized whitespace preserves Indic combining marks, which
    # Python's regular-expression ``\w`` class does not include consistently.
    return preprocess_text(text).split()
