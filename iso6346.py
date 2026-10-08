"""ISO 6346 container number validation + OCR clean-up."""
import re

_LETTER = {}
_v = 10
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if _v % 11 == 0:          # values 11, 22, 33 are skipped
        _v += 1
    _LETTER[_c] = _v
    _v += 1

PATTERN = re.compile(r"([A-Z]{3}[UJZR])\s*(\d{6})\s*(\d)")


def check_digit(first10: str) -> int:
    """Check digit for the 4 letters + 6 digits."""
    total = 0
    for i, ch in enumerate(first10[:10]):
        val = _LETTER[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    return total % 11 % 10


def is_valid(number: str) -> bool:
    n = re.sub(r"[\s-]", "", number.upper())
    if not re.fullmatch(r"[A-Z]{3}[UJZR]\d{7}", n):
        return False
    return check_digit(n[:10]) == int(n[10])


# characters OCR often mixes up, by position type
_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}


def clean_ocr(raw: str) -> str | None:
    """Return a valid container number from noisy OCR text, or None."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if len(s) < 11:
        return None
    for start in range(len(s) - 10):
        w = s[start:start + 11]
        fixed = "".join(_TO_LETTER.get(c, c) for c in w[:4]) + \
                "".join(_TO_DIGIT.get(c, c) for c in w[4:])
        if is_valid(fixed):
            return fixed
    return None
