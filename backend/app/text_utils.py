"""Text helpers shared by ingestion and retrieval: cleaning, Tamil fixes,
language detection and the BM25 tokenizer."""
from __future__ import annotations

import re
import unicodedata

TAMIL_RANGE = (0x0B80, 0x0BFF)
_TAMIL_CONSONANT = "[க-ஹ]"
_PRE_BASE = "[ெேை]"  # ெ ே ை  (written before the consonant visually)

# A pre-base vowel sign that is *not* preceded by a consonant can only appear in
# text that was extracted in visual (glyph) order — common with TN Govt PDFs.
_ORPHAN_PRE_BASE = re.compile(rf"(?<![க-ஹ]){_PRE_BASE}(?={_TAMIL_CONSONANT})")
_SWAP_PRE_BASE = re.compile(rf"({_PRE_BASE})({_TAMIL_CONSONANT})")


def is_tamil_char(ch: str) -> bool:
    return TAMIL_RANGE[0] <= ord(ch) <= TAMIL_RANGE[1]


def fix_tamil_visual_order(text: str) -> tuple[str, bool]:
    """Re-order Tamil pre-base vowel signs extracted in visual order.

    "ேவளாண்ைம" -> "வேளாண்மை". Applied only when the page clearly is in visual
    order (most pre-base signs are orphaned), so correct text is never touched.
    Returns (text, changed)."""
    signs = len(re.findall(_PRE_BASE, text))
    if signs < 5:
        return text, False
    orphans = len(_ORPHAN_PRE_BASE.findall(text))
    if orphans / signs < 0.35:
        return text, False
    fixed = _SWAP_PRE_BASE.sub(r"\2\1", text)
    # ெ+ா -> ொ, ே+ா -> ோ, ெ+ௗ -> ௌ (canonical composition)
    fixed = unicodedata.normalize("NFC", fixed)
    return fixed, True


def legacy_font_ratio(text: str) -> float:
    """Share of letters in Latin-1 Supplement/Latin Extended — a strong hint that a
    Tamil PDF uses a legacy (non-Unicode) font such as Bamini/TAM/TAB."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 50:
        return 0.0
    odd = sum(1 for c in letters if 0x00C0 <= ord(c) <= 0x024F)
    return odd / len(letters)


_PAGE_NUM_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*(of\s*\d+)?\s*$", re.I)
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ​]{2,}")
_MULTI_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = []
    for line in text.split("\n"):
        if _PAGE_NUM_LINE.match(line):
            continue
        lines.append(_MULTI_SPACE.sub(" ", line).strip())
    text = "\n".join(lines)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    letters = [c for c in text[:20000] if c.isalpha()]
    if not letters:
        return "unknown"
    ta = sum(1 for c in letters if is_tamil_char(c))
    r = ta / len(letters)
    if r > 0.6:
        return "ta"
    if r < 0.08:
        return "en"
    return "mixed"


# --------------------------------------------------------------------------- BM25
_TOKEN = re.compile(r"[0-9a-z஀-௿]+")
_EN_STOP = set(
    """a an and are as at be by for from has have in is it its of on or that the to was were will with this
    these those which who whom what when where how than then there their they them we you your our not no
    can may shall should would could also such into per via under any all each other""".split()
)
_TA_STOP = {"மற்றும்", "ஒரு", "இந்த", "என்ற", "ஆகிய", "உள்ள", "போன்ற", "அல்லது", "மூலம்", "வரை", "பற்றி", "என்று", "அந்த", "இது", "அது"}


def _stem_en(tok: str) -> str:
    """Plural normalisation only (schemes->scheme, subsidies->subsidy, boxes->box).
    Dense retrieval covers deeper morphology; BM25 just needs consistency."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("sses"):
        return tok[:-2]
    if tok.endswith("es") and tok[-4:-2] in ("ch", "sh") or tok.endswith(("xes", "zes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Bilingual tokenizer. Tamil is agglutinative (திட்டம் / திட்டத்தின் /
    திட்டங்கள்), so each long Tamil token also emits a 5-code-point prefix token
    ("திட்ட"), which recovers most inflected matches without a full stemmer
    (மதுரை ↔ மதுரையில், மானியம் ↔ மானியத்தில்)."""
    text = unicodedata.normalize("NFC", text.lower())
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if is_tamil_char(tok[0]):
            if tok in _TA_STOP:
                continue
            out.append(tok)
            if len(tok) >= 5:
                out.append("ta~" + tok[:5])
        else:
            if tok in _EN_STOP or len(tok) == 1:
                continue
            out.append(_stem_en(tok))
    return out
