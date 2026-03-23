"""
ROMANIZER — Converts Devanagari script to clean Roman Hindi for TTS.
Uses indic_transliteration IAST then strips diacritics for natural pronunciation.
Also handles mixed text (some Roman + some Devanagari).
"""
import re, logging, unicodedata
logger = logging.getLogger(__name__)

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False
    logger.warning("[ROMAN] indic_transliteration not installed, using fallback")

# Devanagari Unicode range
_DEV_RE = re.compile(r'[\u0900-\u097F]')

# IAST diacritic → simple Roman mapping
_DIACRITIC_MAP = {
    'ā': 'aa', 'ī': 'ee', 'ū': 'oo', 'ṛ': 'ri', 'ṝ': 'ri',
    'ṃ': 'n', 'ṁ': 'n', 'ḥ': 'h',
    'ṭ': 't', 'ḍ': 'd', 'ṇ': 'n', 'ñ': 'n',
    'ś': 'sh', 'ṣ': 'sh',
    'Ā': 'Aa', 'Ī': 'Ee', 'Ū': 'Oo',
    'Ṭ': 'T', 'Ḍ': 'D', 'Ṇ': 'N',
    'Ś': 'Sh', 'Ṣ': 'Sh',
    '|': '.', '||': '.',  # Devanagari danda → period
}

def _strip_diacritics(text):
    """Convert IAST diacritics to simple Roman letters."""
    for src, dst in _DIACRITIC_MAP.items():
        text = text.replace(src, dst)
    # Catch any remaining diacritics via NFD decomposition
    result = []
    for ch in unicodedata.normalize('NFD', text):
        if unicodedata.category(ch) != 'Mn':  # skip combining marks
            result.append(ch)
    return ''.join(result)

def _has_devanagari(text):
    return bool(_DEV_RE.search(text))

def romanize(text):
    """Convert any Devanagari in text to clean Roman Hindi."""
    if not text or not _has_devanagari(text):
        return text
    
    if not HAS_INDIC:
        # Fallback: just flag it
        logger.warning(f"[ROMAN] Cannot romanize without indic_transliteration")
        return text
    
    # Handle mixed text: split into Devanagari and non-Devanagari chunks
    parts = re.split(r'([\u0900-\u097F\u0964\u0965]+)', text)
    result = []
    for part in parts:
        if _has_devanagari(part):
            iast = transliterate(part, sanscript.DEVANAGARI, sanscript.IAST)
            roman = _strip_diacritics(iast)
            result.append(roman)
        else:
            result.append(part)
    
    return ''.join(result)

def romanize_segments(segments):
    """Romanize all dubbed_text in segments list. Returns count of segments fixed."""
    fixed = 0
    for s in segments:
        text = s.get("dubbed_text", "")
        if _has_devanagari(text):
            s["dubbed_text"] = romanize(text)
            fixed += 1
    if fixed:
        logger.info(f"[ROMAN] Romanized {fixed}/{len(segments)} segments (Devanagari → Roman)")
    return fixed
