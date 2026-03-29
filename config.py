"""
CONFIG — Single source of truth for all configuration.
All modules import from here. No duplication.
"""
import os

# ═══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Words per second by language (for timing constraints in lip-sync dubbing)
WORDS_PER_SEC = {
    "Hindi": 2.5, "Tamil": 2.0, "Telugu": 2.2, "Bengali": 2.3,
    "Marathi": 2.4, "Gujarati": 2.4, "Kannada": 2.1, "Malayalam": 1.9,
    "Nepali": 2.5, "Urdu": 2.5, "English": 3.0, "Spanish": 2.8,
    "French": 2.7, "Portuguese": 2.6, "German": 2.5, "Japanese": 3.5,
    "Korean": 2.8, "Arabic": 2.5, "Turkish": 2.6, "Chinese": 3.5,
}

# Language-specific translation instructions
LANG_INSTRUCTIONS = {
    "Hindi": "Translate to natural Hinglish (Hindi+English mix as Indians actually speak). Keep English words Indians commonly use: okay, sorry, please, time, market, phone, office, school. Write in ROMAN script (NOT Devanagari).",
    "Tamil": "Translate to natural spoken Tamil (Tanglish where natural). Roman script only.",
    "Telugu": "Translate to natural spoken Telugu. Roman script only.",
    "Bengali": "Translate to natural spoken Bengali. Roman script only.",
    "Marathi": "Translate to natural spoken Marathi. Roman script only.",
    "Gujarati": "Translate to natural spoken Gujarati. Roman script only.",
    "Kannada": "Translate to natural spoken Kannada. Roman script only.",
    "Malayalam": "Translate to natural spoken Malayalam. Roman script only.",
    "Nepali": "Translate to natural spoken Nepali (Nepali+English mix). Use Hajur for respect. Roman script.",
    "English": "Translate to natural spoken English. Conversational, engaging.",
    "Spanish": "Translate to natural Latin American Spanish.",
    "French": "Translate to natural spoken French.",
    "Portuguese": "Translate to natural Brazilian Portuguese.",
    "German": "Translate to natural spoken German.",
    "Japanese": "Translate to natural spoken Japanese in Romaji.",
    "Korean": "Translate to natural spoken Korean in Romanized form.",
    "Arabic": "Translate to natural spoken Arabic in Romanized form.",
    "Turkish": "Translate to natural spoken Turkish.",
    "Urdu": "Translate to natural spoken Urdu in Roman script.",
}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM PROVIDER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

# Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "4"))

# Model configurations
LLM_MODELS = {
    "primary": "llama-3.3-70b-versatile",
    "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
    "fast": "llama-3.1-8b-instant",
    "whisper": "whisper-large-v3",
}

# Whisper language codes mapping
WHISPER_LANGUAGE_MAP = {
    "zh": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "auto": None,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TTS CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

FISH_SPEECH_URL = os.environ.get("FISH_SPEECH_URL", "http://localhost:8080")
TTS_WORKERS = int(os.environ.get("TTS_WORKERS", "5"))
TARGET_LUFS = -14

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Director chunk settings
DIRECTOR_CHUNK_SIZE = 30
DIRECTOR_OVERLAP = 5

# Translator batch settings
TRANSLATOR_BATCH_SIZE = 20
TRANSLATOR_POLISH_BATCH = 30

# Sentence merger settings
MAX_TTS_DUR = 10.0
MIN_MERGE_DUR = 2.0
MAX_GAP = 0.5

# Preprocessor settings
MIN_SEGMENT_DUR = 1.5
MAX_SEGMENT_DUR = 8.0

# Timestamp aligner settings
ALIGN_TOLERANCE = 0.15
MAX_STRETCH = 1.35
BREATHING_GAP_MS = 20

# Assembler settings
DUBBED_LUFS = -14
BG_LUFS = -26
BG_SPEECH_VOL = 0.08
BG_GAP_VOL = 0.35
DUCK_FADE_MS = 200
FADE_IN_MS = 30
FADE_OUT_MS = 50

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/content/drive/MyDrive/DubbedVideos")
COOKIE_FILE = os.environ.get("COOKIE_FILE", "")

# ═══════════════════════════════════════════════════════════════════════════════
# VOICE PROFILES (SSML prosody differentiation)
# ═══════════════════════════════════════════════════════════════════════════════

VOICE_PROFILES = {
    "NARRATOR":    {"gender": "male",   "rate": "+3%",   "pitch": "+0Hz",   "volume": "+0%"},
    "FATHER":      {"gender": "male",   "rate": "-10%",  "pitch": "-15Hz",  "volume": "+5%"},
    "OLD_MAN":     {"gender": "male",   "rate": "-18%",  "pitch": "-25Hz",  "volume": "-5%"},
    "HERO":        {"gender": "male",   "rate": "+5%",   "pitch": "+5Hz",   "volume": "+5%"},
    "VILLAIN":     {"gender": "male",   "rate": "-8%",   "pitch": "-20Hz",  "volume": "+10%"},
    "YOUNG_MAN":   {"gender": "male",   "rate": "+12%",  "pitch": "+8Hz",   "volume": "+0%"},
    "CHAR_A":      {"gender": "male",   "rate": "+0%",   "pitch": "-5Hz",   "volume": "+0%"},
    "CHAR_C":      {"gender": "male",   "rate": "-5%",   "pitch": "-10Hz",  "volume": "+0%"},
    "MOTHER":      {"gender": "female", "rate": "-3%",   "pitch": "+0Hz",   "volume": "+0%"},
    "OLD_WOMAN":   {"gender": "female", "rate": "-12%",  "pitch": "-10Hz",  "volume": "-5%"},
    "HEROINE":     {"gender": "female", "rate": "+5%",   "pitch": "+8Hz",   "volume": "+5%"},
    "YOUNG_WOMAN": {"gender": "female", "rate": "+8%",   "pitch": "+10Hz",  "volume": "+0%"},
    "DAUGHTER":    {"gender": "female", "rate": "+5%",   "pitch": "+5Hz",   "volume": "+0%"},
    "GIRL":        {"gender": "female", "rate": "+15%",  "pitch": "+18Hz",  "volume": "+5%"},
    "CHAR_B":      {"gender": "female", "rate": "+0%",   "pitch": "+3Hz",   "volume": "+0%"},
    "CHAR_D":      {"gender": "female", "rate": "+8%",   "pitch": "+12Hz",  "volume": "+0%"},
    "BOY":         {"gender": "female", "rate": "+15%",  "pitch": "+25Hz",  "volume": "+5%"},
    "SON":         {"gender": "female", "rate": "+10%",  "pitch": "+18Hz",  "volume": "+0%"},
}

ROLE_GENDER = {
    "NARRATOR": "male", "FATHER": "male", "MOTHER": "female", "SON": "female",
    "DAUGHTER": "female", "OLD_MAN": "male", "OLD_WOMAN": "female",
    "YOUNG_MAN": "male", "YOUNG_WOMAN": "female", "GIRL": "female", "BOY": "female",
    "VILLAIN": "male", "HERO": "male", "HEROINE": "female",
    "CHAR_A": "male", "CHAR_B": "female", "CHAR_C": "male", "CHAR_D": "female",
}

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE TTS VOICE MAP
# ═══════════════════════════════════════════════════════════════════════════════

VOICE_MAP = {
    "Hindi":      {"male": "hi-IN-MadhurNeural",    "female": "hi-IN-SwaraNeural"},
    "Tamil":      {"male": "ta-IN-ValluvarNeural",   "female": "ta-IN-PallaviNeural"},
    "Telugu":     {"male": "te-IN-MohanNeural",       "female": "te-IN-ShrutiNeural"},
    "Bengali":    {"male": "bn-IN-BashkarNeural",     "female": "bn-IN-TanishaaNeural"},
    "Marathi":    {"male": "mr-IN-ManoharNeural",     "female": "mr-IN-AarohiNeural"},
    "Gujarati":   {"male": "gu-IN-NiranjanNeural",    "female": "gu-IN-DhwaniNeural"},
    "Kannada":    {"male": "kn-IN-GaganNeural",        "female": "kn-IN-SapnaNeural"},
    "Malayalam":  {"male": "ml-IN-MidhunNeural",      "female": "ml-IN-SobhanaNeural"},
    "Urdu":       {"male": "ur-IN-SalmanNeural",      "female": "ur-IN-GulNeural"},
    "Nepali":     {"male": "hi-IN-MadhurNeural",      "female": "hi-IN-SwaraNeural"},
    "English":    {"male": "en-US-AndrewNeural",      "female": "en-US-AvaNeural"},
    "Spanish":    {"male": "es-MX-JorgeNeural",       "female": "es-MX-DaliaNeural"},
    "French":     {"male": "fr-FR-RemyMultilingualNeural", "female": "fr-FR-VivienneMultilingualNeural"},
    "Portuguese": {"male": "pt-BR-AntonioNeural",     "female": "pt-BR-D-AlmeidaNeural"},
    "German":     {"male": "de-DE-FlorianMultilingualNeural", "female": "de-DE-SeraphinaMultilingualNeural"},
    "Japanese":   {"male": "ja-JP-KeitaNeural",       "female": "ja-JP-NanamiNeural"},
    "Korean":     {"male": "ko-KR-InJoonNeural",      "female": "ko-KR-SunHiNeural"},
    "Arabic":     {"male": "ar-SA-HamedNeural",       "female": "ar-SA-ZariyahNeural"},
    "Turkish":    {"male": "tr-TR-AhmetNeural",       "female": "tr-TR-EmelNeural"},
}


def get_wps(target_lang):
    """Get words per second for a language."""
    return WORDS_PER_SEC.get(target_lang, 2.5)


def get_lang_instruction(target_lang):
    """Get language-specific translation instruction."""
    return LANG_INSTRUCTIONS.get(target_lang, LANG_INSTRUCTIONS.get("English", ""))


def get_voice_map(target_lang):
    """Get Edge TTS voice map for a language."""
    return VOICE_MAP.get(target_lang, VOICE_MAP.get("English"))
