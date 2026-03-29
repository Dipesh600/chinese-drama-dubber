"""Tests for config module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_wps, get_lang_instruction, get_voice_map,
    WORDS_PER_SEC, VOICE_PROFILES, VOICE_MAP, LANG_INSTRUCTIONS,
    DIRECTOR_CHUNK_SIZE, TRANSLATOR_BATCH_SIZE,
    MAX_TTS_DUR, MIN_MERGE_DUR, MAX_GAP,
)


def test_words_per_sec_defaults():
    """Test that WORDS_PER_SEC has sensible defaults."""
    assert WORDS_PER_SEC["Hindi"] == 2.5
    assert WORDS_PER_SEC["English"] == 3.0
    assert WORDS_PER_SEC["Tamil"] == 2.0
    # Unknown language should fallback to 2.5
    assert get_wps("UnknownLang") == 2.5


def test_get_wps_returns_float():
    """Test get_wps returns correct value."""
    assert isinstance(get_wps("Hindi"), float)
    assert get_wps("Hindi") == 2.5


def test_lang_instructions_exist():
    """Test that key languages have instructions."""
    assert "Hindi" in LANG_INSTRUCTIONS
    assert "English" in LANG_INSTRUCTIONS
    assert "Tamil" in LANG_INSTRUCTIONS
    assert len(LANG_INSTRUCTIONS["Hindi"]) > 0


def test_get_lang_instruction():
    """Test get_lang_instruction returns string."""
    result = get_lang_instruction("Hindi")
    assert isinstance(result, str)
    assert len(result) > 0


def test_voice_map_has_key_languages():
    """Test that voice map has key languages."""
    assert "Hindi" in VOICE_MAP
    assert "English" in VOICE_MAP
    assert "Tamil" in VOICE_MAP
    assert "Telugu" in VOICE_MAP


def test_get_voice_map():
    """Test get_voice_map returns correct structure."""
    vm = get_voice_map("Hindi")
    assert "male" in vm
    assert "female" in vm
    assert vm["male"] == "hi-IN-MadhurNeural"
    assert vm["female"] == "hi-IN-SwaraNeural"


def test_voice_profiles_have_required_fields():
    """Test that all voice profiles have required fields."""
    required = {"gender", "rate", "pitch", "volume"}
    for vid, profile in VOICE_PROFILES.items():
        assert required.issubset(profile.keys()), f"{vid} missing fields"
        assert profile["gender"] in ("male", "female")


def test_translation_constants():
    """Test translation batch sizes are reasonable."""
    assert DIRECTOR_CHUNK_SIZE == 30
    assert TRANSLATOR_BATCH_SIZE == 20
    assert TRANSLATOR_BATCH_SIZE > 0


def test_merger_constants():
    """Test sentence merger constants are reasonable."""
    assert MAX_TTS_DUR == 10.0
    assert MIN_MERGE_DUR == 2.0
    assert MAX_GAP == 0.5
    assert MAX_TTS_DUR > MIN_MERGE_DUR


if __name__ == "__main__":
    test_words_per_sec_defaults()
    test_get_wps_returns_float()
    test_lang_instructions_exist()
    test_get_lang_instruction()
    test_voice_map_has_key_languages()
    test_get_voice_map()
    test_voice_profiles_have_required_fields()
    test_translation_constants()
    test_merger_constants()
    print("All config tests passed!")
