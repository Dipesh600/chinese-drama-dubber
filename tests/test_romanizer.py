"""Tests for romanizer module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from romanizer import romanize, romanize_segments, _has_devanagari, _DEV_RE


def test_no_devanagari_passthrough():
    """Text without Devanagari should pass through unchanged."""
    assert romanize("Hello world") == "Hello world"
    assert romanize("Namaste") == "Namaste"


def test_has_devanagari_detection():
    """Test Devanagari detection."""
    # Hindi
    assert _has_devanagari("नमस्ते")
    # Mixed
    assert _has_devanagari("Hello नमस्ते world")
    # Not Devanagari
    assert not _has_devanagari("Hello")
    assert not _has_devanagari("こんにちは")


def test_romanize_segments_empty():
    """Test with empty segments."""
    result = romanize_segments([])
    assert result == 0


def test_romanize_segments_no_devanagari():
    """Segments without Devanagari should not be modified."""
    segments = [
        {"id": 0, "dubbed_text": "Hello"},
        {"id": 1, "dubbed_text": "Namaste"},
    ]
    result = romanize_segments(segments)
    assert result == 0
    assert segments[0]["dubbed_text"] == "Hello"


if __name__ == "__main__":
    test_no_devanagari_passthrough()
    test_has_devanagari_detection()
    test_romanize_segments_empty()
    test_romanize_segments_no_devanagari()
    print("All romanizer tests passed!")
