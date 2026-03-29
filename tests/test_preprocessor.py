"""Tests for preprocessor module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import merge_short_segments, expand_dubbed_to_subsegments


def make_segment(id, start, end, text, speaker="NARRATOR", mood="neutral"):
    return {
        "id": id,
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "mood": mood,
    }


def test_merge_empty():
    """Test with empty segment list."""
    result = merge_short_segments([])
    assert result == []


def test_no_merge_needed():
    """Segments > MIN_SEGMENT_DUR should not merge."""
    segs = [
        make_segment(0, 0, 2.0, "This is a normal length segment"),
        make_segment(1, 2.0, 4.0, "Another normal length segment"),
    ]
    result = merge_short_segments(segs)
    assert len(result) == 2


def test_short_segment_merged():
    """Short segments should be merged."""
    segs = [
        make_segment(0, 0, 0.8, "Hi"),
        make_segment(1, 0.8, 1.5, "there friend"),
    ]
    result = merge_short_segments(segs)
    assert len(result) == 1
    assert result[0]["text"] == "Hi there friend"
    assert result[0]["merged_count"] == 2


def test_different_speakers_not_merged():
    """Different speakers should not merge."""
    segs = [
        make_segment(0, 0, 0.8, "Hi", speaker="FATHER"),
        make_segment(1, 0.8, 1.5, "there", speaker="MOTHER"),
    ]
    result = merge_short_segments(segs)
    assert len(result) == 2


def test_expand_single_segment():
    """Single segment should expand back to itself."""
    segs = [{
        "id": 0,
        "start": 0,
        "end": 2.0,
        "text": "Hello world",
        "dubbed_text": "Namaste duniya",
        "speaker": "NARRATOR",
        "mood": "neutral",
    }]
    result = expand_dubbed_to_subsegments(segs)
    assert len(result) == 1
    assert result[0]["dubbed_text"] == "Namaste duniya"


def test_expand_multiple_merged():
    """Test expanding merged segments distributes text."""
    segs = [{
        "id": 0,
        "start": 0,
        "end": 2.0,
        "text": "Hello world",
        "dubbed_text": "Namaste duniya",
        "speaker": "NARRATOR",
        "mood": "neutral",
        "sub_segments": [
            {"id": 0, "start": 0, "end": 1.0, "text": "Hello"},
            {"id": 1, "start": 1.0, "end": 2.0, "text": "world"},
        ]
    }]
    result = expand_dubbed_to_subsegments(segs)
    assert len(result) == 2
    # Both should have dubbed_text
    assert result[0]["dubbed_text"]
    assert result[1]["dubbed_text"]


if __name__ == "__main__":
    test_merge_empty()
    test_no_merge_needed()
    test_short_segment_merged()
    test_different_speakers_not_merged()
    test_expand_single_segment()
    test_expand_multiple_merged()
    print("All preprocessor tests passed!")
