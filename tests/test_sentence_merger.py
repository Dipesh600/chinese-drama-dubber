"""Tests for sentence_merger module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_merger import merge_for_tts


def make_segment(id, start, end, text, speaker="NARRATOR", mood="neutral"):
    return {
        "id": id,
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "mood": mood,
        "dubbed_text": text,
    }


def test_empty_segments():
    """Test with empty segment list."""
    result = merge_for_tts([])
    assert result == []


def test_single_segment_no_merge():
    """Single segment should not be merged."""
    segs = [make_segment(0, 0, 1.0, "Hello world")]
    result = merge_for_tts(segs)
    assert len(result) == 1
    assert result[0]["tts_merged_count"] == 1


def test_adjacent_same_speaker_short_merged():
    """Two short adjacent segments from same speaker should merge."""
    segs = [
        make_segment(0, 0, 0.8, "Hi"),
        make_segment(1, 0.8, 1.5, "there"),
    ]
    result = merge_for_tts(segs)
    # Should be merged into one
    assert len(result) == 1
    assert result[0]["tts_merged_count"] == 2
    assert result[0]["tts_text"] == "Hi there"


def test_different_speakers_not_merged():
    """Segments from different speakers should not merge."""
    segs = [
        make_segment(0, 0, 0.8, "Hi", speaker="FATHER"),
        make_segment(1, 0.8, 1.5, "there", speaker="MOTHER"),
    ]
    result = merge_for_tts(segs)
    # Should NOT be merged (different speakers)
    assert len(result) == 2


def test_gap_too_large_not_merged():
    """Segments with gap > MAX_GAP should not merge."""
    segs = [
        make_segment(0, 0, 0.8, "Hi"),
        make_segment(1, 1.5, 2.0, "there"),  # 0.7s gap > 0.5s
    ]
    result = merge_for_tts(segs)
    assert len(result) == 2


def test_combined_too_long_not_merged():
    """Merged segments exceeding MAX_TTS_DUR should not merge."""
    segs = [
        make_segment(0, 0, 3.0, "This is a longer sentence"),
        make_segment(1, 3.0, 8.0, "That continues for quite a while"),  # combined > 10s
    ]
    result = merge_for_tts(segs)
    assert len(result) == 2


def test_tts_text_combined():
    """Test that tts_text is properly combined."""
    segs = [
        make_segment(0, 0, 1.0, "Hello"),
        make_segment(1, 1.0, 1.8, "world"),
    ]
    result = merge_for_tts(segs)
    assert result[0]["tts_text"] == "Hello world"


def test_tts_group_preserved():
    """Test that original segments are preserved in tts_group."""
    segs = [
        make_segment(0, 0, 0.5, "One"),
        make_segment(1, 0.5, 1.0, "Two"),
    ]
    result = merge_for_tts(segs)
    assert len(result[0]["tts_group"]) == 2
    assert result[0]["tts_group"][0]["id"] == 0
    assert result[0]["tts_group"][1]["id"] == 1


def test_tts_duration_calculated():
    """Test that tts_duration is set correctly."""
    segs = [
        make_segment(0, 0, 0.5, "One"),
        make_segment(1, 0.5, 1.0, "Two"),
    ]
    result = merge_for_tts(segs)
    assert result[0]["tts_duration"] == 1.0


if __name__ == "__main__":
    test_empty_segments()
    test_single_segment_no_merge()
    test_adjacent_same_speaker_short_merged()
    test_different_speakers_not_merged()
    test_gap_too_large_not_merged()
    test_combined_too_long_not_merged()
    test_tts_text_combined()
    test_tts_group_preserved()
    test_tts_duration_calculated()
    print("All sentence_merger tests passed!")
