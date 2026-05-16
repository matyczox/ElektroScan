from core.detector_pdf import _word_matches_token


def test_word_matches_token_requires_full_short_code():
    assert _word_matches_token("L3", "L3")
    assert _word_matches_token(" l3 ", "L3")

    assert not _word_matches_token("RL3", "L3")
    assert not _word_matches_token("PL3", "L3")
    assert not _word_matches_token("SL3", "L3")


def test_word_matches_token_allows_full_normalized_token_only():
    assert _word_matches_token("TB1.1", "TB11")
    assert not _word_matches_token("ATB11", "TB11")
