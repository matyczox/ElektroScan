"""Tests for detector_config constants and env-var reading."""
from __future__ import annotations

import importlib
import os

import pytest


def _reload_config():
    import core.detector_config as mod
    return importlib.reload(mod)


def test_threshold_precise_in_range():
    import core.detector_config as cfg
    assert 0.0 < cfg.THRESHOLD_PRECISE < 1.0


def test_threshold_dilated_in_range():
    import core.detector_config as cfg
    assert 0.0 < cfg.THRESHOLD_DILATED < 1.0


def test_threshold_precise_gt_dilated():
    import core.detector_config as cfg
    assert cfg.THRESHOLD_PRECISE > cfg.THRESHOLD_DILATED


def test_scales_count():
    import core.detector_config as cfg
    assert len(cfg.SCALES) == 3


def test_scales_contain_unity():
    import core.detector_config as cfg
    assert 1.0 in cfg.SCALES


def test_rotations_count():
    import core.detector_config as cfg
    assert len(cfg.ROTATIONS) == 4


def test_rotations_degrees():
    import core.detector_config as cfg
    degrees = [r[0] for r in cfg.ROTATIONS]
    assert set(degrees) == {0, 90, 180, 270}


def test_mirrored_variant_prefixes_is_set():
    import core.detector_config as cfg
    assert isinstance(cfg.MIRRORED_VARIANT_PREFIXES, set)
    assert len(cfg.MIRRORED_VARIANT_PREFIXES) > 0


def test_min_verification_score_in_range():
    import core.detector_config as cfg
    assert 0.0 < cfg.MIN_VERIFICATION_SCORE < 1.0


def test_env_int_reads_valid_value(monkeypatch):
    monkeypatch.setenv("ELEKTROSCAN_DETECTOR_SCAN_WORKERS", "3")
    cfg = _reload_config()
    assert cfg.DETECTOR_SCAN_MAX_WORKERS == 3


def test_env_int_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ELEKTROSCAN_OPENCV_THREADS", "notanumber")
    cfg = _reload_config()
    assert cfg.OPENCV_NUM_THREADS >= 1


def test_env_int_empty_uses_default(monkeypatch):
    monkeypatch.setenv("ELEKTROSCAN_OPENCV_THREADS", "")
    cfg = _reload_config()
    assert cfg.OPENCV_NUM_THREADS >= 1


def test_env_int_enforces_minimum(monkeypatch):
    monkeypatch.setenv("ELEKTROSCAN_DETECTOR_SCAN_WORKERS", "0")
    cfg = _reload_config()
    assert cfg.DETECTOR_SCAN_MAX_WORKERS >= 1


def test_runtime_limits_are_positive():
    import core.detector_config as cfg
    assert cfg.MAX_PEAKS_PER_VARIANT > 0
    assert cfg.GRAY_RAW_MAX_HITS_PER_TEMPLATE > 0
    assert cfg.GRAY_STRONG_TRACE_MAX_ITEMS > 0
