"""Tests for detector dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from core.detector_models import (
    CandidateHit,
    Detection,
    DetectionResult,
    TargetedPromotionRule,
    TemplateInfo,
    TemplateVariant,
)


class TestDetection:
    def test_required_fields(self):
        d = Detection(symbol_name="06_test", x=10, y=20, width=30, height=40)
        assert d.symbol_name == "06_test"
        assert d.x == 10
        assert d.y == 20
        assert d.width == 30
        assert d.height == 40

    def test_default_values(self):
        d = Detection(symbol_name="06_test", x=0, y=0, width=1, height=1)
        assert d.confidence == 0.0
        assert d.source == "template"
        assert d.rotation == 0
        assert d.scale == 1.0
        assert d.mirrored is False
        assert d.coverage == 0.0
        assert d.purity == 0.0
        assert d.context_purity == 0.0
        assert d.color_similarity == 1.0
        assert d.verification_score == 0.0
        assert d.is_text_label is False
        assert d.content_score == 0.0

    def test_custom_values(self):
        d = Detection(
            symbol_name="12_test",
            x=100,
            y=200,
            width=48,
            height=31,
            confidence=0.9,
            source="template_promoted_x",
            rotation=90,
            scale=1.1,
            mirrored=True,
        )
        assert d.confidence == 0.9
        assert d.source == "template_promoted_x"
        assert d.rotation == 90
        assert d.scale == 1.1
        assert d.mirrored is True


class TestDetectionResult:
    def test_defaults(self):
        r = DetectionResult(symbol_name="06_test", count=3)
        assert r.symbol_name == "06_test"
        assert r.count == 3
        assert r.detections == []

    def test_detections_list_is_independent(self):
        r1 = DetectionResult(symbol_name="a", count=1)
        r2 = DetectionResult(symbol_name="b", count=2)
        r1.detections.append(Detection(symbol_name="a", x=0, y=0, width=1, height=1))
        assert r2.detections == [], "detections lists must not share the same object"

    def test_default_color(self):
        r = DetectionResult(symbol_name="test", count=0)
        assert r.color.startswith("#")


class TestCandidateHit:
    def _make(self, **kwargs) -> CandidateHit:
        mask = np.zeros((10, 10), dtype=np.uint8)
        defaults = dict(
            template_id=1,
            scale=1.0,
            rotation=0,
            mirrored=False,
            transformed_mask=mask,
            content_mask=None,
            pixel_count=100,
            content_pixel_count=0,
            content_bbox=None,
            bbox=(10, 20, 30, 40),
            match_score=0.75,
            dominant_hsv=None,
        )
        defaults.update(kwargs)
        return CandidateHit(**defaults)

    def test_defaults(self):
        hit = self._make()
        assert hit.source == "template"
        assert hit.is_text_label is False
        assert hit.coverage == 0.0
        assert hit.purity == 0.0
        assert hit.context_purity == 1.0
        assert hit.color_similarity == 1.0
        assert hit.verification_score == 0.0
        assert hit.content_score == 0.0
        assert hit.promoted_from_template_id is None

    def test_bbox_stored(self):
        hit = self._make(bbox=(5, 10, 20, 30))
        assert hit.bbox == (5, 10, 20, 30)

    def test_dominant_hsv_none_allowed(self):
        hit = self._make(dominant_hsv=None)
        assert hit.dominant_hsv is None

    def test_promoted_from_id(self):
        hit = self._make(promoted_from_template_id=42)
        assert hit.promoted_from_template_id == 42


class TestTemplateVariant:
    def test_creation(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        v = TemplateVariant(
            template_id=0,
            scale=1.0,
            rotation=0,
            mirrored=False,
            transformed_mask=mask,
            content_mask=None,
            pixel_count=200,
            content_pixel_count=0,
            content_bbox=None,
            width=30,
            height=20,
        )
        assert v.width == 30
        assert v.height == 20
        assert v.pixel_count == 200
