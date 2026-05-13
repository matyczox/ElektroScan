"""Dataclasses shared by detector modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Detection:
    """Single detected symbol on the plan."""

    symbol_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    source: str = "template"
    rotation: int = 0
    scale: float = 1.0
    mirrored: bool = False
    coverage: float = 0.0
    purity: float = 0.0
    context_purity: float = 0.0
    color_similarity: float = 1.0
    verification_score: float = 0.0
    is_text_label: bool = False
    content_score: float = 0.0
    content_bbox: tuple[int, int, int, int] | None = None
    content_source: str = ""
    roi_strategy: str = ""
    visual_bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DetectionResult:
    """Grouped detections for one symbol type."""

    symbol_name: str
    count: int
    color: str = "#10b981"
    detections: list[Detection] = field(default_factory=list)


@dataclass(slots=True)
class TemplateInfo:
    """Loaded template with metadata used during matching."""

    path: str
    name: str
    pixel_count: int
    mask: np.ndarray
    requires_precision: bool
    image_bgr: np.ndarray
    dominant_hsv: tuple[int, int, int] | None
    text_tokens: list[str] = field(default_factory=list)
    content_mask: np.ndarray | None = None
    content_pixel_count: int = 0
    content_bbox: tuple[int, int, int, int] | None = None
    is_text_label: bool = False


@dataclass(slots=True)
class TemplateVariant:
    """One concrete template variant after scale and rotation."""

    template_id: int
    scale: float
    rotation: int
    mirrored: bool
    transformed_mask: np.ndarray
    content_mask: np.ndarray | None
    pixel_count: int
    content_pixel_count: int
    content_bbox: tuple[int, int, int, int] | None
    width: int
    height: int


@dataclass(slots=True)
class CandidateHit:
    """Candidate detection produced by template matching or PDF text lookup."""

    template_id: int
    scale: float
    rotation: int
    mirrored: bool
    transformed_mask: np.ndarray | None
    content_mask: np.ndarray | None
    pixel_count: int
    content_pixel_count: int
    content_bbox: tuple[int, int, int, int] | None
    bbox: tuple[int, int, int, int]
    match_score: float
    dominant_hsv: tuple[int, int, int] | None
    source: str = "template"
    is_text_label: bool = False
    coverage: float = 0.0
    purity: float = 0.0
    context_purity: float = 1.0
    color_similarity: float = 1.0
    verification_score: float = 0.0
    content_score: float = 0.0
    promoted_from_template_id: int | None = None
    roi_strategy: str = ""


@dataclass(slots=True)
class TargetedPromotionRule:
    """Pair-specific promotion from a smaller template to a larger one."""

    child_template_id: int
    parent_template_id: int
    scale: float
    rotation: int
    mirrored: bool
    offset_x: int
    offset_y: int
    extension_mask: np.ndarray
    extension_pixels: int
    min_extra_coverage: float
    allow_rotation_mismatch: bool = False
