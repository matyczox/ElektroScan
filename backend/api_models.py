"""Pydantic request/response models for the FastAPI app."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AuthRegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class AuthLoginRequest(BaseModel):
    email: str
    password: str

class AuthProfileUpdateRequest(BaseModel):
    name: Optional[str] = None

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str

class ProjectCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""

class ProjectUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class LegendZone(BaseModel):
    page: Optional[int] = 0
    x: float
    y: float
    width: float
    height: float

class ExtractRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []
    legend_zone: Optional[LegendZone] = None
    detector_profile: Optional[str] = "auto"
    legend_engine: Optional[str] = "auto"
    include_legend_debug: Optional[bool] = False

class RenderRequest(BaseModel):
    hidden_layers: Optional[List[str]] = []
    preview: Optional[bool] = False

class AnalyzeRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []
    include_debug: Optional[bool] = None
    include_image: Optional[bool] = None
    detector_profile: Optional[str] = "auto"
    legend_zone: Optional[LegendZone] = None
    plan_zone: Optional[LegendZone] = None

class AnalysisExportResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = ""
    count: int = 0
    color: Optional[str] = None

class AnalysisExportBox(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    symbol_name: str = Field(default="", alias="symbolName")
    color: Optional[str] = None

class AnalysisExportRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    results: list[AnalysisExportResult] = Field(default_factory=list)
    boxes: list[AnalysisExportBox] = Field(default_factory=list)
    analysis_context: dict = Field(default_factory=dict, alias="analysisContext")
    symbol_labels: dict[str, str] = Field(default_factory=dict, alias="symbolLabels")

class RoiInspectRequest(BaseModel):
    hidden_layers: Optional[List[str]] = []
    detector_profile: Optional[str] = "auto"
    roi: LegendZone
    top_n: Optional[int] = 15

class GrayDebugZonesRequest(BaseModel):
    excluded_zones: Optional[List[dict]] = []
    hidden_layers: Optional[List[str]] = []
    detector_profile: Optional[str] = "auto"
    legend_zone: Optional[LegendZone] = None
    plan_zone: Optional[LegendZone] = None

class TemplateCropRequest(BaseModel):
    session_id: str
    x: float
    y: float
    width: float
    height: float
    name: Optional[str] = None
    hidden_layers: Optional[List[str]] = []

class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
