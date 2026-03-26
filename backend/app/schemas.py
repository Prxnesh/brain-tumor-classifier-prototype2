from typing import Literal

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str
    probability: float


class ReportSection(BaseModel):
    title: str
    body: str


class PredictionResponse(BaseModel):
    modality: Literal["mri", "ct", "fusion"]
    predicted_label: str
    confidence: float
    probabilities: list[ClassProbability]
    gradcam_overlay: str
    original_preview: str
    report: list[ReportSection]
    model_ready: bool = True
    notes: list[str] = Field(default_factory=list)


class AppConfigResponse(BaseModel):
    supported_modalities: list[str]
    available_now: list[str]
    pending_datasets: list[str]
