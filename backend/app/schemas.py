from typing import Literal

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str
    probability: float


class ReportSection(BaseModel):
    title: str
    body: str


class TumorLocation(BaseModel):
    cx: float
    cy: float
    radius: float
    quadrant: str
    description: str


class PredictionResponse(BaseModel):
    modality: Literal["mri", "ct", "fusion"]
    predicted_label: str
    confidence: float
    probabilities: list[ClassProbability]
    gradcam_overlay: str
    original_preview: str
    report: list[ReportSection]
    report_provider: Literal["template", "ollama"] = "template"
    model_ready: bool = True
    notes: list[str] = Field(default_factory=list)
    tumor_location: TumorLocation | None = None


class AppConfigResponse(BaseModel):
    supported_modalities: list[str]
    available_now: list[str]
    pending_datasets: list[str]
    ollama_available: bool = False
    ollama_model: str | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    context: dict | None = None


class ChatResponse(BaseModel):
    message: str
    role: Literal["assistant"] = "assistant"
