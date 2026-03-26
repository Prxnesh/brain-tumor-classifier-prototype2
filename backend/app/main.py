from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AppConfigResponse, PredictionResponse
from app.services.mri_service import MRIInferenceService


mri_service: MRIInferenceService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global mri_service
    if not settings.model_path.exists():
        raise RuntimeError(f"Model weights not found at {settings.model_path}")
    mri_service = MRIInferenceService()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config", response_model=AppConfigResponse)
async def get_config() -> AppConfigResponse:
    return AppConfigResponse(
        supported_modalities=["mri", "ct", "fusion"],
        available_now=["mri"],
        pending_datasets=["ct", "fusion"],
    )


@app.post("/predict/mri", response_model=PredictionResponse)
async def predict_mri(file: UploadFile = File(...)) -> PredictionResponse:
    if mri_service is None:
        raise HTTPException(status_code=503, detail="MRI model is still loading.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = mri_service.predict(image_bytes)
    return PredictionResponse(
        modality="mri",
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_overlay=result.gradcam_overlay,
        original_preview=result.original_preview,
        report=result.report,
        notes=result.notes,
    )


@app.post("/predict/ct", response_model=PredictionResponse)
async def predict_ct() -> PredictionResponse:
    raise HTTPException(
        status_code=501,
        detail="CT inference is scaffolded but needs a CT dataset and trained weights.",
    )


@app.post("/predict/fusion", response_model=PredictionResponse)
async def predict_fusion() -> PredictionResponse:
    raise HTTPException(
        status_code=501,
        detail="CT+MRI fusion is scaffolded but needs paired CT/MRI data and a fusion model.",
    )
