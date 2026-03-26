from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AppConfigResponse, PredictionResponse
from app.services.classifier_service import ClassifierService


mri_service: ClassifierService | None = None
ct_service: ClassifierService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global mri_service, ct_service
    if settings.mri_model_path.exists():
        mri_service = ClassifierService(settings.mri_model_path, "mri")
    if settings.ct_model_path.exists():
        ct_service = ClassifierService(settings.ct_model_path, "ct")
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
    available_now = []
    pending = []

    if mri_service is not None:
        available_now.append("mri")
    else:
        pending.append("mri")

    if ct_service is not None:
        available_now.append("ct")
    else:
        pending.append("ct")

    if mri_service is not None and ct_service is not None:
        available_now.append("fusion")
    else:
        pending.append("fusion")

    return AppConfigResponse(
        supported_modalities=["mri", "ct", "fusion"],
        available_now=available_now,
        pending_datasets=pending,
    )


@app.post("/predict/mri", response_model=PredictionResponse)
async def predict_mri(file: UploadFile = File(...)) -> PredictionResponse:
    if mri_service is None:
        raise HTTPException(status_code=503, detail="MRI model weights are missing.")
    image_bytes = await _read_upload(file)

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
async def predict_ct(file: UploadFile = File(...)) -> PredictionResponse:
    if ct_service is None:
        raise HTTPException(
            status_code=503,
            detail=f"CT dataset is present, but trained weights are missing at {settings.ct_model_path.name}.",
        )

    image_bytes = await _read_upload(file)
    result = ct_service.predict(
        image_bytes,
        extra_notes=["CT dataset is present. Train and save CT weights to enable this path permanently."],
    )
    return PredictionResponse(
        modality="ct",
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_overlay=result.gradcam_overlay,
        original_preview=result.original_preview,
        report=result.report,
        notes=result.notes,
    )


@app.post("/predict/fusion", response_model=PredictionResponse)
async def predict_fusion(
    mri_file: UploadFile = File(...),
    ct_file: UploadFile = File(...),
) -> PredictionResponse:
    if mri_service is None or ct_service is None:
        raise HTTPException(
            status_code=503,
            detail="Fusion needs both MRI and CT trained weights. MRI is ready; CT still needs a trained model file.",
        )

    mri_bytes = await _read_upload(mri_file)
    ct_bytes = await _read_upload(ct_file)
    result = mri_service.build_fusion_result(mri_bytes, ct_bytes)
    return PredictionResponse(
        modality="fusion",
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_overlay=result.gradcam_overlay,
        original_preview=result.original_preview,
        report=result.report,
        notes=result.notes,
    )


async def _read_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return image_bytes
