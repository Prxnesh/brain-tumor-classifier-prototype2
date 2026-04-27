from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AppConfigResponse, PredictionResponse
from app.services.classifier_service import ClassifierService
from app.services.ollama_service import OllamaService


mri_service: ClassifierService | None = None
ct_service: ClassifierService | None = None
ollama_service: OllamaService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global mri_service, ct_service, ollama_service
    if settings.mri_model_path.exists():
        mri_service = ClassifierService(settings.mri_model_path, "mri")
    if settings.ct_model_path.exists():
        ct_service = ClassifierService(settings.ct_model_path, "ct")
    if settings.ollama_enabled:
        ollama_service = OllamaService(
            settings.ollama_base_url,
            settings.ollama_model,
            settings.ollama_timeout_seconds,
        )
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
    resolved_ollama_model = ollama_service.resolve_model() if ollama_service is not None else None
    ollama_available = resolved_ollama_model is not None

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
        ollama_available=ollama_available,
        ollama_model=resolved_ollama_model,
    )


@app.post("/predict/mri", response_model=PredictionResponse)
async def predict_mri(
    file: UploadFile = File(...),
    use_ollama: bool = Form(False),
) -> PredictionResponse:
    if mri_service is None:
        raise HTTPException(status_code=503, detail="MRI model weights are missing.")
    image_bytes = await _read_upload(file)

    result = mri_service.predict(image_bytes)
    return _build_prediction_response("mri", result, use_ollama)


@app.post("/predict/ct", response_model=PredictionResponse)
async def predict_ct(
    file: UploadFile = File(...),
    use_ollama: bool = Form(False),
) -> PredictionResponse:
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
    return _build_prediction_response("ct", result, use_ollama)


@app.post("/predict/fusion", response_model=PredictionResponse)
async def predict_fusion(
    mri_file: UploadFile = File(...),
    ct_file: UploadFile = File(...),
    use_ollama: bool = Form(False),
) -> PredictionResponse:
    if mri_service is None or ct_service is None:
        raise HTTPException(
            status_code=503,
            detail="Fusion needs both MRI and CT trained weights. MRI is ready; CT still needs a trained model file.",
        )

    mri_bytes = await _read_upload(mri_file)
    ct_bytes = await _read_upload(ct_file)
    result = mri_service.build_fusion_result(mri_bytes, ct_bytes)
    return _build_prediction_response("fusion", result, use_ollama)


async def _read_upload(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return image_bytes


def _build_prediction_response(
    modality: str,
    result,
    use_ollama: bool,
) -> PredictionResponse:
    report = result.report
    notes = list(result.notes)
    report_provider = "template"

    if use_ollama:
        if ollama_service is None:
            notes.append("Ollama report enhancement was requested, but Ollama integration is disabled in backend settings.")
        elif not ollama_service.is_available():
            notes.append(
                f"Ollama report enhancement was requested, but no responsive Ollama server was detected at {settings.ollama_base_url}."
            )
        else:
            try:
                resolved_model = ollama_service.resolve_model()
                report = ollama_service.generate_report(
                    modality=modality,
                    predicted_label=result.predicted_label,
                    confidence=result.confidence,
                    probabilities=result.probabilities,
                    template_report=result.report,
                )
                report_provider = "ollama"
                notes.append(
                    f"Structured report enhanced locally with Ollama model {resolved_model or settings.ollama_model}."
                )
            except Exception as exc:
                notes.append(f"Ollama report enhancement failed, so the template report was used instead: {exc}")

    return PredictionResponse(
        modality=modality,
        predicted_label=result.predicted_label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        gradcam_overlay=result.gradcam_overlay,
        original_preview=result.original_preview,
        report=report,
        report_provider=report_provider,
        notes=notes,
    )
