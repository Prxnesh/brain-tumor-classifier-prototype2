# NeuroVision AI Backend

FastAPI service for MRI inference, Grad-CAM visualization, and report generation.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Environment overrides

- `BRAIN_TUMOR_MODEL_PATH`
- `BRAIN_TUMOR_MRI_DATASET_PATH`
- `BRAIN_TUMOR_ALLOWED_ORIGINS`
