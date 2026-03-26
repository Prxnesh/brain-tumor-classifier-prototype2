# NeuroVision AI Backend

FastAPI service for MRI inference, Grad-CAM visualization, and report generation.

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

CT support becomes active automatically once `brain_tumor_ct_resnet50.pth` exists in the project root.

## Environment overrides

- `BRAIN_TUMOR_MRI_MODEL_PATH`
- `BRAIN_TUMOR_CT_MODEL_PATH`
- `BRAIN_TUMOR_MRI_DATASET_PATH`
- `BRAIN_TUMOR_CT_DATASET_PATH`
- `BRAIN_TUMOR_ALLOWED_ORIGINS`
