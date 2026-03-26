# NeuroVision AI

Full-stack brain tumor detection prototype with:

- MRI upload and classification
- Grad-CAM visual explanation
- Clinical-style summary report
- CT and CT+MRI fusion endpoints scaffolded for future datasets

## Current status

- `MRI`: implemented end-to-end
- `CT`: app and API placeholders ready, dataset and trained weights still needed
- `Fusion`: app and API placeholders ready, paired CT+MRI data still needed

## Project structure

- [backend](D:/Devloper/minor%20project/brain_tumor_ai/backend): FastAPI API for inference and reports
- [frontend](D:/Devloper/minor%20project/brain_tumor_ai/frontend): React client for upload and visualization
- [archive MRI](D:/Devloper/minor%20project/brain_tumor_ai/archive%20MRI): MRI training and testing dataset
- [brain_tumor_resnet50.pth](D:/Devloper/minor%20project/brain_tumor_ai/brain_tumor_resnet50.pth): trained MRI classifier weights

## Backend setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`

Backend default URL: `http://127.0.0.1:8000`

## Notes

- The frontend build has been verified successfully in this workspace.
- The backend source compiles successfully, but the current Python environment still needs packages from `backend/requirements.txt` before the API can run.
- This is a decision-support prototype and not a clinically validated medical device.
