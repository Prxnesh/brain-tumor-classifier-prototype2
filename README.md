# NeuroVision AI

Full-stack brain tumor detection prototype with:

- MRI upload and classification
- CT upload path wired to its own model slot
- Grad-CAM visual explanation
- Clinical-style summary report
- CT and CT+MRI fusion endpoints scaffolded and ready for model weights

## Current status

- `MRI`: implemented end-to-end
- `CT`: dataset added, app/API wired, trained CT weights still needed
- `Fusion`: dual-upload flow added, runs once both MRI and CT weights are available

## Project structure

- [backend](D:/Devloper/minor%20project/brain_tumor_ai/backend): FastAPI API for inference and reports
- [frontend](D:/Devloper/minor%20project/brain_tumor_ai/frontend): React client for upload and visualization
- [archive MRI](D:/Devloper/minor%20project/brain_tumor_ai/archive%20MRI): MRI training and testing dataset
- [archive ct](D:/Devloper/minor%20project/brain_tumor_ai/archive%20ct): CT training and testing dataset
- [brain_tumor_resnet50.pth](D:/Devloper/minor%20project/brain_tumor_ai/brain_tumor_resnet50.pth): trained MRI classifier weights
- [train_modality_classifier.py](D:/Devloper/minor%20project/brain_tumor_ai/train_modality_classifier.py): train MRI or CT classifiers

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
- The backend API boots successfully from the local `backend/.venv`.
- MRI inference has been smoke-tested successfully.
- CT inference will go live after training and saving `brain_tumor_ct_resnet50.pth`.
- This is a decision-support prototype and not a clinically validated medical device.

## Train CT weights

```bash
cd backend
.venv\Scripts\activate
cd ..
python train_modality_classifier.py --dataset "archive ct" --output brain_tumor_ct_resnet50.pth --epochs 5 --batch-size 16
```
