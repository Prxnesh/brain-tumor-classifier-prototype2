import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm import create_model
from torchvision import transforms


CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    probabilities: list[dict[str, float | str]]
    gradcam_overlay: str
    original_preview: str
    report: list[dict[str, str]]
    notes: list[str]


class ClassifierService:
    def __init__(self, model_path: Path, modality_label: str) -> None:
        self.model_path = model_path
        self.modality_label = modality_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )
        self.model = create_model("resnet50", pretrained=False, num_classes=4)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.cam = GradCAM(model=self.model, target_layers=[self.model.layer3[-1]])

    def predict(self, image_bytes: bytes, extra_notes: list[str] | None = None) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = self._predict_probabilities(input_tensor)
        pred_class = int(np.argmax(probabilities))
        confidence = float(probabilities[pred_class])
        predicted_label = CLASS_NAMES[pred_class]

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0]
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), 0)

        rgb_img = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        notes = [
            f"{self.modality_label.upper()} inference is active in this build.",
            "This is a decision-support prototype and requires clinician review.",
        ]
        if extra_notes:
            notes.extend(extra_notes)

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=[
                {"label": label, "probability": float(prob)}
                for label, prob in zip(CLASS_NAMES, probabilities.tolist(), strict=True)
            ],
            gradcam_overlay=self._to_data_url(Image.fromarray(overlay)),
            original_preview=self._to_data_url(Image.fromarray((rgb_img * 255).astype(np.uint8))),
            report=self._build_report(predicted_label, confidence, probabilities),
            notes=notes,
        )

    def predict_probabilities_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return self._predict_probabilities(input_tensor)

    def build_fusion_result(self, mri_bytes: bytes, ct_bytes: bytes) -> PredictionResult:
        mri_image = Image.open(io.BytesIO(mri_bytes)).convert("RGB")
        mri_tensor = self.transform(mri_image).unsqueeze(0).to(self.device)
        ct_probabilities = self.predict_probabilities_from_bytes(ct_bytes)
        mri_probabilities = self._predict_probabilities(mri_tensor)
        fused_probabilities = (mri_probabilities + ct_probabilities) / 2

        pred_class = int(np.argmax(fused_probabilities))
        confidence = float(fused_probabilities[pred_class])
        predicted_label = CLASS_NAMES[pred_class]

        grayscale_cam = self.cam(
            input_tensor=mri_tensor,
            targets=[ClassifierOutputTarget(pred_class)],
        )[0]
        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), 0)

        rgb_img = np.array(mri_image.resize((224, 224)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=[
                {"label": label, "probability": float(prob)}
                for label, prob in zip(CLASS_NAMES, fused_probabilities.tolist(), strict=True)
            ],
            gradcam_overlay=self._to_data_url(Image.fromarray(overlay)),
            original_preview=self._to_data_url(Image.fromarray((rgb_img * 255).astype(np.uint8))),
            report=[
                {
                    "title": "Fusion summary",
                    "body": "This result averages MRI and CT class probabilities as a first-pass multimodal baseline.",
                },
                {
                    "title": "Primary finding",
                    "body": f"Combined MRI and CT evidence is most aligned with {predicted_label.replace('_', ' ')}.",
                },
                {
                    "title": "Clinical caution",
                    "body": "This fusion flow is a heuristic baseline. A dedicated paired-modality fusion model would be stronger once paired studies are available.",
                },
            ],
            notes=[
                "Fusion currently uses late-probability averaging across MRI and CT models.",
                "Grad-CAM is generated from the MRI branch in the current prototype.",
            ],
        )

    def _predict_probabilities(self, input_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(input_tensor)
            return torch.softmax(outputs, dim=1)[0].cpu().numpy()

    @staticmethod
    def _build_report(label: str, confidence: float, probabilities: np.ndarray) -> list[dict[str, str]]:
        highest_other = sorted(
            ((CLASS_NAMES[idx], float(value)) for idx, value in enumerate(probabilities) if CLASS_NAMES[idx] != label),
            key=lambda item: item[1],
            reverse=True,
        )[0]

        findings = {
            "glioma_tumor": "Pattern is most consistent with a glioma-like presentation in the uploaded scan.",
            "meningioma_tumor": "Model attention favors meningioma-like morphology in the visible region.",
            "pituitary_tumor": "Prediction suggests a pituitary-region abnormality in the provided image.",
            "no_tumor": "The uploaded scan appears more consistent with the no-tumor class.",
        }

        return [
            {
                "title": "Primary finding",
                "body": findings[label],
            },
            {
                "title": "Confidence summary",
                "body": f"Top-class confidence is {confidence:.1%}. The next strongest alternative is {highest_other[0]} at {highest_other[1]:.1%}.",
            },
            {
                "title": "Clinical caution",
                "body": "This output is a decision-support prototype and should be reviewed by a qualified clinician before any diagnosis or treatment decision.",
            },
        ]

    @staticmethod
    def _to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
