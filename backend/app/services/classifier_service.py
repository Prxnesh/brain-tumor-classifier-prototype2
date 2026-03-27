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
            report=self._build_report(predicted_label, confidence, probabilities, self.modality_label),
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
                    "title": "Why the model leaned this way",
                    "body": self._build_reasoning_summary(predicted_label, confidence, fused_probabilities, "fusion"),
                },
                {
                    "title": "Primary finding",
                    "body": f"Combined MRI and CT evidence is most aligned with {predicted_label.replace('_', ' ')}.",
                },
                {
                    "title": "Detailed clinical note",
                    "body": self._build_clinical_note(predicted_label, confidence, fused_probabilities, "fusion"),
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

    @classmethod
    def _build_report(
        cls,
        label: str,
        confidence: float,
        probabilities: np.ndarray,
        modality_label: str,
    ) -> list[dict[str, str]]:
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
                "title": "Why the model leaned this way",
                "body": cls._build_reasoning_summary(label, confidence, probabilities, modality_label),
            },
            {
                "title": "Detailed clinical note",
                "body": cls._build_clinical_note(label, confidence, probabilities, modality_label),
            },
            {
                "title": "Clinical caution",
                "body": "This output is a decision-support prototype and should be reviewed by a qualified clinician before any diagnosis or treatment decision.",
            },
        ]

    @staticmethod
    def _build_reasoning_summary(
        label: str,
        confidence: float,
        probabilities: np.ndarray,
        modality_label: str,
    ) -> str:
        ranked = sorted(
            ((CLASS_NAMES[idx], float(value)) for idx, value in enumerate(probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        top_label, top_score = ranked[0]
        runner_up_label, runner_up_score = ranked[1]
        confidence_gap = top_score - runner_up_score

        attention_text = {
            "glioma_tumor": "the attention map tends to concentrate over an irregular focal region rather than diffuse normal parenchyma",
            "meningioma_tumor": "the attention map favors a more circumscribed extra-axial looking region with comparatively cleaner boundaries",
            "pituitary_tumor": "the attention map is drawn toward a central sellar or parasellar appearing focus",
            "no_tumor": "the attention map is relatively less concentrated on a discrete mass-like region and the class balance stays closer to normal tissue patterns",
        }

        strength_text = (
            "The decision is relatively strong"
            if confidence_gap >= 0.25
            else "The decision is moderate"
            if confidence_gap >= 0.12
            else "The decision is comparatively soft"
        )

        return (
            f"For this {modality_label.upper()} study, the model ranked {top_label.replace('_', ' ')} highest at {confidence:.1%}. "
            f"The nearest alternative was {runner_up_label.replace('_', ' ')} at {runner_up_score:.1%}, giving a margin of {confidence_gap:.1%}. "
            f"{strength_text}, and {attention_text[label]}. This means the classifier is not only choosing the top class, "
            f"but also separating it from the next-best explanation by a measurable probability gap."
        )

    @staticmethod
    def _build_clinical_note(
        label: str,
        confidence: float,
        probabilities: np.ndarray,
        modality_label: str,
    ) -> str:
        ranked = sorted(
            ((CLASS_NAMES[idx], float(value)) for idx, value in enumerate(probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        differential = ", ".join(
            f"{name.replace('_', ' ')} ({score:.1%})" for name, score in ranked[:3]
        )

        impression_map = {
            "glioma_tumor": "Impression: focal appearance is most compatible with a glioma-pattern class prediction. In a real workflow this would justify correlation with lesion location, margins, edema, mass effect, and contrast behavior if available.",
            "meningioma_tumor": "Impression: imaging pattern is most compatible with a meningioma-pattern class prediction. In practice this would merit review for extra-axial features, dural attachment, adjacent edema, and displacement of nearby structures.",
            "pituitary_tumor": "Impression: imaging pattern is most compatible with a pituitary-pattern class prediction. In a real review this would prompt closer inspection of the sellar region, suprasellar extension, and relationship to surrounding anatomy.",
            "no_tumor": "Impression: no-tumor class is favored on this image. That does not exclude subtle pathology, small lesions, motion-related obscuration, or findings that require multi-slice or multi-sequence review.",
        }

        certainty_note = (
            "Confidence is high enough to support a focused review of the leading class first."
            if confidence >= 0.75
            else "Confidence is intermediate, so the leading class should be interpreted together with the secondary differential."
            if confidence >= 0.5
            else "Confidence is limited, so this output should be treated as a weak suggestion rather than a stable conclusion."
        )

        return (
            f"{impression_map[label]} {certainty_note} "
            f"Differential ranking from the current model is: {differential}. "
            f"Recommended next step: correlate this {modality_label.upper()} output with the full study, clinical history, and expert radiology review before any diagnostic or treatment decision."
        )

    @staticmethod
    def _to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
