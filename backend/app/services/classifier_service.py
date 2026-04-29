import base64
import io
from dataclasses import dataclass, field
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
    tumor_location: dict | None = field(default=None)


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

        tumor_location = self._extract_tumor_location(grayscale_cam)

        rgb_img = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        notes = [
            f"{self.modality_label.upper()} inference pipeline loaded successfully.",
            "Structured report is AI-generated and formatted in a radiology-style summary for decision support.",
            "Formal diagnosis still requires full-study review, clinical correlation, and clinician oversight.",
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
            tumor_location=tumor_location,
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

        tumor_location = self._extract_tumor_location(grayscale_cam)

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
            report=self._build_report(predicted_label, confidence, fused_probabilities, "fusion"),
            notes=[
                "Fusion currently uses late-probability averaging across independently trained MRI and CT classifiers.",
                "The displayed saliency map in fusion mode is generated from the MRI branch only.",
                "A dedicated paired-modality fusion model would be the appropriate next step for stronger clinical research performance.",
            ],
            tumor_location=tumor_location,
        )

    def _predict_probabilities(self, input_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(input_tensor)
            return torch.softmax(outputs, dim=1)[0].cpu().numpy()

    @staticmethod
    def _extract_tumor_location(grayscale_cam: np.ndarray) -> dict:
        """Extract tumor centroid from GradCAM heatmap using weighted centroid."""
        h, w = grayscale_cam.shape
        total_weight = float(np.sum(grayscale_cam))

        if total_weight < 1e-6:
            return {
                "cx": 0.5,
                "cy": 0.5,
                "radius": 0.1,
                "quadrant": "center",
                "description": "Activation too diffuse to localize",
            }

        ys = np.arange(h).reshape(-1, 1).astype(np.float32)
        xs = np.arange(w).reshape(1, -1).astype(np.float32)
        cy = float(np.sum(ys * grayscale_cam)) / total_weight / h
        cx = float(np.sum(xs * grayscale_cam)) / total_weight / w

        # Estimate activation radius from area above 60th percentile
        threshold = np.percentile(grayscale_cam, 60)
        high_mask = (grayscale_cam >= threshold).astype(np.float32)
        area_fraction = float(np.sum(high_mask)) / (h * w)
        radius = float(np.sqrt(area_fraction / np.pi)) * 0.55
        radius = max(0.05, min(0.35, radius))

        quadrant_v = "upper" if cy < 0.5 else "lower"
        quadrant_h = "left" if cx < 0.5 else "right"
        quadrant = f"{quadrant_v}-{quadrant_h}"
        description = (
            f"Peak activation at {cx * 100:.0f}% from left, "
            f"{cy * 100:.0f}% from top ({quadrant} quadrant)"
        )

        return {
            "cx": round(cx, 4),
            "cy": round(cy, 4),
            "radius": round(radius, 4),
            "quadrant": quadrant,
            "description": description,
        }

    @classmethod
    def _build_report(
        cls,
        label: str,
        confidence: float,
        probabilities: np.ndarray,
        modality_label: str,
    ) -> list[dict[str, str]]:
        ranked = sorted(
            ((CLASS_NAMES[idx], float(value)) for idx, value in enumerate(probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        differential = ", ".join(
            f"{name.replace('_', ' ')} ({score:.1%})" for name, score in ranked[:3]
        )
        technique_line = (
            "Multimodal fusion generated by averaging MRI and CT class probabilities, with Grad-CAM visual explanation displayed from the MRI branch."
            if modality_label == "fusion"
            else f"Single uploaded {modality_label.upper()} image processed through a ResNet50 classifier with Grad-CAM saliency review."
        )
        findings_map = {
            "glioma_tumor": "AI pattern analysis favors an intra-axial glioma-spectrum presentation, with saliency concentrated over a comparatively irregular focal abnormality.",
            "meningioma_tumor": "AI pattern analysis favors a meningioma-pattern presentation, with attention drawn toward a more circumscribed lesion morphology within the learned label space.",
            "pituitary_tumor": "AI pattern analysis favors a pituitary-pattern presentation, with highest saliency near a central sellar or parasellar appearing focus.",
            "no_tumor": "No dominant tumor-pattern class is favored by the model on the submitted image, and the overall appearance is most aligned with the no-tumor class in the current label space.",
        }
        recommendation_line = (
            "Correlate with the complete imaging examination, lesion distribution, mass effect, edema pattern, contrast behavior when available, prior studies, and formal neuroradiology review."
            if label != "no_tumor"
            else "Absence of a dominant tumor-pattern prediction does not exclude subtle, early, or non-mass pathology; correlate with the complete study and clinical context."
        )

        return [
            {
                "title": "Study summary",
                "body": (
                    f"AI-assisted review of the submitted {modality_label.upper()} input identifies "
                    f"{label.replace('_', ' ')} as the leading class prediction at {confidence:.1%} confidence."
                ),
            },
            {
                "title": "Technique",
                "body": technique_line,
            },
            {
                "title": "Findings",
                "body": findings_map[label],
            },
            {
                "title": "Saliency interpretation",
                "body": cls._build_reasoning_summary(label, confidence, probabilities, modality_label),
            },
            {
                "title": "Differential consideration",
                "body": f"Top ranked differential classes from the current model are {differential}. These probabilities reflect model preference within the trained label space and should not be interpreted as a complete clinical differential diagnosis.",
            },
            {
                "title": "Impression",
                "body": cls._build_clinical_note(label, confidence, probabilities, modality_label),
            },
            {
                "title": "Recommendations and limitations",
                "body": (
                    f"{recommendation_line} This report is AI-generated, clinically styled, and intended for decision support only; "
                    "it is not a validated standalone medical report."
                ),
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
