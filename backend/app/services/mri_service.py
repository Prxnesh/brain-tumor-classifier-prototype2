import base64
import io
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from timm import create_model
from torchvision import transforms

from app.config import settings


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


class MRIInferenceService:
    def __init__(self) -> None:
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
        state_dict = torch.load(settings.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        # Multi-layer CAM captures both coarse context and sharper lesion boundaries.
        self.target_layers = [self.model.layer3[-1], self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
        self.cam_pp = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        predicted_label = CLASS_NAMES[pred_class]

        grayscale_cam = self._build_robust_cam(input_tensor, pred_class)

        rgb_img = np.array(image.resize((224, 224)), dtype=np.float32) / 255.0
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return PredictionResult(
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=[
                {"label": label, "probability": float(prob)}
                for label, prob in zip(CLASS_NAMES, probs.tolist(), strict=True)
            ],
            gradcam_overlay=self._to_data_url(Image.fromarray(overlay)),
            original_preview=self._to_data_url(Image.fromarray((rgb_img * 255).astype(np.uint8))),
            report=self._build_report(predicted_label, confidence, probs),
            notes=[
                "MRI inference is active in this build.",
                "Heatmap is generated with Grad-CAM + Grad-CAM++ + test-time augmentation averaging.",
                "CT-only and CT+MRI fusion flows are scaffolded and ready for datasets.",
            ],
        )

    def _build_robust_cam(self, input_tensor: torch.Tensor, pred_class: int) -> np.ndarray:
        targets = [ClassifierOutputTarget(pred_class)]
        tta_inputs = self._generate_tta_inputs(input_tensor)
        cam_maps: list[np.ndarray] = []

        for variant in tta_inputs:
            cam_basic = self.cam(input_tensor=variant, targets=targets)[0]
            cam_pp = self.cam_pp(input_tensor=variant, targets=targets)[0]
            fused = 0.45 * cam_basic + 0.55 * cam_pp
            cam_maps.append(self._undo_tta_if_needed(fused, variant, input_tensor))

        stacked = np.stack(cam_maps, axis=0)
        cam = np.mean(stacked, axis=0)
        cam = self._normalize_cam(cam)
        cam = cv2.GaussianBlur(cam, (7, 7), 0)
        cam = self._normalize_cam(cam)
        return cam

    def _generate_tta_inputs(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        # Use horizontal and vertical flips to stabilize saliency without changing image scale.
        return [
            input_tensor,
            torch.flip(input_tensor, dims=[3]),
            torch.flip(input_tensor, dims=[2]),
        ]

    def _undo_tta_if_needed(
        self,
        cam_map: np.ndarray,
        variant_tensor: torch.Tensor,
        original_tensor: torch.Tensor,
    ) -> np.ndarray:
        result = cam_map
        if torch.equal(variant_tensor, torch.flip(original_tensor, dims=[3])):
            result = np.flip(result, axis=1)
        elif torch.equal(variant_tensor, torch.flip(original_tensor, dims=[2])):
            result = np.flip(result, axis=0)
        return np.ascontiguousarray(result)

    @staticmethod
    def _normalize_cam(cam_map: np.ndarray) -> np.ndarray:
        cam = cam_map.astype(np.float32)
        cam -= cam.min()
        max_value = cam.max()
        if max_value < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        return cam / max_value

    def _build_report(self, label: str, confidence: float, probs: np.ndarray) -> list[dict[str, str]]:
        highest_other = sorted(
            ((CLASS_NAMES[idx], float(value)) for idx, value in enumerate(probs) if CLASS_NAMES[idx] != label),
            key=lambda item: item[1],
            reverse=True,
        )[0]

        findings = {
            "glioma_tumor": "Pattern is most consistent with a glioma-like presentation in the uploaded MRI slice.",
            "meningioma_tumor": "Model attention favors meningioma-like morphology in the visible region.",
            "pituitary_tumor": "Prediction suggests a pituitary-region abnormality in the provided image.",
            "no_tumor": "The uploaded MRI slice appears more consistent with the no-tumor class.",
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
