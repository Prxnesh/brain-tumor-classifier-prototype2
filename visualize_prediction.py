import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import timm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= CONFIG =================
model_path = "brain_tumor_resnet50.pth"
image_path = "archive\\Testing\\glioma_tumor\\image(5).jpg"

class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
model = timm.create_model("resnet50", pretrained=False, num_classes=4)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()

print("Model loaded successfully.")
print("Using device:", device)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ================= PREDICTION =================
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

print(f"Predicted Class: {class_names[pred_class]}")
print(f"Confidence: {confidence:.4f}")

# ================= GRAD-CAM =================
# Using layer3 for better spatial resolution
target_layers = [model.layer3[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=[ClassifierOutputTarget(pred_class)]
)

grayscale_cam = grayscale_cam[0]

# Smooth heatmap slightly
grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), 0)

# ================= OVERLAY =================
rgb_img = np.array(image.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# ================= DESCRIPTION =================
def generate_description(label, conf):
    if label == "glioma_tumor":
        return f"High probability of Glioma detected.\nIrregular tumor mass observed.\nConfidence: {conf:.2f}"
    elif label == "meningioma_tumor":
        return f"Meningioma likely present.\nTumor appears well-defined.\nConfidence: {conf:.2f}"
    elif label == "pituitary_tumor":
        return f"Pituitary tumor detected.\nAbnormal growth in central region.\nConfidence: {conf:.2f}"
    else:
        return f"No tumor detected.\nBrain scan appears normal.\nConfidence: {conf:.2f}"

description = generate_description(class_names[pred_class], confidence)

# ================= DISPLAY =================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(visualization)
plt.title(f"{class_names[pred_class]} ({confidence:.2f})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.barh(class_names, probs.cpu().numpy()[0])
plt.title("Class Probabilities")
plt.xlim(0, 1)

plt.tight_layout()
plt.show()

print("\n--- Diagnosis Report ---")
print(description)
