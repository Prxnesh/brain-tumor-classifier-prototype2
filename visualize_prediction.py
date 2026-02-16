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

# ---- Config ----
model_path = "brain_tumor_resnet50_best.pth"
image_path = "archive/Testing/glioma_tumor/your_image.jpg"

class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load Model ----
model = timm.create_model("resnet50", pretrained=False, num_classes=4)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# ---- Image Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ---- Prediction ----
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_class].item()

# ---- Grad-CAM ----
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
grayscale_cam = grayscale_cam[0]

# ---- Overlay Heatmap ----
rgb_img = np.array(image.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# ---- Description Generator ----
def generate_description(label, conf):
    if label == "glioma_tumor":
        return f"High probability of Glioma detected. Irregular tumor mass observed. Confidence: {conf:.2f}"
    elif label == "meningioma_tumor":
        return f"Meningioma likely present. Tumor appears well-defined. Confidence: {conf:.2f}"
    elif label == "pituitary_tumor":
        return f"Pituitary tumor detected. Abnormal growth in central region. Confidence: {conf:.2f}"
    else:
        return f"No tumor detected. Brain scan appears normal. Confidence: {conf:.2f}"

description = generate_description(class_names[pred_class], confidence)

# ---- Show Result ----
plt.figure(figsize=(6,6))
plt.imshow(visualization)
plt.title(f"{class_names[pred_class]} ({confidence:.2f})")
plt.axis("off")
plt.show()

print("\n--- Diagnosis Report ---")
print(description)
