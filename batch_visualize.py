import torch
import os
import cv2
import numpy as np
import pandas as pd
from torchvision import transforms
import timm
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ================= CONFIG =================
model_path = "brain_tumor_resnet50.pth"
input_folder = "archive/Testing"
output_folder = "batch_results"

class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD MODEL =================
model = timm.create_model("resnet50", pretrained=False, num_classes=4)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.to(device)
model.eval()

target_layers = [model.layer3[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

results = []
correct = 0
total = 0

print("Starting batch processing...\n")

for true_class in os.listdir(input_folder):
    class_folder = os.path.join(input_folder, true_class)
    
    if not os.path.isdir(class_folder):
        continue
    
    save_class_folder = os.path.join(output_folder, true_class)
    os.makedirs(save_class_folder, exist_ok=True)
    
    for img_name in tqdm(os.listdir(class_folder)):
        img_path = os.path.join(class_folder, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            continue
        
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()

        predicted_label = class_names[pred_class]

        # Grad-CAM
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred_class)]
        )[0]

        grayscale_cam = cv2.GaussianBlur(grayscale_cam, (5, 5), 0)

        rgb_img = np.array(image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Save image
        save_path = os.path.join(save_class_folder, img_name)
        cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        # Accuracy tracking
        if predicted_label == true_class:
            correct += 1
        total += 1

        results.append({
            "Image": img_name,
            "True Label": true_class,
            "Predicted Label": predicted_label,
            "Confidence": round(confidence, 4)
        })

# ================= SAVE REPORT =================
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_folder, "diagnosis_report.csv"), index=False)

accuracy = correct / total if total > 0 else 0

print("\n==============================")
print("Batch Processing Complete")
print(f"Total Images: {total}")
print(f"Accuracy: {accuracy:.4f}")
print("Results saved in:", output_folder)
print("==============================")
