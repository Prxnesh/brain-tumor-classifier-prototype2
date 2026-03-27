import torch
import timm

model = timm.create_model("resnet50", pretrained=False, num_classes=4)
state_dict = torch.load("brain_tumor_resnet50.pth", weights_only=True)
model.load_state_dict(state_dict)

print("Model loaded successfully.")\

