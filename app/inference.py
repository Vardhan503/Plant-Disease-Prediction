import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import json
import os

# --- 1. Define Architecture ---
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), nn.Dropout(0.2)
        )
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

# --- 2. Setup Globals ---
# Ensure these paths are correct relative to where you run `uvicorn`
MODEL_PATH = "models/plant_disease_custom_cnn.pth"
CLASSES_PATH = "models/classes.json" 
DEVICE = torch.device("cpu")
class_names = []
model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model():
    global model, class_names
    
    # FIX: Handle the nested JSON structure {"classes": [...]}
    try:
        with open(CLASSES_PATH, "r") as f:
            data = json.load(f)
            # Check if it's a dict with a "classes" key, or just a list
            if isinstance(data, dict) and "classes" in data:
                class_names = data["classes"]
            else:
                class_names = data
        print(f"Loaded {len(class_names)} classes.")
    except FileNotFoundError:
        print(f"WARNING: {CLASSES_PATH} not found. Returning IDs.")
        class_names = [str(i) for i in range(38)]

    # Load Model
    model = PlantDiseaseCNN(num_classes=38)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"ERROR: Model not found at {MODEL_PATH}")

def predict(image_bytes):
    if model is None:
        raise RuntimeError("Model not loaded!")
        
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    idx = predicted_idx.item()
    # FIX: Ensure we don't go out of bounds
    if idx < len(class_names):
        return class_names[idx], confidence.item()
    return str(idx), confidence.item()