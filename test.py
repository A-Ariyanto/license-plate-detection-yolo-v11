import os
from ultralytics import YOLO

# === Testing Trained Yolo Model ===
def test_model(model):
    results = model("demo.mp4", save=True)