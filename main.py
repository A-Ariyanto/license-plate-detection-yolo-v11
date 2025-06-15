import os
from roboflow import Roboflow
from ultralytics import YOLO

# === CONFIGURATION ===
ROBOFLOW_API_KEY = "eifZWS53J7x5kD1drknb"
WORKSPACE = "roboflow-universe-projects"
PROJECT = "license-plate-recognition-rxg4e"
VERSION = 11
EPOCHS = 3
IMAGE_SIZE = 640

# === STEP 1: DOWNLOAD DATASET FROM ROBOFLOW ===
def download_dataset():
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download("yolov11")
    return dataset.location  # Path to dataset directory

# === STEP 2: TRAIN THE YOLO MODEL ===
def train_model():
    print("Training YOLOv11 model...")
    model = YOLO("yolo11n.yaml")
    
    results = model.train(data="coco8.yaml", epochs=3)
    
    results = model.val()

    results = model("https://ultralytics.com/images/bus.jpg")

    success = model.export(format="onnx")
    
    
    
# === MAIN EXECUTION ===
if __name__ == "__main__":
    dataset_path = download_dataset()
    # train_model(dataset_path)