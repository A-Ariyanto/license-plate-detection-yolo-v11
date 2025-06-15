# Before running this script, ensure you have the required packages installed:
# pip3 install ultralytics
# pip3 install roboflow

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
DATA_YAML_PATH = "License-Plate-Recognition-11/data.yaml"

# === STEP 1: DOWNLOAD DATASET FROM ROBOFLOW ===
def download_dataset():
    if os.path.exists("License-Plate-Recognition-11"):
        print("Dataset already exists. Skipping download.")
        return "License-Plate-Recognition-11"
    
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION)
    dataset = version.download("yolov11")
    return dataset.location  # Path to dataset directory

# === STEP 2: TRAIN THE YOLO MODEL ===
def train_model():
    
    
    # Using and prexisting model
    print("Using pre-existing YOLOv11 model...")
    model = YOLO("yolo11n.pt")
    
    # Pure training
    # print("Training YOLOv11 model...")
    # model = YOLO("yolo11n.yaml")
    # results = model.train(data=DATA_YAML_PATH, epochs=3)
    # results = model.val()

    results = model("demo.mp4", save=True)
    
    
    
# === MAIN EXECUTION ===
if __name__ == "__main__":
    dataset_path = download_dataset()
    train_model()