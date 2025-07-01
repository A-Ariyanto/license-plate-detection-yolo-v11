# Author: Abdullah Ariyanto
# This is a script to train a YOLOv11 model for license plate recognition using a dataset from Roboflow.

# Before running this script, ensure you have the required packages installed:
# pip3 install ultralytics
# pip3 install roboflow

import os
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = "eifZWS53J7x5kD1drknb"
WORKSPACE = "roboflow-universe-projects"
PROJECT = "license-plate-recognition-rxg4e"
VERSION = 11
EPOCHS = 3
IMAGE_SIZE = 640
DATA_YAML_PATH = "License-Plate-Recognition-11/data.yaml"

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

def train_model():
    print("Training YOLOv11 model...")
    model = YOLO("yolo11n.yaml")
    results = model.train(data=DATA_YAML_PATH, epochs=EPOCHS, imgsz=IMAGE_SIZE)

    print("Evaluating model...")
    results = model.val()

    return results
    
if __name__ == "__main__":
    dataset_path = download_dataset()
    train_model()

