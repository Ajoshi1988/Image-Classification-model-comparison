import ultralytics
import supervision
import torch
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt





# Load a model

model = YOLO("yolo11l-cls.pt")  # load a pretrained model (recommended for training)


# Train the model
model.train(data="/home/ubuntu/yolov_train_data", epochs=200, imgsz=64, device=0)