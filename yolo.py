import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models.experimental import attempt_load
from utils.general import non_max_suppression

WEIGHTS = "models/yolov7.pt"
DEVICE = "cpu"
IMAGE_SIZE = 640

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "car|rot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"]

allowed = ["person", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "book"]

model = attempt_load(WEIGHTS, map_location=DEVICE)


def predict(image, image_size=640):
    image = np.asarray(image)

    # Resize image to the inference size
    ori_h, ori_w = image.shape[:2]
    image = cv2.resize(image, (image_size, image_size))

    # Transform image from numpy to torch format
    image_pt = torch.from_numpy(image).permute(2, 0, 1).to(DEVICE)
    image_pt = image_pt.float() / 255.0

    # Infer
    with torch.no_grad():
        pred = model(image_pt[None], augment=False)[0]

    # NMS
    pred = non_max_suppression(pred)[0].cpu().numpy()

    # Resize boxes to the original image size
    pred[:, [0, 2]] *= ori_w / image_size
    pred[:, [1, 3]] *= ori_h / image_size

    return pred

