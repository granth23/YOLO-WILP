import cv2
import numpy as np
from PIL import Image
import base64
from yolo import allowed
from yolo import predict, CLASSES

def analyze(IMAGE_FILE):
    IMAGE_FILE = base64.b64decode(IMAGE_FILE)
    pred = predict(IMAGE_FILE)

    Output = []

    for x1, y1, x2, y2, conf, class_id in pred:
        object_text = CLASSES[int(class_id)]
        if object_text in allowed:
            temp = {}
            temp['conf'] = int(conf)
            temp['type'] = object_text
            temp['coords'] = [int(round(x1,1)), int(round(y1,1)), int(round(x2,1)), int(round(y2,1))]
            Output.append(temp)

    return Output

from file import base64file
print(analyze(base64file))
