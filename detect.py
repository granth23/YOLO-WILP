import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from yolo import predict, CLASSES
from blur import blur_check

def analyze(IMAGE_FILE):

    image = Image.open(IMAGE_FILE)
    pred = predict(image)
    draw = ImageDraw.Draw(image)

    per_c = 0
    detected = []

    Output = {}

    Output['detected'] = []

    for x1, y1, x2, y2, conf, class_id in pred:
        object_text = CLASSES[int(class_id)]

        if object_text == "person":
            per_c += 1
        else:
            if round(conf,2) >= .75:
                Output['detected'].append([int(round(x1,1)), int(round(y1,1)), int(round(x2,1)), int(round(y2,1))])

    if per_c == 0:
        Output['content'] = 'No person Detected'
    elif per_c > 1:
        Output['content'] = 'More than 1 person Detected'
    elif len(Output['detected']) > 1:
        Output['content'] = 'Illegal Object Detected'
    else:
        Output['content'] = 'None'

    Output['blur'] = blur_check(IMAGE_FILE)

    return Output
