import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import base64

from yolo import predict, CLASSES
from blur import blur_check

def analyze(IMAGE_FILE):

    IMAGE_FILE = base64.b64decode(IMAGE_FILE)

    pred = predict(IMAGE_FILE)

    im_file = BytesIO(IMAGE_FILE)
    image = Image.open(im_file)
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

    im_arr = np.frombuffer(IMAGE_FILE, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    blur = blur_check(IMAGE_FILE)
    if blur == True:
        Output['content'] = 'Image is Blurry'
    elif per_c == 0:
        Output['content'] = 'No person Detected'
    elif per_c > 1:
        Output['content'] = 'More than 1 person Detected'
    elif len(Output['detected']) > 1:
        Output['content'] = 'Illegal Object Detected'
    else:
        Output['content'] = False

    return Output
