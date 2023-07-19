import numpy as np
import cv2
import os
import base64

def blur_check(IMAGE_FILE):

    im_bytes = base64.b64decode(IMAGE_FILE)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(grey, cv2.CV_64F).var()

    if var < 120:
        return 0
    else:
        return 1
