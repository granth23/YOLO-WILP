import base64
import numpy as np
import cv2

def blur_check(file):

    im_bytes = base64.b64decode(file)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)


    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    var = cv2.Laplacian(grey, cv2.CV_64F).var()

    if var < 20:
        return True
    else:
        return False
