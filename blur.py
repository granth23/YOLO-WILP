import cv2

def blur_check(IMAGE_FILE):

    img = cv2.imread(IMAGE_FILE)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    var = cv2.Laplacian(grey, cv2.CV_64F).var()

    if var < 20:
        return True
    else:
        return False
