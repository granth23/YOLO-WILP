import requests
import os.path

def weights():

    path = 'models/yolov7.pt'

    check_file = os.path.isfile(path)

    if check_file == True:
        pass
    else:
        URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
        response = requests.get(URL)
        open(path, "wb").write(response.content)

weights()