import wget
import os
import torch

if os.path.isfile("yolov7.pt") == False:
    wget.download("https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

import eventlet
import socketio
import base64
from blur import blur_check
from detect import analyze

sio = socketio.Server(cors_allowed_origins='*')
app = socketio.WSGIApp(sio)

MOCK_AI = os.getenv("MOCK_AI") == "true"

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def webcam(sid, data):
    if MOCK_AI:
        return {
            "blurry": False,
            "detected": [{
                "type": "person",
                "coords": [0,0,0,0],
                "conf": 1
            }]
        }

    output = {
        "blurry": blur_check(data),
        "detected": analyze(str(data))
    }
    return output

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 5000)), app)
