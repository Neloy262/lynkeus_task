import numpy as np
import cv2
from ultralytics import YOLO
import requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from bbox import draw_bbox
import datetime
from vidgear.gears import CamGear


names= {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

def save_image(img,classes):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    classes = [names[cls] for cls in classes]
    img = Image.fromarray(img)
    img_name = '_'.join(classes)
    ct = datetime.datetime.now()
    ct.isoformat()
    img_name+="+"+ct.isoformat()+".jpeg"
    img.save("./saved_images/"+img_name)
    

def read_stream(rtsp):
 
    cap = CamGear(source=rtsp).start() 
    
    with ThreadPoolExecutor(12) as pool:
    

        while True:

            frame = cap.read()
         
            if frame is not None:
                
                img_str = cv2.imencode('.jpg', frame)[1].tobytes()
                headers = {'Content-type': 'image/jpeg'}
                try:
                    response = requests.post("http://127.0.0.1:8000/", data=img_str, headers=headers)
                    val_obj = response.json()
                    bboxes = val_obj['bbox']
                    classes = val_obj['class']
                    
                    print(classes)
                    processed_frame = draw_bbox(frame,bboxes,classes,names)

                    if len(classes)>0:
                        pool.submit(save_image,processed_frame,classes)

                
                    cv2.imshow("Frame",processed_frame)
                except:
                    cv2.imshow("Frame",frame)
            
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    
    cv2.destroyAllWindows()
    cap.stop()




