import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# yolo_path = "models/best_v2.pt"
# npwp_path = "models/best (6).pt"
# proto_txt = "models/deploy.prototxt"
# model_face_detection = "models/res10_300x300_ssd_iter_140000.caffemodel"
# net = cv2.dnn.readNetFromCaffe(proto_txt,model_face_detection)
# model = torch.hub.load("yolov5","custom", path=yolo_path,source="local", force_reload=True).eval()
# model.eval()

# model_npwp = torch.hub.load("yolov5","custom", path=npwp_path,source="local", force_reload=True).eval()
# model_npwp.eval()
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp15/weights/last.pt', force_reload=True)

# python E:\Computer vision lab\KTP\npwp\yolov5\detect.py --weights models/best (6).pt --img 640 --conf 0.2 --source 1

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()