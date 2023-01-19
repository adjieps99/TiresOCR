import torch
import cv2

# Model
yolo_path = "models/best_v2.pt"
proto_txt = "models/deploy.prototxt"
model_face_detection = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_txt,model_face_detection)
model = torch.hub.load("yolov5","custom", path=yolo_path,source="local").eval()

# Images
img = 'E:\Computer vision lab\KTP\ktp-face-detection\kekw.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
crops = results.crop(save=True)  # or .show(), .save(), .print(), .pandas(), etc.