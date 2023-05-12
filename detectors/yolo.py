import torch
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
out = model.predict(source=torch.randn(1, 3, 128, 128))
print(out)
