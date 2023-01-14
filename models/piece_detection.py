import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn as fasterrcnn
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt

model = fasterrcnn(pretrained=True)
model.eval()
img = [read_image('skewed.jpg') / 255.0]
out = model(img)[0]

boxes = out['boxes']
print(boxes)
img_with_boxes = draw_bounding_boxes((img[0] * 255).to(torch.uint8), boxes, width=2)
plt.imshow(img_with_boxes.permute(1, 2, 0)) 
plt.show()