import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from dataloader import BoardDetectorDataset, points_to_mask

img_size = 128

MODEL = 'squeezenet'
json_file = 'dataloader/data/board_detector_coco.json'
root_folder = 'dataloader/data/raw/'
PRETRAINED = False

board_data = BoardDetectorDataset(root_folder, json_file, size=img_size, target="mask")

img, mask = board_data[2]

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn as maskrcnn

model = maskrcnn(weights=MaskRCNN_ResNet50_FPN_Weights)
model.eval()
out = model(img.unsqueeze(0))
masks = out[0]['masks'].squeeze(1)
n_masks = masks.shape[0]

fig, ax = plt.subplots(1, n_masks)
for i in range(n_masks):
    if n_masks != 1:
        ax[i].imshow(masks[i].detach().cpu())
    else:
        ax.imshow(masks[i].detach().cpu())
plt.show()

# start = time.time()
# img = cv2.imread('dataloader/data/raw/IMG_0576.jpg')
# img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# def get_edges(img):
#     kernel_size = 5
#     blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
#     edges = cv2.Canny(blur_gray, 50, 150, apertureSize=3)
#     return edges
# edges = get_edges(img)
# mask = np.zeros((img_size + 2, img_size + 2), np.uint8)
# cv2.floodFill(edges, mask, (0, 0), 123)
#
# bg = np.zeros_like(edges)
# bg[edges == 123] = 255
#
# bg = cv2.blur(bg, (3, 3))
# edges = cv2.Canny(bg, 50, 150, apertureSize=3)
# print("--- %s seconds ---" % (time.time() - start))
#
# fig, ax = plt.subplots(1, 1)
# ax.imshow(edges)
# plt.show()

# img, _ = board_data[5]
# layers = []
# for i, c in enumerate(board_detector.model.modules()):
#     print(c)
#     if i == 4:
#         break
# quit()
# model = torch.nn.Sequential(*layers)
# print(model)
# out = model(img.unsqueeze(0).to(device))
# fig, ax = plt.subplots(5, 5)
# count = 0
# for i in range(5):
#     for j in range(5):
#         ax[i][j].imshow(out[0][count].detach().cpu())
#         count += 1
# plt.show()