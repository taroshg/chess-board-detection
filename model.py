import torch
from torch import nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms import transforms

import numpy as np

torch.manual_seed(2)

def showImg(img, save=False, name='out'):
    if save:    
        plt.imshow(img.permute(1, 2, 0))
        plt.savefig(name)
        plt.show()
    else:
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

# model_boxes = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()

resize = transforms.Resize(size=[256, 256])

img = resize(read_image('chess.jpeg')) / 255.0
out = model(img.unsqueeze(0))[0]

disp_img = draw_bounding_boxes(image=(img*255).type(torch.uint8), boxes=out['boxes'], width=1)
showImg(disp_img, save=True, name=f'out')

num_masks = out['masks'].shape[0]
img_size = img.shape[-1]
mask_threshold = 0.6 # higher: more img, lower: less img
for i in range(num_masks):
    # if mask is bigger than 30% of img then its a board
    if torch.sum(out['masks'][i]) / (img_size ** 2) > 0.3:
        new_masked_out = img * (out['masks'][i] > mask_threshold)
        disp_img = (new_masked_out * 255).type(torch.uint8)
        showImg(disp_img, save=True, name=f'masks/mask_{i}')
