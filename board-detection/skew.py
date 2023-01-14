import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.io import read_image
from model import Bd_Model # Bd is short for board detection

import json

def skew(img_path, save_path, coords, save=True, show=False, rotate=0):
    # Zach's Code!!
    """
    Args 
        (String) img_path: path of the image
        (String) save_path: path of the returned image
        (float Array) coords: 0 - 1, normalized coords 
        (bool) save: saves to file
        (bool) show: displays the skewed image
 
    Returns:
        cv2 image of a board that is cropped and skewed to perfect square
    """

    # read in image and resize
    img = cv2.imread(img_path)

    # corner coordinates (will get from outermost coordinates of mask r-cnn board segmentation)
    coords = np.float32(coords)

    # get resized image height and width
    new_height, new_width, _ = img.shape

    # new corner coordinates will stretch to fit image window
    new_coords = np.float32([[0,0], [new_width, 0], [new_width, new_height], [0, new_height]]) # rotate = 0
    if rotate == 90:
        new_coords = np.float32([[0, new_height], [0, 0], [new_width, 0], [new_width, new_height]]) # rotate = 90
    if rotate == -90:
        new_coords = np.float32([[new_width, 0], [new_width, new_height], [0, new_height], [0,0]]) # rotate = -90
    if rotate == 180:
        new_coords = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]]) # rotate = 180

    # use coordinates to compute perspective transform matrix
    ptm = cv2.getPerspectiveTransform(coords, new_coords)

    # use perspective transform matrix to transform resized image
    trans_img = cv2.warpPerspective(img, ptm, (new_width, new_height))

    if save:
        cv2.imwrite(save_path, trans_img)
    if show:
        plt.imshow(trans_img)
        plt.axis('off')
        plt.show()
        print("done")

    return trans_img


VERSION = 5
LOAD_PATH = f'./checkpoint/{VERSION}/model'
device = "cuda" if torch.cuda.is_available() else "cpu"
board_detector = Bd_Model().to(device)
board_detector.load_state_dict(torch.load(LOAD_PATH))
board_detector.eval()
tr = transforms.Resize((320, 320))

for i in range(11):
    IMG_PATH = f"./data/{i}.jpg"
    inp = tr(read_image(IMG_PATH) / 255.0).unsqueeze(0).to(device) # Resizes, normaizes, adds dim at 0, and casts to GPU.
    out = (board_detector(inp)[0] * 320).tolist()   
    coords = np.float32([[out[0],out[1]], [out[2],out[3]], [out[4],out[5]], [out[6],out[7]]])
    skew(img_path=f"./data/{i}.jpg", save_path=f"skewed_out/{i}.jpg", coords=coords, rotate=0)