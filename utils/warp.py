import cv2
import numpy as np
import matplotlib.pyplot as plt
import kornia
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def warp(img : torch.Tensor, coords, rotate=0):
    """
    Args:
        (torch.Tensor) img: batch of image tensors (B, C, H, W)
        (float Array) coords: coordinates of the 4 corners
 
    Returns:
        (torch.Tensor) warped batch of image tensors (B, C, H, W)
    """
    _, _, h, w = img.shape

    # new corner coordinates will stretch to fit image window
    new_coords = torch.Tensor([[[0,0], [w, 0], [w, h], [0, h]]]).to(device) # rotate = 0
    if rotate == 90:
        new_coords = torch.Tensor([[[0, h], [0, 0], [w, 0], [w, h]]]).to(device) # rotate = 90
    if rotate == -90:
        new_coords = torch.Tensor([[[w, 0], [w, h], [0, h], [0,0]]]).to(device) # rotate = -90
    if rotate == 180:
        new_coords = torch.Tensor([[[0, 0], [w, 0], [w, h], [0, h]]]).to(device) # rotate = 180

    # compute perspective transform
    M = get_perspective_transform(coords, new_coords)
    # use perspective transform matrix to transform resized image
    warped_img = warp_perspective(img, M, dsize=(h, w))

    return warped_img