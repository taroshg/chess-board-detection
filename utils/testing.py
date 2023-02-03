from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import matplotlib.pyplot as plt
import torch

import json
import os

from models import PieceDetector, BoardDetector
from dataloader import BoardDetectorDataset, PieceDetectorDataset
from utils import warp

def show_piece_detector_results(idx: int, 
                                piece_detector: PieceDetector,
                                piece_data: PieceDetectorDataset,
                                device):
    """
    using draw_bounding_boxes, the function displays results of predicted boxes

    Args:
        (str) img_path: path of the image being tested
        (PieceDetectorDataset object) piece_data: initialized, PieceDetectorDataset
        (PieceDetector object) piece_detector: initialized, PieceDetector model
        
    Returns:
        None
    """
    img, target = piece_data[idx]
    real_boxes, real_labels = target.values()
    real_labels = [piece_data.classes[l]["name"] for l in real_labels]

    boxes, labels, scores = piece_detector(img.unsqueeze(0).to(device))[0].values()
    labels = [piece_data.classes[l]["name"] for l in labels] # convert int labels to string labels (0 -> white pawn)
    
    og_img = (img * 255).to(torch.uint8)
    out = draw_bounding_boxes(og_img, boxes * 320, labels, width=2)
    real_out = draw_bounding_boxes(og_img, real_boxes * 320, real_labels, width=2)
    
    f, ax = plt.subplots(1,2, figsize=(12, 10))
    ax[0].imshow(real_out.permute(1, 2, 0))
    ax[1].imshow(out.permute(1, 2, 0))
    plt.show()

def show_board_detector_results(idx: int, 
                                board_detector: BoardDetector,
                                board_data: BoardDetectorDataset,
                                device):
    """
    Displays two board dectector results, board detector predition and actual target
    Args:
        (int) idx: index of image to diplay
        (BoardDetector object) board_detector: initalized, board detector model
        (BoardDetectorDataset object) board_data: initalized, board dataset

    Returns:
        None
    """

    img, keypoints_actual = board_data[idx]
    keypoints = board_detector(img.unsqueeze(0).to(device))[0]

    criteron = torch.nn.MSELoss()
    loss = criteron(keypoints.to(device), keypoints_actual.to(device))
    print(f'loss on displayed image: {loss}')

    img = img.cpu()
    keypoints_actual = (keypoints_actual * 320).cpu()
    keypoints = (keypoints * 320).detach().cpu()

    f, ax = plt.subplots(2,2, figsize=(12, 10))
    ax[0, 0].imshow(img.permute(1, 2, 0))
    ax[0, 0].plot(keypoints_actual[0::2], keypoints_actual[1::2], '-o')
    ax[0, 1].imshow(img.permute(1, 2, 0))
    ax[0, 1].plot(keypoints[0::2], keypoints[1::2], '-o')

    # converts (8,) keypoints to coordinates (4, 2) 
    # (x, y, x, y, x, y, x, y) => ((x, y), (x, y), (x, y), (x, y))
    coords = list(zip(keypoints_actual.tolist()[0::2], keypoints_actual.tolist()[1::2]))
    warped_actual = warp(img.unsqueeze(0), torch.tensor(coords).unsqueeze(0))
    ax[1, 0].imshow(warped_actual[0].permute(1, 2, 0))

    coords = list(zip(keypoints.tolist()[0::2], keypoints.tolist()[1::2]))
    warped = warp(img.unsqueeze(0), torch.tensor(coords).unsqueeze(0))
    ax[1, 1].imshow(warped[0].permute(1, 2, 0))
    plt.show()
