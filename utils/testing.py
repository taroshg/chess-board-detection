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

def show_piece_detector_results(idx, weights_path, json_file, device):
    """
    using draw_bounding_boxes, the function displays results of predicted boxes

    Args:
        (str) img_path: path of the image being tested
        (str) weights_path: path of the weights for the piece detector model
        (str) json_file: Labelbox json file with the annotations

    Returns:
        None
    """
    piece_detector = PieceDetector().to(device)
    piece_detector.load_state_dict(torch.load(weights_path))
    piece_detector.eval()

    data = json.load(open(json_file))
    data_folder, _ = os.path.split(json_file)

    classes = data['categories']
    raw_img = read_image(os.path.join(data_folder, data["images"][idx]["file_name"]))
    img = raw_img.unsqueeze(0).to(device) / 255.0
    boxes, labels, scores = piece_detector(img)[0].values()
    labels = [classes[label]["name"] for label in labels]

    objects = [obj for obj in data["annotations"] if obj.get('image_id') == idx]
    real_boxes = [obj["bbox"] for obj in objects]
    real_labels = [classes[obj["category_id"]]["name"] for obj in objects]

    real_boxes = torch.tensor(real_boxes, dtype=torch.float)
    real_boxes = box_convert(real_boxes, 'xywh', 'xyxy')

    out = draw_bounding_boxes(raw_img, boxes * 320, labels, width=2)
    real_out = draw_bounding_boxes(raw_img, real_boxes, real_labels, width=2)
    
    f, ax = plt.subplots(1,2)
    ax[0].imshow(out.permute(1, 2, 0))
    ax[1].imshow(real_out.permute(1, 2, 0))
    plt.show()

def show_board_detector_results(idx: int, 
                                board_detector: object,
                                board_data: object,
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
    img = img.cpu()
    keypoints_actual = (keypoints_actual * 320).cpu()

    keypoints = board_detector(img.unsqueeze(0).to(device))
    keypoints = (keypoints[0] * 320).detach().cpu()

    f, ax = plt.subplots(2,2)
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
