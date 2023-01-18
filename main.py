# installed imports
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch

# default imports
import glob
import json

# local imports
from dataloader import BoardDetectorDataset
from dataloader import PieceDetectorDataset
from utils import generate_warped_board_images
from models import PieceDetector
from models import BoardDetector
from utils import download_data
from utils import warp



SAVE = False # Toggle this to avoid override
device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    # piece_detector = PieceDetector().to(device)
    # B = 2 # batch size
    # n_classes = 32
    # images, boxes = torch.rand(B, 3, 320, 320, device=device), torch.rand(B, 10, 4, device=device) # box: (N, n_boxes, coords) Note: coords have (x1, y1, x2, y2)
    # boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
    # labels = torch.randint(n_classes, (B, 11), device=device) 
    # targets = []
    # for i in range(B):
    #     d = {}
    #     d['boxes'] = boxes[i]
    #     d['labels'] = labels[i]
    #     targets.append(d)
    # out = piece_detector(images, targets)
    # print(out)
    dataloader = DataLoader(PieceDetectorDataset(), batch_size=2, shuffle=True)
    img, box, label = next(iter(dataloader))
    print(box.shape)
    print(label.shape)

if __name__ == '__main__':
    main()