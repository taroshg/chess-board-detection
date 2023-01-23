# installed imports
from torchvision.utils import save_image, draw_bounding_boxes
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import matplotlib.pyplot as plt
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
from trainers import train_piece_detector

SAVE = False # Toggle this to avoid override
device = "cuda" if torch.cuda.is_available() else "cpu"
def main():
    # epochs = 10
    # batch_size = 2
    # lr = 3e-4
    # train_piece_detector(batch_size=batch_size,
    #                      epochs=epochs,
    #                      learning_rate=lr,
    #                      weights_save_folder='./models/checkpoints/piece_detector/0')
    test_piece_detector(img_path='dataloader/data/warped_target/0_target.jpg',
                        weights_path='models/checkpoints/piece_detector/0/weight')


def test_piece_detector(img_path, weights_path):
    piece_detector = PieceDetector().to(device)
    piece_detector.load_state_dict(torch.load(weights_path))
    piece_detector.eval()

    raw_img = read_image(img_path)
    img = raw_img.unsqueeze(0).to(device) / 255.0
    boxes, labels, scores = piece_detector(img)[0].values()

    out = draw_bounding_boxes(raw_img, boxes * 320, width=3)
    plt.imshow(out.permute(1, 2, 0))    
    plt.show()


if __name__ == '__main__':
    main()