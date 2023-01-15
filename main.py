from asyncore import read
import torch
import torch.nn as nn
from models import BoardDetector # Bd is short for board detection
from dataloader import BoardDetectorDataset # Bd (Board Detection)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image
from utils import download_data
from utils import warp

import numpy as np

SAVE = False # Toggle this to avoid override
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("running...")
    generate_board_focused_images()

# def run_model():
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     dataset = Bd_Data()
#     dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#     model = Bd_Model().to(device)
#     VERSION = 3
#     LOAD_PATH = f'./checkpoint/{VERSION}/model'
#     model.load_state_dict(torch.load(LOAD_PATH))
#     model.eval()

#     for x, y in dataloader:
#         out = model(x)

#         pred_coords = (out[0] * 320).tolist()
#         real_coords = (y[0] * 320).tolist()

#         print(pred_coords)
#         print(real_coords)
        
#         plt.imshow(x[0].cpu().permute(1, 2, 0))
#         plt.plot(pred_coords[::2]+[pred_coords[0]], pred_coords[1::2]+[pred_coords[1]], '.r-')
#         plt.plot(real_coords[::2]+[real_coords[0]], real_coords[1::2]+[real_coords[1]], '.g-', alpha=0.5)
#         if SAVE:
#             plt.savefig(f'./checkpoint/{VERSION}/out.png')
#         plt.show()
#         quit()

from torchvision.utils import save_image

def generate_board_focused_images():
    VERSION = 5
    LOAD_PATH = f'./checkpoint/{VERSION}/model'
    board_detector = BoardDetector().to(device)
    board_detector.load_state_dict(torch.load(LOAD_PATH))
    board_detector.eval()
    tr = transforms.Resize((320, 320))

    i = 0

    IMG_PATH = f"dataloader/data/{i}.jpg"
    inp = tr(read_image(IMG_PATH) / 255.0).unsqueeze(0).to(device) # Resizes, normaizes, adds dim at 0, and casts to GPU.
    out = (board_detector(inp)[0] * 320).tolist()   
    coords = torch.Tensor([[[out[0],out[1]], [out[2],out[3]], [out[4],out[5]], [out[6], out[7]]]]).to(device)
    # skew(img_path=f"./data/{i}.jpg", save_path=f"skewed_out/{i}.jpg", coords=coords, rotate=0)
    out = warp(img=inp, coords=coords)
    save_image(out[0], f"dataloader/data/skewed_out/{i}_test.jpg")
    

    # for i in range(11):
    #     IMG_PATH = f"./data/{i}.jpg"
    #     inp = tr(read_image(IMG_PATH) / 255.0).unsqueeze(0).to(device) # Resizes, normaizes, adds dim at 0, and casts to GPU.
    #     out = (board_detector(inp)[0] * 320).tolist()   
    #     coords = np.float32([[out[0],out[1]], [out[2],out[3]], [out[4],out[5]], [out[6],out[7]]])
    #     skew(img_path=f"./data/{i}.jpg", save_path=f"skewed_out/{i}.jpg", coords=coords, rotate=0)

if __name__ == '__main__':
    main()