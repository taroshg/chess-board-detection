from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch
import glob
import json

from models import PieceDetector
from models import BoardDetector
from dataloader import BoardDetectorDataset
from dataloader import PieceDetectorDataset
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


def generate_board_warped_images(data_path="dataloader/data", from_model=False, model_version=5):
    tr = transforms.Resize((320, 320))
    for i in range(len(glob.glob1(data_path, '*.jpg'))):
        img_path = data_path + f"/{i}.jpg"
        if not from_model:
            data = json.load(open(data_path + '/data.json'))
            coords = data[i]["Label"]["objects"][0]["polygon"]
            coords = torch.tensor([[list(coords[0].values()), list(coords[1].values()), list(coords[2].values()),
                                        list(coords[3].values())]]).to(device);
            out = warp(img=tr(read_image(img_path) / 255.0).unsqueeze(0).to(device), coords=coords)
            save_image(out[0], data_path + f"/warped_target/{i}_target.jpg")
        else:
            load_path = f'checkpoint/{model_version}/model'
            board_detector = BoardDetector().to(device)
            board_detector.load_state_dict(torch.load(load_path))
            board_detector.eval()
            inp = tr(read_image(img_path) / 255.0).unsqueeze(0).to(device) # Resizes, normaizes, adds dim at 0, and casts to GPU.
            out = (board_detector(inp)[0] * 320).tolist() 
            save_image(out[0], data_path + f"/warped_pred/{i}_pred.jpg")

if __name__ == '__main__':
    main()