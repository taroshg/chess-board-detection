from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

import json

# original size: 320 x 320
device = "cuda" if torch.cuda.is_available() else "cpu"
class Bd_Data(Dataset):
    def __init__(self, json_file="data/data.json", size=(320, 320)) -> None:
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Resize(size)
        super().__init__()
        print("Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        coords = self.data[i]["Label"]["objects"][0]["polygon"]
        img = self.tr(read_image(f'data/{i}.jpg') / 255.0).to(device) # applies transfromations to img
        h = self.s[0]
        w = self.s[1]
        # gets the coords and normalizes it to 0 - 1.
        label = torch.tensor([coords[0]['x'] / h, coords[0]['y'] / w, coords[1]['x'] / h, coords[1]['y'] / w,
                             coords[2]['x'] / h, coords[2]['y'] / w, coords[3]['x']/ h, coords[3]['y'] / w]).to(device);
        return img, label