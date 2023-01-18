# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# default imports
import json

# original size: 320 x 320
device = "cuda" if torch.cuda.is_available() else "cpu"
class BoardDetectorDataset(Dataset):
    """
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project
        (str) data_folder: folder with all normal chess board images
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, data_folder, size=(320, 320)) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Resize(size)
        print("Board Detector Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        coords = self.data[i]["Label"]["objects"][0]["polygon"]
        img = self.tr(read_image(self.data_folder + f'/{i}.jpg') / 255.0).to(device) # applies transfromations to img
        h = self.s[0]
        w = self.s[1]
        # gets the coords and normalizes it to 0 - 1.
        label = torch.tensor([coords[0]['x'] / h, coords[0]['y'] / w, coords[1]['x'] / h, coords[1]['y'] / w,
                             coords[2]['x'] / h, coords[2]['y'] / w, coords[3]['x']/ h, coords[3]['y'] / w]).to(device);
        return img, label

class PieceDetectorDataset(Dataset):
    """
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Piece Detection" project
        (str) data_folder: folder with all warped chess board images
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, data_folder, size=(320, 320)):
        super().__init__()
        self.data_folder = data_folder
        self.data = json.load(open(json_file))
        self.h = size[0]
        self.w = size[1]
        self.tr = transforms.Resize(size)
        self.max_objects = 32 # total of 32 pieces at anygiven position
        self.classification = ['whiteking', 'whitequeen', 'whiterook', 'whitebishop', 'whiteknight', 'whitepawn',
                               'blackking', 'blackqueen', 'blackrook', 'blackbishop', 'blackknight', 'blackpawn']
        print("Piece Detector Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.tr(read_image(self.data_folder + f'/{i}.jpg') / 255.0).to(device) # applies transfromations to img

        boxes = torch.zeros(self.max_objects, 4, device=device)
        labels = torch.zeros(self.max_objects, device=device)
        for i, piece in enumerate(self.data[i]["Label"]["objects"]):
            box = torch.tensor(list(piece['bbox'].values()), dtype=torch.float)

            # normalizes bbox coords to (0 - 1)
            box[0:2] /= self.h
            box[2:4] /= self.w

            box[2:4] = box[0:2] + box[2:4] # (add height to x1 --> x2) and (add width to y1 --> y2)

            boxes[i, :] = box

            color = piece["classifications"][0]['answer']['value'] # ex: white
            piece_type = piece["classifications"][1]['answer']['value'] # ex: pawn

            # finds index of "whitepawn" and appends
            # +1 is added, because classification of 0, means it does not exist
            labels[i] = torch.tensor(self.classification.index(color + piece_type) + 1)
        return img, boxes, labels
