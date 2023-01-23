# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_convert
import torch

# default imports
import json

# original size: 320 x 320
class BoardDetectorDataset(Dataset):
    """
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project
        (str) data_folder: folder with all normal chess board images
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, data_folder, size=(320, 320)) -> None:
        super().__init__()
        assert(json_file != None), "json_file not provided for board detector dataset"
        assert(data_folder != None), "data_folder not provided for board detctor dataset"
        self.data_folder = data_folder
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Resize(size)
        print("Board Detector Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        coords = self.data[i]["Label"]["objects"][0]["polygon"]
        img = self.tr(read_image(self.data_folder + f'/{i}.jpg') / 255.0) # applies transfromations to img
        h = self.s[0]
        w = self.s[1]
        # gets the coords and normalizes it to 0 - 1.
        label = torch.tensor([coords[0]['x'] / h, coords[0]['y'] / w, coords[1]['x'] / h, coords[1]['y'] / w,
                             coords[2]['x'] / h, coords[2]['y'] / w, coords[3]['x']/ h, coords[3]['y'] / w]);
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
        assert(json_file != None), "json_file not provided for piece detector dataset"
        assert(data_folder != None), "data_folder not provided for piece detctor dataset"
        self.data_folder = data_folder
        self.data = json.load(open(json_file))
        self.h = size[0]
        self.w = size[1]
        self.tr = transforms.Resize(size)
        self.classification = ['whiteking', 'whitequeen', 'whiterook', 'whitebishop', 'whiteknight', 'whitepawn',
                               'blackking', 'blackqueen', 'blackrook', 'blackbishop', 'blackknight', 'blackpawn']
        print("Piece Detector Dataset initalized!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.tr(read_image(self.data_folder + f'/{i}.jpg') / 255.0) # applies transfromations to img

        boxes = [] # (32, 4) boxes
        labels = [] # (32) labels for each box
        for i, piece in enumerate(self.data[i]["Label"]["objects"]):
            boxes.append(list(piece['bbox'].values()))

            color = piece["classifications"][0]['answer']['value'] # ex: white
            piece_type = piece["classifications"][1]['answer']['value'] # ex: pawn
            # finds index of "whitepawn" and appends
            # +1 is added, because classification of 0, means it does not exist
            labels.append(self.classification.index(color + piece_type) + 1)

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        # convert box xywh format to xyxy
        boxes = box_convert(boxes, 'xywh', 'xyxy')
        # normalizes bbox coords to (0 - 1)
        boxes[:, :2] /= self.h
        boxes[:, 2:] /= self.w

        target = {"boxes": boxes, "labels": labels}
        
        return img, target
