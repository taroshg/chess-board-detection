# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_convert
import torch

# default imports
import json
import os

class PieceDetectorDataset(Dataset):
    """
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Piece Detection" project
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, size=(320, 320)):
        super().__init__()
        assert(json_file != None), "json_file not provided for piece detector dataset"
        self.data_folder, _ = os.path.split(json_file)
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.h = size[0]
        self.w = size[1]
        self.tr = transforms.Resize(size)
        self.classes = self.data['categories']
        print("Piece Detector Dataset initalized!")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, i):
        img_path = os.path.join(self.data_folder, self.data["images"][i]["file_name"])
        img = self.tr(read_image(img_path) / 255.0) # applies transfromations to img

        # gets all corresponding bounding boxes and labels for image i
        boxes = [] # (32, 4)
        labels = [] # (32)
        for obj in self.data["annotations"]:
            if obj["image_id"] == i:
                boxes.append(obj["bbox"])
                labels.append(obj["category_id"])

        # converts lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)

        # convert box xywh format to xyxy
        boxes = box_convert(boxes, 'xywh', 'xyxy')
        # normalizes bbox coords to (0 - 1)
        boxes[:, :2] /= self.h
        boxes[:, 2:] /= self.w

        target = {"boxes": boxes, "labels": labels}
        
        return img, target
