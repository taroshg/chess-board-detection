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
    def __init__(self, json_file, size=(1000, 1000)):
        super().__init__()
        assert(json_file is not None), "json_file not provided for piece detector dataset"
        self.data_folder, _ = os.path.split(json_file)
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.h = size[0]
        self.w = size[1]
        self.tr = transforms.Resize(size)
        self.classes = self.data['categories']
        print("Piece Detector Dataset initialized!")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, i):
        working_img = self.data["images"][i]
        img_path = os.path.join(self.data_folder, working_img["file_name"])
        img = self.tr(read_image(img_path) / 255.0) # applies transformations to img

        # gets all corresponding bounding boxes and labels for image i
        boxes = []  # (32, 4)
        labels = []  # (32)
        areas = []
        for obj in self.data["annotations"]:
            if obj["image_id"] == working_img["id"]:
                boxes.append(obj["bbox"])
                labels.append(obj["category_id"])
                areas.append(obj["area"])
        # convert lists to tensors
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        areas = torch.tensor(areas, dtype=torch.float)
        # convert box xywh format to xyxy
        boxes = box_convert(boxes, 'xywh', 'xyxy')

        target = {"boxes": boxes, "labels": labels, "areas": areas}
        
        return img, target
