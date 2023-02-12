# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# default imports
import json
import os

# original size: 320 x 320
class BoardDetectorDataset(Dataset):
    """
    Note:
        Images have to have a square ratio meaning height = width
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project
        (str) data_folder: folder with all normal chess board images
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, size=(320, 320)) -> None:
        super().__init__()
        assert(json_file != None), "json_file not provided for board detector dataset"
        assert(size[0] == size[1]), "images need to be perfect square ratio"
        self.data_folder, _ = os.path.split(json_file)
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Compose([transforms.Resize(size)])
        self.classes = self.data['categories'] # [a8, h8, h1, a1]
        print("Board Detector Dataset initalized!")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, i):
        img_path = os.path.join(self.data_folder, self.data["images"][i]["file_name"])
        img = self.tr(read_image(img_path) / 255.0) # applies transfromations to img

        # there are only 4 "bounding box" keypoints per image and are in order.
        # therefore, this is more effecient than a loop to find "bounding box" keypoints
        box_i = i * 4
        output = [0, 0, 0, 0, 0, 0, 0, 0] # [a8, h8, h1, a1]
        for k in range(box_i, box_i + 4):
            keypoints = self.data["annotations"][k]["keypoints"] 
            category_id = self.data["annotations"][k]["category_id"]
            if category_id == 1: # if a8
                output[0] = keypoints[0]
                output[1] = keypoints[1]
            if category_id == 2: # if h8
                output[2] = keypoints[0]
                output[3] = keypoints[1]
            if category_id == 3: # if h1
                output[4] = keypoints[0]
                output[5] = keypoints[1]
            if category_id == 4: # if a1
                output[6] = keypoints[0]
                output[7] = keypoints[1]

        output = torch.tensor(output, dtype=torch.float) / self.data["images"][i]["width"]

        return img, output

