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
    Args:
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project
        (str) data_folder: folder with all normal chess board images
        (tuple) size: resize shape of the images
    """
    def __init__(self, json_file, size=(320, 320)) -> None:
        super().__init__()
        assert(json_file != None), "json_file not provided for board detector dataset"
        self.data_folder, _ = os.path.split(json_file)
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.s = size
        self.tr = transforms.Compose([transforms.Resize(size)])
        self.classes = self.data['categories'] # [a1, a8, h1, h8]
        print("Board Detector Dataset initalized!")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, i):
        img_path = os.path.join(self.data_folder, self.data["images"][i]["file_name"])
        img = self.tr(read_image(img_path) / 255.0) # applies transfromations to img

        # there are only 4 "bounding box" keypoints per image and are in order.
        # therefore, this is more effecient than a loop to find "bounding box" keypoints
        box_i = i * 4
        keypoints = [0, 0, 0, 0] # [a8, h8, h1, a1]
        for k in range(box_i, box_i + 4):
            keypoint = self.data["annotations"][k] 
            if keypoint["category_id"] == 1: # if a1
                keypoints[3] = keypoint["bbox"][:2] # we just need x y points for keypoint
            if keypoint["category_id"] == 2: # if a8
                keypoints[0] = keypoint["bbox"][:2]
            if keypoint["category_id"] == 3: # if h1
                keypoints[2] = keypoint["bbox"][:2]
            if keypoint["category_id"] == 4: # if h8
                keypoints[1] = keypoint["bbox"][:2]

        keypoints = torch.tensor(keypoints, dtype=torch.float).flatten() / self.s[0]

        return img, keypoints
