# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image, ImageDraw
import torch

# default imports
import json
import os


def points_to_mask(points: torch.Tensor, img_size: tuple) -> torch.Tensor:
    """
    Args:
        points: list of points [x1, y1, x2, y2, ...]
        img_size: tuple of (width, height)

    Returns:
        Mask of the points in the image

    thanks https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
    """
    mask = Image.new('L', img_size, 0)
    ImageDraw.Draw(mask).polygon(points.tolist(), outline=1, fill=1)

    return transforms.functional.pil_to_tensor(mask)


class BoardDetectorDataset(Dataset):
    def __init__(self, root, json_file, size: int = 320, target: str ='points') -> None:
        """

        Args:
            json_file: coco json file with keypoints
            size:
            target: "points" or "mask"
        """
        super().__init__()
        assert(json_file is not None), "json_file not provided for board detector dataset"
        self.data_folder = root
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.size = size
        self.target = target
        self.tr = transforms.Compose([transforms.Resize(size)])
        self.classes = self.data['categories'] # [a8, h8, h1, a1]
        print("Board Detector Dataset initalized!")

    def __len__(self):
        # 4 coords per image so, total_boxes / 4 is len(images_annotated)
        return len(self.data["annotations"]) // 4

    def __getitem__(self, i):
        img_path = os.path.join(self.data_folder, self.data["images"][i]["file_name"])
        img = self.tr(read_image(img_path) / 255.0) # applies transfromations to img

        # there are only 4 "bounding box" keypoints per image and are in order.
        # therefore, this is more efficient than a loop to find "bounding box" keypoints
        box_i = i * 4
        points = [0, 0, 0, 0, 0, 0, 0, 0] # [a8, h8, h1, a1]
        for k in range(box_i, box_i + 4):
            keypoints = self.data["annotations"][k]["keypoints"] 
            category_id = self.data["annotations"][k]["category_id"]
            if category_id == 1: # if a8
                points[0] = keypoints[0]
                points[1] = keypoints[1]
            if category_id == 2: # if h8
                points[2] = keypoints[0]
                points[3] = keypoints[1]
            if category_id == 3: # if h1
                points[4] = keypoints[0]
                points[5] = keypoints[1]
            if category_id == 4: # if a1
                points[6] = keypoints[0]
                points[7] = keypoints[1]
        points = torch.tensor(points, dtype=torch.float) / self.data["images"][i]["width"]

        target = points_to_mask(points * self.size, (self.size, self.size)) if self.target == "mask" else points

        return img, target


