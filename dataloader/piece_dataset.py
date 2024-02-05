# installed imports
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from PIL import Image
import torch

# default imports
from typing import Any, List
import json
import os
import glob
import sqlite3
import bisect

class PieceDetectorDatasetDB(Dataset):
    """
    reads .db files
    Args:
        (str) root: directory with all the images
        (str) db_file: sql database (.db) file location
        (tuple) size: resize shape of the images
    """
    def __init__(self, root, db_file, size=(320, 320)):
        super().__init__()
        assert(db_file is not None), "db_file not provided for piece detector dataset"
        self.root = root
        self.w = size[0]
        self.h = size[1]
        self.tr = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

        db_conn = sqlite3.connect(db_file)
        db_cur = db_conn.cursor()
        db_cur.row_factory = lambda cursor, row: (row[1], row[2], json.loads(row[3]))
        self.anns = db_cur.execute("SELECT * FROM piece_annotations").fetchall()
        db_cur.row_factory = None
        self.imgs_data = db_cur.execute("SELECT * FROM images").fetchall()
        db_conn.close()
        
        print("Piece Detector Dataset initialized!")

    def __len__(self):
        return len(self.imgs_data)
    
    def filter_annotations(self, image_id):
        # Find the index of the first occurrence of target_id
        start_index = bisect.bisect_left(self.anns, (image_id,))

        rel_anns = []

        # Iterate from the start_index while the id is equal to target_id
        for i in range(start_index, len(self.anns)):
            if self.anns[i][0] == image_id:
                rel_anns.append(self.anns[i])
            else:
                break

        return rel_anns

    def __getitem__(self, index: int):

        rel_anns = self.filter_annotations(index)
        rel_img = self.imgs_data[index - 1]
        img = self.tr(Image.open(os.path.join(self.root, rel_img[1])).convert("RGB"))

        if len(rel_anns) == 0:
            return img, {'boxes': torch.tensor([[[],[],[],[]]], dtype=torch.float), 'labels': torch.tensor([], dtype=torch.int64)}
        
        boxes = []
        labels = []

        for ann in rel_anns:
            boxes.append(ann[-1])
            labels.append(ann[-2])
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes[:, 0::2] *= (320 / rel_img[-1])
        boxes[:, 1::2] *= (320 / rel_img[-2])
        boxes = box_convert(boxes, 'xywh', 'xyxy')

        target = {'boxes': boxes, 'labels': labels}

        return img, target

class PieceDetectorDataset(Dataset):
    """
    Args:
        (str) root: directory with all the images
        (str) json_file: coco json file with boxes information
        (tuple) size: resize shape of the images
    """
    def __init__(self, root, json_file, size=(320, 320)):
        super().__init__()
        assert(json_file is not None), "json_file not provided for piece detector dataset"
        self.data_folder = root
        self.json_file = json_file
        self.data = json.load(open(json_file))
        self.w = size[0]
        self.h = size[1]
        self.tr = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.classes = self.data['categories']
        self.coco = CocoDetection(root, json_file)
        print("Piece Detector Dataset initialized!")

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, index: int):
        img, target = self.coco[index]
        h, w = img.size
        img = self.tr(img)
        if len(target) == 0:
            return img, {'boxes': torch.tensor([[[],[],[],[]]], dtype=torch.float), 'labels': torch.tensor([], dtype=torch.int64)}

        boxes = torch.tensor([t['bbox'] for t in target], dtype=torch.float)
        labels = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)

        boxes[:, 0::2] *= (self.h / h)
        boxes[:, 1::2] *= (self.w / w)

        boxes = box_convert(boxes, 'xywh', 'xyxy')

        target = {'boxes': boxes, 'labels': labels}

        return img, target


class PieceDetectorCOGDataset(Dataset):
    """
    Args:
        (str) root: directory with all the images
        (str) json_file: coco json file with boxes information
        (tuple) size: resize shape of the images
    """
    def __init__(self, root, size=(320, 320)):
        super().__init__()
        self.root = root
        self.image_files = glob.glob1(root, '*.png')
        json_files = glob.glob1(root, '*.json')
        self.all_data = [json.load(open(os.path.join(root, j))) for j in json_files]
        self.h = size[1]
        self.w = size[0]
        self.tr = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
        self.classes = [{'name': 'none'}, {'name': 'P'}, {'name': 'N'}, {'name': 'B'}, {'name': 'R'},
                        {'name': 'Q'}, {'name': 'K'}, {'name': 'p'}, {'name': 'n'}, {'name': 'b'},
                        {'name': 'r'}, {'name': 'q'}, {'name': 'k'}]
        self.labels_dict = {'none': 0, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                            'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
        print("Piece COG Detector Dataset initialized!")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int):
        img = Image.open(os.path.join(self.root, self.image_files[index])).convert('RGB')
        data = self.all_data[index]
        h, w = img.size
        img = self.tr(img)

        labels = torch.tensor([self.labels_dict[d['piece']] for d in data['pieces']], dtype=torch.int64)
        boxes = torch.tensor([d['box'] for d in data['pieces']], dtype=torch.float)

        boxes[:, 0::2] *= (self.h / h)
        boxes[:, 1::2] *= (self.w / w)

        boxes = box_convert(boxes, 'xywh', 'xyxy')

        target = {'boxes': boxes, 'labels': labels}

        return img, target

