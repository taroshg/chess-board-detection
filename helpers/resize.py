from tqdm.auto import tqdm
from glob import glob
from PIL import Image
import json
import os

def resize_dir(load_folder: str, save_folder: str, size: tuple):
    """
        resizes images in a folder
    """
    image_paths = glob(f'{load_folder}/*jpg')
    
    for image in tqdm(image_paths):
        filename = os.path.basename(image)
        im = Image.open(image)
        im = im.resize(size, Image.ANTIALIAS)
        im.save(f'{save_folder}/{filename}', 'JPEG', quality=90)

def resize_coco_bbox_annotations(coco_json: str, from_size: tuple, size: tuple):
    """
        resizes bounding boxes coco json annotations
    """
    assert(from_size[0] == from_size[1]), "images have to have square ratio"
    assert(size[0] == size[1]), "images have to have square ratio"

    data = json.load(open(coco_json))

    for annotation in tqdm(data['annotations']):
        annotation['bbox'] = [int((val * size[0]) / from_size[0]) for val in annotation['bbox']]

    path = os.path.splitext(coco_json)[0] + f'_{size[0]}.json' # takes path and appends _{size}.json to differenciate files
    json.dump(data,open(path, 'w'))
