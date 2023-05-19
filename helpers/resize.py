from tqdm.auto import tqdm
from glob import glob
from PIL import Image
import json
import os


def resize_img(from_path: str, size: tuple, to_path: str = None):
    """

    Args:
        from_path: path of image being resized
        to_path: path of the new resized images (optional), if None, image saves in from_path
        size: new size of image

    Returns:
        None

    """
    im = Image.open(from_path)
    im = im.resize(size, Image.ANTIALIAS)
    if to_path is None:
        im.save(from_path, 'JPEG', quality=90) # rewrites the same file
    else:
        im.save(to_path, 'JPEG', quality=90)


def resize_dir(load_folder: str, save_folder: str, size: tuple):
    """
        resizes images in a folder
    """
    image_paths = glob(f'{load_folder}/*jpg')
    
    for image_path in tqdm(image_paths):
        filename = os.path.basename(image_path)
        save_path = f'{save_folder}/{filename}'
        resize_img(image_path, size, save_path)


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

def resize_coco_keypoint_annotations(coco_json: str, from_size: tuple, size: tuple):
    """
        resizes bounding boxes coco json annotations
    """
    assert(from_size[0] == from_size[1]), "images have to have square ratio"
    assert(size[0] == size[1]), "images have to have square ratio"

    data = json.load(open(coco_json))

    for annotation in tqdm(data['annotations']):
        annotation['keypoints'][0] *= (size[0] / from_size[0])
        annotation['keypoints'][1] *= (size[1] / from_size[1])

    path = os.path.splitext(coco_json)[0] + f'_{size[0]}.json' # takes path and appends _{size}.json to differenciate files
    json.dump(data,open(path, 'w'))


