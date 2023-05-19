# installed imports
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective, transform_points
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision import transforms
import torch
from tqdm.auto import tqdm

from pathlib import Path
from glob import glob
import json

# local imports
from dataloader import BoardDetectorDataset
from models import BoardDetector


def warp(img : torch.Tensor, coords, rotate=0, device='cpu'):
    """
    takes a section of an image (four coordinates) and warps to fill the entire image with it
    Args:
        (torch.Tensor) img: batch of image tensors (B, C, H, W)
        (torch.Tensor) coords: coordinates of the 4 corners torch.Size([4, 2])
 
    Returns:
        (torch.Tensor) warped batch of image tensors (B, C, H, W)
    """
    _, _, h, w = img.shape

    # new corner coordinates will stretch to fit image window
    new_coords = torch.Tensor([[[0,0], [w, 0], [w, h], [0, h]]]).to(device) # rotate = 0
    if rotate == 90:
        new_coords = torch.Tensor([[[0, h], [0, 0], [w, 0], [w, h]]]).to(device) # rotate = 90
    if rotate == -90:
        new_coords = torch.Tensor([[[w, 0], [w, h], [0, h], [0,0]]]).to(device) # rotate = -90
    if rotate == 180:
        new_coords = torch.Tensor([[[0, 0], [w, 0], [w, h], [0, h]]]).to(device) # rotate = 180

    # compute perspective transform
    M = get_perspective_transform(coords, new_coords)
    # use perspective transform matrix to transform resized image
    warped_img = warp_perspective(img, M, dsize=(h, w))

    return warped_img, M


def warp_points(points, coords, img_size=(320, 320), device='cpu'):
    """

    Args:
        points (torch.Tensor): point locations for every piece (batch, n_points, 2)
        coords (torch.Tensor): (1, 8) corner points of the board (x1, y1, x2, y2, ...)
        img_size (tuple): height and width of image
        device: device

    Returns:
        new points that are warped to img_size

    """
    w, h = img_size
    new_coords = torch.Tensor([[[0, 0], [w, 0], [w, h], [0, h]]]).to(device)
    M = get_perspective_transform(reshape_coords(coords), new_coords)

    return transform_points(M, points)


def generate_warped_board_images(load_folder: str, save_folder: str, size: tuple,
                                 board_detector: BoardDetector = None,
                                 board_data: BoardDetectorDataset = None, 
                                 device='cpu'):
    """
    takes all normal board images in a folder and warps them using the warp function (see warp function for more details)

    Args:
        (str) load_folder: folder with normal board images
        (str) save_folder: folder where the warped board images go
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project (not needed if from_model is True)
        (bool) from_model: if True, prediction of the coordinates of the board image are done by the BoardDetector model
        (str) model_path: path of the BoardDetector model used to detect the coordinates of the board image
 
    """
    tr = transforms.Resize((320, 320)) # for board detector
    out_tr = transforms.Resize(size)
    raw_image_paths = glob(f'{load_folder}/*jpg')
    for i in tqdm(range(len(raw_image_paths))):
        img_path = raw_image_paths[i]
        filename = Path(img_path).stem
        if board_detector is None:
            assert(board_data != None), "no board_data"
            img, coords = board_data[i]
            raw_inp = (read_image(img_path) / 255.0).unsqueeze(0).to(device)
            coords = reshape_coords(coords * size[0]).to(device)
            out = warp(img=out_tr(raw_inp), coords=coords, device=device)
            save_image(out[0], save_folder + f"/{filename}_warped.jpg")
        else:
            assert(board_detector != None), "no board_detector"
            raw_inp = (read_image(img_path) / 255.0).unsqueeze(0).to(device)
            inp = tr(raw_inp)
            coords = reshape_coords(board_detector(inp)[0] * size[0])
            out = warp(img=out_tr(raw_inp), coords=coords, device=device) 
            save_image(out[0], save_folder + f"/{filename}_warped_pred.jpg")


def reshape_coords(out: torch.Tensor) -> torch.Tensor:
    """
    reshapes output of (8,) from BoardDetector to coords of torch.Size([4, 2]), for the warp function
    """
    assert(out.shape == torch.Size([1, 8]) or out.shape == torch.Size([8,])), f"incorrect input size expected ({torch.Size[1, 8]}, got {out.shape}"
    return out.reshape((1, 4, 2))
