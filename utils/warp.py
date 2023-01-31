# installed imports
from kornia.geometry.transform.imgwarp import get_perspective_transform, warp_perspective
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision import transforms
import torch
import glob
import json

# local imports
from dataloader import BoardDetectorDataset
from models import BoardDetector

def warp(img : torch.Tensor, coords, rotate=0, device='cpu'):
    """
    takes a section of an image (four coordinates) and warps to fill the entire image with it
    Args:
        (torch.Tensor) img: batch of image tensors (B, C, H, W)
        (float Array) coords: coordinates of the 4 corners
 
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

    return warped_img

def generate_warped_board_images(from_model, load_folder, save_folder, json_file, model_path, device):
    """
    takes all normal board images in a folder and warps them using the warp function (see warp function for more details)

    Args:
        (str) load_folder: folder with normal board images
        (str) save_folder: folder where the warped board images go
        (str) json_file: json file downloaded from labelbox "Chess Board Detection" project (not needed if from_model is True)
        (bool) from_model: if True, prediction of the coordinates of the board image are done by the BoardDetector model
        (str) model_path: path of the BoardDetector model used to detect the coordinates of the board image
 
    """
    tr = transforms.Resize((320, 320))
    for i in range(len(glob.glob1(load_folder, '*.jpg'))):
        img_path = load_folder + f"/{i}.jpg"
        if not from_model:
            assert(json_file != ""), "json_file needs to be defined if from_model is False"
            data = json.load(open(json_file))
            coords = data[i]["Label"]["objects"][0]["polygon"]
            coords = torch.tensor([[list(coords[0].values()), list(coords[1].values()), list(coords[2].values()),
                                        list(coords[3].values())]]).to(device);
            out = warp(img=tr(read_image(img_path) / 255.0).unsqueeze(0).to(device), coords=coords)
            save_image(out[0], save_folder + f"{i}_target.jpg")
        else:
            assert(model_path != ""), "model_path needs to be defined if from_model is True"
            load_path = model_path
            board_detector = BoardDetector().to(device)
            board_detector.load_state_dict(torch.load(load_path))
            board_detector.eval()
            inp = tr(read_image(img_path) / 255.0).unsqueeze(0).to(device) # Resizes, normaizes, adds dim at 0, and casts to GPU.
            out = (board_detector(inp)[0] * 320).tolist() 
            save_image(out[0], save_folder + f"{i}_pred.jpg")