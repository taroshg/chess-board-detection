# installed imports
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn as nn
import torch

# default imports
import math

# local imports
from models import BoardDetector
from models import PieceDetector
from dataloader import BoardDetectorDataset
from dataloader import PieceDetectorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_board_detector(weights_load_path=None, weights_save_folder=None, weights_name="weight",
                         batch_size=64, learning_rate=3e-4, epochs=30):

    """
    This function trains the Board Detector
    Args: 
        (str) weights_load_path: path to the Board Detector model weights that needs to be loaded
        (str) weights_save_folder: folder where the Board Detector weights will be saved
        (int) batch_size: size of each batch trained at once
        (float) learning_rate: lr of the model
        (int) epochs: # of iterations through dataset

        in function args:
        (str) loss_save_path: path where losses plot fig (losses.jpg) will be stored

    Returns:
        None, displays losses at the end
    """

    weights_save_path = weights_save_folder + f"/{weights_name}"
    loss_save_path = weights_save_folder + "/losses.jpg"

    board_detector = BoardDetector().to(device)
    if weights_load_path != None:
        board_detector.load_state_dict(torch.load(weights_load_path)) 
        print('loaded weights for board detector!')
    else:
        print('no weights are loaded for board detector')

    if weights_save_folder != None:
        print('weights will be saved at' + weights_save_path + ' for board detector')
    else:
        print('no weights will be saved for board detector')

    optim = torch.optim.Adam(board_detector.parameters(), lr=learning_rate) 
    board_detector_dataset = BoardDetectorDataset()

    dataloader = DataLoader(board_detector_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()

    losses = []
    lowest_loss = math.inf
    for epoch in tqdm(range(epochs)):
        for data in dataloader:
            x, y = data

            optim.zero_grad()

            out = board_detector(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()

            losses += [loss.item()]
            if weights_save_folder != None and loss.item() < lowest_loss:
                lowest_loss = loss.item()
                torch.save(board_detector.state_dict(), weights_save_path)

    plt.plot(losses)
    plt.show()
    plt.savefig(loss_save_path)

def train_piece_detector(weights_load_path=None, weights_save_folder=None, weights_name="weight",
                         batch_size=2, learning_rate=3e-4, epochs=10):

    piece_detector = PieceDetector().to(device)

    weights_save_path = weights_save_folder + f"/{weights_name}"
    loss_save_path = weights_save_folder + "/losses.jpg"

    if weights_save_folder != None:
        weights_save_path = weights_save_folder + f"/{weights_name}"

    if weights_load_path != None:
        piece_detector.load_state_dict(torch.load(weights_load_path))
        print('loaded weights for piece detector!')
    else:
        print('no weights are loaded for piece detector')

    if weights_save_path != None:
        print('weights will be saved at' + weights_save_path + ' for piece detector')
    else:
        print('no weights will be saved for piece detector')

    optim = torch.optim.Adam(piece_detector.parameters(), lr=learning_rate)
    dataset = PieceDetectorDataset(data_folder='dataloader/piece_detector_data',
                                   json_file='dataloader/piece_detector_data/data.json')
    
    # the collate_fn function is needed to create batches without using default stack mehtod,
    # the default collate_fn requires all boxes to be of same size, this is not possible since
    # each image has variable number of objects
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=lambda b: tuple(zip(*b))) 

    losses = []
    lowest_loss = math.inf    
    for epoch in tqdm(range(epochs)):
        for imgs, targets in dataloader:
            # converts tensor batch of images to an iterable list of images
            imgs = list(image.to(device) for image in imgs)

            # converts to a list of objects {boxes, lables}
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = piece_detector(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses += [loss.item()]
            if weights_save_folder != None and loss.item() < lowest_loss:
                lowest_loss = loss
                torch.save(piece_detector.state_dict(), weights_save_path)

    plt.plot(losses)
    plt.show()
    plt.savefig(loss_save_path)
