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
from dataloader import BoardDetectorDataset

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
        print('loaded weights!')
    else:
        print('no weights are loaded')

    if weights_save_folder != None:
        print('weights will be saved at' + weights_save_path)
    else:
        print('no weights will be saved')

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
            if(weights_save_folder != None and loss.item() < lowest_loss):
                lowest_loss = loss.item()
                torch.save(board_detector.state_dict(), weights_save_path)

    plt.plot(losses)
    plt.show()
    plt.savefig(loss_save_path)


# TODO: write code to train piece detector
