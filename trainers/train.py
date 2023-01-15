import torch
import torch.nn as nn
from models import BoardDetector # Bd is short for board detection
from dataloader import BoardDetectorDataset # Bd (Board Detection)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_board_detector(training_version, load_version=None, 
                         batch_size=64, learning_rate=3e-4, epochs=30):

    """
    This function trains the Board Detector
    Args: 
        (int) training_version: saves in checkpoints/board_detector/{training_version}
        (int) load_version: if not None then, pre-loads from checkpoints/board_detector/{load_version}
        (int) batch_size: size of each batch trained at once
        (float) learning_rate: lr of the model
        (int) epochs: # of iterations through dataset

    Returns:
        None, displays losses at the end
    """

    model_save_path = f"./checkpoint/{training_version}/model"
    loss_save_path = f"./checkpoint/{training_version}/losses.jpg"

    board_detector = BoardDetector().to(device)
    if load_version != None:
        board_detector.load_state_dict(torch.load(f"./checkpoint/{load_version}/model")) 
        print('loaded model!')

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
            if(loss.item() < lowest_loss):
                lowest_loss = loss.item()
                torch.save(board_detector.state_dict(), model_save_path)

    plt.plot(losses)
    plt.show()
    plt.savefig(loss_save_path)


# TODO: write code to train piece detector
