# installed imports
from torch.utils.tensorboard import SummaryWriter
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


def train_board_detector(model='densenet',
                         load=None,
                         save=None,
                         json_file='dataloader/data/board_data/board_dataset_coco.json',
                         root_folder='dataloader/data/raw',
                         batch_size=64, 
                         learning_rate=3e-4, 
                         epochs=30, 
                         from_pretrained=True,
                         mixed_precision_training=True,
                         writer=None,
                         step=0,
                         device='cpu'):

    """
    This function trains the Board Detector
    Args: 
        (str) weights_load_path: path to the Board Detector model weights that needs to be loaded
        (str) weights_save_folder: folder where the Board Detector weights will be saved
        (str) weights_name: name for the weights that will be saved
        (int) batch_size: size of each batch trained at once
        (float) learning_rate: lr of the model
        (int) epochs: # of iterations through dataset
        (bool) from_pretrained: should the Board Detector be loader with pretrained weights
        (bool) mixed_precision_training: reduces the float32 to float16, checkout pytorch documentation for more details

        in function args:
        (str) loss_save_path: path where losses plot fig (losses.jpg) will be stored

    Returns:
        last loss and lowest loss
    """

    board_detector = BoardDetector(pretrained=from_pretrained, model=model, target='points').to(device)
    if load != None:
        board_detector.load_state_dict(torch.load(load))
        print('loaded weights for board detector!')
    else:
        print('no weights are loaded for board detector')

    optim = torch.optim.Adam(board_detector.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.1, verbose=True)

    board_dataset = BoardDetectorDataset(root=root_folder, json_file=json_file, target='points')
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    dataloader = DataLoader(board_dataset, batch_size=batch_size, shuffle=True)
    criterion = board_detector.loss_function()

    losses = []
    lowest_loss = math.inf
    for epoch in tqdm(range(epochs)):
        for data in dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)

            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                out = board_detector(x)
                loss = criterion(out, y)

            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            if writer is not None:
                writer.add_scalar(f'MSE Loss b:{batch_size}, lr: {learning_rate}', loss, global_step=step)
                step += 1

            losses += [loss.item()]
            if save is not None and loss.item() < lowest_loss:
                lowest_loss = loss.item()
                torch.save(board_detector.state_dict(), save)

        scheduler.step(sum(losses) / len(losses))

    return losses[-1], lowest_loss