# installed imports
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch

# default imports
import math

# local imports
from models import PieceDetector
from dataloader import PieceDetectorDataset

def train_piece_detector(weights_load_path=None, weights_save_folder=None, weights_name="weight",
                         batch_size=2,
                         learning_rate=0.01, 
                         weight_decay=1e-4,
                         epochs=10, 
                         from_pretrained=True, 
                         mixed_precision_training=False, 
                         device='cpu'):
    """
    This function trains the Piece Detector
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
        None, displays losses at the end
    """

    piece_detector = PieceDetector(pretrained=from_pretrained).to(device)

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

    optim = torch.optim.SGD(piece_detector.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    dataset = PieceDetectorDataset(json_file='dataloader/data/piece_data/train/_annotations.coco.json')
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None
    
    # the collate_fn function is needed to create batches without using default stack mehtod,
    # the default collate_fn requires all boxes to be of same size, this is not possible since
    # each image has variable number of objects
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=lambda b: tuple(zip(*b))) 

    losses = []
    lowest_loss = math.inf    
    for epoch in tqdm(range(epochs)):
        for imgs, targets in dataloader:
            # converts tensor batch of images to an iterable list of images
            imgs = list(image.to(device) for image in imgs)

            # converts to a list of objects {boxes, lables}
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                loss_dict = piece_detector(imgs, targets)
                loss = sum(loss for loss in loss_dict.values())

            # backward
            optim.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            losses += [loss.item()]
            if weights_save_folder != None and loss.item() < lowest_loss:
                lowest_loss = loss
                torch.save(piece_detector.state_dict(), weights_save_path)

    print(f"last loss: {losses[-1]}")
    print(f"lowest loss: {lowest_loss}")
    plt.plot(losses)
    plt.savefig(loss_save_path)
    plt.show()
